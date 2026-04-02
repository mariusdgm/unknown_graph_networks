import contextlib
import io
import time
from typing import Any

import numpy as np
import pandas as pd

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from ..identify_freeprop import (
    GraphIdentifierEnv,
    pairs_from_intermediate,
    train_graph_identifier,
)
from ..utils.env_setup import EnvironmentFactory
from ..baseline import centrality_based_continuous_control

from rollouts import rollout_with_v, make_env_with_dynamics



def _env_template_kwargs(env, fallback_seed: int | None = None) -> dict[str, Any]:
    return dict(
        connectivity_matrix=np.array(env.connectivity_matrix, copy=True),
        num_agents=int(env.num_agents),
        max_u=np.array(env.max_u, copy=True),
        desired_opinion=float(env.desired_opinion),
        t_campaign=float(env.t_campaign),
        t_s=float(env.t_s),
        dynamics_model=str(getattr(env, "dynamics_model", "laplacian")),
        control_resistance=np.array(getattr(env, "control_resistance", np.zeros(env.num_agents)), copy=True),
        max_steps=int(getattr(env, "max_steps", 10_000)),
        opinion_end_tolerance=float(getattr(env, "opinion_end_tolerance", 0.01)),
        control_beta=float(getattr(env, "control_beta", 0.4)),
        normalize_reward=bool(getattr(env, "normalize_reward", False)),
        terminal_reward=float(getattr(env, "terminal_reward", 0.0)),
        terminate_when_converged=bool(getattr(env, "terminate_when_converged", True)),
        seed=int(getattr(env, "seed", fallback_seed)) if getattr(env, "seed", None) is not None or fallback_seed is not None else None,
    )



def _fresh_env_from_template(env_template, repeat_seed: int | None):
    EnvCls = env_template.__class__
    kwargs = _env_template_kwargs(env_template, fallback_seed=repeat_seed)
    kwargs["seed"] = repeat_seed
    return EnvCls(**kwargs), kwargs



def run_repeated_online_id_on_env(
    env_template,
    *,
    num_repeats=10,
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,
    fit_max_steps=50_000,
    fit_mae_stop=1e-3,
    fit_batch_size=64,
    fit_check_every=200,
    identifier_kwargs=None,
    repeat_seed_stride: int = 10_000,
):
    """
    Keep the same graph topology across repeats, but rebuild a fresh env each repeat
    with a different seed so reset()/stochastic initial conditions can vary.

    Crucially, the graph learner and the accumulated transition buffer persist across repeats.
    """
    N = int(env_template.num_agents)
    A_true = np.asarray(env_template.connectivity_matrix, dtype=float)
    v_true = compute_eigenvector_centrality(compute_laplacian(A_true))

    base_seed = int(getattr(env_template, "seed", 0) or 0)

    gi = GraphIdentifierEnv(
        N=N,
        s=float(env_template.t_s),
        l2_lambda=l2_lambda,
        zero_diag=True,
        **({} if identifier_kwargs is None else identifier_kwargs),
    )

    buf_x: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    repeat_rows: list[dict[str, Any]] = []
    repeat_artifacts: list[dict[str, Any]] = []

    A_hat = None
    v_hat = None

    def _fit_current():
        X = np.concatenate(buf_x, axis=0)
        Y = np.concatenate(buf_y, axis=0)
        if suppress_fit_logs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                A_local = train_graph_identifier(
                    gi,
                    X,
                    Y,
                    lr=lr,
                    batch_size=fit_batch_size,
                    max_steps=fit_max_steps,
                    mae_stop=fit_mae_stop,
                    device=device,
                    fit_check_every=fit_check_every,
                )
        else:
            A_local = train_graph_identifier(
                gi,
                X,
                Y,
                lr=lr,
                batch_size=fit_batch_size,
                max_steps=fit_max_steps,
                mae_stop=fit_mae_stop,
                device=device,
                fit_check_every=fit_check_every,
            )
        return A_local

    for repeat_idx in range(int(num_repeats)):
        repeat_seed = base_seed + repeat_seed_stride * (repeat_idx + 1)
        env, env_kwargs = _fresh_env_from_template(env_template, repeat_seed=repeat_seed)
        ubar_vec = np.asarray(env.max_u, dtype=float)

        states, actions, rewards = [], [], []
        intermediate_states_list = []
        intermediate_times_list = []
        A_hats = []
        v_hats = []

        timing = {
            "fit_time": 0.0,
            "step_time": 0.0,
            "fit_calls": 0,
            "step_calls": 0,
        }

        x, _ = env.reset()
        states.append(np.array(x, copy=True))

        def _record_intermediate(info_obj):
            inter = info_obj.get("intermediate_states", None)
            if inter is None:
                intermediate_states_list.append(None)
                intermediate_times_list.append(None)
                return
            inter_arr = np.asarray(inter, dtype=float)
            dt = getattr(env, "t_s", None)
            t = np.arange(inter_arr.shape[0], dtype=float) if dt is None else dt * np.arange(inter_arr.shape[0], dtype=float)
            intermediate_states_list.append(inter_arr)
            intermediate_times_list.append(t)

        def _step(u):
            t0 = time.perf_counter()
            out = env.step(u)
            timing["step_time"] += time.perf_counter() - t0
            timing["step_calls"] += 1
            return out

        def _fit():
            nonlocal A_hat, v_hat
            t0 = time.perf_counter()
            A_hat = _fit_current()
            v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))
            timing["fit_time"] += time.perf_counter() - t0
            timing["fit_calls"] += 1
            return A_hat, v_hat

        # campaign 0: zero control and accumulate data
        u0 = np.zeros(N, dtype=float)
        x1, r0, done, trunc, info0 = _step(u0)
        actions.append(u0.copy())
        rewards.append(float(r0))
        states.append(np.array(x1, copy=True))
        _record_intermediate(info0)

        inter0 = info0.get("intermediate_states", None)
        if inter0 is None:
            raise RuntimeError("env.step did not return info['intermediate_states']")
        Xp, Yp = pairs_from_intermediate(inter0)
        buf_x.append(Xp)
        buf_y.append(Yp)

        A_hat, v_hat = _fit()
        A_hats.append(np.array(A_hat, copy=True))
        v_hats.append(np.array(v_hat, copy=True))

        for _k in range(1, int(num_campaigns_total)):
            if done or trunc:
                break

            beta_k = min(float(B_campaign), float(ubar_vec.sum()))
            uk, _ = centrality_based_continuous_control(env, beta_k, v=v_hat)
            x_next, r, done, trunc, info_k = _step(uk)
            actions.append(np.array(uk, copy=True))
            rewards.append(float(r))
            states.append(np.array(x_next, copy=True))
            _record_intermediate(info_k)

            inter = info_k.get("intermediate_states", None)
            if inter is not None:
                Xp, Yp = pairs_from_intermediate(inter)
                buf_x.append(Xp)
                buf_y.append(Yp)

            if update_A_each_campaign:
                A_hat, v_hat = _fit()
                A_hats.append(np.array(A_hat, copy=True))
                v_hats.append(np.array(v_hat, copy=True))

        states_arr = np.asarray(states, dtype=float)
        A_hat_final = np.asarray(A_hats[-1], dtype=float)
        v_hat_final = compute_eigenvector_centrality(compute_laplacian(A_hat_final))

        x0 = states_arr[0].copy()
        K_total = states_arr.shape[0] - 1
        states_or = rollout_with_v(env, x0, K_total, B_campaign, v_true)
        states_nc = rollout_with_v(env, x0, K_total, B_campaign, None)

        T = min(states_arr.shape[0], states_or.shape[0], states_nc.shape[0])
        sl = states_arr[:T]
        so = np.asarray(states_or[:T], dtype=float)
        sn = np.asarray(states_nc[:T], dtype=float)

        mean_le = sl.mean(axis=1)
        mean_or = so.mean(axis=1)
        mean_nc = sn.mean(axis=1)
        vx_le = sl @ v_true
        vx_or = so @ v_true

        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)

        row = dict(
            seed=int(base_seed),
            repeat=int(repeat_idx),
            repeat_seed=int(repeat_seed),
            dynamics=str(getattr(env, "dynamics_model", "laplacian")),
            N=int(N),
            train_pairs_total=int(sum(x.shape[0] for x in buf_x)),
            v_L1_final=float(np.sum(np.abs(v_hat_final - v_true))),
            A_Fro_final=float(np.linalg.norm(A_hat_final - A_true, ord="fro")),
            A_MAE_final=float(np.mean(np.abs(A_hat_final - A_true))),
            mean_oracle_end=float(mean_or[-1]),
            mean_learn_end=float(mean_le[-1]),
            mean_noc_end=float(mean_nc[-1]),
            mean_gap_to_oracle_end=float(np.abs(mean_le[-1] - mean_or[-1])),
            mean_err_avg=float(mean_err.mean()),
            mean_err_max=float(mean_err.max()),
            vx_gap_to_oracle_end=float(np.abs(vx_le[-1] - vx_or[-1])),
            vx_err_avg=float(vx_err.mean()),
            vx_err_max=float(vx_err.max()),
            time_fit_inner=float(timing["fit_time"]),
            time_step_inner=float(timing["step_time"]),
            fit_calls_inner=int(timing["fit_calls"]),
            step_calls_inner=int(timing["step_calls"]),
        )
        repeat_rows.append(row)
        repeat_artifacts.append(
            dict(
                env=env,
                env_template_kwargs=env_kwargs,
                states=states_arr,
                actions=np.asarray(actions, dtype=float),
                rewards=np.asarray(rewards, dtype=float),
                A_hats=[np.asarray(A, dtype=float) for A in A_hats],
                v_hats=[np.asarray(v, dtype=float) for v in v_hats],
                A_hat_final=A_hat_final,
                v_hat_final=v_hat_final,
                A_true=A_true,
                v_true=v_true,
                states_oracle=np.asarray(states_or, dtype=float),
                states_nocontrol=np.asarray(states_nc, dtype=float),
                intermediate_states_list=intermediate_states_list,
                intermediate_times_list=intermediate_times_list,
                train_pairs_total=row["train_pairs_total"],
                timing=timing,
            )
        )

    return {
        "rows": repeat_rows,
        "artifacts": repeat_artifacts,
        "A_true": A_true,
        "v_true": v_true,
        "A_hat_final": np.asarray(A_hat, dtype=float) if A_hat is not None else None,
        "v_hat_final": np.asarray(v_hat, dtype=float) if v_hat is not None else None,
        "train_pairs_total": int(sum(x.shape[0] for x in buf_x)),
    }



def run_multi_seed_repeated_experiment_dynamics(
    *,
    seeds=range(10),
    repeats_per_seed=10,
    dynamics_model="laplacian",
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,
    fit_max_steps=None,
    fit_mae_stop=None,
    fit_batch_size=None,
    fit_check_every=None,
    identifier_kwargs: dict | None = None,
    repeat_seed_stride: int = 10_000,
    return_artifacts: bool = False,
):
    if fit_max_steps is None:
        fit_max_steps = 50_000 if dynamics_model == "laplacian" else 2_000
    if fit_mae_stop is None:
        fit_mae_stop = 1e-3 if dynamics_model == "laplacian" else 5e-3
    if fit_batch_size is None:
        fit_batch_size = 64 if dynamics_model == "laplacian" else 256
    if fit_check_every is None:
        fit_check_every = 200

    env_factory = EnvironmentFactory()
    rows = []
    artifacts_by_seed = {} if return_artifacts else None

    for seed in seeds:
        env_template = make_env_with_dynamics(env_factory, seed=int(seed), dynamics_model=dynamics_model)
        out = run_repeated_online_id_on_env(
            env_template,
            num_repeats=repeats_per_seed,
            B_campaign=B_campaign,
            num_campaigns_total=num_campaigns_total,
            lr=lr,
            l2_lambda=l2_lambda,
            device=device,
            update_A_each_campaign=update_A_each_campaign,
            suppress_fit_logs=suppress_fit_logs,
            fit_max_steps=fit_max_steps,
            fit_mae_stop=fit_mae_stop,
            fit_batch_size=fit_batch_size,
            fit_check_every=fit_check_every,
            identifier_kwargs=identifier_kwargs,
            repeat_seed_stride=repeat_seed_stride,
        )
        rows.extend(out["rows"])
        if return_artifacts:
            artifacts_by_seed[int(seed)] = out

    df = pd.DataFrame(rows).sort_values(["seed", "repeat"]).reset_index(drop=True)
    if return_artifacts:
        return df, artifacts_by_seed
    return df
