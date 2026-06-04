"""Online nonlinear identification experiment runners.

Parity-preserving extraction from the baseline notebook.
"""

from __future__ import annotations

import contextlib
import copy
import io
import time
from typing import Any

import numpy as np
import pandas as pd

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from opinion_dynamics.identify_nonlinear import (
    GraphIdentifierEnv,
    pairs_from_intermediate,
    train_graph_identifier,
)
from opinion_dynamics.utils.env_setup import EnvironmentFactory
from opinion_dynamics.experiments.metrics import state_distance_to_oracle
from opinion_dynamics.experiments.rollouts import (
    _fresh_env_from_template,
    sample_init_opinions,
    rollout_with_model_derived_control_intermediate,
    rollout_with_uniform_intermediate,
    rollout_with_v_intermediate,
    make_env_with_dynamics,
)

# These are notebook-configurable globals. The modularized notebook sets them
# immediately after defining the config cell, preserving the original notebook behavior.
EVAL_ZERO_FIRST_CAMPAIGN = False
FIT_BY_DYNAMICS = {}
IDENTIFIER_KW = {}


def run_repeated_nocontrol_singlecampaign_id_on_env(
    env_template,
    *,
    num_repeats=100,
    init_mode="uniform",
    init_low=0.01,
    init_high=0.99,
    init_seed_base: int = 12345,
    learn_num_campaigns=1,
    eval_num_campaigns=5,
    B_campaign=1.0,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    suppress_fit_logs=True,
    fit_max_steps=50_000,
    fit_mae_stop=1e-3,
    fit_batch_size=64,
    fit_check_every=200,
    identifier_kwargs=None,
    repeat_seed_stride: int = 10_000,
    eval_seed_offset: int = 999_999,
):
    """
    Learning phase:
      - same graph topology
      - fresh random x0 each repeat
      - NO CONTROL only
      - collect data from learn_num_campaigns (default: 1) campaign rollouts
      - accumulate transition pairs and refit after each repeat

    Evaluation phase:
      - fixed evaluation x0
      - run eval_num_campaigns with:
          * learned control (using v_hat from A_hat)
          * oracle control (using v_true)
          * no control
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
    train_x0s: list[np.ndarray] = []

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

    eval_rng = np.random.default_rng(base_seed + eval_seed_offset)
    eval_x0 = sample_init_opinions(
        N,
        eval_rng,
        mode=init_mode,
        low=init_low,
        high=init_high,
    )

    for repeat_idx in range(int(num_repeats)):
        rng = np.random.default_rng(init_seed_base + repeat_idx)
        x0_train = sample_init_opinions(
            N,
            rng,
            mode=init_mode,
            low=init_low,
            high=init_high,
        )
        train_x0s.append(np.array(x0_train, copy=True))

        repeat_seed = base_seed + repeat_seed_stride * (repeat_idx + 1)
        env, env_kwargs = _fresh_env_from_template(
            env_template,
            repeat_seed=repeat_seed,
            initial_opinions=x0_train,
        )

        states_train = []
        actions_train = []
        rewards_train = []
        intermediate_states_list = []
        intermediate_times_list = []

        timing = {
            "fit_time": 0.0,
            "step_time": 0.0,
            "fit_calls": 0,
            "step_calls": 0,
        }

        x, _ = env.reset()
        states_train.append(np.array(x, copy=True))

        def _record_intermediate(info_obj):
            inter = info_obj.get("intermediate_states", None)
            if inter is None:
                intermediate_states_list.append(None)
                intermediate_times_list.append(None)
                return
            inter_arr = np.asarray(inter, dtype=float)
            dt = getattr(env, "t_s", None)
            t = (
                np.arange(inter_arr.shape[0], dtype=float)
                if dt is None
                else dt * np.arange(inter_arr.shape[0], dtype=float)
            )
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

        for _k in range(int(learn_num_campaigns)):
            u0 = np.zeros(N, dtype=float)
            x_next, r, done, trunc, info_k = _step(u0)

            actions_train.append(u0.copy())
            rewards_train.append(float(r))
            states_train.append(np.array(x_next, copy=True))
            _record_intermediate(info_k)

            inter = info_k.get("intermediate_states", None)
            if inter is None:
                raise RuntimeError(
                    "env.step did not return info['intermediate_states']; "
                    "cannot build transition pairs."
                )

            Xp, Yp = pairs_from_intermediate(inter)
            buf_x.append(Xp)
            buf_y.append(Yp)

            if done or trunc:
                break

        A_hat, v_hat = _fit()

        states_train_arr = np.asarray(states_train, dtype=float)
        A_hat_final = np.asarray(A_hat, dtype=float)
        v_hat_final = np.asarray(v_hat, dtype=float)

        env_eval_learn, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset,
            initial_opinions=eval_x0,
        )
        env_eval_oracle, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset,
            initial_opinions=eval_x0,
        )
        env_eval_noc, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset,
            initial_opinions=eval_x0,
        )
        env_eval_uni, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset,
            initial_opinions=eval_x0,
        )

        learn_eval = rollout_with_model_derived_control_intermediate(
            env_eval_learn,
            gi,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            device=device,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )
        states_learn = learn_eval["states"]

        oracle_eval = rollout_with_v_intermediate(
            env_eval_oracle,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            v_true,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )
        noc_eval = rollout_with_v_intermediate(
            env_eval_noc,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            None,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )
        uniform_eval = rollout_with_uniform_intermediate(
            env_eval_uni,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )

        states_oracle = oracle_eval["states"]
        states_noc = noc_eval["states"]
        states_uniform = uniform_eval["states"]

        T = min(states_learn.shape[0], states_oracle.shape[0], states_noc.shape[0], states_uniform.shape[0])
        sl = np.asarray(states_learn[:T], dtype=float)
        so = np.asarray(states_oracle[:T], dtype=float)
        sn = np.asarray(states_noc[:T], dtype=float)
        su = np.asarray(states_uniform[:T], dtype=float)

        mean_le = sl.mean(axis=1)
        mean_or = so.mean(axis=1)
        mean_nc = sn.mean(axis=1)
        mean_uni = su.mean(axis=1)
        vx_le = sl @ v_true
        vx_or = so @ v_true

        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)
        state_oracle_metrics = state_distance_to_oracle(
            sl,
            so,
            states_uniform=su,
            states_nocontrol=sn,
        )
        last_fit_info = getattr(gi, "last_fit_info", {}) or {}

        row = dict(
            seed=int(base_seed),
            repeat=int(repeat_idx),
            repeat_seed=int(repeat_seed),
            dynamics=str(getattr(env, "dynamics_model", "laplacian")),
            init_mode=str(init_mode),
            learn_num_campaigns=int(learn_num_campaigns),
            eval_num_campaigns=int(eval_num_campaigns),
            N=int(N),
            train_pairs_total=int(sum(x.shape[0] for x in buf_x)),
            v_L1_final=float(np.sum(np.abs(v_hat_final - v_true))),
            A_Fro_final=float(np.linalg.norm(A_hat_final - A_true, ord="fro")),
            A_MAE_final=float(np.mean(np.abs(A_hat_final - A_true))),
            mean_oracle_end=float(mean_or[-1]),
            mean_learn_end=float(mean_le[-1]),
            mean_noc_end=float(mean_nc[-1]),
            mean_uniform_end=float(mean_uni[-1]),
            mean_gap_to_oracle_end=float(np.abs(mean_le[-1] - mean_or[-1])),
            mean_gain_vs_noc_end=float(mean_le[-1] - mean_nc[-1]),
            mean_gain_vs_uniform_end=float(mean_le[-1] - mean_uni[-1]),
            mean_err_avg=float(mean_err.mean()),
            mean_err_max=float(mean_err.max()),
            vx_gap_to_oracle_end=float(np.abs(vx_le[-1] - vx_or[-1])),
            vx_err_avg=float(vx_err.mean()),
            vx_err_max=float(vx_err.max()),
            **state_oracle_metrics,
            fit_max_steps_used=int(fit_max_steps),
            fit_mae_stop_used=float(fit_mae_stop),
            fit_batch_size_used=int(fit_batch_size),
            fit_check_every_used=int(fit_check_every),
            fit_steps_run_last=int(last_fit_info.get("steps_run", -1)),
            fit_stop_reason_last=str(last_fit_info.get("stop_reason", "unknown")),
            fit_mae_last=float(last_fit_info.get("mae", np.nan)),
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
                x0_train=np.asarray(x0_train, dtype=float),
                x0_eval=np.asarray(eval_x0, dtype=float),
                states_nocontrol_train=states_train_arr,
                actions_nocontrol_train=np.asarray(actions_train, dtype=float),
                rewards_nocontrol_train=np.asarray(rewards_train, dtype=float),
                A_hat_final=A_hat_final,
                v_hat_final=v_hat_final,
                A_true=A_true,
                v_true=v_true,
                states_learn=np.asarray(states_learn, dtype=float),
                states_oracle=np.asarray(states_oracle, dtype=float),
                states_nocontrol_eval=np.asarray(states_noc, dtype=float),
                states_uniform_eval=np.asarray(states_uniform, dtype=float),
                actions_learn_eval=np.asarray(learn_eval["actions"], dtype=float),
                actions_oracle_eval=np.asarray(oracle_eval["actions"], dtype=float),
                actions_nocontrol_eval=np.asarray(noc_eval["actions"], dtype=float),
                actions_uniform_eval=np.asarray(uniform_eval["actions"], dtype=float),
                eval_zero_first_campaign=bool(EVAL_ZERO_FIRST_CAMPAIGN),
                learned_effective_adjacencies=[
                    np.asarray(Ae, dtype=float) for Ae in learn_eval["effective_adjacencies"]
                ],
                learned_effective_centralities=[
                    np.asarray(ve, dtype=float) for ve in learn_eval["effective_centralities"]
                ],
                learned_intermediate_states_list=learn_eval["intermediate_states_list"],
                learned_intermediate_times_list=learn_eval["intermediate_times_list"],
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
        "x0_eval": np.asarray(eval_x0, dtype=float),
        "train_x0s": np.stack(train_x0s, axis=0),
        "model_final": copy.deepcopy(gi).cpu(),
    }


def run_multi_seed_nocontrol_singlecampaign_experiment_dynamics(
    *,
    seeds=range(10),
    repeats_per_seed=100,
    dynamics_model="laplacian",
    init_mode="uniform",
    learn_num_campaigns=1,
    eval_num_campaigns=5,
    B_campaign=1.0,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
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
        env_template = make_env_with_dynamics(
            env_factory,
            seed=int(seed),
            dynamics_model=dynamics_model,
        )
        out = run_repeated_nocontrol_singlecampaign_id_on_env(
            env_template,
            num_repeats=repeats_per_seed,
            init_mode=init_mode,
            learn_num_campaigns=learn_num_campaigns,
            eval_num_campaigns=eval_num_campaigns,
            B_campaign=B_campaign,
            lr=lr,
            l2_lambda=l2_lambda,
            device=device,
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
            artifacts_by_seed[int(seed)] = {
                "rows": out["rows"],
                "final_row": out["rows"][-1],
                "final_artifact": out["artifacts"][-1],
                "train_x0s": out["train_x0s"],
                "x0_eval": out["x0_eval"],
                "model_final": out["model_final"],
            }

    df = pd.DataFrame(rows).sort_values(["seed", "repeat"]).reset_index(drop=True)
    if return_artifacts:
        return df, artifacts_by_seed
    return df


def make_validation_x0_set(
    N: int,
    *,
    n_validation: int,
    seed: int,
    init_mode: str = "uniform",
    low: float = 0.01,
    high: float = 0.99,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.stack(
        [sample_init_opinions(N, rng, mode=init_mode, low=low, high=high) for _ in range(int(n_validation))],
        axis=0,
    )


def evaluate_model_on_validation_x0s(
    env_template,
    model,
    validation_x0s: np.ndarray,
    *,
    B_campaign: float,
    eval_num_campaigns: int,
    device: str = "cpu",
    eval_seed_offset: int = 999_999,
) -> pd.DataFrame:
    """
    Evaluate learned/oracle/uniform/no-control on several held-out x0s.
    Returns one row per validation x0.
    """
    N = int(env_template.num_agents)
    A_true = np.asarray(env_template.connectivity_matrix, dtype=float)
    v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
    base_seed = int(getattr(env_template, "seed", 0) or 0)

    rows = []
    for val_idx, x0 in enumerate(np.asarray(validation_x0s, dtype=float)):
        env_learn, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset + int(val_idx),
            initial_opinions=x0,
        )
        env_or, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset + int(val_idx),
            initial_opinions=x0,
        )
        env_nc, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset + int(val_idx),
            initial_opinions=x0,
        )
        env_uni, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=base_seed + eval_seed_offset + int(val_idx),
            initial_opinions=x0,
        )

        le = rollout_with_model_derived_control_intermediate(
            env_learn,
            model,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            device=device,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )
        oracle = rollout_with_v_intermediate(
            env_or,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            v_true,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )
        noc = rollout_with_v_intermediate(
            env_nc,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            None,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )
        uni = rollout_with_uniform_intermediate(
            env_uni,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            zero_first_campaign=EVAL_ZERO_FIRST_CAMPAIGN,
        )

        sl = np.asarray(le["states"], dtype=float)
        so = np.asarray(oracle["states"], dtype=float)
        sn = np.asarray(noc["states"], dtype=float)
        su = np.asarray(uni["states"], dtype=float)
        T = min(sl.shape[0], so.shape[0], sn.shape[0], su.shape[0])
        sl, so, sn, su = sl[:T], so[:T], sn[:T], su[:T]

        mean_le = sl.mean(axis=1)
        mean_or = so.mean(axis=1)
        mean_nc = sn.mean(axis=1)
        mean_uni = su.mean(axis=1)
        vx_le = sl @ v_true
        vx_or = so @ v_true
        state_oracle_metrics = state_distance_to_oracle(
            sl,
            so,
            states_uniform=su,
            states_nocontrol=sn,
        )

        rows.append(
            dict(
                validation_idx=int(val_idx),
                mean_learn_end=float(mean_le[-1]),
                mean_oracle_end=float(mean_or[-1]),
                mean_noc_end=float(mean_nc[-1]),
                mean_uniform_end=float(mean_uni[-1]),
                mean_gap_to_oracle_end=float(abs(mean_le[-1] - mean_or[-1])),
                mean_gain_vs_noc_end=float(mean_le[-1] - mean_nc[-1]),
                mean_gain_vs_uniform_end=float(mean_le[-1] - mean_uni[-1]),
                vx_gap_to_oracle_end=float(abs(vx_le[-1] - vx_or[-1])),
                mean_err_avg=float(np.abs(mean_le - mean_or).mean()),
                mean_err_max=float(np.abs(mean_le - mean_or).max()),
                **state_oracle_metrics,
            )
        )

    return pd.DataFrame(rows)


def run_data_budget_sweep_on_env(
    env_template,
    *,
    trial_counts: list[int],
    n_validation_x0: int = 8,
    init_mode: str = "uniform",
    init_low: float = 0.01,
    init_high: float = 0.99,
    init_seed_base: int = 12345,
    validation_seed_offset: int = 555_555,
    learn_num_campaigns: int = 1,
    eval_num_campaigns: int = 5,
    B_campaign: float = 1.0,
    lr: float = 1e-3,
    l2_lambda: float = 0.0,
    device: str = "cpu",
    suppress_fit_logs: bool = True,
    fit_max_steps: int = 2_000,
    fit_mae_stop: float = 2e-2,
    fit_batch_size: int = 256,
    fit_check_every: int = 1_000,
    identifier_kwargs: dict | None = None,
    repeat_seed_stride: int = 10_000,
) -> pd.DataFrame:
    """
    Train cumulatively and evaluate only at selected trial budgets on multiple validation x0s.
    """
    trial_counts = sorted({int(t) for t in trial_counts if int(t) > 0})
    max_trials = max(trial_counts)
    trial_set = set(trial_counts)

    N = int(env_template.num_agents)
    A_true = np.asarray(env_template.connectivity_matrix, dtype=float)
    v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
    base_seed = int(getattr(env_template, "seed", 0) or 0)

    validation_x0s = make_validation_x0_set(
        N,
        n_validation=n_validation_x0,
        seed=base_seed + validation_seed_offset,
        init_mode=init_mode,
        low=init_low,
        high=init_high,
    )

    gi = GraphIdentifierEnv(
        N=N,
        s=float(env_template.t_s),
        l2_lambda=l2_lambda,
        zero_diag=True,
        **({} if identifier_kwargs is None else identifier_kwargs),
    )

    buf_x, buf_y = [], []
    rows = []
    A_hat = None

    def fit_current_model():
        X = np.concatenate(buf_x, axis=0)
        Y = np.concatenate(buf_y, axis=0)
        if suppress_fit_logs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                return train_graph_identifier(
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
        return train_graph_identifier(
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

    for trial_idx in range(max_trials):
        rng = np.random.default_rng(init_seed_base + trial_idx)
        x0_train = sample_init_opinions(N, rng, mode=init_mode, low=init_low, high=init_high)
        repeat_seed = base_seed + repeat_seed_stride * (trial_idx + 1)
        env, _ = _fresh_env_from_template(
            env_template,
            repeat_seed=repeat_seed,
            initial_opinions=x0_train,
        )
        env.reset()

        for _k in range(int(learn_num_campaigns)):
            u0 = np.zeros(N, dtype=float)
            _x_next, _r, done, trunc, info_k = env.step(u0)
            inter = info_k.get("intermediate_states", None)
            if inter is None:
                raise RuntimeError("env.step did not return info['intermediate_states']")
            Xp, Yp = pairs_from_intermediate(inter)
            buf_x.append(Xp)
            buf_y.append(Yp)
            if done or trunc:
                break

        trial_count = trial_idx + 1
        if trial_count not in trial_set:
            continue

        t0 = time.perf_counter()
        A_hat = fit_current_model()
        fit_time = time.perf_counter() - t0
        v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))

        val_rows = evaluate_model_on_validation_x0s(
            env_template,
            gi,
            validation_x0s,
            B_campaign=B_campaign,
            eval_num_campaigns=eval_num_campaigns,
            device=device,
        )
        val_mean = val_rows.mean(numeric_only=True).to_dict()
        val_std = val_rows.std(numeric_only=True).add_suffix("_std").to_dict()

        rows.append(
            dict(
                seed=int(base_seed),
                dynamics=str(getattr(env_template, "dynamics_model", "laplacian")),
                trial_count=int(trial_count),
                repeat=int(trial_count - 1),
                train_pairs_total=int(sum(x.shape[0] for x in buf_x)),
                n_validation_x0=int(n_validation_x0),
                A_MAE_final=float(np.mean(np.abs(np.asarray(A_hat, dtype=float) - A_true))),
                v_L1_final=float(np.sum(np.abs(np.asarray(v_hat, dtype=float) - v_true))),
                fit_time_inner=float(fit_time),
                fit_max_steps_used=int(fit_max_steps),
                fit_mae_stop_used=float(fit_mae_stop),
                fit_batch_size_used=int(fit_batch_size),
                fit_check_every_used=int(fit_check_every),
                **{k: float(v) for k, v in val_mean.items()},
                **{k: float(v) for k, v in val_std.items()},
            )
        )

    return pd.DataFrame(rows)


def run_multi_seed_data_budget_sweep(
    *,
    seeds,
    dynamics_list,
    trial_counts,
    n_validation_x0,
    init_mode,
    learn_num_campaigns,
    eval_num_campaigns,
    B_campaign,
    device="cpu",
):
    env_factory = EnvironmentFactory()
    rows = []
    for dyn in dynamics_list:
        for seed in seeds:
            print(f"=== data-budget sweep | {dyn} | seed={seed} ===")
            env_template = make_env_with_dynamics(env_factory, seed=int(seed), dynamics_model=dyn)
            fit_cfg = FIT_BY_DYNAMICS[dyn]
            df_budget = run_data_budget_sweep_on_env(
                env_template,
                trial_counts=trial_counts,
                n_validation_x0=n_validation_x0,
                init_mode=init_mode,
                learn_num_campaigns=learn_num_campaigns,
                eval_num_campaigns=eval_num_campaigns,
                B_campaign=B_campaign,
                device=device,
                identifier_kwargs=IDENTIFIER_KW,
                **fit_cfg,
            )
            rows.append(df_budget)
    return pd.concat(rows, axis=0).reset_index(drop=True) if rows else pd.DataFrame()


