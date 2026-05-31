import contextlib
import io
import time

import numpy as np
import pandas as pd

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from ..identify_nonlinear import (
    GraphIdentifierEnv,
    pairs_from_intermediate,
    train_graph_identifier,
)

from ..utils.env_setup import EnvironmentFactory
from ..baseline import centrality_based_continuous_control

from .rollouts import rollout_with_v, make_env_with_dynamics


def run_single_seed_experiment(
    *,
    seed: int,
    dynamics_model: str = "laplacian",   # NEW
    B_campaign: float = 1.0,
    num_campaigns_total: int = 5,
    lr: float = 1e-3,
    l2_lambda: float = 0.0,
    device: str = "cpu",
    update_A_each_campaign: bool = True,
    suppress_fit_logs: bool = True,
    return_artifacts: bool = True,

    # NEW fit-budget knobs
    fit_max_steps: int | None = None,
    fit_mae_stop: float | None = None,
    fit_batch_size: int | None = None,
    fit_check_every: int | None = None,  # kept for signature consistency
    identifier_kwargs: dict | None = None,
):
    import numpy as np

    from rl_envs_forge.envs.network_graph.graph_utils import (
        compute_laplacian,
        compute_eigenvector_centrality,
    )
    from ..utils.env_setup import EnvironmentFactory
    from .rollouts import make_env_with_dynamics, rollout_with_v

    # sensible defaults if caller didn't set them
    if fit_max_steps is None:
        fit_max_steps = 50_000 if dynamics_model == "laplacian" else 1_000
    if fit_mae_stop is None:
        fit_mae_stop = 1e-3 if dynamics_model == "laplacian" else 1e-2
    if fit_batch_size is None:
        fit_batch_size = 64 if dynamics_model == "laplacian" else 512
    if fit_check_every is None:
        fit_check_every = 200

    env_factory = EnvironmentFactory()
    env = make_env_with_dynamics(env_factory, seed=int(seed), dynamics_model=str(dynamics_model))

    out = run_single_paper_experiment_per_campaign_budget_on_env(
        env,
        B_campaign=B_campaign,
        num_campaigns_total=num_campaigns_total,
        lr=lr,
        l2_lambda=l2_lambda,
        device=device,
        update_A_each_campaign=update_A_each_campaign,
        suppress_fit_logs=suppress_fit_logs,

        # NEW
        fit_max_steps=fit_max_steps,
        fit_mae_stop=fit_mae_stop,
        fit_batch_size=fit_batch_size,
        fit_check_every=fit_check_every,
        identifier_kwargs=identifier_kwargs,
    )

    env0 = out["env"]
    env_kwargs = out.get("env_template_kwargs", None)

    states_learn = np.asarray(out["states"], dtype=float)
    actions = np.asarray(out["actions"], dtype=float)
    rewards = np.asarray(out["rewards"], dtype=float)
    A_hats = out["A_hats"]
    v_hats = out["v_hats"]

    A_hat_final = np.asarray(A_hats[-1], dtype=float)

    A_true = np.asarray(env0.connectivity_matrix, dtype=float)
    v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
    v_hat_final = compute_eigenvector_centrality(compute_laplacian(A_hat_final))

    # Rollouts on fresh env for clean campaign-0 overlap
    EnvCls = env0.__class__
    if env_kwargs is None:
        env_kwargs = dict(
            connectivity_matrix=np.array(env0.connectivity_matrix, copy=True),
            num_agents=int(env0.num_agents),
            max_u=np.array(env0.max_u, copy=True),
            desired_opinion=float(env0.desired_opinion),
            t_campaign=float(env0.t_campaign),
            t_s=float(env0.t_s),
            dynamics_model=str(getattr(env0, "dynamics_model", dynamics_model)),
            control_resistance=np.array(getattr(env0, "control_resistance", np.zeros(env0.num_agents)), copy=True),
            max_steps=int(getattr(env0, "max_steps", 10_000)),
            opinion_end_tolerance=float(getattr(env0, "opinion_end_tolerance", 0.01)),
            control_beta=float(getattr(env0, "control_beta", 0.4)),
            normalize_reward=bool(getattr(env0, "normalize_reward", False)),
            terminal_reward=float(getattr(env0, "terminal_reward", 0.0)),
            terminate_when_converged=bool(getattr(env0, "terminate_when_converged", True)),
            seed=int(getattr(env0, "seed", seed)) if getattr(env0, "seed", None) is not None else None,
        )
    env_base = EnvCls(**env_kwargs)

    x0 = states_learn[0].copy()
    K_total = states_learn.shape[0] - 1
    states_or = rollout_with_v(env_base, x0, K_total, B_campaign, v_true)  # rollout_with_v already zeros campaign0 in your version :contentReference[oaicite:1]{index=1}
    states_nc = rollout_with_v(env_base, x0, K_total, B_campaign, None)

    T = min(states_learn.shape[0], states_or.shape[0], states_nc.shape[0])
    sl = states_learn[:T]
    so = np.asarray(states_or[:T], dtype=float)
    sn = np.asarray(states_nc[:T], dtype=float)

    mean_le = sl.mean(axis=1)
    mean_or = so.mean(axis=1)
    mean_nc = sn.mean(axis=1)

    vx_le = sl @ v_true
    vx_or = so @ v_true

    mean_err = np.abs(mean_le - mean_or)
    vx_err = np.abs(vx_le - vx_or)

    metrics = dict(
        seed=int(seed),
        dynamics=str(dynamics_model),
        N=int(env0.num_agents),

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
    )

    if not return_artifacts:
        return metrics

    artifacts = dict(
        env=env0,
        env_template_kwargs=env_kwargs,
        dynamics=str(dynamics_model),

        A_true=A_true,
        v_true=v_true,
        x0=x0,

        A_hat_final=A_hat_final,
        v_hat_final=v_hat_final,
        A_hats=[np.asarray(A, dtype=float) for A in A_hats],
        v_hats=[np.asarray(v, dtype=float) for v in v_hats],

        states_learn=states_learn,
        states_oracle=np.asarray(states_or, dtype=float),
        states_nocontrol=np.asarray(states_nc, dtype=float),

        intermediate_states_list=out.get("intermediate_states_list", []),
        intermediate_times_list=out.get("intermediate_times_list", []),

        actions=actions,
        rewards=rewards,

        params=dict(
            seed=int(seed),
            dynamics_model=str(dynamics_model),
            B_campaign=float(B_campaign),
            num_campaigns_total=int(num_campaigns_total),
            lr=float(lr),
            l2_lambda=float(l2_lambda),
            device=str(device),
            update_A_each_campaign=bool(update_A_each_campaign),

            fit_max_steps=int(fit_max_steps),
            fit_mae_stop=float(fit_mae_stop),
            fit_batch_size=int(fit_batch_size),
            fit_check_every=int(fit_check_every),
        ),
    )

    return metrics, artifacts


def run_single_paper_experiment_per_campaign_budget_on_env(
    env,
    *,
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
):
    """
    Online experiment on a PROVIDED env instance.

    Returns:
      env
      states (K+1, N)
      actions (K, N)
      rewards (K,)
      A_hats (list of (N,N))
      v_hats (list of (N,))
      intermediate_states_list: list length K, each is (T_k, N)
      intermediate_times_list:  list length K, each is (T_k,)
      env_template_kwargs: kwargs that can recreate the same env
    """
    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    states, actions, rewards = [], [], []
    A_hats, v_hats = [], []
    buf_x, buf_y = [], []

    intermediate_states_list = []
    intermediate_times_list = []

    # helpful for “fresh env” reconstruction in single-seed plotting
    env_template_kwargs = dict(
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
        seed=int(getattr(env, "seed", 0)) if getattr(env, "seed", None) is not None else None,
    )

    x, _ = env.reset()
    states.append(x.copy())

    def _record_intermediate(info_obj):
        inter = info_obj.get("intermediate_states", None)
        if inter is None:
            intermediate_states_list.append(None)
            intermediate_times_list.append(None)
            return

        inter_arr = np.asarray(inter, dtype=float)
        dt = getattr(env, "t_s", None)
        if dt is None:
            t = np.arange(inter_arr.shape[0], dtype=float)
        else:
            t = dt * np.arange(inter_arr.shape[0], dtype=float)

        intermediate_states_list.append(inter_arr)
        intermediate_times_list.append(t)

    # -----------------------
    # Campaign 0: zero control
    # -----------------------
    u0 = np.zeros(N, dtype=float)
    x1, r0, done, trunc, info0 = env.step(u0)

    actions.append(u0.copy())
    rewards.append(float(r0))
    states.append(x1.copy())
    _record_intermediate(info0)

    inter0 = info0.get("intermediate_states", None)
    if inter0 is None:
        raise RuntimeError("env.step did not return info['intermediate_states']")
    Xp, Yp = pairs_from_intermediate(inter0)
    buf_x.append(Xp)
    buf_y.append(Yp)

    gi = GraphIdentifierEnv(
        N=N, s=env.t_s, l2_lambda=l2_lambda, zero_diag=True,
        **({} if identifier_kwargs is None else identifier_kwargs),
    )

    def _fit_current():
        X = np.concatenate(buf_x, axis=0)
        Y = np.concatenate(buf_y, axis=0)

        if suppress_fit_logs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                A_hat = train_graph_identifier(
                    gi, X, Y,
                    lr=lr,
                    batch_size=fit_batch_size,
                    max_steps=fit_max_steps,
                    mae_stop=fit_mae_stop,
                    device=device,
                    fit_check_every=fit_check_every,
                )
        else:
            A_hat = train_graph_identifier(
                gi, X, Y,
                lr=lr,
                batch_size=fit_batch_size,
                max_steps=fit_max_steps,
                mae_stop=fit_mae_stop,
                device=device,
                fit_check_every=fit_check_every,
            )
        return A_hat

    # initial fit after campaign 0
    A_hat = _fit_current()
    A_hats.append(A_hat)
    v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))
    v_hats.append(v_hat)

    # -----------------------
    # Campaigns 1..K-1
    # -----------------------
    for k in range(1, num_campaigns_total):
        if done or trunc:
            break

        beta_k = min(float(B_campaign), float(ubar_vec.sum()))
        uk, _ = centrality_based_continuous_control(env, beta_k, v=v_hat)

        x_next, r, done, trunc, info_k = env.step(uk)
        actions.append(uk.copy())
        rewards.append(float(r))
        states.append(x_next.copy())
        _record_intermediate(info_k)

        inter = info_k.get("intermediate_states", None)
        if inter is not None:
            Xp, Yp = pairs_from_intermediate(inter)
            buf_x.append(Xp)
            buf_y.append(Yp)

        if update_A_each_campaign:
            A_hat = _fit_current()
            v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))
            A_hats.append(A_hat)
            v_hats.append(v_hat)

    return {
        "env": env,
        "env_template_kwargs": env_template_kwargs,
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "A_hats": A_hats,
        "v_hats": v_hats,
        "intermediate_states_list": intermediate_states_list,
        "intermediate_times_list": intermediate_times_list,
    }


def run_multi_seed_experiment(
    *,
    seeds=range(10),
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,
):
    env_factory = EnvironmentFactory()
    rows = []

    for seed in seeds:
        env = env_factory.get_randomized_env(seed=int(seed))

        # run learned-online on THIS env
        out = run_single_paper_experiment_per_campaign_budget_on_env(
            env,
            B_campaign=B_campaign,
            num_campaigns_total=num_campaigns_total,
            lr=lr,
            l2_lambda=l2_lambda,
            device=device,
            update_A_each_campaign=update_A_each_campaign,
            suppress_fit_logs=suppress_fit_logs,
        )

        env0 = out["env"]
        states_learn = out["states"]
        A_hat_final = np.asarray(out["A_hats"][-1], dtype=float)

        # true A/v on this env
        A_true = np.asarray(env0.connectivity_matrix, dtype=float)
        v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
        v_hat_final = compute_eigenvector_centrality(compute_laplacian(A_hat_final))

        # oracle & no-control on same graph, same x0
        x0 = states_learn[0].copy()
        K_total = states_learn.shape[0] - 1
        states_or = rollout_with_v(env0, x0, K_total, B_campaign, v_true)
        states_nc = rollout_with_v(env0, x0, K_total, B_campaign, None)

        # trajectory metrics (boundary-level)
        T = min(states_learn.shape[0], states_or.shape[0], states_nc.shape[0])
        mean_le = states_learn[:T].mean(axis=1)
        mean_or = states_or[:T].mean(axis=1)
        mean_nc = states_nc[:T].mean(axis=1)

        vx_le = states_learn[:T] @ v_true
        vx_or = states_or[:T] @ v_true
        vx_nc = states_nc[:T] @ v_true

        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)

        rows.append(
            dict(
                seed=int(seed),
                N=int(env0.num_agents),
                # identification
                v_L1_final=float(np.sum(np.abs(v_hat_final - v_true))),
                A_Fro_final=float(np.linalg.norm(A_hat_final - A_true, ord="fro")),
                A_MAE_final=float(np.mean(np.abs(A_hat_final - A_true))),
                # trajectories
                mean_oracle_end=float(mean_or[-1]),
                mean_learn_end=float(mean_le[-1]),
                mean_noc_end=float(mean_nc[-1]),
                mean_gap_to_oracle_end=float(np.abs(mean_le[-1] - mean_or[-1])),
                mean_err_avg=float(mean_err.mean()),
                mean_err_max=float(mean_err.max()),
                vx_gap_to_oracle_end=float(np.abs(vx_le[-1] - vx_or[-1])),
                vx_err_avg=float(vx_err.mean()),
                vx_err_max=float(vx_err.max()),
            )
        )

    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def run_single_online_id_on_env(
    env,
    *,
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,
    # NEW (optional): make COCA runs sane + easier debugging
    fit_max_steps=50_000,
    fit_mae_stop=1e-3,
    fit_batch_size=64,
    identifier_kwargs=None,
):
    """
    Campaign0=zero control to collect data, then centrality control with learned v.
    Returns minimal stuff for metrics + inner timing counters.

    Adds:
      out["timing"] = {
         fit_time, step_time, fit_calls, step_calls
      }
    """
    import time
    import io
    import contextlib
    import numpy as np

    timing = {
        "fit_time": 0.0,
        "step_time": 0.0,
        "fit_calls": 0,
        "step_calls": 0,
    }

    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    states, actions, rewards = [], [], []
    A_hats = []
    buf_x, buf_y = [], []

    x, _ = env.reset()
    states.append(x.copy())

    # --- helper: timed env.step ---
    def _step(u):
        t0 = time.perf_counter()
        out = env.step(u)
        timing["step_time"] += (time.perf_counter() - t0)
        timing["step_calls"] += 1
        return out

    # --- Campaign 0: no control for learning ---
    u0 = np.zeros(N, dtype=float)
    x1, r0, done, trunc, info0 = _step(u0)
    actions.append(u0.copy())
    rewards.append(float(r0))
    states.append(x1.copy())

    inter0 = info0.get("intermediate_states", None)
    if inter0 is None:
        raise RuntimeError("env.step did not return info['intermediate_states']")
    Xp, Yp = pairs_from_intermediate(inter0)
    buf_x.append(Xp)
    buf_y.append(Yp)

    gi = GraphIdentifierEnv(
        N=N, s=env.t_s, l2_lambda=l2_lambda, zero_diag=True,
        **({} if identifier_kwargs is None else identifier_kwargs),
    )

    # --- helper: timed fit ---
    def _fit():
        X = np.concatenate(buf_x, axis=0)
        Y = np.concatenate(buf_y, axis=0)

        t0 = time.perf_counter()
        if suppress_fit_logs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                A_hat = train_graph_identifier(
                    gi,
                    X,
                    Y,
                    lr=lr,
                    batch_size=fit_batch_size,
                    max_steps=fit_max_steps,
                    mae_stop=fit_mae_stop,
                    device=device,
                )
        else:
            A_hat = train_graph_identifier(
                gi,
                X,
                Y,
                lr=lr,
                batch_size=fit_batch_size,
                max_steps=fit_max_steps,
                mae_stop=fit_mae_stop,
                device=device,
            )

        timing["fit_time"] += (time.perf_counter() - t0)
        timing["fit_calls"] += 1
        return A_hat

    # --- initial fit ---
    A_hat = _fit()
    A_hats.append(A_hat)
    v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))

    # --- Campaigns 1.. ---
    for k in range(1, num_campaigns_total):
        if done or trunc:
            break

        beta_k = min(float(B_campaign), float(ubar_vec.sum()))
        uk, _ = centrality_based_continuous_control(env, beta_k, v=v_hat)

        x_next, r, done, trunc, info_k = _step(uk)
        actions.append(uk.copy())
        rewards.append(float(r))
        states.append(x_next.copy())

        inter = info_k.get("intermediate_states", None)
        if inter is not None:
            Xp, Yp = pairs_from_intermediate(inter)
            buf_x.append(Xp)
            buf_y.append(Yp)

        if update_A_each_campaign:
            A_hat = _fit()
            v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))
            A_hats.append(A_hat)

    return {
        "env": env,
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "A_hats": A_hats,
        "timing": timing,  # <-- NEW: consumed by the multi-seed wrapper
    }


def run_multi_seed_experiment_dynamics(
    *,
    seeds=range(10),
    dynamics_model="laplacian",
    B_campaign=1.0,
    num_campaigns_total=5,
    lr=1e-3,
    l2_lambda=0.0,
    device="cpu",
    update_A_each_campaign=True,
    suppress_fit_logs=True,

    # NEW: fit-budget controls (threaded into run_single_online_id_on_env -> train_graph_identifier)
    fit_max_steps=None,
    fit_mae_stop=None,
    fit_batch_size=None,
    fit_check_every=None,

    # NEW: timing controls
    timing: bool = True,
    timing_print: bool = False,
    identifier_kwargs: dict | None = None,
):
    """
    Multi-seed runner comparing dynamics ("laplacian" vs "coca") while keeping the
    identification model the same, but allowing you to control identification compute
    via fit_* arguments.

    If fit_* are None, we apply sensible defaults:
      - laplacian: looser budget
      - coca: tighter budget (to avoid maxing out steps due to model mismatch)
    """
    # -------------------------
    # Default fit budgets by dynamics (only if caller didn't specify)
    # -------------------------
    if fit_max_steps is None:
        fit_max_steps = 50_000 if dynamics_model == "laplacian" else 2_000
    if fit_mae_stop is None:
        fit_mae_stop = 1e-3 if dynamics_model == "laplacian" else 5e-3
    if fit_batch_size is None:
        fit_batch_size = 64 if dynamics_model == "laplacian" else 256
    if fit_check_every is None:
        fit_check_every = 200  # matches your existing behavior

    env_factory = EnvironmentFactory()
    rows = []

    for seed in seeds:
        t_total0 = time.perf_counter()

        # -------------------------
        # 1) Make env
        # -------------------------
        t0 = time.perf_counter()
        env = make_env_with_dynamics(
            env_factory, seed=int(seed), dynamics_model=dynamics_model
        )
        t_make_env = time.perf_counter() - t0

        # -------------------------
        # 2) Online identification run
        # -------------------------
        t0 = time.perf_counter()
        out = run_single_online_id_on_env(
            env,
            B_campaign=B_campaign,
            num_campaigns_total=num_campaigns_total,
            lr=lr,
            l2_lambda=l2_lambda,
            device=device,
            update_A_each_campaign=update_A_each_campaign,
            suppress_fit_logs=suppress_fit_logs,

            # NEW: pass fit budgets down
            fit_max_steps=fit_max_steps,
            fit_mae_stop=fit_mae_stop,
            fit_batch_size=fit_batch_size,
            identifier_kwargs=identifier_kwargs,
        )
        t_online = time.perf_counter() - t0

        env0 = out["env"]
        states_learn = np.asarray(out["states"], dtype=float)
        A_hat_final = np.asarray(out["A_hats"][-1], dtype=float)

        # Optional inner timing (only if run_single_online_id_on_env returns it)
        inner_timing = out.get("timing", {}) if isinstance(out, dict) else {}
        t_fit_inner = float(inner_timing.get("fit_time", np.nan)) if timing else np.nan
        t_step_inner = float(inner_timing.get("step_time", np.nan)) if timing else np.nan
        fit_calls_inner = int(inner_timing.get("fit_calls", 0)) if timing else 0
        step_calls_inner = int(inner_timing.get("step_calls", 0)) if timing else 0

        # -------------------------
        # 3) True graph metrics
        # -------------------------
        t0 = time.perf_counter()
        A_true = np.asarray(env0.connectivity_matrix, dtype=float)
        v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
        v_hat_final = compute_eigenvector_centrality(compute_laplacian(A_hat_final))
        t_ident_metrics = time.perf_counter() - t0

        # -------------------------
        # 4) Rollouts (oracle / no-control)
        # -------------------------
        x0 = states_learn[0].copy()
        K_total = states_learn.shape[0] - 1

        t0 = time.perf_counter()
        states_or = rollout_with_v(env0, x0, K_total, B_campaign, v_true)
        t_oracle = time.perf_counter() - t0

        t0 = time.perf_counter()
        states_nc = rollout_with_v(env0, x0, K_total, B_campaign, None)
        t_noc = time.perf_counter() - t0

        # -------------------------
        # 5) Trajectory metrics
        # -------------------------
        t0 = time.perf_counter()
        T = min(states_learn.shape[0], states_or.shape[0], states_nc.shape[0])
        mean_le = states_learn[:T].mean(axis=1)
        mean_or = states_or[:T].mean(axis=1)
        mean_nc = states_nc[:T].mean(axis=1)

        vx_le = states_learn[:T] @ v_true
        vx_or = states_or[:T] @ v_true

        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)
        t_traj_metrics = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total0

        row = dict(
            seed=int(seed),
            dynamics=str(dynamics_model),
            N=int(env0.num_agents),

            # identification
            v_L1_final=float(np.sum(np.abs(v_hat_final - v_true))),
            A_Fro_final=float(np.linalg.norm(A_hat_final - A_true, ord="fro")),
            A_MAE_final=float(np.mean(np.abs(A_hat_final - A_true))),

            # trajectories
            mean_gap_to_oracle_end=float(np.abs(mean_le[-1] - mean_or[-1])),
            mean_err_avg=float(mean_err.mean()),
            mean_err_max=float(mean_err.max()),
            vx_gap_to_oracle_end=float(np.abs(vx_le[-1] - vx_or[-1])),
            vx_err_avg=float(vx_err.mean()),
            vx_err_max=float(vx_err.max()),
            mean_oracle_end=float(mean_or[-1]),
            mean_learn_end=float(mean_le[-1]),
            mean_noc_end=float(mean_nc[-1]),

            # record fit budget used (so it’s visible in your DF)
            fit_max_steps=int(fit_max_steps),
            fit_mae_stop=float(fit_mae_stop),
            fit_batch_size=int(fit_batch_size),
            fit_check_every=int(fit_check_every),
        )

        if timing:
            row.update(dict(
                time_total=float(t_total),
                time_make_env=float(t_make_env),
                time_online=float(t_online),
                time_ident_metrics=float(t_ident_metrics),
                time_oracle_rollout=float(t_oracle),
                time_noc_rollout=float(t_noc),
                time_traj_metrics=float(t_traj_metrics),

                # optional inner breakdown
                time_fit_inner=float(t_fit_inner),
                time_step_inner=float(t_step_inner),
                fit_calls_inner=int(fit_calls_inner),
                step_calls_inner=int(step_calls_inner),
            ))

        rows.append(row)

        if timing_print:
            msg = (
                f"[{dynamics_model}] seed={seed} "
                f"total={t_total:.2f}s online={t_online:.2f}s "
                f"oracle={t_oracle:.2f}s noc={t_noc:.2f}s "
                f"(make_env={t_make_env:.3f}s)"
            )
            if np.isfinite(t_fit_inner):
                msg += (
                    f" | inner_fit={t_fit_inner:.2f}s/{fit_calls_inner} "
                    f"inner_step={t_step_inner:.4f}s/{step_calls_inner}"
                )
            msg += (
                f" | fit_budget: steps={fit_max_steps} mae_stop={fit_mae_stop} "
                f"batch={fit_batch_size} check={fit_check_every}"
            )
            print(msg)

    return pd.DataFrame(rows).sort_values(["dynamics", "seed"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Repeated no-control random-init experiment utilities
# -----------------------------------------------------------------------------

from typing import Any
import copy

from ..identify_nonlinear import train_graph_identifier_with_info
from .rollouts import (
    fresh_env_from_template,
    rollout_with_model_derived_control_intermediate,
    rollout_with_uniform_intermediate,
    rollout_with_v_aligned_intermediate,
)
from .metrics import (
    model_prediction_mae_on_arrays,
    model_prediction_mae_on_intermediates,
    identity_prediction_mae_on_arrays,
    identity_prediction_mae_on_intermediates,
    safe_ratio,
    action_alignment_metrics,
    effective_centrality_alignment_metrics,
)


def sample_init_opinions(N, rng, mode="uniform", low=0.01, high=0.99):
    """Sample bounded initial opinions for repeated identification trials."""
    if mode == "permuted_linspace":
        x0 = np.linspace(low, high, int(N), dtype=float)
        rng.shuffle(x0)
        return x0
    if mode == "uniform":
        return rng.uniform(low, high, size=int(N)).astype(float)
    raise ValueError(f"Unknown init_mode: {mode}")


def summarize_training_inits(x0_list):
    """Return per-repeat summary statistics for sampled training initial states."""
    rows = []
    x0_arr = np.asarray(x0_list, dtype=float)
    for r in range(x0_arr.shape[0]):
        x = x0_arr[r]
        rows.append(
            dict(
                repeat=r,
                min=float(x.min()),
                q10=float(np.quantile(x, 0.10)),
                q25=float(np.quantile(x, 0.25)),
                median=float(np.median(x)),
                q75=float(np.quantile(x, 0.75)),
                q90=float(np.quantile(x, 0.90)),
                max=float(x.max()),
                mean=float(x.mean()),
                std=float(x.std()),
                range=float(x.max() - x.min()),
            )
        )
    return pd.DataFrame(rows)


def _prefix_dict(d, prefix):
    return {f"{prefix}{k}": v for k, v in d.items()}


def run_repeated_nocontrol_singlecampaign_id_on_env(
    env_template,
    *,
    num_repeats=100,
    init_mode="uniform",
    init_low=0.01,
    init_high=0.99,
    init_seed_base=12345,
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
    repeat_seed_stride=10_000,
    eval_seed_offset=999_999,
    eval_zero_first_campaign=False,
):
    """
    Learn from many fresh no-control short rollouts on one fixed graph.

    After each new trial, fit/update the identifier and evaluate the induced
    learned-ranking policy against oracle, uniform, flat-rank, and no-control
    baselines. Returns per-repeat rows for amount-of-data learning curves.
    """
    N = int(env_template.num_agents)
    A_true = np.asarray(env_template.connectivity_matrix, dtype=float)
    v_true = compute_eigenvector_centrality(compute_laplacian(A_true))
    v_flat = np.ones(N, dtype=float) / float(N)
    base_seed = int(getattr(env_template, "seed", 0) or 0)

    gi = GraphIdentifierEnv(
        N=N,
        s=float(env_template.t_s),
        l2_lambda=l2_lambda,
        zero_diag=True,
        **({} if identifier_kwargs is None else identifier_kwargs),
    )

    buf_x, buf_y = [], []
    repeat_rows, repeat_artifacts, train_x0s = [], [], []
    A_hat = v_hat = None

    def _fit_current():
        X = np.concatenate(buf_x, axis=0)
        Y = np.concatenate(buf_y, axis=0)
        if suppress_fit_logs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                return train_graph_identifier_with_info(
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
        return train_graph_identifier_with_info(
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

    eval_rng = np.random.default_rng(base_seed + eval_seed_offset)
    eval_x0 = sample_init_opinions(N, eval_rng, mode=init_mode, low=init_low, high=init_high)

    for repeat_idx in range(int(num_repeats)):
        rng = np.random.default_rng(init_seed_base + repeat_idx)
        x0_train = sample_init_opinions(N, rng, mode=init_mode, low=init_low, high=init_high)
        train_x0s.append(np.array(x0_train, copy=True))

        repeat_seed = base_seed + int(repeat_seed_stride) * (repeat_idx + 1)
        env, env_kwargs = fresh_env_from_template(
            env_template,
            repeat_seed=repeat_seed,
            initial_opinions=x0_train,
        )

        states_train, actions_train, rewards_train = [], [], []
        intermediate_states_list, intermediate_times_list = [], []
        timing = {"fit_time": 0.0, "step_time": 0.0, "fit_calls": 0, "step_calls": 0}

        x, _ = env.reset()
        states_train.append(np.array(x, copy=True))

        def _step(u):
            t0 = time.perf_counter()
            out = env.step(u)
            timing["step_time"] += time.perf_counter() - t0
            timing["step_calls"] += 1
            return out

        for _k in range(int(learn_num_campaigns)):
            u0 = np.zeros(N, dtype=float)
            x_next, r, done, trunc, info_k = _step(u0)
            actions_train.append(u0.copy())
            rewards_train.append(float(r))
            states_train.append(np.array(x_next, copy=True))
            inter = info_k.get("intermediate_states", None)
            if inter is None:
                raise RuntimeError("env.step did not return info['intermediate_states']; cannot build transition pairs.")
            inter_arr = np.asarray(inter, dtype=float)
            intermediate_states_list.append(inter_arr.copy())
            intermediate_times_list.append(float(getattr(env, "t_s", 1.0)) * np.arange(inter_arr.shape[0], dtype=float))
            Xp, Yp = pairs_from_intermediate(inter_arr)
            buf_x.append(Xp)
            buf_y.append(Yp)
            if done or trunc:
                break

        t0 = time.perf_counter()
        A_hat, fit_info = _fit_current()
        timing["fit_time"] += time.perf_counter() - t0
        timing["fit_calls"] += 1
        v_hat = compute_eigenvector_centrality(compute_laplacian(A_hat))

        # Fresh eval envs with identical x0.
        env_eval_learn, _ = fresh_env_from_template(env_template, repeat_seed=base_seed + eval_seed_offset, initial_opinions=eval_x0)
        env_eval_oracle, _ = fresh_env_from_template(env_template, repeat_seed=base_seed + eval_seed_offset, initial_opinions=eval_x0)
        env_eval_flat, _ = fresh_env_from_template(env_template, repeat_seed=base_seed + eval_seed_offset, initial_opinions=eval_x0)
        env_eval_noc, _ = fresh_env_from_template(env_template, repeat_seed=base_seed + eval_seed_offset, initial_opinions=eval_x0)
        env_eval_uni, _ = fresh_env_from_template(env_template, repeat_seed=base_seed + eval_seed_offset, initial_opinions=eval_x0)

        learn_eval = rollout_with_model_derived_control_intermediate(
            env_eval_learn,
            gi,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            device=device,
            zero_first_campaign=eval_zero_first_campaign,
        )
        oracle_eval = rollout_with_v_aligned_intermediate(env_eval_oracle, eval_x0, int(eval_num_campaigns), B_campaign, v_true, zero_first_campaign=eval_zero_first_campaign)
        flat_eval = rollout_with_v_aligned_intermediate(env_eval_flat, eval_x0, int(eval_num_campaigns), B_campaign, v_flat, zero_first_campaign=eval_zero_first_campaign)
        noc_eval = rollout_with_v_aligned_intermediate(env_eval_noc, eval_x0, int(eval_num_campaigns), B_campaign, None, zero_first_campaign=eval_zero_first_campaign)
        uniform_eval = rollout_with_uniform_intermediate(env_eval_uni, eval_x0, int(eval_num_campaigns), B_campaign, zero_first_campaign=eval_zero_first_campaign)

        states_learn = learn_eval["states"]
        states_oracle = oracle_eval["states"]
        states_flat = flat_eval["states"]
        states_noc = noc_eval["states"]
        states_uniform = uniform_eval["states"]
        T = min(states_learn.shape[0], states_oracle.shape[0], states_flat.shape[0], states_noc.shape[0], states_uniform.shape[0])
        sl, so, sf, sn, su = [np.asarray(z[:T], dtype=float) for z in [states_learn, states_oracle, states_flat, states_noc, states_uniform]]
        mean_le, mean_or, mean_flat, mean_nc, mean_uni = [z.mean(axis=1) for z in [sl, so, sf, sn, su]]
        vx_le = sl @ v_true
        vx_or = so @ v_true
        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)

        train_X_all = np.concatenate(buf_x, axis=0)
        train_Y_all = np.concatenate(buf_y, axis=0)
        one_step_train_mae = model_prediction_mae_on_arrays(gi, train_X_all, train_Y_all, device=device)
        train_identity_mae = identity_prediction_mae_on_arrays(train_X_all, train_Y_all)
        train_model_over_identity = safe_ratio(one_step_train_mae, train_identity_mae)
        one_step_val_mae = model_prediction_mae_on_intermediates(gi, noc_eval["intermediate_states_list"], device=device)
        val_identity_mae = identity_prediction_mae_on_intermediates(noc_eval["intermediate_states_list"])
        val_model_over_identity = safe_ratio(one_step_val_mae, val_identity_mae)

        action_metrics = action_alignment_metrics(learn_eval["actions"], oracle_eval["actions"])
        flat_action_metrics = _prefix_dict(action_alignment_metrics(learn_eval["actions"], flat_eval["actions"]), "flat_rank_")
        v_eff_metrics = effective_centrality_alignment_metrics(learn_eval["effective_centralities"], v_true)

        A_hat_final = np.asarray(A_hat, dtype=float)
        v_hat_final = np.asarray(v_hat, dtype=float)
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
            mean_flat_rank_end=float(mean_flat[-1]),
            mean_noc_end=float(mean_nc[-1]),
            mean_uniform_end=float(mean_uni[-1]),
            mean_gap_to_oracle_end=float(np.abs(mean_le[-1] - mean_or[-1])),
            mean_gain_vs_noc_end=float(mean_le[-1] - mean_nc[-1]),
            mean_gain_vs_uniform_end=float(mean_le[-1] - mean_uni[-1]),
            mean_gain_vs_flat_rank_end=float(mean_le[-1] - mean_flat[-1]),
            oracle_gain_vs_flat_rank_end=float(mean_or[-1] - mean_flat[-1]),
            mean_err_avg=float(mean_err.mean()),
            mean_err_max=float(mean_err.max()),
            vx_gap_to_oracle_end=float(np.abs(vx_le[-1] - vx_or[-1])),
            vx_err_avg=float(vx_err.mean()),
            vx_err_max=float(vx_err.max()),
            mean_auc_learn=float(mean_le.mean()),
            mean_auc_oracle=float(mean_or.mean()),
            mean_auc_uniform=float(mean_uni.mean()),
            mean_auc_noc=float(mean_nc.mean()),
            mean_auc_flat_rank=float(mean_flat.mean()),
            mean_auc_gap_to_oracle=float(mean_err.mean()),
            one_step_train_mae=float(one_step_train_mae),
            train_identity_mae=float(train_identity_mae),
            train_model_over_identity=float(train_model_over_identity),
            train_improvement_over_identity=(1.0 - train_model_over_identity) if np.isfinite(train_model_over_identity) else float("nan"),
            one_step_val_mae=float(one_step_val_mae),
            val_identity_mae=float(val_identity_mae),
            val_model_over_identity=float(val_model_over_identity),
            val_improvement_over_identity=(1.0 - val_model_over_identity) if np.isfinite(val_model_over_identity) else float("nan"),
            fit_mae_stop_used=float(fit_mae_stop),
            fit_max_steps_used=int(fit_max_steps),
            fit_batch_size_used=int(fit_batch_size),
            fit_check_every_used=int(fit_check_every),
            **fit_info,
            **action_metrics,
            **flat_action_metrics,
            **v_eff_metrics,
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
                states_nocontrol_train=np.asarray(states_train, dtype=float),
                actions_nocontrol_train=np.asarray(actions_train, dtype=float),
                rewards_nocontrol_train=np.asarray(rewards_train, dtype=float),
                A_hat_final=A_hat_final,
                v_hat_final=v_hat_final,
                A_true=A_true,
                v_true=v_true,
                states_learn=sl,
                states_oracle=so,
                states_flat_rank=sf,
                states_nocontrol_eval=sn,
                states_uniform_eval=su,
                actions_learn_eval=np.asarray(learn_eval["actions"], dtype=float),
                actions_oracle_eval=np.asarray(oracle_eval["actions"], dtype=float),
                actions_flat_rank_eval=np.asarray(flat_eval["actions"], dtype=float),
                actions_nocontrol_eval=np.asarray(noc_eval["actions"], dtype=float),
                actions_uniform_eval=np.asarray(uniform_eval["actions"], dtype=float),
                eval_zero_first_campaign=bool(eval_zero_first_campaign),
                learned_effective_adjacencies=[np.asarray(Ae, dtype=float) for Ae in learn_eval["effective_adjacencies"]],
                learned_effective_centralities=[np.asarray(ve, dtype=float) for ve in learn_eval["effective_centralities"]],
                learned_intermediate_states_list=learn_eval["intermediate_states_list"],
                learned_intermediate_times_list=learn_eval["intermediate_times_list"],
                intermediate_states_list=intermediate_states_list,
                intermediate_times_list=intermediate_times_list,
                train_pairs_total=row["train_pairs_total"],
                timing=timing,
                fit_info=fit_info,
            )
        )

    return dict(
        rows=repeat_rows,
        artifacts=repeat_artifacts,
        A_true=A_true,
        v_true=v_true,
        A_hat_final=np.asarray(A_hat, dtype=float) if A_hat is not None else None,
        v_hat_final=np.asarray(v_hat, dtype=float) if v_hat is not None else None,
        train_pairs_total=int(sum(x.shape[0] for x in buf_x)),
        x0_eval=np.asarray(eval_x0, dtype=float),
        train_x0s=np.stack(train_x0s, axis=0),
        model_final=copy.deepcopy(gi).cpu(),
    )


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
    identifier_kwargs=None,
    repeat_seed_stride=10_000,
    return_artifacts=False,
    eval_zero_first_campaign=False,
):
    """Multi-seed wrapper for the repeated no-control amount-of-data experiment."""
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
            eval_zero_first_campaign=eval_zero_first_campaign,
        )
        rows.extend(out["rows"])
        if return_artifacts:
            artifacts_by_seed[int(seed)] = dict(
                rows=out["rows"],
                final_row=out["rows"][-1],
                final_artifact=out["artifacts"][-1],
                train_x0s=out["train_x0s"],
                x0_eval=out["x0_eval"],
                model_final=out["model_final"],
            )
    df = pd.DataFrame(rows).sort_values(["seed", "repeat"]).reset_index(drop=True)
    if return_artifacts:
        return df, artifacts_by_seed
    return df

# =============================================================================
# Source-of-truth no-control random-init experiment helpers
# Extracted from the user-maintained notebook to keep test notebooks small.
# =============================================================================
import copy
from typing import Any
from .rollouts import (
    _fresh_env_from_template,
    fresh_env_from_template,
    rollout_with_model_derived_control_intermediate,
    rollout_with_v_intermediate,
    rollout_with_uniform_intermediate,
)
from .metrics import (
    model_prediction_mae_on_arrays,
    model_prediction_mae_on_intermediates,
    action_alignment_metrics,
    effective_centrality_alignment_metrics,
    _safe_fraction_of_oracle,
)

def sample_init_opinions(
    N: int,
    rng: np.random.Generator,
    mode: str = "uniform",
    low: float = 0.01,
    high: float = 0.99,
) -> np.ndarray:
    if mode == "permuted_linspace":
        x0 = np.linspace(low, high, N, dtype=float)
        rng.shuffle(x0)
        return x0
    if mode == "uniform":
        return rng.uniform(low, high, size=N).astype(float)
    raise ValueError(f"Unknown init_mode: {mode}")


def summarize_training_inits(x0_list: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in range(x0_list.shape[0]):
        x = np.asarray(x0_list[r], dtype=float)
        rows.append(
            dict(
                repeat=r,
                min=float(x.min()),
                q10=float(np.quantile(x, 0.10)),
                q25=float(np.quantile(x, 0.25)),
                median=float(np.median(x)),
                q75=float(np.quantile(x, 0.75)),
                q90=float(np.quantile(x, 0.90)),
                max=float(x.max()),
                mean=float(x.mean()),
                std=float(x.std()),
                range=float(x.max() - x.min()),
            )
        )
    return pd.DataFrame(rows)

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
    eval_zero_first_campaign: bool = False,
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
            zero_first_campaign=eval_zero_first_campaign,
        )
        states_learn = learn_eval["states"]

        oracle_eval = rollout_with_v_intermediate(
            env_eval_oracle,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            v_true,
            zero_first_campaign=eval_zero_first_campaign,
        )
        noc_eval = rollout_with_v_intermediate(
            env_eval_noc,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            None,
            zero_first_campaign=eval_zero_first_campaign,
        )
        uniform_eval = rollout_with_uniform_intermediate(
            env_eval_uni,
            eval_x0,
            int(eval_num_campaigns),
            B_campaign,
            zero_first_campaign=eval_zero_first_campaign,
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

        # Policy-centered diagnostics for data-need curves.
        # Raw A_hat-vs-A_true is still logged below, but it is not reliable as a
        # stopping metric for state-dependent/nonlinear dynamics.
        train_X_all = np.concatenate(buf_x, axis=0)
        train_Y_all = np.concatenate(buf_y, axis=0)
        one_step_train_mae = model_prediction_mae_on_arrays(
            gi, train_X_all, train_Y_all, device=device
        )
        one_step_val_mae = model_prediction_mae_on_intermediates(
            gi, noc_eval["intermediate_states_list"], device=device
        )
        action_metrics = action_alignment_metrics(
            learn_eval["actions"], oracle_eval["actions"]
        )
        v_eff_metrics = effective_centrality_alignment_metrics(
            learn_eval["effective_centralities"], v_true
        )

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
            mean_auc_learn=float(mean_le.mean()),
            mean_auc_oracle=float(mean_or.mean()),
            mean_auc_uniform=float(mean_uni.mean()),
            mean_auc_noc=float(mean_nc.mean()),
            mean_auc_gap_to_oracle=float(mean_err.mean()),
            policy_frac_oracle_vs_uniform_end=_safe_fraction_of_oracle(mean_le[-1], mean_uni[-1], mean_or[-1]),
            policy_frac_oracle_vs_noc_end=_safe_fraction_of_oracle(mean_le[-1], mean_nc[-1], mean_or[-1]),
            policy_frac_oracle_vs_uniform_auc=_safe_fraction_of_oracle(mean_le.mean(), mean_uni.mean(), mean_or.mean()),
            policy_frac_oracle_vs_noc_auc=_safe_fraction_of_oracle(mean_le.mean(), mean_nc.mean(), mean_or.mean()),
            one_step_train_mae=float(one_step_train_mae),
            one_step_val_mae=float(one_step_val_mae),
            **action_metrics,
            **v_eff_metrics,
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
                eval_zero_first_campaign=bool(eval_zero_first_campaign),
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
    eval_zero_first_campaign: bool = False,
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
            eval_zero_first_campaign=eval_zero_first_campaign,
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
    eval_zero_first_campaign: bool = False,
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
            zero_first_campaign=eval_zero_first_campaign,
        )
        oracle = rollout_with_v_intermediate(
            env_or,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            v_true,
            zero_first_campaign=eval_zero_first_campaign,
        )
        noc = rollout_with_v_intermediate(
            env_nc,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            None,
            zero_first_campaign=eval_zero_first_campaign,
        )
        uni = rollout_with_uniform_intermediate(
            env_uni,
            x0,
            int(eval_num_campaigns),
            B_campaign,
            zero_first_campaign=eval_zero_first_campaign,
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
        mean_err = np.abs(mean_le - mean_or)
        vx_err = np.abs(vx_le - vx_or)
        action_metrics = action_alignment_metrics(le["actions"], oracle["actions"])
        v_eff_metrics = effective_centrality_alignment_metrics(le["effective_centralities"], v_true)
        one_step_val_mae = model_prediction_mae_on_intermediates(
            model, noc["intermediate_states_list"], device=device
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
                vx_err_avg=float(vx_err.mean()),
                vx_err_max=float(vx_err.max()),
                mean_err_avg=float(mean_err.mean()),
                mean_err_max=float(mean_err.max()),
                mean_auc_learn=float(mean_le.mean()),
                mean_auc_oracle=float(mean_or.mean()),
                mean_auc_uniform=float(mean_uni.mean()),
                mean_auc_noc=float(mean_nc.mean()),
                mean_auc_gap_to_oracle=float(mean_err.mean()),
                policy_frac_oracle_vs_uniform_end=_safe_fraction_of_oracle(mean_le[-1], mean_uni[-1], mean_or[-1]),
                policy_frac_oracle_vs_noc_end=_safe_fraction_of_oracle(mean_le[-1], mean_nc[-1], mean_or[-1]),
                policy_frac_oracle_vs_uniform_auc=_safe_fraction_of_oracle(mean_le.mean(), mean_uni.mean(), mean_or.mean()),
                policy_frac_oracle_vs_noc_auc=_safe_fraction_of_oracle(mean_le.mean(), mean_nc.mean(), mean_or.mean()),
                one_step_val_mae=float(one_step_val_mae),
                **action_metrics,
                **v_eff_metrics,
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
    eval_zero_first_campaign: bool = False,
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
            eval_zero_first_campaign=eval_zero_first_campaign,
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
    fit_by_dynamics=None,
    identifier_kwargs=None,
    eval_zero_first_campaign: bool = False,
):
    env_factory = EnvironmentFactory()
    rows = []
    for dyn in dynamics_list:
        for seed in seeds:
            print(f"=== data-budget sweep | {dyn} | seed={seed} ===")
            env_template = make_env_with_dynamics(env_factory, seed=int(seed), dynamics_model=dyn)
            fit_cfg = (fit_by_dynamics or {})[dyn]
            df_budget = run_data_budget_sweep_on_env(
                env_template,
                trial_counts=trial_counts,
                n_validation_x0=n_validation_x0,
                init_mode=init_mode,
                learn_num_campaigns=learn_num_campaigns,
                eval_num_campaigns=eval_num_campaigns,
                B_campaign=B_campaign,
                device=device,
                identifier_kwargs=identifier_kwargs,
                eval_zero_first_campaign=eval_zero_first_campaign,
                **fit_cfg,
            )
            rows.append(df_budget)
    return pd.concat(rows, axis=0).reset_index(drop=True) if rows else pd.DataFrame()
