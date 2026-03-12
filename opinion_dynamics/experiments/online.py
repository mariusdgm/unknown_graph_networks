import contextlib
import io
import time

import numpy as np
import pandas as pd

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from ..identify import (
    GraphIdentifierEnv,
    pairs_from_intermediate,
    train_graph_identifier,
)
from ..utils.env_setup import EnvironmentFactory
from ..baseline import centrality_based_continuous_control

from rollouts import rollout_with_v, make_env_with_dynamics


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

    # NEW fit budget knobs (threaded into train_graph_identifier)
    fit_max_steps=50_000,
    fit_mae_stop=1e-3,
    fit_batch_size=64,
    fit_check_every=200,   # NOTE: only used if your train_graph_identifier supports it
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
    import contextlib, io
    import numpy as np

    from rl_envs_forge.envs.network_graph.graph_utils import (
        compute_laplacian,
        compute_eigenvector_centrality,
    )
    from ..identify import GraphIdentifierEnv, pairs_from_intermediate, train_graph_identifier
    from ..baseline import centrality_based_continuous_control

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

    gi = GraphIdentifierEnv(N=N, s=env.t_s, l2_lambda=l2_lambda, zero_diag=True)

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
                )
        else:
            A_hat = train_graph_identifier(
                gi, X, Y,
                lr=lr,
                batch_size=fit_batch_size,
                max_steps=fit_max_steps,
                mae_stop=fit_mae_stop,
                device=device,
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

    gi = GraphIdentifierEnv(N=N, s=env.t_s, l2_lambda=l2_lambda, zero_diag=True)

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