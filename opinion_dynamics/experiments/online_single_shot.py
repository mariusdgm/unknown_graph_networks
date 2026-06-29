"""
Single-shot online identification experiments with exploratory control.

Intended placement in the repo:
    opinion_dynamics/experiments/online_single_shot.py

The runner keeps one continuous trajectory of a fixed environment:
  - campaign 0: no control, collect passive dynamics
  - campaigns 1..exploration_campaigns: epsilon-mixed random/learned control
  - remaining campaigns: pure learned lambda-mix centrality control

The learned policy remains in the centrality-ranking family.  It builds

    M_lambda(x) = (1 - lambda) A_hat + lambda (A_hat * alpha(x))

without row-normalizing after applying alpha, then ranks nodes by
centrality(M_lambda(x)) * |desired - x_i|.
"""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from opinion_dynamics.baseline import centrality_based_continuous_control
from opinion_dynamics.identify_nonlinear import (
    GraphIdentifierEnv,
    pairs_from_intermediate,
    train_graph_identifier,
)


Array = np.ndarray


def as_max_u_vec(max_u: Any, n: int) -> Array:
    """Return max_u as a length-n float vector."""
    u = np.asarray(max_u, dtype=float)
    if u.ndim == 0:
        return np.full(n, float(u), dtype=float)
    u = u.reshape(-1).astype(float)
    if u.shape != (n,):
        raise ValueError(f"max_u must be scalar or shape ({n},), got {u.shape}")
    return u


def env_kwargs_from_env(env: Any, *, t_campaign: Optional[float] = None, t_s: Optional[float] = None) -> Dict[str, Any]:
    """Extract kwargs that recreate the same NetworkGraph, with optional timing overrides."""
    return dict(
        connectivity_matrix=np.array(env.connectivity_matrix, copy=True),
        num_agents=int(env.num_agents),
        max_u=np.array(env.max_u, copy=True),
        desired_opinion=float(env.desired_opinion),
        t_campaign=float(env.t_campaign if t_campaign is None else t_campaign),
        t_s=float(env.t_s if t_s is None else t_s),
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


def make_env_from_template(env_template: Any, *, t_campaign: Optional[float] = None, t_s: Optional[float] = None) -> Any:
    """Clone an environment template, optionally overriding t_campaign/t_s."""
    kwargs = env_kwargs_from_env(env_template, t_campaign=t_campaign, t_s=t_s)
    return env_template.__class__(**kwargs)


def set_initial_state(env: Any, x0: Array) -> None:
    """Force the env opinion state to x0 after reset."""
    env.opinions = np.array(x0, dtype=float, copy=True)
    # Some env variants may mirror the state in a separate attribute.
    if hasattr(env, "state"):
        try:
            env.state = np.array(x0, dtype=float, copy=True)
        except Exception:
            pass


def normalize_scores(scores: Array, *, eps: float = 1e-12) -> Array:
    """Scale nonnegative scores to [0, 1] while preserving ranking."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.maximum(s, 0.0)
    m = float(s.max()) if s.size else 0.0
    if m <= eps:
        return np.ones_like(s)
    return s / m


def waterfill_from_scores(scores: Array, max_u: Array, budget: float) -> Array:
    """Greedy budget allocation by descending score with per-node max_u constraints."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    max_u = np.asarray(max_u, dtype=float).reshape(-1)
    if scores.shape != max_u.shape:
        raise ValueError(f"scores and max_u shape mismatch: {scores.shape} vs {max_u.shape}")

    u = np.zeros_like(scores, dtype=float)
    remaining = min(float(budget), float(max_u.sum()))
    if remaining <= 0:
        return u

    order = np.argsort(scores)[::-1]
    for i in order:
        if remaining <= 1e-12:
            break
        if scores[i] <= 0 and np.any(scores > 0):
            break
        assign = min(float(max_u[i]), remaining)
        if assign > 0:
            u[i] = assign
            remaining -= assign
    return u


def uniform_budget_action(max_u: Array, budget: float) -> Array:
    """Spread budget as uniformly as possible while respecting max_u."""
    max_u = np.asarray(max_u, dtype=float).reshape(-1)
    n = max_u.size
    u = np.zeros(n, dtype=float)
    remaining = min(float(budget), float(max_u.sum()))
    active = np.ones(n, dtype=bool)

    while remaining > 1e-12 and active.any():
        share = remaining / int(active.sum())
        idxs = np.where(active)[0]
        progressed = False
        for i in idxs:
            add = min(share, max(0.0, float(max_u[i] - u[i])))
            if add > 0:
                u[i] += add
                remaining -= add
                progressed = True
            if u[i] >= max_u[i] - 1e-12:
                active[i] = False
        if not progressed:
            break
    return u


def learned_lambda_mix_matrix(model: GraphIdentifierEnv, x: Array, *, lambda_mix: float, device: str = "cpu") -> Array:
    """Build M_lambda(x) = (1-lambda) A_hat + lambda (A_hat * alpha(x))."""
    model.to(device)
    model.eval()
    x_np = np.asarray(x, dtype=float).reshape(-1)
    with torch.no_grad():
        A = model.A_hat().detach().cpu().numpy()
        xt = torch.tensor(x_np, dtype=torch.float32, device=device).view(1, -1)
        xi = xt.unsqueeze(2)
        xj = xt.unsqueeze(1)
        alpha = model.alpha(xi, xj).squeeze(0).detach().cpu().numpy()
    lam = float(lambda_mix)
    M = (1.0 - lam) * A + lam * (A * alpha)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.maximum(M, 0.0)
    np.fill_diagonal(M, 0.0)
    return M


def centrality_from_matrix(M: Array) -> Array:
    """Compute the same graph centrality used elsewhere in the experiments."""
    M = np.asarray(M, dtype=float)
    v = compute_eigenvector_centrality(compute_laplacian(M))
    v = np.asarray(v, dtype=float).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if np.abs(v).sum() > 0:
        # centrality implementations can have arbitrary scale/sign; keep nonnegative weights.
        v = np.maximum(v, 0.0)
        if v.sum() > 0:
            v = v / v.sum()
    return v


def learned_lambda_mix_scores(
    model: GraphIdentifierEnv,
    x: Array,
    *,
    desired_opinion: float,
    lambda_mix: float,
    device: str = "cpu",
) -> Tuple[Array, Array, Array]:
    """Return policy scores, centrality, and M_lambda for the learned lambda-mix policy."""
    x = np.asarray(x, dtype=float).reshape(-1)
    M = learned_lambda_mix_matrix(model, x, lambda_mix=lambda_mix, device=device)
    v = centrality_from_matrix(M)
    scores = v * np.abs(float(desired_opinion) - x)
    return scores, v, M


def exploratory_lambda_mix_action(
    model: Optional[GraphIdentifierEnv],
    x: Array,
    *,
    desired_opinion: float,
    max_u: Array,
    budget: float,
    epsilon: float,
    lambda_mix: float,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Tuple[Array, Dict[str, Any]]:
    """Epsilon-mix learned policy scores with random exploration scores, then waterfill."""
    x = np.asarray(x, dtype=float).reshape(-1)
    max_u = np.asarray(max_u, dtype=float).reshape(-1)
    random_scores = rng.random(x.shape[0])

    if model is None:
        learned_scores = np.zeros_like(random_scores)
        v = np.full_like(random_scores, 1.0 / len(random_scores))
        M = np.zeros((len(random_scores), len(random_scores)), dtype=float)
    else:
        learned_scores, v, M = learned_lambda_mix_scores(
            model,
            x,
            desired_opinion=desired_opinion,
            lambda_mix=lambda_mix,
            device=device,
        )

    eps = float(np.clip(epsilon, 0.0, 1.0))
    combined_scores = (1.0 - eps) * normalize_scores(learned_scores) + eps * normalize_scores(random_scores)
    action = waterfill_from_scores(combined_scores, max_u=max_u, budget=budget)
    info = dict(
        epsilon=eps,
        learned_scores=learned_scores,
        random_scores=random_scores,
        combined_scores=combined_scores,
        learned_centrality=v,
        learned_matrix=M,
    )
    return action, info


def _fit_identifier(
    model: GraphIdentifierEnv,
    buf_x: List[Array],
    buf_y: List[Array],
    *,
    lr: float,
    batch_size: int,
    max_steps: int,
    mae_stop: float,
    fit_check_every: int,
    device: str,
    suppress_fit_logs: bool = True,
) -> Tuple[Array, Dict[str, Any]]:
    """Fit/re-fit identifier and return A_hat plus lightweight fit info."""
    X = np.concatenate(buf_x, axis=0)
    Y = np.concatenate(buf_y, axis=0)
    t0 = time.perf_counter()
    if suppress_fit_logs:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            A_hat = train_graph_identifier(
                model,
                X,
                Y,
                lr=lr,
                batch_size=batch_size,
                max_steps=max_steps,
                mae_stop=mae_stop,
                device=device,
                fit_check_every=fit_check_every,
                verbose_every=0,
            )
    else:
        A_hat = train_graph_identifier(
            model,
            X,
            Y,
            lr=lr,
            batch_size=batch_size,
            max_steps=max_steps,
            mae_stop=mae_stop,
            device=device,
            fit_check_every=fit_check_every,
        )
    elapsed = time.perf_counter() - t0
    # Compute post-fit MAE and identity MAE for logging.
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)
        pred = model.predict_next(Xt)
        mae = float((pred - Yt).abs().mean().detach().cpu().item())
        identity_mae = float((Xt - Yt).abs().mean().detach().cpu().item())
    return A_hat, dict(
        n_pairs=int(X.shape[0]),
        fit_elapsed_s=float(elapsed),
        train_mae=float(mae),
        identity_mae=float(identity_mae),
        model_over_identity=float(mae / identity_mae) if identity_mae > 1e-12 else np.nan,
        max_steps=int(max_steps),
        mae_stop=float(mae_stop),
        batch_size=int(batch_size),
        fit_check_every=int(fit_check_every),
    )


def default_epsilon_schedule(num_campaigns_total: int = 10, exploration_campaigns: int = 5) -> List[float]:
    """Campaign 0 is no-control; campaigns 1..exploration_campaigns decay to exploitation."""
    eps = [0.0 for _ in range(int(num_campaigns_total))]
    if exploration_campaigns <= 0:
        return eps
    vals = np.linspace(1.0, 0.2, int(exploration_campaigns))
    for offset, val in enumerate(vals, start=1):
        if offset < len(eps):
            eps[offset] = float(val)
    return eps


def run_single_shot_online_identification(
    env_template: Any,
    *,
    x0: Optional[Array] = None,
    random_initial_opinions: bool = True,
    initial_opinion_low: float = 0.0,
    initial_opinion_high: float = 1.0,
    num_campaigns_total: int = 10,
    t_campaign: Optional[float] = 1.0,
    t_s: Optional[float] = 0.1,
    B_campaign: float = 0.5,
    lambda_mix: float = 0.70,
    exploration_campaigns: int = 5,
    epsilon_schedule: Optional[Iterable[float]] = None,
    lr: float = 1e-3,
    l2_lambda: float = 0.0,
    fit_max_steps: int = 2_000,
    fit_mae_stop: float = 5e-3,
    fit_batch_size: int = 256,
    fit_check_every: int = 200,
    identifier_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    rng_seed: int = 0,
    suppress_fit_logs: bool = True,
) -> Dict[str, Any]:
    """Run one online identification/control trajectory on a cloned env."""
    rng = np.random.default_rng(int(rng_seed))
    env = make_env_from_template(env_template, t_campaign=t_campaign, t_s=t_s)
    N = int(env.num_agents)
    max_u = as_max_u_vec(env.max_u, N)

    state, _ = env.reset()
    if x0 is None and random_initial_opinions:
        x0 = rng.uniform(float(initial_opinion_low), float(initial_opinion_high), size=N)
    elif x0 is None:
        x0 = np.asarray(state, dtype=float).copy()
    else:
        x0 = np.asarray(x0, dtype=float).reshape(N)
    set_initial_state(env, x0)
    state = np.asarray(env.opinions, dtype=float).copy()

    eps_schedule = list(default_epsilon_schedule(num_campaigns_total, exploration_campaigns) if epsilon_schedule is None else epsilon_schedule)
    if len(eps_schedule) < int(num_campaigns_total):
        eps_schedule = eps_schedule + [0.0] * (int(num_campaigns_total) - len(eps_schedule))

    states: List[Array] = [state.copy()]
    actions: List[Array] = []
    rewards: List[float] = []
    inter_states: List[Array] = []
    A_hats: List[Array] = []
    v_hats_static: List[Array] = []
    v_hats_lambda: List[Array] = []
    fit_infos: List[Dict[str, Any]] = []
    policy_infos: List[Dict[str, Any]] = []
    buf_x: List[Array] = []
    buf_y: List[Array] = []

    model: Optional[GraphIdentifierEnv] = None
    done = trunc = False

    for k in range(int(num_campaigns_total)):
        if k == 0:
            action = np.zeros(N, dtype=float)
            policy_info = dict(epsilon=np.nan, phase="passive_initial")
        else:
            eps_k = float(eps_schedule[k])
            action, policy_info = exploratory_lambda_mix_action(
                model,
                state,
                desired_opinion=float(env.desired_opinion),
                max_u=max_u,
                budget=float(B_campaign),
                epsilon=eps_k,
                lambda_mix=float(lambda_mix),
                rng=rng,
                device=device,
            )
            policy_info["phase"] = "explore" if eps_k > 0 else "exploit"

        x_next, reward, done, trunc, info = env.step(action)
        inter = info.get("intermediate_states", None)
        if inter is None:
            raise RuntimeError("env.step did not return info['intermediate_states']")
        inter = np.asarray(inter, dtype=float)
        Xp, Yp = pairs_from_intermediate(inter)
        buf_x.append(Xp)
        buf_y.append(Yp)

        actions.append(np.asarray(action, dtype=float).copy())
        rewards.append(float(reward))
        inter_states.append(inter.copy())
        states.append(np.asarray(x_next, dtype=float).copy())
        policy_infos.append(policy_info)

        if model is None:
            model = GraphIdentifierEnv(
                N=N,
                s=float(env.t_s),
                l2_lambda=float(l2_lambda),
                zero_diag=True,
                **({} if identifier_kwargs is None else identifier_kwargs),
            )

        A_hat, fit_info = _fit_identifier(
            model,
            buf_x,
            buf_y,
            lr=float(lr),
            batch_size=int(fit_batch_size),
            max_steps=int(fit_max_steps),
            mae_stop=float(fit_mae_stop),
            fit_check_every=int(fit_check_every),
            device=device,
            suppress_fit_logs=bool(suppress_fit_logs),
        )
        fit_info["campaign"] = int(k)
        fit_infos.append(fit_info)
        A_hats.append(np.asarray(A_hat, dtype=float).copy())
        v_static = centrality_from_matrix(A_hat)
        v_hats_static.append(v_static)
        _, v_lambda, _ = learned_lambda_mix_scores(
            model,
            np.asarray(x_next, dtype=float),
            desired_opinion=float(env.desired_opinion),
            lambda_mix=float(lambda_mix),
            device=device,
        )
        v_hats_lambda.append(v_lambda)

        state = np.asarray(x_next, dtype=float).copy()
        if done or trunc:
            break

    return dict(
        env=env,
        model=model,
        x0=np.asarray(x0, dtype=float),
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        intermediate_states_list=inter_states,
        A_hats=A_hats,
        v_hats_static=v_hats_static,
        v_hats_lambda=v_hats_lambda,
        fit_infos=fit_infos,
        policy_infos=policy_infos,
        epsilon_schedule=np.asarray(eps_schedule[: len(actions)], dtype=float),
        env_template_kwargs=env_kwargs_from_env(env, t_campaign=t_campaign, t_s=t_s),
        params=dict(
            num_campaigns_total=int(num_campaigns_total),
            t_campaign=float(env.t_campaign),
            t_s=float(env.t_s),
            B_campaign=float(B_campaign),
            lambda_mix=float(lambda_mix),
            exploration_campaigns=int(exploration_campaigns),
            rng_seed=int(rng_seed),
            fit_max_steps=int(fit_max_steps),
            fit_mae_stop=float(fit_mae_stop),
            fit_batch_size=int(fit_batch_size),
            fit_check_every=int(fit_check_every),
        ),
    )


def rollout_fixed_action_policy(
    env_template: Any,
    x0: Array,
    *,
    num_campaigns_total: int,
    B_campaign: float,
    policy: str,
    v_true: Optional[Array] = None,
    zero_first_campaign: bool = True,
    t_campaign: Optional[float] = None,
    t_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Roll out oracle/uniform/no-control baselines on a fresh cloned environment."""
    env = make_env_from_template(env_template, t_campaign=t_campaign, t_s=t_s)
    N = int(env.num_agents)
    max_u = as_max_u_vec(env.max_u, N)
    env.reset()
    set_initial_state(env, x0)
    state = np.asarray(env.opinions, dtype=float).copy()

    states = [state.copy()]
    actions = []
    rewards = []
    inter_states = []

    for k in range(int(num_campaigns_total)):
        if policy == "no_control" or (zero_first_campaign and k == 0):
            action = np.zeros(N, dtype=float)
        elif policy == "uniform":
            action = uniform_budget_action(max_u, B_campaign)
        elif policy == "oracle_true_v":
            if v_true is None:
                raise ValueError("v_true is required for policy='oracle_true_v'")
            action, _ = centrality_based_continuous_control(env, float(B_campaign), v=np.asarray(v_true, dtype=float))
        else:
            raise ValueError(f"Unknown baseline policy: {policy!r}")

        x_next, reward, done, trunc, info = env.step(action)
        actions.append(np.asarray(action, dtype=float).copy())
        rewards.append(float(reward))
        states.append(np.asarray(x_next, dtype=float).copy())
        inter = info.get("intermediate_states", None)
        if inter is not None:
            inter_states.append(np.asarray(inter, dtype=float).copy())
        state = np.asarray(x_next, dtype=float).copy()
        if done or trunc:
            break

    return dict(
        env=env,
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        intermediate_states_list=inter_states,
    )


def trajectory_summary_metrics(states: Array, *, desired_opinion: float = 1.0) -> Dict[str, float]:
    """Basic rollout performance metrics."""
    X = np.asarray(states, dtype=float)
    return dict(
        mean_end=float(X[-1].mean()),
        min_end=float(X[-1].min()),
        max_end=float(X[-1].max()),
        mean_avg=float(X.mean(axis=1).mean()),
        min_avg=float(X.min(axis=1).mean()),
        final_mean_distance_to_desired=float(np.mean(np.abs(float(desired_opinion) - X[-1]))),
        final_max_distance_to_desired=float(np.max(np.abs(float(desired_opinion) - X[-1]))),
    )


def state_distance_metrics(states: Array, oracle_states: Array, *, prefix: str = "") -> Dict[str, float]:
    """State-level distance metrics against oracle trajectory."""
    X = np.asarray(states, dtype=float)
    O = np.asarray(oracle_states, dtype=float)
    T = min(len(X), len(O))
    X = X[:T]
    O = O[:T]
    D = X - O
    p = f"{prefix}" if prefix else ""
    return {
        f"{p}state_rmse_to_oracle_end": float(np.sqrt(np.mean(D[-1] ** 2))),
        f"{p}state_rmse_to_oracle_avg": float(np.mean(np.sqrt(np.mean(D ** 2, axis=1)))),
        f"{p}state_linf_to_oracle_end": float(np.max(np.abs(D[-1]))),
        f"{p}state_linf_to_oracle_max": float(np.max(np.abs(D))),
        f"{p}mean_delta_vs_oracle_end": float(X[-1].mean() - O[-1].mean()),
        f"{p}min_delta_vs_oracle_end": float(X[-1].min() - O[-1].min()),
    }
