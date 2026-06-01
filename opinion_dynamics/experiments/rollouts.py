"""Rollout and control-allocation helpers used by the network opinion experiments.

This file intentionally contains one canonical definition per helper.  The
implementations below were extracted from the maintained inline notebook so the
modularized notebook follows the same control/evaluation behavior.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from opinion_dynamics.baseline import centrality_based_continuous_control


def _clone_env_from_template(env_template):
    """Create a fresh env with the same graph/parameters as env_template."""
    env, _ = _fresh_env_from_template(env_template, repeat_seed=getattr(env_template, "seed", None))
    return env


def make_env_with_dynamics(env_factory, seed: int, dynamics_model: str):
    """Build an env for one dynamics model, preserving the generated graph when needed."""
    try:
        return env_factory.get_randomized_env(seed=int(seed), dynamics_model=str(dynamics_model))
    except TypeError:
        base = env_factory.get_randomized_env(seed=int(seed))
        EnvCls = base.__class__
        kwargs = _env_template_kwargs_full(base, fallback_seed=int(seed))
        kwargs["dynamics_model"] = str(dynamics_model)
        return EnvCls(**kwargs)


def rollout_with_v(
    env_template,
    x0,
    num_campaigns_total,
    B_campaign,
    v_used,
    *,
    zero_first_campaign: bool = True,
):
    """
    Roll out centrality-based control from x0.

    The default preserves the source notebook/module behavior: controlled
    centrality rollouts use a zero-control campaign first, then control on
    campaigns 1..K-1. Set zero_first_campaign=False only for aligned
    campaign-0 ablations.
    """
    env = _clone_env_from_template(env_template)
    env.reset()
    env.opinions = np.array(x0, copy=True)

    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    states = [env.opinions.copy()]
    done = trunc = False

    for k in range(int(num_campaigns_total)):
        if (zero_first_campaign and k == 0) or v_used is None:
            uk = np.zeros(N, dtype=float)
        else:
            beta_k = min(float(B_campaign), float(ubar_vec.sum()))
            uk, _ = centrality_based_continuous_control(env, beta_k, v=v_used)

        x_next, _r, done, trunc, _info = env.step(uk)
        states.append(x_next.copy())

        if done or trunc:
            break

    return np.asarray(states)


def rollout_with_v_intermediate(
    env_template,
    x0,
    K_total,
    B_campaign,
    v_used,
    *,
    zero_first_campaign: bool = True,
):
    """Centrality/no-control rollout that also returns intermediate states."""
    env = _clone_env_from_template(env_template)

    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    env.reset()
    env.opinions = np.array(x0, copy=True)

    states = [env.opinions.copy()]
    actions, rewards = [], []
    inter_list, time_list = [], []
    dt = float(getattr(env, "t_s", 1.0))

    for k in range(int(K_total)):
        if (zero_first_campaign and k == 0) or v_used is None:
            uk = np.zeros(N, dtype=float)
        else:
            beta_k = min(float(B_campaign), float(ubar_vec.sum()))
            uk, _ = centrality_based_continuous_control(env, beta_k, v=v_used)

        x_next, r, done, trunc, info = env.step(uk)
        actions.append(uk.copy())
        rewards.append(float(r))
        states.append(x_next.copy())

        inter = info.get("intermediate_states", None)
        if inter is None:
            inter_list.append(None)
            time_list.append(None)
        else:
            inter_arr = np.asarray(inter, dtype=float)
            inter_list.append(inter_arr)
            time_list.append(dt * np.arange(inter_arr.shape[0], dtype=float))

        if done or trunc:
            break

    return {
        "states": np.asarray(states, dtype=float),
        "actions": np.asarray(actions, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "intermediate_states_list": inter_list,
        "intermediate_times_list": time_list,
        "env": env,
    }

def _maybe_copy(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return np.array(v, copy=True)
    try:
        return np.array(v, copy=True)
    except Exception:
        return v


def _env_template_kwargs_full(env, fallback_seed: int | None = None) -> dict[str, Any]:
    kwargs = dict(
        connectivity_matrix=np.array(env.connectivity_matrix, copy=True),
        num_agents=int(env.num_agents),
        max_u=np.array(env.max_u, copy=True),
        desired_opinion=float(env.desired_opinion),
        t_campaign=float(env.t_campaign),
        t_s=float(env.t_s),
        dynamics_model=str(getattr(env, "dynamics_model", "laplacian")),
        control_resistance=np.array(
            getattr(env, "control_resistance", np.zeros(env.num_agents)),
            copy=True,
        ),
        max_steps=int(getattr(env, "max_steps", 10_000)),
        opinion_end_tolerance=float(getattr(env, "opinion_end_tolerance", 0.01)),
        control_beta=float(getattr(env, "control_beta", 0.4)),
        normalize_reward=bool(getattr(env, "normalize_reward", False)),
        terminal_reward=float(getattr(env, "terminal_reward", 0.0)),
        terminate_when_converged=bool(getattr(env, "terminate_when_converged", True)),
        seed=int(getattr(env, "seed", fallback_seed))
        if getattr(env, "seed", None) is not None or fallback_seed is not None
        else None,
    )

    # Preserve dynamics-specific parameters when recreating fresh envs.
    for name in [
        "fj_lambda",
        "fj_prejudice",
        "hk_epsilon",
        "hk_include_self",
        "nonlinear_beta",
        "repulsion_epsilon",
        "repulsion_strength",
    ]:
        if hasattr(env, name):
            val = getattr(env, name)
            if val is not None:
                kwargs[name] = _maybe_copy(val)

    return kwargs


def _fresh_env_from_template(
    env_template,
    *,
    repeat_seed: int | None,
    initial_opinions: np.ndarray | None = None,
):
    EnvCls = env_template.__class__
    kwargs = _env_template_kwargs_full(env_template, fallback_seed=repeat_seed)
    kwargs["seed"] = repeat_seed
    if initial_opinions is not None:
        kwargs["initial_opinions"] = np.array(initial_opinions, copy=True)
    env = EnvCls(**kwargs)
    return env, kwargs


def waterfill_from_scores(
    scores: np.ndarray,
    max_u: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Greedy water-filling / top-score allocation with per-node caps.
    """
    scores = np.asarray(scores, dtype=float)
    max_u = np.asarray(max_u, dtype=float)
    u = np.zeros_like(max_u, dtype=float)

    remaining = min(float(beta), float(max_u.sum()))
    order = np.argsort(-scores)

    for idx in order:
        if remaining <= 1e-12:
            break
        alloc = min(float(max_u[idx]), remaining)
        u[idx] = alloc
        remaining -= alloc

    return u


def uniform_budget_action(max_u: np.ndarray, beta: float) -> np.ndarray:
    """
    Split budget as uniformly as possible across all nodes, while respecting max_u.
    """
    max_u = np.asarray(max_u, dtype=float)
    u = np.zeros_like(max_u, dtype=float)

    remaining = min(float(beta), float(max_u.sum()))
    active = np.ones(max_u.shape[0], dtype=bool)

    while remaining > 1e-12 and active.any():
        share = remaining / active.sum()
        capped = active & (max_u <= share + 1e-12)

        if capped.any():
            u[capped] = max_u[capped]
            remaining -= float(max_u[capped].sum())
            active[capped] = False
        else:
            u[active] = share
            remaining = 0.0

    return u


def centrality_budget_action_from_state(
    x: np.ndarray,
    *,
    v: np.ndarray,
    max_u: np.ndarray,
    beta: float,
    desired_opinion: float,
) -> np.ndarray:
    """
    Centrality-based campaign allocation using current state x and centrality vector v.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    scores = v * np.abs(float(desired_opinion) - x)
    return waterfill_from_scores(scores, max_u=np.asarray(max_u, dtype=float), beta=float(beta))


def apply_impulse_control(
    x: np.ndarray,
    u: np.ndarray,
    desired_opinion: float,
) -> np.ndarray:
    """
    Campaign impulse:
        x_ctrl = u * desired_opinion + (1 - u) * x
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    d = float(desired_opinion)
    return u * d + (1.0 - u) * x


def effective_adjacency_from_model_state(
    model,
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """
    Build a state-dependent effective adjacency by combining the learned static
    adjacency A_hat with the learned nonlinear gate alpha(x_i, x_j), then
    row-normalizing the result.

    A_eff(x) = row_normalize( A_hat ⊙ alpha(x) )
    """
    mdl = model.to(device).eval()
    x_arr = np.asarray(x, dtype=float)

    with torch.no_grad():
        x_t = torch.tensor(x_arr[None, :], dtype=torch.float32, device=device)
        A_hat = mdl.A_hat()  # (N, N)

        xi = x_t.unsqueeze(2)  # (1, N, 1)
        xj = x_t.unsqueeze(1)  # (1, 1, N)
        alpha = mdl.alpha(xi, xj).squeeze(0)  # (N, N)

        A_eff = A_hat * alpha

        if getattr(mdl, "zero_diag", False):
            A_eff = A_eff * mdl._diag_mask

        rs = A_eff.sum(dim=1, keepdim=True)
        rs = torch.where(rs > 0, rs, torch.ones_like(rs))
        A_eff = A_eff / rs

    return A_eff.detach().cpu().numpy()


def effective_centrality_from_model_state(
    model,
    x: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (A_eff(x), v_eff(x)) where v_eff is the centrality computed from the
    derived state-dependent effective adjacency.
    """
    A_eff = effective_adjacency_from_model_state(model, x, device=device)
    v_eff = compute_eigenvector_centrality(compute_laplacian(A_eff))
    return A_eff, np.asarray(v_eff, dtype=float)


def rollout_with_model_derived_control_intermediate(
    env,
    model,
    x0: np.ndarray,
    num_campaigns_total: int,
    B_campaign: float,
    *,
    device: str = "cpu",
):
    """
    Roll out the TRUE environment using a learned state-dependent control policy.

    At each campaign boundary:
      1) derive A_eff(x_k) from the learned model
      2) compute v_eff(x_k) from A_eff(x_k)
      3) allocate control using that v_eff(x_k)

    This uses the true environment for propagation, not the learned model.
    """
    x, _ = env.reset()

    mdl = copy.deepcopy(model).to(device).eval()

    states = [np.asarray(x, dtype=float).copy()]
    actions = []
    rewards = []
    intermediate_states_list = []
    intermediate_times_list = []
    effective_adjacencies = []
    effective_centralities = []

    dt = float(getattr(env, "t_s", 1.0))
    desired_opinion = float(getattr(env, "desired_opinion", 1.0))
    max_u = np.asarray(getattr(env, "max_u", np.ones_like(states[0])), dtype=float)

    for _k in range(int(num_campaigns_total)):
        x_curr = np.asarray(states[-1], dtype=float)

        A_eff, v_eff = effective_centrality_from_model_state(mdl, x_curr, device=device)
        u = centrality_budget_action_from_state(
            x_curr,
            v=v_eff,
            max_u=max_u,
            beta=B_campaign,
            desired_opinion=desired_opinion,
        )

        x_next, r, done, trunc, info = env.step(np.asarray(u, dtype=float))

        effective_adjacencies.append(np.asarray(A_eff, dtype=float))
        effective_centralities.append(np.asarray(v_eff, dtype=float))
        actions.append(np.asarray(u, dtype=float).copy())
        rewards.append(float(r))
        states.append(np.asarray(x_next, dtype=float).copy())

        inter = info.get("intermediate_states", None)
        if inter is None:
            intermediate_states_list.append(None)
            intermediate_times_list.append(None)
        else:
            inter_arr = np.asarray(inter, dtype=float)
            intermediate_states_list.append(inter_arr)
            intermediate_times_list.append(dt * np.arange(inter_arr.shape[0], dtype=float))

        if done or trunc:
            break

    return {
        "states": np.asarray(states, dtype=float),
        "actions": np.asarray(actions, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "intermediate_states_list": intermediate_states_list,
        "intermediate_times_list": intermediate_times_list,
        "effective_adjacencies": effective_adjacencies,
        "effective_centralities": effective_centralities,
    }


def rollout_with_policy_intermediate(
    env,
    x0: np.ndarray,
    num_campaigns_total: int,
    action_fn,
):
    """
    Roll out the TRUE environment using an external policy action_fn(x_current).
    """
    x, _ = env.reset()

    states = [np.asarray(x, dtype=float).copy()]
    actions = []
    rewards = []
    intermediate_states_list = []
    intermediate_times_list = []

    dt = float(getattr(env, "t_s", 1.0))

    for _k in range(int(num_campaigns_total)):
        x_curr = np.asarray(states[-1], dtype=float)
        u = np.asarray(action_fn(x_curr), dtype=float)

        x_next, r, done, trunc, info = env.step(u)

        actions.append(np.array(u, copy=True))
        rewards.append(float(r))
        states.append(np.asarray(x_next, dtype=float).copy())

        inter = info.get("intermediate_states", None)
        if inter is None:
            intermediate_states_list.append(None)
            intermediate_times_list.append(None)
        else:
            inter_arr = np.asarray(inter, dtype=float)
            intermediate_states_list.append(inter_arr)
            intermediate_times_list.append(dt * np.arange(inter_arr.shape[0], dtype=float))

        if done or trunc:
            break

    return {
        "states": np.asarray(states, dtype=float),
        "actions": np.asarray(actions, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "intermediate_states_list": intermediate_states_list,
        "intermediate_times_list": intermediate_times_list,
    }


def rollout_with_uniform_intermediate(
    env,
    x0: np.ndarray,
    num_campaigns_total: int,
    B_campaign: float,
):
    max_u = np.asarray(env.max_u, dtype=float)
    return rollout_with_policy_intermediate(
        env,
        x0,
        num_campaigns_total,
        action_fn=lambda x: uniform_budget_action(max_u=max_u, beta=B_campaign),
    )


def rollout_identifier_model_with_policy(
    model,
    env_template,
    x0: np.ndarray,
    num_campaigns_total: int,
    B_campaign: float,
    *,
    policy: str = "centrality",
    v: np.ndarray | None = None,
    device: str = "cpu",
):
    """
    Roll out the LEARNED model only:
      - apply campaign impulse in state space
      - then evolve using model.predict_next for the substeps inside each campaign
    """
    mdl = copy.deepcopy(model).to(device).eval()

    x = np.asarray(x0, dtype=float).copy()
    N = x.shape[0]
    desired_opinion = float(getattr(env_template, "desired_opinion", 1.0))
    max_u = np.asarray(getattr(env_template, "max_u", np.ones(N)), dtype=float)
    dt = float(getattr(env_template, "t_s", 1.0))
    t_campaign = float(getattr(env_template, "t_campaign", dt))
    steps_per_campaign = max(1, int(round(t_campaign / dt)))

    states = [x.copy()]
    actions = []
    intermediate_states_list = []
    intermediate_times_list = []

    for _k in range(int(num_campaigns_total)):
        if policy == "centrality":
            if v is None:
                raise ValueError("policy='centrality' requires a centrality vector v.")
            u = centrality_budget_action_from_state(
                x,
                v=np.asarray(v, dtype=float),
                max_u=max_u,
                beta=B_campaign,
                desired_opinion=desired_opinion,
            )
        elif policy == "uniform":
            u = uniform_budget_action(max_u=max_u, beta=B_campaign)
        elif policy == "none":
            u = np.zeros(N, dtype=float)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        x_ctrl = apply_impulse_control(x, u, desired_opinion=desired_opinion)

        campaign_states = [x_ctrl.copy()]
        xt = torch.tensor(x_ctrl[None, :], dtype=torch.float32, device=device)

        with torch.no_grad():
            for _ in range(steps_per_campaign):
                xt = mdl.predict_next(xt)
                x_sub = xt[0].detach().cpu().numpy()
                x_sub = np.clip(x_sub, 0.0, 1.0)
                campaign_states.append(x_sub.copy())

        x = campaign_states[-1].copy()
        states.append(x.copy())
        actions.append(u.copy())
        intermediate_states_list.append(np.asarray(campaign_states, dtype=float))
        intermediate_times_list.append(dt * np.arange(len(campaign_states), dtype=float))

    return {
        "states": np.asarray(states, dtype=float),
        "actions": np.asarray(actions, dtype=float),
        "intermediate_states_list": intermediate_states_list,
        "intermediate_times_list": intermediate_times_list,
    }


# Public aliases without leading underscores for reuse outside notebooks.
env_template_kwargs_full = _env_template_kwargs_full
fresh_env_from_template = _fresh_env_from_template
