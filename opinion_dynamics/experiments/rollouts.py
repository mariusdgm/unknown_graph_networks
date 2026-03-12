import numpy as np

from opinion_dynamics.baseline import centrality_based_continuous_control


def _clone_env_from_template(env_template):
    """Create a fresh env with the *same* graph/params as env_template."""
    EnvCls = env_template.__class__
    kwargs = dict(
        connectivity_matrix=np.array(env_template.connectivity_matrix, copy=True),
        num_agents=int(env_template.num_agents),
        max_u=np.array(env_template.max_u, copy=True),
        desired_opinion=float(env_template.desired_opinion),
        t_campaign=float(env_template.t_campaign),
        t_s=float(env_template.t_s),
        dynamics_model=str(env_template.dynamics_model),
        control_resistance=np.array(env_template.control_resistance, copy=True),
        max_steps=int(getattr(env_template, "max_steps", 10_000)),
        opinion_end_tolerance=float(getattr(env_template, "opinion_end_tolerance", 0.01)),
        control_beta=float(getattr(env_template, "control_beta", 0.4)),
        normalize_reward=bool(getattr(env_template, "normalize_reward", False)),
        terminal_reward=float(getattr(env_template, "terminal_reward", 0.0)),
        terminate_when_converged=bool(getattr(env_template, "terminate_when_converged", True)),
        seed=int(getattr(env_template, "seed", 0)) if getattr(env_template, "seed", None) is not None else None,
    )
    return EnvCls(**kwargs)

def make_env_with_dynamics(env_factory, seed: int, dynamics_model: str):
    """
    Best-effort:
    1) try to ask the factory directly for coca/laplacian
    2) if factory doesn't accept it, build env normally then reconstruct with same graph but new dynamics
    """
    try:
        return env_factory.get_randomized_env(seed=int(seed), dynamics_model=str(dynamics_model))
    except TypeError:
        # factory signature doesn't accept dynamics_model; fallback:
        base = env_factory.get_randomized_env(seed=int(seed))
        EnvCls = base.__class__
        kwargs = dict(
            connectivity_matrix=np.array(base.connectivity_matrix, copy=True),
            num_agents=base.num_agents,
            max_u=np.array(base.max_u, copy=True),
            desired_opinion=float(base.desired_opinion),
            t_campaign=float(base.t_campaign),
            t_s=float(base.t_s),
            dynamics_model=str(dynamics_model),
            control_resistance=np.array(base.control_resistance, copy=True),
            max_steps=int(getattr(base, "max_steps", 10_000)),
            opinion_end_tolerance=float(getattr(base, "opinion_end_tolerance", 0.01)),
            control_beta=float(getattr(base, "control_beta", 0.4)),
            normalize_reward=bool(getattr(base, "normalize_reward", False)),
            terminal_reward=float(getattr(base, "terminal_reward", 0.0)),
            terminate_when_converged=bool(getattr(base, "terminate_when_converged", True)),
            seed=int(getattr(base, "seed", seed)) if getattr(base, "seed", None) is not None else int(seed),
        )
        return EnvCls(**kwargs)
    
def rollout_with_v(env_template, x0, num_campaigns_total, B_campaign, v_used):
    """
    campaign0: zero control
    campaign1..: centrality control using v_used (or None -> no control)
    returns states at campaign boundaries (K+1, N)
    """
    env = _clone_env_from_template(env_template)
    env.reset()
    env.opinions = np.array(x0, copy=True)

    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    states = [env.opinions.copy()]

    # campaign0: no control
    u0 = np.zeros(N, dtype=float)
    x1, r0, done, trunc, info0 = env.step(u0)
    states.append(x1.copy())

    for k in range(1, num_campaigns_total):
        if done or trunc:
            break
        if v_used is None:
            uk = np.zeros(N, dtype=float)
        else:
            beta_k = min(float(B_campaign), float(ubar_vec.sum()))
            uk, _ = centrality_based_continuous_control(env, beta_k, v=v_used)

        x_next, r, done, trunc, info = env.step(uk)
        states.append(x_next.copy())

    return np.asarray(states)

def rollout_with_policy_intermediate(env_template, x0, *, num_campaigns_total, B_campaign, v_used, mode="oracle"):
    """
    mode:
      - "oracle": campaign0 is zero-control, then centrality-based control using v_used each campaign
      - "nocontrol": always zero action
    Returns:
      env, states_boundaries, actions, rewards, inter_states, inter_times
    """
    EnvCls = env_template.__class__
    kwargs = dict(
        connectivity_matrix=np.array(env_template.connectivity_matrix, copy=True),
        num_agents=env_template.num_agents,
        max_u=np.array(env_template.max_u, copy=True),
        desired_opinion=float(env_template.desired_opinion),
        t_campaign=float(env_template.t_campaign),
        t_s=float(env_template.t_s),
        dynamics_model=str(env_template.dynamics_model),
        control_resistance=np.array(env_template.control_resistance, copy=True),
        max_steps=int(getattr(env_template, "max_steps", 10_000)),
        opinion_end_tolerance=float(getattr(env_template, "opinion_end_tolerance", 0.01)),
        control_beta=float(getattr(env_template, "control_beta", 0.4)),
        normalize_reward=bool(getattr(env_template, "normalize_reward", False)),
        terminal_reward=float(getattr(env_template, "terminal_reward", 0.0)),
        terminate_when_converged=bool(getattr(env_template, "terminate_when_converged", True)),
        seed=int(getattr(env_template, "seed", 0)) if getattr(env_template, "seed", None) is not None else None,
    )
    env = EnvCls(**kwargs)

    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    x, _ = env.reset()
    env.opinions = np.array(x0, copy=True)

    states = [env.opinions.copy()]
    actions = []
    rewards = []

    inter_states = []
    inter_times  = []

    for k in range(num_campaigns_total):
        if mode == "nocontrol" or (k == 0):
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
            raise RuntimeError("env.step did not return info['intermediate_states']")
        inter = np.asarray(inter)
        inter_states.append(inter.copy())

        Ksub = inter.shape[0] - 1
        t0 = k * float(env.t_campaign)
        times_k = t0 + np.arange(Ksub + 1) * float(env.t_s)
        inter_times.append(times_k)

        if done or trunc:
            break

    return env, np.array(states), np.array(actions), np.array(rewards), np.array(inter_states), np.array(inter_times)


def rollout_with_v_intermediate(env_template, x0, K_total, B_campaign, v_used, *, zero_first_campaign=True):
    """
    Roll out on a FRESH env cloned from env_template, starting from EXACTLY x0,
    collecting info['intermediate_states'] each campaign.

    IMPORTANT: If zero_first_campaign=True, campaign 0 uses u=0 even for oracle,
    matching the learned experiment's data-collection campaign.
    """
    # Always clone so we don't reuse a stepped env
    env = _clone_env_from_template(env_template)

    N = env.num_agents
    ubar_vec = np.asarray(env.max_u, dtype=float)

    # Reset then FORCE the starting state to x0 (NetworkGraph uses env.opinions / env.state)
    env.reset()
    env.opinions = np.array(x0, copy=True)

    states = [env.opinions.copy()]
    actions, rewards = [], []

    inter_list, time_list = [], []
    dt = float(getattr(env, "t_s", 1.0))

    for k in range(K_total):
        # --- CONTROL CHOICE ---
        if zero_first_campaign and k == 0:
            uk = np.zeros(N, dtype=float)
        else:
            if v_used is None:
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
            inter_arr = np.asarray(inter, dtype=float)  # (T_k, N)
            t = dt * np.arange(inter_arr.shape[0], dtype=float)
            inter_list.append(inter_arr)
            time_list.append(t)

        if done or trunc:
            break

    return {
        "states": np.asarray(states, dtype=float),          # (K+1, N)
        "actions": np.asarray(actions, dtype=float),        # (K, N)
        "rewards": np.asarray(rewards, dtype=float),        # (K,)
        "intermediate_states_list": inter_list,             # list of (T_k, N)
        "intermediate_times_list": time_list,               # list of (T_k,)
        "env": env,                                         # helpful for debugging
    }