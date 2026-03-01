import numpy as np
import torch

from opinion_dynamics.experiments.algos import centrality_based_continuous_control
from dynamic_programming.opinion_dynamics.algo_infinite_horizon import (
    value_iteration,
    extract_policy,
    create_state_grid,
)


def run_policy(env, policy, nx, max_steps=1000):
    """
    Run the simulation using a given policy with t_s and t_campaign dynamics.

    Args:
        env: The environment with network properties.
        policy: The control policy to be used (mapping from grid indices to actions).
        nx: Number of grid points per dimension (for discretizing state space).
        max_steps: Maximum number of control steps to run.

    Returns:
        opinions_over_time (np.ndarray): Opinions at each t_campaign boundary.
        time_points (np.ndarray): Corresponding times.
        rewards_over_time (np.ndarray): Rewards at each control step.
        actions_over_time (np.ndarray): Actions at each control step.
        all_intermediate_states (np.ndarray): Fine-grained state trajectories within each t_campaign.
    """
    N = env.num_agents
    t_campaign = env.t_campaign
    t_s = env.t_s

    current_time = 0.0
    time_points = [current_time]
    rewards_over_time = []
    actions_over_time = []
    all_intermediate_states = []

    grids = create_state_grid(N, nx)
    state, _ = env.reset()
    opinions_over_time = [state.copy()]

    for step in range(max_steps):
        # Discretize state to grid indices for policy lookup
        idx = tuple(np.abs(grids[i] - state[i]).argmin() for i in range(N))
        action = policy.get(idx, np.zeros(N))

        # Apply action and get next state + intermediate states
        next_state, reward, done, truncated, info = env.step(action)

        opinions_over_time.append(next_state.copy())
        rewards_over_time.append(reward)
        actions_over_time.append(action.copy())
        time_points.append(current_time)

        intermediate_states = info.get("intermediate_states")
        if intermediate_states is not None:
            all_intermediate_states.append(intermediate_states.copy())

        current_time += t_campaign
        state = next_state

        if done or truncated:
            print(
                f"Policy run terminated after {step+1} steps. Done={done}, Truncated={truncated}"
            )
            break

    return (
        np.array(opinions_over_time),  # (steps+1, agents)
        np.array(time_points),  # (steps+1,)
        np.array(rewards_over_time),  # (steps,)
        np.array(actions_over_time),  # (steps, agents)
        np.array(all_intermediate_states),  # (steps, num_substeps+1, agents)
    )


def flatten_intermediate_states_np(all_intermediate_states, t_campaign, t_s):
    """
    Flatten intermediate states and generate correctly aligned time points.

    Args:
        all_intermediate_states (List[np.ndarray]): Each array shape (num_substeps+1, num_agents)
        t_campaign (float): Duration per control step.
        t_s (float): Sampling time.

    Returns:
        flat_states (np.ndarray): Shape (total_samples, num_agents)
        flat_times (np.ndarray): Shape (total_samples,)
    """
    stacked = np.array(
        all_intermediate_states
    )  # (num_blocks, num_substeps+1, num_agents)
    S, K, N = stacked.shape

    # Flatten states: (S * K, N)
    flat_states = stacked.reshape(S * K, N)

    # Build times:
    per_block_times = np.arange(K) * t_s  # shape (K,)
    block_offsets = np.arange(S) * t_campaign  # shape (S,)
    full_times = (block_offsets[:, None] + per_block_times[None, :]).reshape(-1)

    # Fix: the last time of each block (except the very last one) should match the next block's impulse time
    # Identify indices of the last state in each block
    last_indices = np.arange(K - 1, S * K, K)
    for idx in last_indices[:-1]:  # skip the very last one (no next block)
        full_times[idx] = full_times[idx + 1]  # Set time equal to the next impulse

    return flat_states, full_times


def run_centrality_policy(env, available_budget, max_steps=1000):
    """
    Run the simulation using dynamic centrality-based continuous control.

    Args:
        env: The environment instance.
        available_budget (float): Total control budget per step.
        max_steps (int): Maximum number of steps to run.

    Returns:
        opinions_over_time (np.ndarray): Opinions at each t_campaign boundary (shape: steps+1, agents).
        time_points (np.ndarray): Time points at each t_campaign.
        rewards_over_time (np.ndarray): Rewards at each control step.
        actions_over_time (List[np.ndarray]): List of actions at each step.
        all_intermediate_states (List[np.ndarray]): All intermediate states at t_s granularity.
    """
    time_points = [0.0]
    rewards_over_time = []
    actions_over_time = []
    all_intermediate_states = []

    current_time = 0.0
    state, _ = env.reset()
    opinions_over_time = [state.copy()]

    for step in range(max_steps):
        action, controlled_nodes = centrality_based_continuous_control(
            env, available_budget
        )
        actions_over_time.append(action.copy())

        next_state, reward, done, truncated, info = env.step(action)

        # Store main opinions at each t_campaign boundary
        opinions_over_time.append(next_state.copy())
        rewards_over_time.append(reward)
        time_points.append(current_time)

        # Store intermediate states from info
        intermediate_states = info.get("intermediate_states")
        if intermediate_states is not None:
            all_intermediate_states.append(intermediate_states.copy())

        current_time += env.t_campaign
        state = next_state

        if done or truncated:
            print(
                f"Centrality-based policy finished after {step} steps. Done={done}, Truncated={truncated}"
            )
            break

    return (
        np.array(opinions_over_time),  # shape: (num_control_steps+1, num_agents)
        np.array(time_points),  # shape: (num_control_steps+1,)
        np.array(rewards_over_time),  # shape: (num_control_steps,)
        np.array(actions_over_time),  # list of length num_control_steps
        np.array(
            all_intermediate_states
        ),  # list of arrays, each (num_substeps+1, num_agents)
    )


def run_policy_agent(agent, max_steps=1000):
    """
    Run the simulation using the agent’s policy (exploitation only).

    Args:
        agent: An already-trained AgentDQN instance.
        max_steps: Maximum number of steps to run.

    Returns:
        opinions_over_time (np.ndarray): Array of opinions (states) over time (at t_campaign boundaries).
        time_points (np.ndarray): Array of time stamps at each control step.
        rewards_over_time (np.ndarray): Array of rewards collected at each step.
        actions_over_time (np.ndarray): Array of actions taken at each step.
        all_intermediate_states (np.ndarray): All intermediate states at t_s granularity.
    """
    time_points = []
    rewards_over_time = []
    actions_over_time = []
    opinions_over_time = []
    all_intermediate_states = []

    env = agent.validation_env
    current_time = 0.0

    # Reset environment
    state, _ = env.reset()
    opinions_over_time.append(state.copy())

    for step in range(max_steps):
        # Convert state to batched tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Agent selects action (exploitation only)
        action, _, _, _ = agent.select_action(
            state_tensor,
            epsilon=agent.validation_epsilon,
            random_action=False,
            action_noise=False,
        )
        action = np.squeeze(action)  # shape: (num_agents,)
        actions_over_time.append(action.copy())

        # Apply action
        next_state, reward, done, truncated, info = env.step(action)
        opinions_over_time.append(next_state.copy())
        rewards_over_time.append(reward)
        time_points.append(current_time)

        # Collect intermediate fine-grained states
        intermediate_states = info.get("intermediate_states")
        if intermediate_states is not None:
            all_intermediate_states.append(intermediate_states.copy())

        current_time += env.t_campaign
        state = next_state

        if done or truncated:
            break

    print(f"Simulation ended at step {step}: done={done}, truncated={truncated}")
    return (
        np.array(opinions_over_time),  # shape: (steps+1, num_agents)
        np.array(time_points),  # shape: (steps+1,)
        np.array(rewards_over_time),  # shape: (steps,)
        np.array(actions_over_time),  # shape: (steps, num_agents)
        np.array(all_intermediate_states),  # shape: (steps, num_substeps+1, num_agents)
    )


def run_no_control_policy(env, max_steps=1000):
    """
    Run the simulation with zero control at every step.

    Mirrors run_centrality_policy() outputs so you can compare apples-to-apples.

    Args:
        env: The environment instance.
        max_steps (int): Maximum number of steps to run.

    Returns:
        opinions_over_time (np.ndarray): Opinions at each t_campaign boundary (steps+1, agents).
        time_points (np.ndarray): Time points at each t_campaign boundary (steps+1,).
        rewards_over_time (np.ndarray): Rewards at each control step (steps,).
        actions_over_time (np.ndarray): Actions at each step (steps, agents).
        all_intermediate_states (np.ndarray): Intermediate states (steps, num_substeps+1, agents) if present.
    """
    time_points = [0.0]
    rewards_over_time = []
    actions_over_time = []
    all_intermediate_states = []

    current_time = 0.0
    state, _ = env.reset()
    opinions_over_time = [state.copy()]

    # Zero action (shape must match action space)
    zero_action = np.zeros(env.num_agents, dtype=np.float32)

    for step in range(max_steps):
        actions_over_time.append(zero_action.copy())

        next_state, reward, done, truncated, info = env.step(zero_action)

        opinions_over_time.append(next_state.copy())
        rewards_over_time.append(float(reward))

        current_time += env.t_campaign
        time_points.append(current_time)

        intermediate_states = info.get("intermediate_states")
        if intermediate_states is not None:
            all_intermediate_states.append(intermediate_states.copy())

        if done or truncated:
            print(
                f"No-control policy finished after {step} steps. Done={done}, Truncated={truncated}"
            )
            break

    return (
        np.array(opinions_over_time),
        np.array(time_points),
        np.array(rewards_over_time),
        np.array(actions_over_time),
        (
            np.array(all_intermediate_states)
            if len(all_intermediate_states) > 0
            else np.array([])
        ),
    )
