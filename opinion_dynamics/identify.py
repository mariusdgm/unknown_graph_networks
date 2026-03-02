import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)


class GraphIdentifier(nn.Module):
    def __init__(
        self, N: int, s: float, diag_penalty: float = 1.0, l2_lambda: float = 0.0
    ):
        super().__init__()
        self.N = N
        self.s = float(s)
        self.diag_penalty = float(diag_penalty)
        self.l2_lambda = float(l2_lambda)

        self.Theta = nn.Parameter(torch.zeros(N, N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

    def A_hat(self) -> torch.Tensor:
        # Row-stochastic
        return F.softmax(self.Theta, dim=1)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        Euler step over sample interval s:
            x_{l+1} = x_l + s * (A x_l - x_l)   (since rows sum to 1)
        x: (B,N)
        """
        A = self.A_hat()  # (N,N)
        Ax = x @ A.T  # (B,N)
        return x + self.s * (Ax - x)

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)

        diag_pen = torch.diagonal(self.Theta).sum()
        l2 = (self.Theta**2).sum()

        total = mse + self.diag_penalty * diag_pen + self.l2_lambda * l2
        return total, {
            "mse": mse.detach(),
            "diag_pen": diag_pen.detach(),
            "l2": l2.detach(),
        }


def train_graph_identifier(
    model: GraphIdentifier,
    data_x: np.ndarray,  # (T,N)
    data_x_next: np.ndarray,  # (T,N)
    lr: float = 1e-3,
    batch_size: int = 64,
    max_steps: int = 50_000,
    mae_stop: float = 1e-3,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor(data_x, dtype=torch.float32, device=device)
    Y = torch.tensor(data_x_next, dtype=torch.float32, device=device)

    n = X.shape[0]
    if n == 0:
        raise ValueError("No training pairs provided.")

    for step in range(max_steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        xb, yb = X[idx], Y[idx]

        loss, parts = model.loss(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Check MAE on full buffer every so often (cheap for N=15)
        if step % 200 == 0 or step == max_steps - 1:
            with torch.no_grad():
                yhat = model.predict_next(X)
                mae = (yhat - Y).abs().mean().item()
            if mae <= mae_stop:
                break

    with torch.no_grad():
        A_hat = model.A_hat().detach().cpu().numpy()
    return A_hat


def pairs_from_intermediate(intermediate_states: np.ndarray):
    # intermediate_states: (K+1,N)
    x = intermediate_states[:-1]
    x_next = intermediate_states[1:]
    return x, x_next


def centrality_based_continuous_control(env, available_budget):
    """
    Compute a control action distributing a continuous budget based on centrality * deviation heuristic.

    Args:
        env: The environment instance with `opinions`, `desired_opinion`, `centralities`, and `max_u` attributes.
        available_budget (float): Total control budget to distribute.

    Returns:
        control_action (np.array): Control action array (N,) where 0 <= control_action[i] <= max_u
        controlled_nodes (list): List of indices of nodes that received some control
    """
    N = env.num_agents
    deviations = np.abs(env.opinions - env.desired_opinion)  # (N,)
    influence_powers = env.centralities * deviations  # (N,)
    agent_order = np.argsort(influence_powers)[::-1]  # Sort descending by power

    control_action = np.zeros(N)
    remaining_budget = available_budget

    controlled_nodes = []

    for agent_idx in agent_order:
        if remaining_budget <= 0:
            break

        assign_amount = min(float(env.max_u[agent_idx]), remaining_budget)
        control_action[agent_idx] = assign_amount
        controlled_nodes.append(agent_idx)
        remaining_budget -= assign_amount

    return control_action, controlled_nodes


def run_paper_baseline_like(
    env,
    B_total: float,
    num_campaigns: int = 5,  # paper: 5
    lr: float = 1e-3,
    diag_penalty: float = 1.0,
    l2_lambda: float = 0.0,
    device: str = "cpu",
    learn_first_without_control: bool = True,
):
    """
    Returns:
      A_hats: list of learned A_hat per campaign (after training at that stage)
      actions: list of actions applied
      states: list of boundary states (x at campaign boundaries)
    """
    N = env.num_agents
    ubar_vec = env.max_u
    desired = float(env.desired_opinion)

    # Reset
    x, _ = env.reset()
    states = [x.copy()]
    actions = []
    A_hats = []

    # Data buffer of observed pairs
    buf_x = []
    buf_y = []

    # Optional: initial observation window with zero control (paper uses t0>0 idea)
    if learn_first_without_control:
        zero = np.zeros(N, dtype=float)
        x_next, r, done, trunc, info = env.step(zero)
        inter = info["intermediate_states"]
        Xp, Yp = pairs_from_intermediate(inter)
        buf_x.append(Xp)
        buf_y.append(Yp)
        states.append(x_next.copy())

    B_rem = float(B_total)

    # Initialize graph learner
    gi = GraphIdentifier(N=N, s=env.t_s, diag_penalty=diag_penalty, l2_lambda=l2_lambda)

    for k in range(num_campaigns):
        # ---- 1) Train/update A_hat from all data so far
        X = np.concatenate(buf_x, axis=0) if len(buf_x) else None
        Y = np.concatenate(buf_y, axis=0) if len(buf_y) else None
        if X is None or len(X) == 0:
            # if no data yet, just use uniform A_hat (softmax(0))
            A_hat = np.full((N, N), 1.0 / N)
        else:
            A_hat = train_graph_identifier(gi, X, Y, lr=lr, device=device)
        A_hats.append(A_hat)

        # ---- 2) Compute v from learned graph
        L_hat = compute_laplacian(A_hat)
        v_hat = compute_eigenvector_centrality(L_hat)

        # ---- 3) Choose a per-campaign budget (placeholder)
        # Paper brute-forces beta_k under remaining budget; start simple:
        beta_k = min(
            B_rem, float(np.sum(ubar_vec))
        )  # can't spend more than saturating all nodes
        if beta_k <= 0:
            action = np.zeros(N, dtype=float)
        else:
            action, _ = centrality_based_continuous_control(env, beta_k, v=v_hat)

        # Apply and track spending
        spent = float(np.sum(action))
        B_rem -= spent
        actions.append(action.copy())

        # ---- 4) Step env, collect new observations
        x_next, r, done, trunc, info = env.step(action)
        inter = info["intermediate_states"]
        Xp, Yp = pairs_from_intermediate(inter)
        buf_x.append(Xp)
        buf_y.append(Yp)
        states.append(x_next.copy())

        if done or trunc:
            break

    return {
        "A_hats": A_hats,
        "actions": np.array(actions),
        "states": np.array(states),
    }
