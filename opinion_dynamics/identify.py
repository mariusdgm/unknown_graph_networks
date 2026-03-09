import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

from rl_envs_forge.envs.network_graph.graph_utils import (
    compute_laplacian,
    compute_eigenvector_centrality,
)

from opinion_dynamics.baseline import centrality_based_continuous_control


class GraphIdentifierEnv(nn.Module):
    """
    Learn nonnegative adjacency A_hat (not row-stochastic) to match env's Laplacian dynamics.
    Euler step: x_{t+s} = x_t + s * (A_hat x_t - D_hat x_t), where D_hat = diag(row_sums(A_hat)).
    """
    def __init__(self, N: int, s: float, l2_lambda: float = 0.0, zero_diag: bool = True):
        super().__init__()
        self.N = int(N)
        self.s = float(s)
        self.l2_lambda = float(l2_lambda)
        self.zero_diag = bool(zero_diag)

        self.Theta = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

        self.register_buffer("_diag_mask", (1.0 - torch.eye(self.N)))

    def A_hat(self) -> torch.Tensor:
        # row-stochastic, nonnegative
        A = F.softmax(self.Theta, dim=1)  # rows sum to 1

        # zero diagonal
        if self.zero_diag:
            A = A * self._diag_mask

            # renormalize
            rs = A.sum(dim=1, keepdim=True)
            rs = torch.where(rs > 0, rs, torch.ones_like(rs))
            A = A / rs

        return A

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A_hat()  # row-stochastic, diag=0
        I = torch.eye(self.N, device=x.device, dtype=x.dtype)
        L = I - A
        M = torch.matrix_exp(-L * self.s)   # (N,N)
        return x @ M.T                  # (B,N)

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        l2 = (self.Theta ** 2).sum()
        total = mse + self.l2_lambda * l2
        return total, {"mse": mse.detach(), "l2": l2.detach()}


def train_graph_identifier(
    model: GraphIdentifierEnv,
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
            
        if step % 2000 == 0:
            with torch.no_grad():
                yhat = model.predict_next(X)
                mae_dbg = (yhat - Y).abs().mean().item()
                A = model.A_hat()
                rs = A.sum(dim=1)
                print(
                    f"[fit] step={step} mae={mae_dbg:.4g} "
                    f"| A_row_sum min/mean/max={rs.min().item():.3g}/{rs.mean().item():.3g}/{rs.max().item():.3g} "
                    f"| A min/max={A.min().item():.3g}/{A.max().item():.3g}"
                )
    with torch.no_grad():
        A_hat = model.A_hat().detach().cpu().numpy()
    return A_hat


def pairs_from_intermediate(intermediate_states: np.ndarray):
    # intermediate_states: (K+1,N)
    x = intermediate_states[:-1]
    x_next = intermediate_states[1:]
    return x, x_next



def v_from_P(L: np.ndarray, Tc: float, tol: float = 1e-12) -> np.ndarray:
    """
    Stationary consensus-weight vector for discrete-time propagation P = exp(-L Tc).
    Returns v >= 0, sum(v)=1 such that v^T P = v^T.
    """
    P = expm(-L * Tc)
    w, VL = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    v = np.real(VL[:, idx])
    v = np.maximum(v, 0.0)
    s = v.sum()
    if s <= tol:
        # fallback (should be rare): use abs then normalize
        v = np.abs(np.real(VL[:, idx]))
        s = v.sum()
    return v / (s + tol)

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
    gi = GraphIdentifierEnv(N=N, s=env.t_s, diag_penalty=diag_penalty, l2_lambda=l2_lambda)

    for k in range(num_campaigns):
        # ---- 1) Train/update A_hat from all data so far
        X = np.concatenate(buf_x, axis=0) if len(buf_x) else None
        Y = np.concatenate(buf_y, axis=0) if len(buf_y) else None
        
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
