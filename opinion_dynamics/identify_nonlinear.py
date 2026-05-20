import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphIdentifierEnvNonlinear(nn.Module):
    """
    Simplified nonlinear graph identifier.

    Learn a static row-stochastic adjacency A_hat together with a shared
    state-dependent interaction law:

        x_i^+ = x_i + s * sum_j A_ij * alpha_phi(x_i, x_j) * (x_j - x_i)

    Compared with the previous version, this simplified variant keeps:
      - row-stochastic nonnegative A_hat
      - optional zero diagonal
      - bounded shared nonlinear modulator alpha_phi

    and removes:
      - structural regularization terms
      - staged / warmup training
      - separate learning rates for Theta and alpha_net
      - init-time source-location debug printing
    """

    def __init__(
        self,
        N: int,
        s: float,
        zero_diag: bool = True,
        hidden_dim: int = 16,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        device: str | None = None,
    ):
        super().__init__()

        self.N = int(N)
        self.s = float(s)
        self.zero_diag = bool(zero_diag)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        # Static adjacency parameterization.
        self.Theta = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

        # Shared pairwise state-dependent modulation of influence strength.
        # Input features per edge: [x_i, x_j, x_j - x_i].
        self.alpha_net = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.register_buffer("_diag_mask", 1.0 - torch.eye(self.N))

        if device is not None:
            self.to(device)

    def A_hat(self) -> torch.Tensor:
        """Row-stochastic, nonnegative adjacency with optional zero diagonal."""
        A = F.softmax(self.Theta, dim=1)
        if self.zero_diag:
            A = A * self._diag_mask
            rs = A.sum(dim=1, keepdim=True)
            rs = torch.where(rs > 0, rs, torch.ones_like(rs))
            A = A / rs
        return A

    def alpha(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        """
        xi, xj: tensors broadcastable to the same shape.
        Returns a modulation factor in [alpha_min, alpha_max].
        """
        xi, xj = torch.broadcast_tensors(xi, xj)
        diff = xj - xi
        feats = torch.stack([xi, xj, diff], dim=-1)
        raw = self.alpha_net(feats).squeeze(-1)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N)
        returns: (B, N)
        """
        A = self.A_hat()  # (N, N)

        xi = x.unsqueeze(2)  # (B, N, 1)
        xj = x.unsqueeze(1)  # (B, 1, N)
        diff = xj - xi       # (B, N, N), entry (i, j) = x_j - x_i

        alpha = self.alpha(xi, xj)  # (B, N, N)
        weighted_diff = alpha * diff

        if self.zero_diag:
            weighted_diff = weighted_diff * self._diag_mask.unsqueeze(0)

        agg = (A.unsqueeze(0) * weighted_diff).sum(dim=2)  # (B, N)
        return x + self.s * agg

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        """
        Plain one-step prediction loss (no regularizers).
        """
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        return mse, {"mse": mse.detach()}


# Optional alias so you can import it with a familiar name in experiments.
GraphIdentifierEnv = GraphIdentifierEnvNonlinear


def train_graph_identifier(
    model: GraphIdentifierEnvNonlinear,
    data_x: np.ndarray,
    data_x_next: np.ndarray,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_steps: int = 50_000,
    mae_stop: float = 1e-3,
    device: str = "cpu",
    fit_check_every: int = 200,
    verbose_every: int = 2000,
):
    """
    Simplified training loop:
      - single Adam optimizer over all parameters
      - no warmup schedule
      - no split learning rates
      - early stopping on full-dataset MAE
    """
    model.to(device)

    X = torch.tensor(data_x, dtype=torch.float32, device=device)
    Y = torch.tensor(data_x_next, dtype=torch.float32, device=device)

    n = X.shape[0]
    if n == 0:
        raise ValueError("No training pairs provided.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(max_steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        xb, yb = X[idx], Y[idx]

        loss, _ = model.loss(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % fit_check_every == 0 or step == max_steps - 1:
            with torch.no_grad():
                yhat = model.predict_next(X)
                mae = (yhat - Y).abs().mean().item()
            if mae <= mae_stop:
                break

        if verbose_every and step % verbose_every == 0:
            with torch.no_grad():
                yhat = model.predict_next(X)
                mae_dbg = (yhat - Y).abs().mean().item()
                A = model.A_hat()
                rs = A.sum(dim=1)
                alpha0 = model.alpha(
                    torch.zeros(1, device=device),
                    torch.zeros(1, device=device),
                ).item()
                print(
                    f"[fit-nonlinear-simple] step={step} mae={mae_dbg:.4g} "
                    f"| A_row_sum min/mean/max={rs.min().item():.3g}/{rs.mean().item():.3g}/{rs.max().item():.3g} "
                    f"| A min/max={A.min().item():.3g}/{A.max().item():.3g} "
                    f"| alpha(0,0)={alpha0:.3g}"
                )

    with torch.no_grad():
        A_hat = model.A_hat().detach().cpu().numpy()
    return A_hat


def pairs_from_intermediate(intermediate_states: np.ndarray):
    x = intermediate_states[:-1]
    x_next = intermediate_states[1:]
    return x, x_next
