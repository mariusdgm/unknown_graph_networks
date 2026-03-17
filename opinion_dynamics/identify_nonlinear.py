import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphIdentifierEnvNonlinear(nn.Module):
    """
    Learn a static row-stochastic adjacency A_hat together with a shared
    state-dependent interaction law.

    Update rule:
        x_i^+ = x_i + s * sum_j A_ij * alpha_phi(x_i, x_j) * (x_j - x_i)

    where:
      - A_hat is nonnegative, row-stochastic, and optionally zero-diagonal.
      - alpha_phi is a shared neural modulator applied to every pair (i, j).

    This keeps A_hat interpretable as the base influence graph while allowing
    nonlinear / state-dependent interaction strength, which is useful when the
    true dynamics are not well captured by the fixed linear consensus model.
    """

    def __init__(
        self,
        N: int,
        s: float,
        l2_lambda: float = 0.0,
        zero_diag: bool = True,
        hidden_dim: int = 32,
        alpha_min: float = 0.0,
        alpha_max: float = 2.0,
        anchor_lambda: float = 1e-3,
        odd_lambda: float = 1e-3,
        device: str | None = None,
    ):
        super().__init__()
        self.N = int(N)
        self.s = float(s)
        self.l2_lambda = float(l2_lambda)
        self.zero_diag = bool(zero_diag)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.anchor_lambda = float(anchor_lambda)
        self.odd_lambda = float(odd_lambda)

        # Static adjacency parameterization.
        self.Theta = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

        # Shared pairwise state-dependent modulation of influence strength.
        # Input features per edge: [x_i, x_j, x_j - x_i].
        self.alpha_net = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.register_buffer("_diag_mask", 1.0 - torch.eye(self.N))
        self.register_buffer("_eps", torch.tensor(1e-3))

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
        xi, xj: (...,) tensors broadcastable to same shape.
        Returns a modulation factor in [alpha_min, alpha_max].
        """
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
        diff = xj - xi       # (B, N, N), entry (i,j) = x_j - x_i

        alpha = self.alpha(xi, xj)  # (B, N, N)
        weighted_diff = alpha * diff

        if self.zero_diag:
            weighted_diff = weighted_diff * self._diag_mask.unsqueeze(0)

        agg = (A.unsqueeze(0) * weighted_diff).sum(dim=2)  # (B, N)
        return x + self.s * agg

    def regularization_loss(self) -> tuple[torch.Tensor, dict]:
        """
        Small structural regularizers to keep the nonlinear interaction close to
        a consensus-like law near zero disagreement.
        """
        l2 = (self.Theta ** 2).sum()

        # Anchor local linear case: alpha(x, x) ~= 1 near the diagonal.
        z = torch.zeros(16, device=self.Theta.device)
        alpha0 = self.alpha(z, z)
        anchor = ((alpha0 - 1.0) ** 2).mean()

        # Mild symmetry preference around small disagreements.
        # For equal and opposite perturbations around 0, alpha should not vary wildly.
        eps = self._eps.to(self.Theta.device)
        x_left = -eps * torch.ones(16, device=self.Theta.device)
        x_mid = torch.zeros(16, device=self.Theta.device)
        x_right = eps * torch.ones(16, device=self.Theta.device)
        oddish = ((self.alpha(x_left, x_mid) - self.alpha(x_mid, x_right)) ** 2).mean()

        reg = self.l2_lambda * l2 + self.anchor_lambda * anchor + self.odd_lambda * oddish
        parts = {
            "l2": l2.detach(),
            "anchor": anchor.detach(),
            "oddish": oddish.detach(),
        }
        return reg, parts

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        reg, reg_parts = self.regularization_loss()
        total = mse + reg
        parts = {"mse": mse.detach(), **reg_parts}
        return total, parts


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
                alpha0 = model.alpha(torch.zeros(1, device=device), torch.zeros(1, device=device)).item()
                print(
                    f"[fit-nonlinear] step={step} mae={mae_dbg:.4g} "
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
