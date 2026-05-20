import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect


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
        l2_lambda: float = 1e-4,
        zero_diag: bool = True,
        hidden_dim: int = 16,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        anchor_lambda: float = 1e-2,
        odd_lambda: float = 1e-3,
        entropy_lambda: float = 1e-3,
        alpha_var_lambda: float = 1e-3,
        device: str | None = None,
    ):
        super().__init__()
        print(
            f"[identifier-init] class={self.__class__.__name__} "
            f"module={self.__class__.__module__} "
            f"file={inspect.getsourcefile(self.__class__)}:"
            f"{inspect.getsourcelines(self.__class__)[1]}"
        )
        
        self.N = int(N)
        self.s = float(s)
        self.l2_lambda = float(l2_lambda)
        self.zero_diag = bool(zero_diag)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.anchor_lambda = float(anchor_lambda)
        self.odd_lambda = float(odd_lambda)
        self.entropy_lambda = float(entropy_lambda)
        self.alpha_var_lambda = float(alpha_var_lambda)

        # Static adjacency parameterization.
        self.Theta = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

        # Shared pairwise state-dependent modulation of influence strength.
        # Input features per edge: [x_i, x_j, x_j - x_i].
        self.alpha_net = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.Tanh(),
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
        diff = xj - xi       # (B, N, N), entry (i,j) = x_j - x_i

        alpha = self.alpha(xi, xj)  # (B, N, N)
        weighted_diff = alpha * diff

        if self.zero_diag:
            weighted_diff = weighted_diff * self._diag_mask.unsqueeze(0)

        agg = (A.unsqueeze(0) * weighted_diff).sum(dim=2)  # (B, N)
        return x + self.s * agg

    def regularization_loss(self, sample_x: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Structural regularizers that keep the nonlinear interaction close to a
        consensus-like law unless the data strongly demands otherwise.
        """
        l2 = (self.Theta ** 2).sum()

        # Encourage sharper / lower-entropy rows in A_hat.
        A = self.A_hat()
        row_entropy = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(dim=1).mean()

        # Anchor local linear case: alpha(x, x) ~= 1 near the diagonal.
        z = torch.zeros(16, device=self.Theta.device)
        alpha0 = self.alpha(z, z)
        anchor = ((alpha0 - 1.0) ** 2).mean()

        # Mild symmetry preference around small disagreements.
        eps = self._eps.to(self.Theta.device)
        x_left = -eps * torch.ones(16, device=self.Theta.device)
        x_mid = torch.zeros(16, device=self.Theta.device)
        x_right = eps * torch.ones(16, device=self.Theta.device)
        oddish = ((self.alpha(x_left, x_mid) - self.alpha(x_mid, x_right)) ** 2).mean()

        # Keep alpha from varying too wildly over observed pairs.
        if sample_x is not None and sample_x.numel() > 0:
            xi = sample_x.unsqueeze(2)
            xj = sample_x.unsqueeze(1)
            alpha_vals = self.alpha(xi, xj)
            if self.zero_diag:
                alpha_vals = alpha_vals * self._diag_mask.unsqueeze(0)
                denom = self._diag_mask.sum() * max(sample_x.shape[0], 1)
                alpha_mean = alpha_vals.sum() / denom.clamp_min(1.0)
                alpha_var = ((alpha_vals - alpha_mean) ** 2).sum() / denom.clamp_min(1.0)
            else:
                alpha_var = alpha_vals.var(unbiased=False)
        else:
            alpha_var = torch.tensor(0.0, device=self.Theta.device)

        reg = (
            self.l2_lambda * l2
            + self.entropy_lambda * row_entropy
            + self.anchor_lambda * anchor
            + self.odd_lambda * oddish
            + self.alpha_var_lambda * alpha_var
        )
        parts = {
            "l2": l2.detach(),
            "row_entropy": row_entropy.detach(),
            "anchor": anchor.detach(),
            "oddish": oddish.detach(),
            "alpha_var": alpha_var.detach(),
        }
        return reg, parts

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        reg, reg_parts = self.regularization_loss(sample_x=x)
        total = mse + reg
        parts = {"mse": mse.detach(), **reg_parts}
        return total, parts


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
    alpha_warmup_steps: int = 1000,
):
    model.to(device)

    X = torch.tensor(data_x, dtype=torch.float32, device=device)
    Y = torch.tensor(data_x_next, dtype=torch.float32, device=device)

    n = X.shape[0]
    if n == 0:
        raise ValueError("No training pairs provided.")

    alpha_params = list(model.alpha_net.parameters())
    theta_params = [model.Theta]
    if alpha_warmup_steps > 0:
        opt = torch.optim.Adam(theta_params, lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(max_steps):
        if step == alpha_warmup_steps and alpha_warmup_steps > 0:
            opt = torch.optim.Adam(
                [
                    {"params": theta_params, "lr": lr},
                    {"params": alpha_params, "lr": lr * 0.5},
                ]
            )

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
