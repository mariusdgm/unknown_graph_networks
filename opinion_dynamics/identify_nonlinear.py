import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect


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
      - optional L2 penalty on Theta (kept for compatibility with older callers)

    and removes:
      - structural regularization terms beyond optional L2
      - staged / warmup training
      - separate learning rates for Theta and alpha_net
      - init-time source-location debug printing
    """

    def __init__(
        self,
        N: int,
        s: float,
        l2_lambda: float = 0.0,
        zero_diag: bool = True,
        hidden_dim: int = 8,
        alpha_min: float = 0,
        alpha_max: float = 1,
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
        abs_diff = abs(xj - xi)
        feats = torch.stack([xi, xj, abs_diff], dim=-1)
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
        One-step prediction loss with optional L2 penalty on Theta.

        The L2 term is kept only for compatibility with older code paths that
        instantiate the identifier with l2_lambda=... as in the freeprop / linear
        identifier variants.
        """
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        l2 = (self.Theta ** 2).sum()
        total = mse + self.l2_lambda * l2
        return total, {"mse": mse.detach(), "l2": l2.detach()}


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
      - supports optional l2_lambda passed at model construction
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


def _safe_ratio_for_training(numer: float, denom: float, *, eps: float = 1e-12) -> float:
    denom = float(denom)
    if abs(denom) <= eps:
        return float("nan")
    return float(numer) / denom


def train_graph_identifier_with_info(
    model: GraphIdentifierEnvNonlinear,
    data_x: np.ndarray,
    data_x_next: np.ndarray,
    *,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_steps: int = 50_000,
    mae_stop: float = 1e-3,
    device: str = "cpu",
    fit_check_every: int = 200,
):
    """
    Train a graph identifier and return (A_hat, info).

    This is intentionally separate from train_graph_identifier so older notebooks
    keep working. The info dict is useful for amount-of-data diagnostics.
    """
    model.to(device)
    X = torch.tensor(np.asarray(data_x), dtype=torch.float32, device=device)
    Y = torch.tensor(np.asarray(data_x_next), dtype=torch.float32, device=device)
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("No training pairs provided.")

    def _flat_params(named_filter):
        vals = []
        with torch.no_grad():
            for name, p in model.named_parameters():
                if named_filter(name):
                    vals.append(p.detach().reshape(-1).cpu())
        return torch.cat(vals) if vals else torch.empty(0)

    with torch.no_grad():
        y0 = model.predict_next(X)
        mae_before = float((y0 - Y).abs().mean().item())
        identity_mae = float((Y - X).abs().mean().item())
        A_before = model.A_hat().detach().cpu().numpy()
        theta_before = _flat_params(lambda name: "Theta" in name)
        alpha_before = _flat_params(lambda name: "alpha_net" in name)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stop_reason = "max_steps"
    last_checked_mae = mae_before
    steps_run = 0

    for step in range(int(max_steps)):
        idx = torch.randint(0, n, (min(int(batch_size), n),), device=device)
        xb, yb = X[idx], Y[idx]
        loss, _ = model.loss(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps_run = step + 1

        if step % int(fit_check_every) == 0 or step == int(max_steps) - 1:
            with torch.no_grad():
                last_checked_mae = float((model.predict_next(X) - Y).abs().mean().item())
            if last_checked_mae <= float(mae_stop):
                stop_reason = "mae_stop"
                break

    with torch.no_grad():
        A_after = model.A_hat().detach().cpu().numpy()
        mae_after = float((model.predict_next(X) - Y).abs().mean().item())
        theta_after = _flat_params(lambda name: "Theta" in name)
        alpha_after = _flat_params(lambda name: "alpha_net" in name)

    ratio_before = _safe_ratio_for_training(mae_before, identity_mae)
    ratio_after = _safe_ratio_for_training(mae_after, identity_mae)
    info = dict(
        fit_steps_run=int(steps_run),
        fit_stop_reason=str(stop_reason),
        fit_mae_before=float(mae_before),
        fit_mae_after=float(mae_after),
        fit_mae_last_checked=float(last_checked_mae),
        fit_identity_mae=float(identity_mae),
        fit_model_over_identity_before=float(ratio_before),
        fit_model_over_identity_after=float(ratio_after),
        fit_improvement_over_identity_after=(1.0 - ratio_after) if np.isfinite(ratio_after) else float("nan"),
        fit_A_l1_change=float(np.sum(np.abs(A_after - A_before))),
        fit_A_linf_change=float(np.max(np.abs(A_after - A_before))),
        fit_theta_l2_change=float(torch.linalg.vector_norm(theta_after - theta_before).item()) if theta_after.numel() else float("nan"),
        fit_alpha_l2_change=float(torch.linalg.vector_norm(alpha_after - alpha_before).item()) if alpha_after.numel() else float("nan"),
    )
    return A_after, info
