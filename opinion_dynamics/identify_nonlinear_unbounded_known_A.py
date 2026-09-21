from __future__ import annotations

import inspect
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphIdentifierEnvNonlinearKnownA(nn.Module):
    """
    Known-A ablation of the final unbounded nonlinear identifier.

    The true weighted influence matrix A is supplied at construction time and
    registered as a non-trainable buffer. Only the shared nonlinear alpha MLP
    is optimized.

    Kept identical to the final 2026-09-17 formulation:
      - alpha_phi(z) = softplus(f_phi(z)) / log(2)
      - pairwise features [x_i, x_j, |x_j-x_i|]
      - same hidden dimension and Euler prediction model

    Ablated component:
      - A_hat is NOT learned. A_hat() returns the exact fixed A_true.
    """

    def __init__(
        self,
        N: int,
        s: float,
        A_fixed: np.ndarray,
        l2_lambda: float = 0.0,
        zero_diag: bool = True,
        hidden_dim: int = 8,
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
        self.l2_lambda = float(l2_lambda)  # API compatibility; no Theta exists.
        self.zero_diag = bool(zero_diag)
        self.hidden_dim = int(hidden_dim)

        A = np.asarray(A_fixed, dtype=np.float32).copy()
        if A.shape != (self.N, self.N):
            raise ValueError(f"A_fixed must have shape ({self.N}, {self.N}), got {A.shape}")
        if np.any(~np.isfinite(A)):
            raise ValueError("A_fixed contains non-finite values")
        if np.any(A < -1e-8):
            raise ValueError("A_fixed must be nonnegative")
        if self.zero_diag and not np.allclose(np.diag(A), 0.0, atol=1e-7):
            raise ValueError("A_fixed must have zero diagonal for this ablation")
        if not np.allclose(A.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("A_fixed must be row-stochastic")

        self.register_buffer("_A_fixed", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("_diag_mask", 1.0 - torch.eye(self.N))

        self.alpha_net = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.last_fit_info: dict[str, object] = {}

        if device is not None:
            self.to(device)

    def A_hat(self) -> torch.Tensor:
        """Return the exact, fixed ground-truth weighted influence matrix."""
        return self._A_fixed

    def alpha(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        xi, xj = torch.broadcast_tensors(xi, xj)
        abs_diff = torch.abs(xj - xi)
        feats = torch.stack([xi, xj, abs_diff], dim=-1)
        raw = self.alpha_net(feats).squeeze(-1)
        return F.softplus(raw) / math.log(2.0)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A_hat()
        xi = x.unsqueeze(2)
        xj = x.unsqueeze(1)
        diff = xj - xi
        alpha = self.alpha(xi, xj)
        weighted_diff = alpha * diff

        if self.zero_diag:
            weighted_diff = weighted_diff * self._diag_mask.unsqueeze(0)

        agg = (A.unsqueeze(0) * weighted_diff).sum(dim=2)
        return x + self.s * agg

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        # The final learned-A model regularized only Theta. In the known-A
        # ablation Theta does not exist, so no replacement regularizer is added.
        l2 = torch.zeros((), dtype=mse.dtype, device=mse.device)
        return mse, {"mse": mse.detach(), "l2": l2.detach()}


GraphIdentifierEnv = GraphIdentifierEnvNonlinearKnownA


def train_graph_identifier(
    model: GraphIdentifierEnvNonlinearKnownA,
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
    """Optimize only alpha_net parameters; A_fixed is a registered buffer."""
    model.to(device)
    X = torch.tensor(data_x, dtype=torch.float32, device=device)
    Y = torch.tensor(data_x_next, dtype=torch.float32, device=device)
    n = X.shape[0]
    if n == 0:
        raise ValueError("No training pairs provided.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Known-A identifier has no trainable alpha parameters")
    opt = torch.optim.Adam(trainable, lr=lr)

    stop_reason = "max_steps"
    steps_run = 0
    last_checked_mae = float("nan")

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
                yhat = model.predict_next(X)
                last_checked_mae = float((yhat - Y).abs().mean().item())
            if last_checked_mae <= float(mae_stop):
                stop_reason = "mae_stop"
                break

        if verbose_every and step % int(verbose_every) == 0:
            with torch.no_grad():
                alpha0 = model.alpha(
                    torch.zeros(1, device=device),
                    torch.zeros(1, device=device),
                ).item()
            print(
                f"[fit-known-A-unbounded] step={step} mae={last_checked_mae:.4g} "
                f"| alpha(0,0)={alpha0:.3g}"
            )

    model.last_fit_info = {
        "steps_run": int(steps_run),
        "stop_reason": str(stop_reason),
        "last_checked_mae": float(last_checked_mae),
    }

    with torch.no_grad():
        return model.A_hat().detach().cpu().numpy()


def pairs_from_intermediate(intermediate_states: np.ndarray):
    x = intermediate_states[:-1]
    x_next = intermediate_states[1:]
    return x, x_next
