from __future__ import annotations

import inspect
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphIdentifierEnvNonlinearRegularized(nn.Module):
    """
    Final unbounded-alpha identifier with an optional graph regularizer.

    Base model (unchanged):
        A_hat = row-softmax(Theta), zero diagonal, row renormalized
        alpha_phi = softplus(f_phi) / log(2)
        M_phi(x) = A_hat * alpha_phi(x)

    regularizer_kind:
        "none"
            no regularization

        "theta_l2"
            literal paper-inspired penalty:
                R = sum_ij Theta_ij^2

        "entropy"
            normalized mean row entropy of A_hat:
                R = mean_i H(A_i) / log(N-1)
            Minimizing R favors concentrated rows.

        "gini"
            normalized mean Gini impurity of A_hat:
                R = mean_i [1 - sum_j A_ij^2] / [1 - 1/(N-1)]
            Minimizing R favors concentrated rows.

    The entropy and Gini penalties are bounded approximately in [0, 1] for
    zero-diagonal row-stochastic A_hat, which makes their lambda scales easier
    to interpret than the raw theta-L2 penalty.
    """

    VALID_REGULARIZERS = {"none", "theta_l2", "entropy", "gini"}

    def __init__(
        self,
        N: int,
        s: float,
        *,
        regularizer_kind: str = "none",
        regularizer_lambda: float = 0.0,
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
        self.zero_diag = bool(zero_diag)
        self.hidden_dim = int(hidden_dim)
        self.regularizer_kind = str(regularizer_kind)
        self.regularizer_lambda = float(regularizer_lambda)

        if self.regularizer_kind not in self.VALID_REGULARIZERS:
            raise ValueError(
                f"Unknown regularizer_kind={self.regularizer_kind!r}; "
                f"expected one of {sorted(self.VALID_REGULARIZERS)}"
            )
        if self.regularizer_lambda < 0:
            raise ValueError("regularizer_lambda must be nonnegative.")

        self.Theta = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

        self.alpha_net = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.register_buffer("_diag_mask", 1.0 - torch.eye(self.N))

        if device is not None:
            self.to(device)

    def A_hat(self) -> torch.Tensor:
        A = F.softmax(self.Theta, dim=1)
        if self.zero_diag:
            A = A * self._diag_mask
        rs = A.sum(dim=1, keepdim=True)
        rs = torch.where(rs > 0, rs, torch.ones_like(rs))
        return A / rs

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

    def regularizer_value(self) -> torch.Tensor:
        kind = self.regularizer_kind
        if kind == "none" or self.regularizer_lambda == 0.0:
            return self.Theta.new_zeros(())

        if kind == "theta_l2":
            # Literal form used in the reference paper.
            return (self.Theta ** 2).sum()

        A = self.A_hat()
        if kind == "entropy":
            eps = torch.finfo(A.dtype).eps
            entropy = -(A * torch.log(A.clamp_min(eps))).sum(dim=1)
            denom = math.log(max(2, self.N - 1))
            return (entropy / denom).mean()

        if kind == "gini":
            impurity = 1.0 - (A ** 2).sum(dim=1)
            denom = 1.0 - 1.0 / max(2, self.N - 1)
            return (impurity / denom).mean()

        raise AssertionError(kind)

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        reg = self.regularizer_value()
        total = mse + self.regularizer_lambda * reg
        return total, {
            "mse": mse.detach(),
            "regularizer": reg.detach(),
            "regularizer_weighted": (self.regularizer_lambda * reg).detach(),
        }


GraphIdentifierEnv = GraphIdentifierEnvNonlinearRegularized


def train_graph_identifier(
    model: GraphIdentifierEnvNonlinearRegularized,
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
    X = torch.tensor(data_x, dtype=torch.float32, device=device)
    Y = torch.tensor(data_x_next, dtype=torch.float32, device=device)
    n = X.shape[0]
    if n == 0:
        raise ValueError("No training pairs provided.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stop_reason = "max_steps"
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
                yhat = model.predict_next(X)
                mae = (yhat - Y).abs().mean().item()
            if mae <= float(mae_stop):
                stop_reason = "mae_stop"
                break

        if verbose_every and step % int(verbose_every) == 0:
            with torch.no_grad():
                yhat = model.predict_next(X)
                mae_dbg = (yhat - Y).abs().mean().item()
                A = model.A_hat()
                reg = float(model.regularizer_value().item())
            print(
                f"[fit-regularized] kind={model.regularizer_kind} "
                f"lambda={model.regularizer_lambda:g} step={step} "
                f"mae={mae_dbg:.4g} reg={reg:.4g} "
                f"Amax={A.max().item():.4g}"
            )

    model.last_fit_info = {
        "steps_run": int(steps_run),
        "stop_reason": str(stop_reason),
    }

    with torch.no_grad():
        return model.A_hat().detach().cpu().numpy()


def pairs_from_intermediate(intermediate_states: np.ndarray):
    x = intermediate_states[:-1]
    x_next = intermediate_states[1:]
    return x, x_next
