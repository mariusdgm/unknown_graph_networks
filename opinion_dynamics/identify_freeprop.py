import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect

class GraphIdentifierEnvFreeProp(nn.Module):
    """
    Learn a static row-stochastic adjacency A_hat together with a shared
    propagation law on pairwise opinion differences.

    Update rule:
        x_i^+ = x_i + s * sum_j A_ij * g_phi(x_j - x_i)

    with
        g_phi(z) = z + residual_scale * h_phi(z)

    so the linear consensus-style law remains the backbone while the residual
    term adds flexibility to cover other propagation behaviors.
    """

    def __init__(
        self,
        N: int,
        s: float,
        l2_lambda: float = 0.0,
        zero_diag: bool = True,
        hidden_dim: int = 16,
        residual_scale: float = 0.25,
        anchor_lambda: float = 1e-2,
        odd_lambda: float = 1e-3,
        entropy_lambda: float = 0.0,
        prop_reg_lambda: float = 1e-3,
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
        self.residual_scale = float(residual_scale)
        self.anchor_lambda = float(anchor_lambda)
        self.odd_lambda = float(odd_lambda)
        self.entropy_lambda = float(entropy_lambda)
        self.prop_reg_lambda = float(prop_reg_lambda)

        self.Theta = nn.Parameter(torch.zeros(self.N, self.N))
        nn.init.kaiming_uniform_(self.Theta, a=0.0)

        # Small shared residual on scalar pairwise differences.
        self.prop_net = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
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
        A = F.softmax(self.Theta, dim=1)
        if self.zero_diag:
            A = A * self._diag_mask
            rs = A.sum(dim=1, keepdim=True)
            rs = torch.where(rs > 0, rs, torch.ones_like(rs))
            A = A / rs
        return A

    def g(self, diff: torch.Tensor) -> torch.Tensor:
        """Shared propagation law on pairwise differences."""
        residual = self.prop_net(diff.unsqueeze(-1)).squeeze(-1)
        return diff + self.residual_scale * residual

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N)
        returns: (B, N)
        """
        A = self.A_hat()  # (N, N)
        xi = x.unsqueeze(2)  # (B, N, 1)
        xj = x.unsqueeze(1)  # (B, 1, N)
        diff = xj - xi       # (B, N, N)

        msg = self.g(diff)
        if self.zero_diag:
            msg = msg * self._diag_mask.unsqueeze(0)

        agg = (A.unsqueeze(0) * msg).sum(dim=2)  # (B, N)
        return x + self.s * agg

    def regularization_loss(self, x_batch: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        l2 = (self.Theta ** 2).sum()

        # Encourage the learned propagation law to stay close to the linear one
        # near zero disagreement: g(0) ~= 0 and local odd symmetry.
        z = torch.zeros(32, device=self.Theta.device)
        g0 = self.g(z)
        anchor = (g0 ** 2).mean()

        eps = self._eps.to(self.Theta.device)
        zp = eps * torch.ones(32, device=self.Theta.device)
        zn = -eps * torch.ones(32, device=self.Theta.device)
        oddish = ((self.g(zp) + self.g(zn)) ** 2).mean()

        # Penalize row entropy lightly if requested to avoid very diffuse rows.
        A = self.A_hat()
        row_entropy = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(dim=1).mean()

        # Penalize propagation residual size on observed disagreements when available.
        if x_batch is not None:
            xi = x_batch.unsqueeze(2)
            xj = x_batch.unsqueeze(1)
            diff = xj - xi
            residual = self.g(diff) - diff
            if self.zero_diag:
                residual = residual * self._diag_mask.unsqueeze(0)
            prop_reg = (residual ** 2).mean()
        else:
            prop_reg = torch.zeros((), device=self.Theta.device)

        reg = (
            self.l2_lambda * l2
            + self.anchor_lambda * anchor
            + self.odd_lambda * oddish
            + self.entropy_lambda * row_entropy
            + self.prop_reg_lambda * prop_reg
        )
        parts = {
            "l2": l2.detach(),
            "anchor": anchor.detach(),
            "oddish": oddish.detach(),
            "row_entropy": row_entropy.detach(),
            "prop_reg": prop_reg.detach(),
        }
        return reg, parts

    def loss(self, x: torch.Tensor, x_next: torch.Tensor):
        x_hat = self.predict_next(x)
        mse = F.mse_loss(x_hat, x_next)
        reg, reg_parts = self.regularization_loss(x)
        total = mse + reg
        parts = {"mse": mse.detach(), **reg_parts}
        return total, parts


GraphIdentifierEnv = GraphIdentifierEnvFreeProp


def train_graph_identifier(
    model: GraphIdentifierEnvFreeProp,
    data_x: np.ndarray,
    data_x_next: np.ndarray,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_steps: int = 50_000,
    mae_stop: float = 1e-3,
    device: str = "cpu",
    fit_check_every: int = 200,
    verbose_every: int = 2000,
    prop_warmup_steps: int = 1000,
):
    model.to(device)

    X = torch.tensor(data_x, dtype=torch.float32, device=device)
    Y = torch.tensor(data_x_next, dtype=torch.float32, device=device)

    n = X.shape[0]
    if n == 0:
        raise ValueError("No training pairs provided.")

    theta_params = [model.Theta]
    prop_params = list(model.prop_net.parameters())

    opt_theta = torch.optim.Adam(theta_params, lr=lr)
    opt_all = torch.optim.Adam([{"params": theta_params}, {"params": prop_params}], lr=lr)

    for step in range(max_steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        xb, yb = X[idx], Y[idx]

        loss, parts = model.loss(xb, yb)
        if step < prop_warmup_steps:
            opt = opt_theta
        else:
            opt = opt_all

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
                z = torch.linspace(-1.0, 1.0, steps=5, device=device)
                gz = model.g(z).detach().cpu().numpy()
                print(
                    f"[fit-freeprop] step={step} mae={mae_dbg:.4g} "
                    f"| A_row_sum min/mean/max={rs.min().item():.3g}/{rs.mean().item():.3g}/{rs.max().item():.3g} "
                    f"| A min/max={A.min().item():.3g}/{A.max().item():.3g} "
                    f"| g([-1,-.5,0,.5,1])={np.round(gz, 3)}"
                )

    with torch.no_grad():
        A_hat = model.A_hat().detach().cpu().numpy()
    return A_hat


def pairs_from_intermediate(intermediate_states: np.ndarray):
    x = intermediate_states[:-1]
    x_next = intermediate_states[1:]
    return x, x_next
