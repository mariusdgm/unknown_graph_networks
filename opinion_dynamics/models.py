import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math

def softplus_inv(y, beta=1.0):
    return (1.0/beta) * math.log(math.expm1(beta * y))

class OpinionNet(nn.Module):
    def __init__(
        self,
        nr_agents: int,
        nr_betas: int = 2,
        lin_hidden_size: int = 64,
        c_tanh_scale: Optional[
            float
        ] = None,  # keep None to match prior "winning" setup
        softplus_beta: float = 1.0,
        wstar_eps: float = 1e-6,
        return_w_star_default: bool = False,
        A_min: Optional[float] = None,  # e.g. 0.02
        A_max: Optional[float] = None,  # e.g. 3.0
        b_tanh_scale: Optional[float] = None,  # e.g. 1.5
        b_bias_scale: float = 1e-2,  # small random asymmetry for b
        use_layernorm: bool = True,
        layernorm_eps: float = 1e-5,
    ):
        super().__init__()
        self.nr_agents = nr_agents
        self.nr_betas = nr_betas
        self.lin_hidden_size = lin_hidden_size
        self.c_tanh_scale = c_tanh_scale
        self.softplus_beta = softplus_beta
        self.wstar_eps = wstar_eps
        self.return_w_star_default = return_w_star_default
        self.A_min = A_min
        self.A_max = A_max
        self.b_tanh_scale = b_tanh_scale
        self.b_bias_scale = b_bias_scale
        self.use_layernorm = use_layernorm

        # Trunk
        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_size, self.lin_hidden_size),
            nn.ReLU(),
        )
        self.trunk_ln = nn.LayerNorm(self.lin_hidden_size, eps=layernorm_eps)

        # Per-β head: [c | A_diag(N) | b(N)]
        self.predict_A_b_c = nn.Linear(
            self.lin_hidden_size, self.nr_betas * (2 * self.nr_agents + 1)
        )

        with torch.no_grad():
            # ----- Trunk init -----
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

            # Make the FIRST layer map any uniform input to 0: row-wise zero column-sum
            W0 = self.fc[0].weight  # shape: (hidden, nr_agents)
            W0.sub_(W0.mean(dim=1, keepdim=True))  # enforce W0 * 1 = 0
            # biases already zeroed above

            # ----- Head init (state-sensitive, but gentle near uniform) -----
            nn.init.kaiming_uniform_(self.predict_A_b_c.weight, a=0.0, nonlinearity="linear")
            self.predict_A_b_c.weight.mul_(0.1)  # small, nonzero => mild sensitivity near uniform

            full_bias = self.predict_A_b_c.bias  # shape: J*(2N+1)
            block = 2 * self.nr_agents + 1

            # Curvature for A so gradients exist but w* ~ 0 when b is tiny
            A_base = 0.5
            if (self.A_min is not None) and (A_base < float(self.A_min)):
                A_base = float(self.A_min) + 1e-6
            A_bias_val = softplus_inv(A_base, beta=self.softplus_beta)

            # Tiny asymmetry in b (symmetry breaking), but small enough that w* ~ 0
            b_eps = min(self.b_bias_scale, 1e-4)

            for j in range(self.nr_betas):
                off = j * block
                full_bias[off + 0] = 0.0  # c_j
                full_bias[off + 1 : off + 1 + self.nr_agents] = A_bias_val  # A diag biases
                full_bias[off + 1 + self.nr_agents : off + block] = torch.empty(self.nr_agents).uniform_(
                    -b_eps, b_eps
                )  # b_j

    def forward(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        return_w_star: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if return_w_star is None:
            return_w_star = self.return_w_star_default

        B = x.shape[0]
        features = self.fc(x)
        if self.use_layernorm:
            features = self.trunk_ln(features)

        A_b_c_net = self.predict_A_b_c(features).reshape(
            -1, self.nr_betas, 2 * self.nr_agents + 1
        )

        c = A_b_c_net[:, :, 0]
        A_raw = A_b_c_net[:, :, 1 : self.nr_agents + 1]
        b = A_b_c_net[:, :, self.nr_agents + 1 :]

        A_diag = F.softplus(A_raw, beta=self.softplus_beta)

        if (self.A_min is not None) or (self.A_max is not None):
            A_diag = torch.clamp(
                A_diag,
                min=self.A_min if self.A_min is not None else -float("inf"),
                max=self.A_max if self.A_max is not None else float("inf"),
            )
        A_diag = A_diag + torch.finfo(A_diag.dtype).eps

        if self.b_tanh_scale is not None:
            b = torch.tanh(b) * self.b_tanh_scale

        if self.c_tanh_scale is not None:
            c = torch.tanh(c) * self.c_tanh_scale

        output = {"A_diag": A_diag, "b": b, "c": c}

        if return_w_star or (w is not None):
            w_star = self.compute_w_star(A_diag, b, eps=self.wstar_eps)
            if return_w_star:
                output["w_star"] = w_star

        if w is not None:
            wJ = w.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)
            q = self.compute_q_values(wJ, A_diag, b, c)
            output["q"] = q

        return output

    @staticmethod
    def compute_w_star(
        A_diag: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        return b / (A_diag + eps)

    @staticmethod
    def compute_q_values(
        w: torch.Tensor, A_diag: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        # Make Q invariant to constant shifts in logits
        w = w - w.mean(dim=-1, keepdim=True)

        assert w.shape == A_diag.shape == b.shape, \
            f"Shape mismatch: w={w.shape}, A_diag={A_diag.shape}, b={b.shape}"
        quad = 0.5 * (A_diag * w.pow(2)).sum(dim=2)  # (B, J)
        lin  = (b * w).sum(dim=2)                    # (B, J)
        return c - quad + lin

    @staticmethod
    def apply_action_noise(w: torch.Tensor, noise_amplitude: float) -> torch.Tensor:
        noise = torch.randn_like(w) * noise_amplitude
        return w + noise

    @staticmethod
    def compute_action_from_w(
        w: torch.Tensor, beta: torch.Tensor, max_u: Optional[float] = None
    ) -> torch.Tensor:
        w_norm = F.softmax(w, dim=-1)
        u = w_norm * beta
        if max_u is not None:
            u = torch.clamp(u, max=max_u)
        return u

class OpinionNetCommonAB(nn.Module):
    def __init__(
        self,
        nr_agents: int,
        nr_betas: int = 2,
        lin_hidden_size: int = 64,
        c_tanh_scale: Optional[float] = None,
        softplus_beta: float = 1.0,  # softness for A
        wstar_eps: float = 1e-6,  # safety in w* division
        return_w_star_default: bool = False,  # default for forward()
        A_min: Optional[float] = None,  # e.g. 0.02
        A_max: Optional[float] = None,  # e.g. 3.0
        b_tanh_scale: Optional[float] = None,  # e.g. 1.5
        b_bias_scale: float = 1e-2,  # small random asymmetry for b
        use_layernorm: bool = True,
        layernorm_eps: float = 1e-5,
    ):
        super().__init__()
        self.nr_agents = nr_agents
        self.nr_betas = nr_betas
        self.lin_hidden_size = lin_hidden_size
        self.c_tanh_scale = c_tanh_scale
        self.softplus_beta = softplus_beta
        self.wstar_eps = wstar_eps
        self.return_w_star_default = return_w_star_default
        self.A_min = A_min
        self.A_max = A_max
        self.b_tanh_scale = b_tanh_scale
        self.b_bias_scale = b_bias_scale
        self.use_layernorm = use_layernorm

        # Trunk
        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_size, self.lin_hidden_size),
            nn.ReLU(),
        )
        self.trunk_ln = nn.LayerNorm(self.lin_hidden_size, eps=layernorm_eps)

        # Heads
        self.predict_shared_A_b = nn.Linear(self.lin_hidden_size, 2 * self.nr_agents)
        self.predict_c = nn.Linear(self.lin_hidden_size, self.nr_betas)

        # ---- Initialization: A≈softplus(0)=0.693, b≈U(±1e-2), c=0 ----
        with torch.no_grad():
            # Trunk
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

            # Heads
            nn.init.kaiming_uniform_(
                self.predict_shared_A_b.weight, a=0.0, nonlinearity="linear"
            )
            nn.init.kaiming_uniform_(
                self.predict_c.weight, a=0.0, nonlinearity="linear"
            )

            # Bias layout for [A(1..N) | b(1..N)]
            ab_bias = torch.zeros(2 * self.nr_agents)
            ab_bias[: self.nr_agents] = 0.0  # A bias -> softplus(0)=0.693
            ab_bias[self.nr_agents :] = torch.empty(self.nr_agents).uniform_(
                -self.b_bias_scale, self.b_bias_scale
            )  # small asymmetry for b
            self.predict_shared_A_b.bias.copy_(ab_bias)

            self.predict_c.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        return_w_star: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if return_w_star is None:
            return_w_star = self.return_w_star_default

        B = x.shape[0]
        features = self.fc(x)
        if self.use_layernorm:
            features = self.trunk_ln(features)

        A_b_shared = self.predict_shared_A_b(features)
        A_raw = A_b_shared[:, : self.nr_agents]
        b_shared = A_b_shared[:, self.nr_agents :]

        A_diag_single = F.softplus(A_raw, beta=self.softplus_beta)

        if (self.A_min is not None) or (self.A_max is not None):
            A_diag_single = torch.clamp(
                A_diag_single,
                min=self.A_min if self.A_min is not None else -float("inf"),
                max=self.A_max if self.A_max is not None else float("inf"),
            )
        A_diag_single = A_diag_single + torch.finfo(A_diag_single.dtype).eps

        if self.b_tanh_scale is not None:
            b_shared = torch.tanh(b_shared) * self.b_tanh_scale

        A_diag = A_diag_single.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)
        b = b_shared.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)

        c = self.predict_c(features)
        if self.c_tanh_scale is not None:
            c = torch.tanh(c) * self.c_tanh_scale

        out = {"A_diag": A_diag, "b": b, "c": c}

        if return_w_star or (w is not None):
            w_star = self.compute_w_star(A_diag, b, eps=self.wstar_eps)
            if return_w_star:
                out["w_star"] = w_star

        if w is not None:
            wJ = w.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)
            q = self.compute_q_values(wJ, A_diag, b, c)
            out["q"] = q

        return out

    @staticmethod
    def compute_w_star(
        A_diag: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        return b / (A_diag + eps)

    @staticmethod
    def compute_q_values(
        w: torch.Tensor, A_diag: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        # Make Q invariant to constant shifts in logits
        w = w - w.mean(dim=-1, keepdim=True)

        assert w.shape == A_diag.shape == b.shape, \
            f"Shape mismatch: w={w.shape}, A_diag={A_diag.shape}, b={b.shape}"
        quad = 0.5 * (A_diag * w.pow(2)).sum(dim=2)  # (B, J)
        lin  = (b * w).sum(dim=2)                    # (B, J)
        return c - quad + lin

    @staticmethod
    def apply_action_noise(w: torch.Tensor, noise_amplitude: float) -> torch.Tensor:
        return w + torch.randn_like(w) * noise_amplitude

    @staticmethod
    def compute_action_from_w(
        w: torch.Tensor, beta: torch.Tensor, max_u: Optional[float] = None
    ) -> torch.Tensor:
        w_norm = F.softmax(w, dim=-1)
        u = w_norm * beta
        if max_u is not None:
            u = torch.clamp(u, max=max_u)
        return u
