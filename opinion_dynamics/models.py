# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GraphIdentifier(nn.Module):
#     def __init__(self, N: int, s: float, diag_penalty: float = 1.0, l2_lambda: float = 0.0):
#         super().__init__()
#         self.N = N
#         self.s = float(s)
#         self.diag_penalty = float(diag_penalty)
#         self.l2_lambda = float(l2_lambda)

#         self.Theta = nn.Parameter(torch.zeros(N, N))
#         nn.init.kaiming_uniform_(self.Theta, a=0.0)

#     def A_hat(self) -> torch.Tensor:
#         # Row-stochastic
#         return F.softmax(self.Theta, dim=1)

#     def predict_next(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Euler step over sample interval s:
#             x_{l+1} = x_l + s * (A x_l - x_l)   (since rows sum to 1)
#         x: (B,N)
#         """
#         A = self.A_hat()           # (N,N)
#         Ax = x @ A.T               # (B,N)
#         return x + self.s * (Ax - x)

#     def loss(self, x: torch.Tensor, x_next: torch.Tensor):
#         x_hat = self.predict_next(x)
#         mse = F.mse_loss(x_hat, x_next)

#         diag_pen = torch.diagonal(self.Theta).sum()
#         l2 = (self.Theta ** 2).sum()

#         total = mse + self.diag_penalty * diag_pen + self.l2_lambda * l2
#         return total, {"mse": mse.detach(), "diag_pen": diag_pen.detach(), "l2": l2.detach()}