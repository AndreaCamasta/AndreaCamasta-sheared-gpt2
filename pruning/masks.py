import torch
import torch.nn as nn
from typing import Optional

class HardConcreteMask(nn.Module):
    """
    Hard-concrete stochastic gates for structured pruning.

    This implements a differentiable relaxation of Bernoulli gates.
    It can be shaped as (n_layers,), (n_layers, n_heads), etc.
    """

    def __init__(
        self,
        shape,
        init_log_alpha: float = 0.0,
        beta: float = 2.0/3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        deterministic: bool = False,
    ):
        """
        Args:
            shape: tuple of ints, the shape of the mask tensor.
            init_log_alpha: initial value for log_alpha parameters.
            beta, gamma, zeta: hard-concrete distribution hyperparameters.
            deterministic: if True, always use the mean gate (no sampling).
        """
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(shape, init_log_alpha))
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.deterministic = deterministic

    def _sample_u(self, like: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(like)

    def _stretched_sigmoid(self, log_alpha: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Logistic noise
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        # Stretch to (gamma, zeta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        # Hard-sigmoid to [0, 1]
        return torch.clamp(s_bar, 0.0, 1.0)

    def sample(self) -> torch.Tensor:
        """
        Sample a mask in (0, 1) with gradient through log_alpha.
        """
        if self.deterministic or not self.training:
            # Use the mean gate: E[z] ≈ sigmoid(log_alpha) stretched & clamped
            u = torch.full_like(self.log_alpha, 0.5)
        else:
            u = self._sample_u(self.log_alpha)

        z = self._stretched_sigmoid(self.log_alpha, u)
        return z

    def forward(self, deterministic: Optional[bool] = None) -> torch.Tensor:
        """
        Return a mask tensor of shape `shape`.

        Args:
            deterministic: override self.deterministic for this call.
        """
        if deterministic is None:
            deterministic = self.deterministic
        if deterministic:
            u = torch.full_like(self.log_alpha, 0.5)
            return self._stretched_sigmoid(self.log_alpha, u)
        else:
            return self.sample()

    @property
    def expected_mask(self) -> torch.Tensor:
        """
        Approximate expected gate under the current log_alpha.
        Useful after training to rank units for pruning.
        """
        with torch.no_grad():
            u = torch.full_like(self.log_alpha, 0.5)
            return self._stretched_sigmoid(self.log_alpha, u)
