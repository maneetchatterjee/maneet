# Low-level controller conditioned on skills
"""Low-level policy that executes skills."""

import torch
import torch.nn as nn
from typing import Tuple


class LowLevelController(nn.Module):
    """Low-level policy conditioned on skill embedding."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        skill_embedding_dim: int = 16,
        hidden_dim: int = 256,
        log_std_min: float = -10,
        log_std_max: float = 2,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Condition on both observation and skill
        input_dim = obs_dim + skill_embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        obs: torch.Tensor,
        skill_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass conditioned on skill."""
        x = torch.cat([obs, skill_embedding], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(
        self,
        obs: torch.Tensor,
        skill_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(obs, skill_embedding)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class PDController(nn.Module):
    """PD controller for low-level stabilization."""
    
    def __init__(
        self,
        action_dim: int,
        kp: float = 1.0,
        kd: float = 0.1,
    ):
        super().__init__()
        
        # Learnable PD gains
        self.kp = nn.Parameter(torch.ones(action_dim) * kp)
        self.kd = nn.Parameter(torch.ones(action_dim) * kd)
        
        self.prev_error = None
    
    def forward(
        self,
        target: torch.Tensor,
        current: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PD control signal."""
        error = target - current
        
        # PD control
        control = self.kp * error - self.kd * velocity
        
        return torch.tanh(control)
    
    def reset(self):
        """Reset controller state."""
        self.prev_error = None
