# Neural network architectures for SAC
"""Actor and Critic networks for SAC algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Actor(nn.Module):
    """Gaussian policy network."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Output layers
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Double Q-network critic."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        
        # Q1 network
        layers1 = []
        input_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            layers1.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        layers1.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers1)
        
        # Q2 network
        layers2 = []
        input_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            layers2.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        layers2.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*layers2)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both Q values."""
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
