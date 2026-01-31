# Actor and critic for policy learning in latent space
"""Policy and value networks operating in latent space."""

import torch
import torch.nn as nn
from typing import Tuple


class LatentActor(nn.Module):
    """Policy network in latent space."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 10,
        hidden_dim: int = 256,
        log_std_min: float = -10,
        log_std_max: float = 2,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log_std."""
        h = self.net(latent)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(latent)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class LatentCritic(nn.Module):
    """Value network in latent space."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict value from latent."""
        return self.net(latent)
