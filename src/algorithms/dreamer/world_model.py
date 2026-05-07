# World model components: encoder, dynamics, decoder
"""Core neural network components for world model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Encoder(nn.Module):
    """Encode observations to latent representations."""
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent distribution."""
        h = self.net(obs)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        log_std = torch.clamp(log_std, -10, 2)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample latent from observation."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        z = mean + std * torch.randn_like(std)
        return z


class DynamicsModel(nn.Module):
    """Recurrent dynamics model (RSSM-style)."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 10,
        hidden_dim: int = 256,
        rnn_hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # RNN for temporal dependencies
        self.rnn = nn.GRUCell(latent_dim + action_dim, rnn_hidden_dim)
        
        # Prior network
        self.prior_net = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )
        
        # Posterior network (for training with observations)
        self.posterior_net = nn.Sequential(
            nn.Linear(rnn_hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """Initialize RNN hidden state."""
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)
    
    def forward(
        self,
        prev_latent: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
        obs_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through dynamics.
        Returns: next_latent, next_hidden, prior_dist, posterior_dist
        """
        # Update RNN hidden state
        rnn_input = torch.cat([prev_latent, action], dim=-1)
        next_hidden = self.rnn(rnn_input, hidden)
        
        # Compute prior
        prior_h = self.prior_net(next_hidden)
        prior_mean, prior_log_std = torch.chunk(prior_h, 2, dim=-1)
        prior_log_std = torch.clamp(prior_log_std, -10, 2)
        
        # If observation available, compute posterior
        if obs_latent is not None:
            posterior_input = torch.cat([next_hidden, obs_latent], dim=-1)
            posterior_h = self.posterior_net(posterior_input)
            posterior_mean, posterior_log_std = torch.chunk(posterior_h, 2, dim=-1)
            posterior_log_std = torch.clamp(posterior_log_std, -10, 2)
            
            # Sample from posterior during training
            std = posterior_log_std.exp()
            next_latent = posterior_mean + std * torch.randn_like(std)
        else:
            # Sample from prior during imagination
            posterior_mean, posterior_log_std = None, None
            std = prior_log_std.exp()
            next_latent = prior_mean + std * torch.randn_like(std)
        
        return next_latent, next_hidden, (prior_mean, prior_log_std), (posterior_mean, posterior_log_std)


class Decoder(nn.Module):
    """Decode latent to observation reconstruction."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        obs_dim: int = 37,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation."""
        return self.net(latent)


class RewardModel(nn.Module):
    """Predict reward from latent state."""
    
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
        """Predict reward from latent."""
        return self.net(latent)


class DoneModel(nn.Module):
    """Predict episode termination from latent state."""
    
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
        """Predict done probability from latent."""
        return torch.sigmoid(self.net(latent))
