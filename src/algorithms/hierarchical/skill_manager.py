# High-level skill manager with latent skill embedding
"""High-level policy that selects skills/options."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SkillManager(nn.Module):
    """High-level policy that selects latent skills."""
    
    def __init__(
        self,
        obs_dim: int,
        skill_dim: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.skill_dim = skill_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.skill_head = nn.Linear(hidden_dim, skill_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Select skill logits."""
        h = self.net(obs)
        logits = self.skill_head(h)
        return logits
    
    def sample_skill(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample skill from categorical distribution."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        skill = dist.sample()
        log_prob = dist.log_prob(skill)
        return skill, log_prob


class SkillEncoder(nn.Module):
    """Encode skill index to continuous embedding."""
    
    def __init__(
        self,
        skill_dim: int = 8,
        embedding_dim: int = 16,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(skill_dim, embedding_dim)
    
    def forward(self, skill: torch.Tensor) -> torch.Tensor:
        """Encode skill to embedding."""
        return self.embedding(skill)
