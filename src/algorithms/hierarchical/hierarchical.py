# Hierarchical RL algorithm implementation
"""Two-level hierarchical control with skill selection and execution."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from .skill_manager import SkillManager, SkillEncoder
from .low_level_controller import LowLevelController


class HierarchicalAgent:
    """Hierarchical agent with skill manager and low-level controller."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = 'cpu',
        skill_dim: int = 8,
        skill_embedding_dim: int = 16,
        hidden_dim: int = 256,
        lr_high: float = 3e-4,
        lr_low: float = 3e-4,
        gamma: float = 0.99,
        skill_duration: int = 10,
    ):
        self.device = device
        self.gamma = gamma
        self.skill_duration = skill_duration
        self.skill_dim = skill_dim
        
        # High-level components
        self.skill_manager = SkillManager(obs_dim, skill_dim, hidden_dim).to(device)
        self.skill_encoder = SkillEncoder(skill_dim, skill_embedding_dim).to(device)
        
        # Low-level controller
        self.low_level = LowLevelController(
            obs_dim, action_dim, skill_embedding_dim, hidden_dim
        ).to(device)
        
        # Value networks for high-level
        self.high_value = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(device)
        
        # Value network for low-level
        self.low_value = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + skill_embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(device)
        
        # Optimizers
        self.high_optimizer = torch.optim.Adam(
            list(self.skill_manager.parameters()) + list(self.high_value.parameters()),
            lr=lr_high
        )
        self.low_optimizer = torch.optim.Adam(
            list(self.low_level.parameters()) + list(self.low_value.parameters()),
            lr=lr_low
        )
        
        # State tracking
        self.current_skill = None
        self.skill_embedding = None
        self.skill_step_count = 0
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using hierarchical policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            
            # Select skill if needed (every skill_duration steps)
            if self.current_skill is None or self.skill_step_count >= self.skill_duration:
                skill, _ = self.skill_manager.sample_skill(obs_tensor)
                self.current_skill = skill
                self.skill_embedding = self.skill_encoder(skill)
                self.skill_step_count = 0
            
            # Execute low-level action
            if deterministic:
                mean, _ = self.low_level(obs_tensor, self.skill_embedding)
                action = torch.tanh(mean)
            else:
                action, _ = self.low_level.sample(obs_tensor, self.skill_embedding)
            
            self.skill_step_count += 1
            
            action = action.cpu().numpy()[0]
        
        return action
    
    def reset(self):
        """Reset hierarchical state."""
        self.current_skill = None
        self.skill_embedding = None
        self.skill_step_count = 0
    
    def update(self, trajectories: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update both levels using collected trajectories."""
        obs = trajectories['obs']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        next_obs = trajectories['next_obs']
        dones = trajectories['dones']
        skills = trajectories['skills']
        
        # Update low-level controller
        low_loss = self._update_low_level(obs, actions, rewards, next_obs, dones, skills)
        
        # Update high-level skill manager (less frequently)
        high_loss = self._update_high_level(obs, rewards, next_obs, dones, skills)
        
        return {**low_loss, **high_loss}
    
    def _update_low_level(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        skills: torch.Tensor,
    ) -> Dict[str, float]:
        """Update low-level controller."""
        # Encode skills
        skill_embeddings = self.skill_encoder(skills)
        
        # Compute values
        with torch.no_grad():
            next_skill_embeddings = self.skill_encoder(skills)  # Assume skill persists
            next_values = self.low_value(torch.cat([next_obs, next_skill_embeddings], dim=-1))
            target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # Value loss
        current_values = self.low_value(torch.cat([obs, skill_embeddings], dim=-1))
        value_loss = F.mse_loss(current_values, target_values)
        
        # Policy loss (actor-critic)
        advantages = (target_values - current_values).detach()
        _, log_probs = self.low_level.sample(obs, skill_embeddings)
        policy_loss = -(log_probs * advantages).mean()
        
        # Total loss
        low_loss = policy_loss + value_loss
        
        self.low_optimizer.zero_grad()
        low_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.low_level.parameters()) + list(self.low_value.parameters()),
            max_norm=10.0
        )
        self.low_optimizer.step()
        
        return {
            'low_policy_loss': policy_loss.item(),
            'low_value_loss': value_loss.item(),
        }
    
    def _update_high_level(
        self,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        skills: torch.Tensor,
    ) -> Dict[str, float]:
        """Update high-level skill manager."""
        # Aggregate rewards over skill duration
        # For simplicity, use standard TD update
        
        with torch.no_grad():
            next_values = self.high_value(next_obs)
            target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # Value loss
        current_values = self.high_value(obs)
        value_loss = F.mse_loss(current_values, target_values)
        
        # Policy loss
        advantages = (target_values - current_values).detach()
        _, log_probs = self.skill_manager.sample_skill(obs)
        policy_loss = -(log_probs.unsqueeze(-1) * advantages).mean()
        
        # Total loss
        high_loss = policy_loss + value_loss
        
        self.high_optimizer.zero_grad()
        high_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.skill_manager.parameters()) + list(self.high_value.parameters()),
            max_norm=10.0
        )
        self.high_optimizer.step()
        
        return {
            'high_policy_loss': policy_loss.item(),
            'high_value_loss': value_loss.item(),
        }
    
    def save(self, path: str):
        """Save model state."""
        state = {
            'skill_manager': self.skill_manager.state_dict(),
            'skill_encoder': self.skill_encoder.state_dict(),
            'low_level': self.low_level.state_dict(),
            'high_value': self.high_value.state_dict(),
            'low_value': self.low_value.state_dict(),
            'high_optimizer': self.high_optimizer.state_dict(),
            'low_optimizer': self.low_optimizer.state_dict(),
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load model state."""
        state = torch.load(path, map_location=self.device)
        
        self.skill_manager.load_state_dict(state['skill_manager'])
        self.skill_encoder.load_state_dict(state['skill_encoder'])
        self.low_level.load_state_dict(state['low_level'])
        self.high_value.load_state_dict(state['high_value'])
        self.low_value.load_state_dict(state['low_value'])
        self.high_optimizer.load_state_dict(state['high_optimizer'])
        self.low_optimizer.load_state_dict(state['low_optimizer'])
