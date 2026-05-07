# SAC algorithm implementation
"""Soft Actor-Critic algorithm for continuous control."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class SAC:
    """Soft Actor-Critic algorithm."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = 'cpu',
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        hidden_dims: tuple = (256, 256),
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dims).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            
            if deterministic:
                mean, _ = self.actor(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(obs_tensor)
            
            action = action.cpu().numpy()[0]
        
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update networks using batch of transitions."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.detach()
            else:
                alpha = self.alpha
            
            target_q = rewards + (1 - dones) * self.gamma * (q_next - alpha * next_log_probs)
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.detach()
        else:
            alpha = self.alpha
        
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Update target networks
        self._update_target_network()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else 0.0,
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'mean_q': q1.mean().item(),
        }
    
    def _update_target_network(self):
        """Soft update target network."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save model state."""
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        
        if self.auto_entropy_tuning:
            state['log_alpha'] = self.log_alpha
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(state, path)
    
    def load(self, path: str):
        """Load model state."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        
        if self.auto_entropy_tuning and 'log_alpha' in state:
            self.log_alpha = state['log_alpha']
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
