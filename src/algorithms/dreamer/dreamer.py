# Dreamer algorithm implementation
"""DreamerV3-inspired algorithm with world model and imagination."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from .world_model import Encoder, DynamicsModel, Decoder, RewardModel, DoneModel
from .actor_critic import LatentActor, LatentCritic


class Dreamer:
    """Dreamer algorithm with world model."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = 'cpu',
        latent_dim: int = 64,
        hidden_dim: int = 256,
        rnn_hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        imagination_horizon: int = 15,
    ):
        self.device = device
        self.gamma = gamma
        self.lambda_ = lambda_
        self.imagination_horizon = imagination_horizon
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # World model components
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim).to(device)
        self.dynamics = DynamicsModel(latent_dim, action_dim, hidden_dim, rnn_hidden_dim).to(device)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim).to(device)
        self.reward_model = RewardModel(latent_dim, hidden_dim).to(device)
        self.done_model = DoneModel(latent_dim, hidden_dim).to(device)
        
        # Policy and value networks
        self.actor = LatentActor(latent_dim, action_dim, hidden_dim).to(device)
        self.critic = LatentCritic(latent_dim, hidden_dim).to(device)
        
        # Optimizers
        world_model_params = (
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.done_model.parameters())
        )
        self.world_model_optimizer = torch.optim.Adam(world_model_params, lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # State tracking
        self.hidden = None
        self.latent = None
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            
            # Encode observation
            latent = self.encoder.sample(obs_tensor)
            
            # Initialize hidden state if needed
            if self.hidden is None:
                self.hidden = self.dynamics.init_hidden(1, self.device)
                self.latent = torch.zeros(1, self.latent_dim, device=self.device)
            
            # Sample action from policy
            if deterministic:
                mean, _ = self.actor(latent)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(latent)
            
            # Update internal state for next step
            self.latent, self.hidden, _, _ = self.dynamics(
                self.latent, action, self.hidden, latent
            )
            
            action = action.cpu().numpy()[0]
        
        return action
    
    def reset(self):
        """Reset internal state."""
        self.hidden = None
        self.latent = None
    
    def update_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update world model using batch of sequences."""
        obs_seq = batch['obs']  # (batch, seq_len, obs_dim)
        action_seq = batch['actions']  # (batch, seq_len, action_dim)
        reward_seq = batch['rewards']  # (batch, seq_len, 1)
        
        batch_size, seq_len = obs_seq.shape[:2]
        
        # Encode observations
        obs_flat = obs_seq.reshape(-1, obs_seq.shape[-1])
        latent_mean, latent_log_std = self.encoder(obs_flat)
        latent_mean = latent_mean.reshape(batch_size, seq_len, -1)
        latent_log_std = latent_log_std.reshape(batch_size, seq_len, -1)
        
        # Sample latents
        std = latent_log_std.exp()
        obs_latents = latent_mean + std * torch.randn_like(std)
        
        # Initialize hidden state
        hidden = self.dynamics.init_hidden(batch_size, self.device)
        latent = torch.zeros(batch_size, self.latent_dim, device=self.device)
        
        # Rollout through sequence
        reconstruction_loss = 0
        kl_loss = 0
        reward_loss = 0
        
        for t in range(seq_len):
            # Dynamics step
            latent, hidden, prior_dist, posterior_dist = self.dynamics(
                latent, action_seq[:, t], hidden, obs_latents[:, t]
            )
            
            # Reconstruction loss
            obs_recon = self.decoder(latent)
            reconstruction_loss += F.mse_loss(obs_recon, obs_seq[:, t])
            
            # KL divergence between posterior and prior
            prior_mean, prior_log_std = prior_dist
            post_mean, post_log_std = posterior_dist
            
            if post_mean is not None:
                kl = self._compute_kl(post_mean, post_log_std, prior_mean, prior_log_std)
                kl_loss += kl.mean()
            
            # Reward prediction loss
            reward_pred = self.reward_model(latent)
            reward_loss += F.mse_loss(reward_pred, reward_seq[:, t])
        
        # Average losses over sequence
        reconstruction_loss /= seq_len
        kl_loss /= seq_len
        reward_loss /= seq_len
        
        # Total world model loss
        world_model_loss = reconstruction_loss + 0.1 * kl_loss + reward_loss
        
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.done_model.parameters()),
            max_norm=100.0
        )
        self.world_model_optimizer.step()
        
        return {
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_loss': reward_loss.item(),
            'world_model_loss': world_model_loss.item(),
        }
    
    def update_actor_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update actor and critic using imagination rollouts."""
        obs = batch['obs'][:, 0]  # Start from first observation
        batch_size = obs.shape[0]
        
        # Encode starting observations
        start_latent = self.encoder.sample(obs)
        hidden = self.dynamics.init_hidden(batch_size, self.device)
        
        # Imagine trajectories
        latents = [start_latent]
        actions = []
        rewards = []
        values = []
        
        latent = start_latent
        for t in range(self.imagination_horizon):
            # Sample action from policy
            action, _ = self.actor.sample(latent)
            actions.append(action)
            
            # Predict value
            value = self.critic(latent)
            values.append(value)
            
            # Imagine next latent using dynamics prior
            latent, hidden, _, _ = self.dynamics(latent, action, hidden, obs_latent=None)
            latents.append(latent)
            
            # Predict reward
            reward = self.reward_model(latent)
            rewards.append(reward)
        
        # Compute lambda returns
        returns = self._compute_lambda_returns(rewards, values)
        
        # Actor loss (maximize expected return)
        actor_loss = -torch.stack(returns).mean()
        
        # Critic loss (fit value to returns)
        values_tensor = torch.stack(values)
        returns_tensor = torch.stack(returns).detach()
        critic_loss = F.mse_loss(values_tensor, returns_tensor)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_value': values_tensor.mean().item(),
        }
    
    def _compute_kl(self, mean1, log_std1, mean2, log_std2):
        """Compute KL divergence between two Gaussian distributions."""
        var1 = log_std1.exp().pow(2)
        var2 = log_std2.exp().pow(2)
        
        kl = (log_std2 - log_std1) + (var1 + (mean1 - mean2).pow(2)) / (2 * var2) - 0.5
        return kl.sum(dim=-1)
    
    def _compute_lambda_returns(self, rewards, values):
        """Compute lambda returns (TD-lambda)."""
        returns = []
        last_value = values[-1] if values else 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_return = last_value
            else:
                next_return = returns[0]
            
            td_target = rewards[t] + self.gamma * next_return
            lambda_return = rewards[t] + self.gamma * (
                self.lambda_ * next_return + (1 - self.lambda_) * values[t]
            )
            returns.insert(0, lambda_return)
        
        return returns
    
    def save(self, path: str):
        """Save model state."""
        state = {
            'encoder': self.encoder.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'done_model': self.done_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load model state."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(state['encoder'])
        self.dynamics.load_state_dict(state['dynamics'])
        self.decoder.load_state_dict(state['decoder'])
        self.reward_model.load_state_dict(state['reward_model'])
        self.done_model.load_state_dict(state['done_model'])
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.world_model_optimizer.load_state_dict(state['world_model_optimizer'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
