# Replay buffer for SAC with n-step returns
"""Experience replay buffer with n-step returns."""

import numpy as np
import torch
from typing import Dict, Tuple
from collections import deque


class ReplayBuffer:
    """Replay buffer for off-policy RL."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 1000000,
        device: str = 'cpu',
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        # Storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.size = 0
        
        # N-step buffer
        if n_step > 1:
            self.n_step_buffer = deque(maxlen=n_step)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        if self.n_step > 1:
            self.n_step_buffer.append((obs, action, reward, next_obs, done))
            
            if len(self.n_step_buffer) < self.n_step:
                return
            
            # Compute n-step return
            obs, action = self.n_step_buffer[0][:2]
            next_obs, done = self.n_step_buffer[-1][3:]
            
            n_step_reward = 0.0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    next_obs = self.n_step_buffer[i][3]
                    done = True
                    break
            
            reward = n_step_reward
        
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'obs': torch.from_numpy(self.obs[indices]).to(self.device),
            'actions': torch.from_numpy(self.actions[indices]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
            'next_obs': torch.from_numpy(self.next_obs[indices]).to(self.device),
            'dones': torch.from_numpy(self.dones[indices]).to(self.device),
        }
        
        return batch
    
    def __len__(self) -> int:
        return self.size
