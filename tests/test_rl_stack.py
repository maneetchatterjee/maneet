# Unit tests for environment, checkpointing, and training
"""Test suite for RL stack."""

import pytest
import numpy as np
import torch
import os
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.biped_env import BipedEnv
from src.algorithms.sac.sac import SAC
from src.algorithms.sac.replay_buffer import ReplayBuffer
from src.utils.seeding import set_seed, get_rng_state, set_rng_state
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from src.utils.logging import Logger


class TestEnvironment:
    """Test environment functionality."""
    
    def test_env_creation(self):
        """Test environment can be created."""
        env = BipedEnv(use_gui=False)
        assert env is not None
        env.close()
    
    def test_env_reset(self):
        """Test environment reset."""
        env = BipedEnv(use_gui=False)
        obs = env.reset()
        assert obs is not None
        assert obs.shape == env.observation_space.shape
        env.close()
    
    def test_env_step(self):
        """Test environment stepping."""
        env = BipedEnv(use_gui=False)
        env.reset()
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_action_clipping(self):
        """Test action clipping to valid range."""
        env = BipedEnv(use_gui=False)
        env.reset()
        
        # Test extreme actions
        action = np.ones(env.action_space.shape) * 10.0  # Way out of bounds
        obs, reward, done, info = env.step(action)
        
        # Should not crash
        assert obs is not None
        env.close()


class TestSeeding:
    """Test seeding and RNG state."""
    
    def test_set_seed(self):
        """Test setting seed."""
        set_seed(42)
        
        # Generate random numbers
        r1 = np.random.rand()
        t1 = torch.rand(1).item()
        
        # Reset seed
        set_seed(42)
        
        # Should get same numbers
        r2 = np.random.rand()
        t2 = torch.rand(1).item()
        
        assert r1 == r2
        assert t1 == t2
    
    def test_rng_state(self):
        """Test RNG state save/restore."""
        set_seed(42)
        
        # Generate some random numbers
        np.random.rand(10)
        torch.rand(10)
        
        # Save state
        state = get_rng_state()
        
        # Generate more numbers
        r1 = np.random.rand()
        t1 = torch.rand(1).item()
        
        # Restore state
        set_rng_state(state)
        
        # Should get same numbers
        r2 = np.random.rand()
        t2 = torch.rand(1).item()
        
        assert r1 == r2
        assert t1 == t2


class TestCheckpointing:
    """Test checkpointing functionality."""
    
    def test_save_load_checkpoint(self):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model state
            model_state = {
                'weights': torch.randn(10, 10),
                'bias': torch.randn(10),
            }
            
            # Save checkpoint
            path = save_checkpoint(
                checkpoint_dir=tmpdir,
                step=1000,
                model_state=model_state,
                metadata={'epoch': 10},
            )
            
            assert os.path.exists(path)
            
            # Load checkpoint
            loaded = load_checkpoint(path, restore_rng=False)
            
            assert loaded['step'] == 1000
            assert torch.allclose(loaded['model_state']['weights'], model_state['weights'])
            assert torch.allclose(loaded['model_state']['bias'], model_state['bias'])
            assert loaded['metadata']['epoch'] == 10


class TestReplayBuffer:
    """Test replay buffer."""
    
    def test_buffer_add_sample(self):
        """Test adding and sampling from buffer."""
        buffer = ReplayBuffer(obs_dim=10, action_dim=3, capacity=100, device='cpu')
        
        # Add transitions
        for _ in range(50):
            obs = np.random.randn(10).astype(np.float32)
            action = np.random.randn(3).astype(np.float32)
            reward = np.random.randn()
            next_obs = np.random.randn(10).astype(np.float32)
            done = False
            
            buffer.add(obs, action, reward, next_obs, done)
        
        assert len(buffer) == 50
        
        # Sample batch
        batch = buffer.sample(32)
        
        assert batch['obs'].shape == (32, 10)
        assert batch['actions'].shape == (32, 3)
        assert batch['rewards'].shape == (32, 1)
        assert batch['next_obs'].shape == (32, 10)
        assert batch['dones'].shape == (32, 1)


class TestSmokeTraining:
    """Smoke test for training."""
    
    def test_sac_training_short(self):
        """Test SAC can train for a few steps without crashing."""
        set_seed(42)
        
        env = BipedEnv(use_gui=False)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device='cpu',
        )
        
        buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=1000,
            device='cpu',
        )
        
        obs = env.reset()
        
        # Run for 100 steps
        for step in range(100):
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)
            
            buffer.add(obs, action, reward, next_obs, done)
            
            if len(buffer) >= 32:
                batch = buffer.sample(32)
                update_info = agent.update(batch)
                assert 'critic_loss' in update_info
                assert 'actor_loss' in update_info
            
            obs = next_obs if not done else env.reset()
        
        env.close()
        
        # Test passed if no crash
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
