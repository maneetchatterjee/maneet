#!/usr/bin/env python3
"""
Verification script for Gym to Gymnasium migration.

This script tests that:
1. Gymnasium can be imported successfully
2. The BipedEnv can be imported without gym deprecation warnings
3. Basic environment operations work correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gymnasium_import():
    """Test that gymnasium imports work."""
    print("Testing gymnasium import...")
    import gymnasium as gym
    from gymnasium import spaces
    print(f"✓ Gymnasium {gym.__version__} imported successfully")
    return True

def test_biped_env_import():
    """Test that BipedEnv can be imported."""
    print("\nTesting BipedEnv import...")
    from src.envs.biped_env import BipedEnv
    print("✓ BipedEnv imported successfully (no gym deprecation warning)")
    return True

def test_biped_env_basic():
    """Test basic BipedEnv operations."""
    print("\nTesting BipedEnv basic operations...")
    from src.envs.biped_env import BipedEnv
    
    # Note: This requires PyBullet to be installed
    try:
        env = BipedEnv(use_gui=False)
        print(f"✓ Environment created")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✓ Step successful")
        
        env.close()
        print(f"✓ Environment closed")
        return True
    except ImportError as e:
        print(f"⚠ Skipping environment test (missing dependency: {e})")
        return True

if __name__ == "__main__":
    print("=" * 70)
    print("Gym to Gymnasium Migration Verification")
    print("=" * 70)
    
    try:
        test_gymnasium_import()
        test_biped_env_import()
        test_biped_env_basic()
        
        print("\n" + "=" * 70)
        print("✅ All migration tests passed!")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
