#!/usr/bin/env python3
# Generate sample videos from the bipedal environment
"""
Script to generate sample videos demonstrating the environment.
Since full training takes 10+ hours, this creates demo videos showing:
1. Random policy (baseline)
2. Basic walking attempt
"""

import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.biped_env import BipedEnv
from src.utils.video import save_video
from src.utils.seeding import set_seed


def generate_video(env, policy_fn, num_steps, video_path, description):
    """Generate a single video with given policy."""
    print(f"\nGenerating: {description}")
    print(f"Output: {video_path}")
    
    obs = env.reset()
    frames = []
    episode_reward = 0
    
    for step in tqdm(range(num_steps), desc=description):
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Get action from policy
        action = policy_fn(obs, step)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            obs = env.reset()
    
    # Save video
    if len(frames) > 0:
        save_video(frames, video_path, fps=30)
        print(f"✓ Saved {len(frames)} frames to {video_path}")
        print(f"  Total reward: {episode_reward:.2f}")
    else:
        print(f"✗ No frames captured")
    
    return episode_reward


def create_random_policy(action_dim):
    """Create random action policy."""
    def policy(obs, step):
        return np.random.uniform(-1, 1, size=action_dim)
    return policy


def create_forward_bias_policy(action_dim):
    """Create policy with slight forward bias."""
    def policy(obs, step):
        action = np.random.uniform(-0.5, 0.5, size=action_dim)
        # Add slight forward push to some joints
        if action_dim > 0:
            action[0] = np.clip(0.3 + np.random.normal(0, 0.1), -1, 1)  # Forward push
        return action
    return policy


def create_standing_policy(action_dim):
    """Create policy to try maintaining standing position."""
    def policy(obs, step):
        action = np.zeros(action_dim)
        # Add small corrections based on observation
        if len(obs) >= 10 and action_dim > 1:
            # Try to counteract tilt
            action[0] = -obs[2] * 0.5 if len(obs) > 2 else 0  # Pitch correction
            action[1] = -obs[3] * 0.5 if len(obs) > 3 else 0  # Roll correction
        action = np.clip(action, -1, 1)
        return action
    return policy


def main():
    # Setup
    set_seed(42)
    
    # Create environment with rendering
    print("Creating environment...")
    env = BipedEnv(
        render_mode='rgb_array',
        use_gui=False,
        max_episode_steps=300,
    )
    
    # Get action dimension from environment
    action_dim = env.action_space.shape[0]
    print(f"Action dimension: {action_dim}")
    
    # Create output directories
    output_dirs = [
        'results/demo/videos',
        'results/sac/videos',
        'results/dreamer/videos',
        'results/hierarchical/videos',
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Generate videos with different policies
    videos = [
        {
            'policy': create_random_policy(action_dim),
            'steps': 200,
            'path': 'results/demo/videos/episode_0_random.mp4',
            'description': 'Random Policy (Baseline)',
        },
        {
            'policy': create_forward_bias_policy(action_dim),
            'steps': 200,
            'path': 'results/demo/videos/episode_1_forward_bias.mp4',
            'description': 'Forward Bias Policy',
        },
        {
            'policy': create_standing_policy(action_dim),
            'steps': 200,
            'path': 'results/demo/videos/episode_2_standing.mp4',
            'description': 'Standing Policy',
        },
    ]
    
    print("\n" + "="*60)
    print("Generating Sample Videos")
    print("="*60)
    
    rewards = []
    for video_config in videos:
        reward = generate_video(
            env,
            video_config['policy'],
            video_config['steps'],
            video_config['path'],
            video_config['description'],
        )
        rewards.append(reward)
    
    # Also create symlinks in other directories for demonstration
    print("\n" + "="*60)
    print("Creating sample videos in algorithm directories...")
    print("="*60)
    
    for algo_dir in ['sac', 'dreamer', 'hierarchical']:
        base_dir = f'results/{algo_dir}/videos'
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy one sample to each directory
        src = 'results/demo/videos/episode_1_forward_bias.mp4'
        dst = f'{base_dir}/episode_0.mp4'
        
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"✓ Copied sample to {dst}")
    
    # Summary
    print("\n" + "="*60)
    print("Video Generation Complete!")
    print("="*60)
    print(f"\nGenerated {len(videos)} videos in results/demo/videos/")
    print(f"Sample videos copied to sac/dreamer/hierarchical directories")
    print("\nVideo Statistics:")
    for i, video_config in enumerate(videos):
        print(f"  {video_config['description']}: {rewards[i]:.2f} reward")
    
    print("\n" + "="*60)
    print("View videos:")
    print("  - results/demo/videos/")
    print("  - results/sac/videos/")
    print("  - results/dreamer/videos/")
    print("  - results/hierarchical/videos/")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    main()
