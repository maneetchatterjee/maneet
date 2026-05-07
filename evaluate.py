# Evaluation script for trained agents
"""Evaluate trained agents and generate videos."""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.envs.biped_env import BipedEnv
from src.algorithms.sac.sac import SAC
from src.algorithms.dreamer.dreamer import Dreamer
from src.algorithms.hierarchical.hierarchical import HierarchicalAgent
from src.utils.config import load_config
from src.utils.video import save_video
from src.utils.seeding import set_seed


def load_agent(config, checkpoint_path, device):
    """Load agent from checkpoint."""
    obs_dim = 37  # Default
    action_dim = 10  # Default
    
    algorithm = config.algorithm.type
    
    if algorithm == "sac":
        agent = SAC(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            lr=config.agent.lr,
            gamma=config.agent.gamma,
        )
    elif algorithm == "dreamer":
        agent = Dreamer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            latent_dim=config.agent.latent_dim,
            hidden_dim=config.agent.hidden_dim,
            lr=config.agent.lr,
            gamma=config.agent.gamma,
        )
    elif algorithm == "hierarchical":
        agent = HierarchicalAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            skill_dim=config.agent.skill_dim,
            skill_embedding_dim=config.agent.skill_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Load checkpoint
    agent.load(checkpoint_path)
    
    return agent


def evaluate_agent(env, agent, num_episodes=10, render=False, save_videos=False, video_dir=None):
    """Evaluate agent and optionally record videos."""
    episode_rewards = []
    episode_lengths = []
    energy_consumptions = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()
        
        episode_reward = 0
        episode_length = 0
        energy = 0
        frames = []
        
        done = False
        while not done:
            if render or save_videos:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            energy += info.get('energy_consumption', 0)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        energy_consumptions.append(energy)
        
        # Save video
        if save_videos and video_dir and len(frames) > 0:
            video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
            save_video(frames, video_path, fps=30)
    
    # Compute statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_energy': np.mean(energy_consumptions),
        'std_energy': np.std(energy_consumptions),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--save_videos", action="store_true", help="Save videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = BipedEnv(
        render_mode='rgb_array',
        use_gui=args.render,
        max_episode_steps=config.training.max_episode_steps,
    )
    
    # Load agent
    print(f"Loading agent from {args.checkpoint}")
    agent = load_agent(config, args.checkpoint, device)
    
    # Create video directory if needed
    video_dir = None
    if args.save_videos:
        video_dir = config.paths.video_dir
        Path(video_dir).mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    print(f"Evaluating for {args.num_episodes} episodes...")
    stats = evaluate_agent(
        env,
        agent,
        num_episodes=args.num_episodes,
        render=args.render,
        save_videos=args.save_videos,
        video_dir=video_dir,
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    print("="*50)
    
    env.close()


if __name__ == "__main__":
    main()
