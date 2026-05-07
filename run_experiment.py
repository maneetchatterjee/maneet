# Main experiment runner with YAML configuration
"""Main training script for RL experiments."""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.envs.biped_env import BipedEnv
from src.algorithms.sac.sac import SAC
from src.algorithms.sac.replay_buffer import ReplayBuffer
from src.algorithms.dreamer.dreamer import Dreamer
from src.algorithms.hierarchical.hierarchical import HierarchicalAgent
from src.utils.seeding import set_seed
from src.utils.logging import Logger
from src.utils.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.utils.config import load_config, save_config
from src.utils.video import save_video


def train_sac(config, logger, env, device):
    """Train SAC agent."""
    # Create SAC agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        lr=config.agent.lr,
        gamma=config.agent.gamma,
        tau=config.agent.tau,
        alpha=config.agent.alpha,
        auto_entropy_tuning=config.agent.auto_entropy_tuning,
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=config.training.buffer_size,
        device=device,
        n_step=config.agent.n_step,
        gamma=config.agent.gamma,
    )
    
    # Training loop
    obs = env.reset()
    episode_reward = 0
    episode_step = 0
    episode_num = 0
    
    for step in tqdm(range(config.training.total_steps), desc="Training SAC"):
        # Select action
        if step < config.training.random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        episode_reward += reward
        episode_step += 1
        
        obs = next_obs
        
        # Update agent
        if step >= config.training.learning_starts and len(replay_buffer) >= config.training.batch_size:
            batch = replay_buffer.sample(config.training.batch_size)
            update_info = agent.update(batch)
            
            # Log training metrics
            if step % config.logging.log_interval == 0:
                for key, value in update_info.items():
                    logger.log_scalar(f"train/{key}", value, step)
        
        # Episode end
        if done or episode_step >= config.training.max_episode_steps:
            logger.log_scalar("train/episode_reward", episode_reward, step)
            logger.log_scalar("train/episode_length", episode_step, step)
            logger.log_episode({
                'step': step,
                'episode': episode_num,
                'reward': episode_reward,
                'length': episode_step,
            })
            
            obs = env.reset()
            episode_reward = 0
            episode_step = 0
            episode_num += 1
        
        # Checkpoint
        if step % config.training.checkpoint_interval == 0 and step > 0:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=config.paths.checkpoint_dir,
                step=step,
                model_state={'agent': agent.actor.state_dict()},
                optimizer_state={'agent_optimizer': agent.actor_optimizer.state_dict()},
                metadata={'episode': episode_num},
            )
            print(f"Saved checkpoint at step {step}")
        
        # Evaluation
        if step % config.training.eval_interval == 0 and step > 0:
            eval_reward = evaluate_agent(env, agent, config.training.eval_episodes)
            logger.log_scalar("eval/mean_reward", eval_reward, step)
            print(f"Step {step}: Eval reward = {eval_reward:.2f}")
    
    # Save final model
    agent.save(os.path.join(config.paths.checkpoint_dir, "final_model.pt"))
    
    return agent


def train_dreamer(config, logger, env, device):
    """Train Dreamer agent."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = Dreamer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        latent_dim=config.agent.latent_dim,
        hidden_dim=config.agent.hidden_dim,
        lr=config.agent.lr,
        gamma=config.agent.gamma,
        imagination_horizon=config.agent.imagination_horizon,
    )
    
    # Simplified training (collect sequences and update)
    sequences = []
    obs = env.reset()
    episode_reward = 0
    episode_step = 0
    episode_num = 0
    
    current_sequence = {'obs': [], 'actions': [], 'rewards': []}
    
    for step in tqdm(range(config.training.total_steps), desc="Training Dreamer"):
        # Select action
        action = agent.select_action(obs, deterministic=False)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store in sequence
        current_sequence['obs'].append(obs)
        current_sequence['actions'].append(action)
        current_sequence['rewards'].append([reward])
        
        episode_reward += reward
        episode_step += 1
        obs = next_obs
        
        # Episode end or sequence length reached
        if done or episode_step >= config.agent.seq_len or episode_step >= config.training.max_episode_steps:
            # Convert to tensors and add to buffer
            if len(current_sequence['obs']) >= config.agent.seq_len:
                seq_data = {
                    'obs': torch.FloatTensor(np.array(current_sequence['obs'][-config.agent.seq_len:])).unsqueeze(0).to(device),
                    'actions': torch.FloatTensor(np.array(current_sequence['actions'][-config.agent.seq_len:])).unsqueeze(0).to(device),
                    'rewards': torch.FloatTensor(np.array(current_sequence['rewards'][-config.agent.seq_len:])).unsqueeze(0).to(device),
                }
                sequences.append(seq_data)
            
            if done or episode_step >= config.training.max_episode_steps:
                logger.log_scalar("train/episode_reward", episode_reward, step)
                logger.log_scalar("train/episode_length", episode_step, step)
                
                obs = env.reset()
                agent.reset()
                episode_reward = 0
                episode_step = 0
                episode_num += 1
                current_sequence = {'obs': [], 'actions': [], 'rewards': []}
        
        # Update world model
        if len(sequences) >= config.training.batch_size and step % config.training.update_every == 0:
            batch_indices = np.random.choice(len(sequences), config.training.batch_size, replace=False)
            batch = {
                'obs': torch.cat([sequences[i]['obs'] for i in batch_indices], dim=0),
                'actions': torch.cat([sequences[i]['actions'] for i in batch_indices], dim=0),
                'rewards': torch.cat([sequences[i]['rewards'] for i in batch_indices], dim=0),
            }
            
            wm_info = agent.update_world_model(batch)
            ac_info = agent.update_actor_critic(batch)
            
            if step % config.logging.log_interval == 0:
                for key, value in {**wm_info, **ac_info}.items():
                    logger.log_scalar(f"train/{key}", value, step)
            
            # Keep buffer size manageable
            if len(sequences) > config.training.buffer_size // config.agent.seq_len:
                sequences = sequences[-config.training.buffer_size // config.agent.seq_len:]
        
        # Checkpoint
        if step % config.training.checkpoint_interval == 0 and step > 0:
            save_checkpoint(
                checkpoint_dir=config.paths.checkpoint_dir,
                step=step,
                model_state={'agent': agent.actor.state_dict()},
                metadata={'episode': episode_num},
            )
        
        # Evaluation
        if step % config.training.eval_interval == 0 and step > 0:
            eval_reward = evaluate_agent(env, agent, config.training.eval_episodes)
            logger.log_scalar("eval/mean_reward", eval_reward, step)
            print(f"Step {step}: Eval reward = {eval_reward:.2f}")
    
    agent.save(os.path.join(config.paths.checkpoint_dir, "final_model.pt"))
    return agent


def train_hierarchical(config, logger, env, device):
    """Train hierarchical agent."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = HierarchicalAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        skill_dim=config.agent.skill_dim,
        skill_embedding_dim=config.agent.skill_embedding_dim,
        lr_high=config.agent.lr_high,
        lr_low=config.agent.lr_low,
        gamma=config.agent.gamma,
        skill_duration=config.agent.skill_duration,
    )
    
    # Collect trajectories and update
    trajectories = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'next_obs': [],
        'dones': [],
        'skills': [],
    }
    
    obs = env.reset()
    episode_reward = 0
    episode_step = 0
    episode_num = 0
    
    for step in tqdm(range(config.training.total_steps), desc="Training Hierarchical"):
        # Select action
        action = agent.select_action(obs, deterministic=False)
        
        # Store skill
        skill = agent.current_skill.item() if agent.current_skill is not None else 0
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        trajectories['obs'].append(obs)
        trajectories['actions'].append(action)
        trajectories['rewards'].append([reward])
        trajectories['next_obs'].append(next_obs)
        trajectories['dones'].append([float(done)])
        trajectories['skills'].append(skill)
        
        episode_reward += reward
        episode_step += 1
        obs = next_obs
        
        # Update when buffer is full
        if len(trajectories['obs']) >= config.training.batch_size:
            batch = {
                'obs': torch.FloatTensor(np.array(trajectories['obs'])).to(device),
                'actions': torch.FloatTensor(np.array(trajectories['actions'])).to(device),
                'rewards': torch.FloatTensor(np.array(trajectories['rewards'])).to(device),
                'next_obs': torch.FloatTensor(np.array(trajectories['next_obs'])).to(device),
                'dones': torch.FloatTensor(np.array(trajectories['dones'])).to(device),
                'skills': torch.LongTensor(trajectories['skills']).to(device),
            }
            
            update_info = agent.update(batch)
            
            if step % config.logging.log_interval == 0:
                for key, value in update_info.items():
                    logger.log_scalar(f"train/{key}", value, step)
            
            # Clear trajectories
            trajectories = {k: [] for k in trajectories.keys()}
        
        # Episode end
        if done or episode_step >= config.training.max_episode_steps:
            logger.log_scalar("train/episode_reward", episode_reward, step)
            logger.log_scalar("train/episode_length", episode_step, step)
            
            obs = env.reset()
            agent.reset()
            episode_reward = 0
            episode_step = 0
            episode_num += 1
        
        # Checkpoint
        if step % config.training.checkpoint_interval == 0 and step > 0:
            save_checkpoint(
                checkpoint_dir=config.paths.checkpoint_dir,
                step=step,
                model_state={'agent': agent.skill_manager.state_dict()},
                metadata={'episode': episode_num},
            )
        
        # Evaluation
        if step % config.training.eval_interval == 0 and step > 0:
            eval_reward = evaluate_agent(env, agent, config.training.eval_episodes)
            logger.log_scalar("eval/mean_reward", eval_reward, step)
            print(f"Step {step}: Eval reward = {eval_reward:.2f}")
    
    agent.save(os.path.join(config.paths.checkpoint_dir, "final_model.pt"))
    return agent


def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate agent performance."""
    episode_rewards = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for bipedal robot")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Create directories
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, os.path.join(config.paths.log_dir, "config.yaml"))
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = BipedEnv(
        render_mode='rgb_array',
        use_gui=False,
        domain_randomization=config.env.get('domain_randomization', None),
        max_episode_steps=config.training.max_episode_steps,
    )
    
    # Create logger
    logger = Logger(config.paths.log_dir, "training")
    
    # Train based on algorithm type
    algorithm = config.algorithm.type
    print(f"Training algorithm: {algorithm}")
    
    if algorithm == "sac":
        agent = train_sac(config, logger, env, device)
    elif algorithm == "dreamer":
        agent = train_dreamer(config, logger, env, device)
    elif algorithm == "hierarchical":
        agent = train_hierarchical(config, logger, env, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Close logger
    logger.close()
    env.close()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
