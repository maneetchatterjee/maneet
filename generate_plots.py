# Utility script to generate comparison plots from TensorBoard logs
"""Generate publication-quality comparison plots."""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_tensorboard_scalars(log_dir: str, tags: List[str]) -> Dict[str, pd.DataFrame]:
    """Load scalars from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        data = {}
        for tag in tags:
            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                df = pd.DataFrame(events)
                data[tag] = df
        
        return data
    except Exception as e:
        print(f"Warning: Could not load TensorBoard data from {log_dir}: {e}")
        return {}


def load_episode_json(json_path: str) -> pd.DataFrame:
    """Load episode data from JSON log."""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)


def plot_learning_curves(
    results_dirs: Dict[str, str],
    output_path: str,
    metric: str = 'episode_reward',
    window: int = 10,
):
    """Plot learning curves for multiple algorithms."""
    plt.figure(figsize=(12, 6))
    
    for algo_name, log_dir in results_dirs.items():
        json_path = os.path.join(log_dir, 'training_episodes.json')
        df = load_episode_json(json_path)
        
        if df is not None and 'reward' in df.columns:
            # Smooth with moving average
            df['reward_smooth'] = df['reward'].rolling(window=window, min_periods=1).mean()
            
            plt.plot(df['step'], df['reward_smooth'], label=algo_name, linewidth=2)
            plt.fill_between(
                df['step'],
                df['reward'].rolling(window=window, min_periods=1).min(),
                df['reward'].rolling(window=window, min_periods=1).max(),
                alpha=0.2
            )
    
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def plot_final_performance(
    results_dirs: Dict[str, str],
    output_path: str,
):
    """Plot final performance comparison."""
    algo_names = []
    mean_rewards = []
    std_rewards = []
    
    for algo_name, log_dir in results_dirs.items():
        json_path = os.path.join(log_dir, 'training_episodes.json')
        df = load_episode_json(json_path)
        
        if df is not None and 'reward' in df.columns:
            # Take last 20% of episodes
            cutoff = int(len(df) * 0.8)
            final_rewards = df['reward'].iloc[cutoff:]
            
            algo_names.append(algo_name)
            mean_rewards.append(final_rewards.mean())
            std_rewards.append(final_rewards.std())
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(algo_names))
    plt.bar(x, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    plt.xticks(x, algo_names)
    plt.ylabel('Mean Episode Reward')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison to {output_path}")
    plt.close()


def plot_sample_efficiency(
    results_dirs: Dict[str, str],
    output_path: str,
    threshold: float = 100.0,
):
    """Plot sample efficiency - steps to reach threshold."""
    algo_names = []
    steps_to_threshold = []
    
    for algo_name, log_dir in results_dirs.items():
        json_path = os.path.join(log_dir, 'training_episodes.json')
        df = load_episode_json(json_path)
        
        if df is not None and 'reward' in df.columns:
            # Smooth rewards
            df['reward_smooth'] = df['reward'].rolling(window=10, min_periods=1).mean()
            
            # Find first step where smoothed reward exceeds threshold
            above_threshold = df[df['reward_smooth'] >= threshold]
            
            if len(above_threshold) > 0:
                steps = above_threshold['step'].iloc[0]
            else:
                steps = df['step'].iloc[-1]  # Didn't reach threshold
            
            algo_names.append(algo_name)
            steps_to_threshold.append(steps)
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(algo_names))
    plt.bar(x, steps_to_threshold, alpha=0.7)
    plt.xticks(x, algo_names)
    plt.ylabel('Steps to Reach Threshold')
    plt.title(f'Sample Efficiency (Threshold = {threshold})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sample efficiency plot to {output_path}")
    plt.close()


def generate_all_plots(output_dir: str = 'results/plots'):
    """Generate all comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define results directories
    results_dirs = {
        'SAC': 'results/sac/logs',
        'Dreamer': 'results/dreamer/logs',
        'Hierarchical': 'results/hierarchical/logs',
    }
    
    # Check which results exist
    existing_results = {
        name: path for name, path in results_dirs.items()
        if Path(path).exists()
    }
    
    if not existing_results:
        print("No results found. Train models first.")
        return
    
    print(f"Found results for: {list(existing_results.keys())}")
    
    # Generate plots
    plot_learning_curves(
        existing_results,
        os.path.join(output_dir, 'learning_curves.png'),
    )
    
    plot_final_performance(
        existing_results,
        os.path.join(output_dir, 'final_performance.png'),
    )
    
    plot_sample_efficiency(
        existing_results,
        os.path.join(output_dir, 'sample_efficiency.png'),
        threshold=100.0,
    )
    
    print(f"\n✅ All plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    
    generate_all_plots(args.output_dir)


if __name__ == "__main__":
    main()
