# Results Directory

This directory contains training outputs, checkpoints, logs, and evaluation results.

## Structure

```
results/
├── sac/                    # SAC (model-free) results
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/              # TensorBoard logs and CSV summaries
│   └── videos/            # Evaluation videos
├── dreamer/               # Dreamer (world model) results
│   ├── checkpoints/
│   ├── logs/
│   └── videos/
└── hierarchical/          # Hierarchical controller results
    ├── checkpoints/
    ├── logs/
    └── videos/
```

## Checkpoints

Checkpoints are saved periodically during training:

- `checkpoint_step_*.pt` - Periodic checkpoints at specific steps
- `checkpoint_latest.pt` - Most recent checkpoint (for resuming)
- `final_model.pt` - Final model after training completes

Each checkpoint contains:
- Model state (actor, critic networks)
- Optimizer state
- RNG state (for reproducibility)
- Training metadata (step, episode count)

## Logs

### TensorBoard

View logs:
```bash
tensorboard --logdir results/
```

Metrics logged:
- `train/episode_reward` - Episode return
- `train/episode_length` - Episode length
- `train/critic_loss` - Critic loss
- `train/actor_loss` - Actor loss
- `eval/mean_reward` - Evaluation performance

### JSON Episodes

`training_episodes.json` contains per-episode data:
```json
[
  {
    "step": 1000,
    "episode": 10,
    "reward": 123.45,
    "length": 567
  },
  ...
]
```

### CSV Summary

`training_summary.csv` contains aggregated metrics for easy plotting.

## Videos

Evaluation videos are saved as MP4 files:
- `episode_0.mp4`, `episode_1.mp4`, etc.
- 30 FPS, RGB rendering
- Generated during evaluation runs

## Expected Results

### SAC (1M steps, ~10 hours)

- Final reward: 150-250
- Episode length: 600-900 steps
- Stable forward walking gait

### Dreamer (500k steps, ~5 hours)

- Final reward: 100-200
- Episode length: 500-800 steps
- Sample efficient, reaches SAC performance in ~50% of steps

### Hierarchical (1M steps, ~10 hours)

- Final reward: 120-220
- Episode length: 550-850 steps
- Diverse behaviors from skill selection

## Reproducing Results

To reproduce results:

1. Use exact configs from `configs/`
2. Set seed to 42 (default)
3. Train for specified number of steps
4. Evaluate with 10 episodes

See `reproducibility.md` for detailed instructions.
