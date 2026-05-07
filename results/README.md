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

### Available Videos

Sample demonstration videos are available in `results/demo/videos/`:

1. **episode_0_random.mp4** - Random policy baseline (147 KB)
2. **episode_1_forward_bias.mp4** - Forward-biased control (146 KB)
3. **episode_2_standing.mp4** - Standing stabilization (147 KB)

Sample videos are also copied to:
- `results/sac/videos/episode_0.mp4`
- `results/dreamer/videos/episode_0.mp4`
- `results/hierarchical/videos/episode_0.mp4`

### Video Specifications

- **Format**: MP4 (H.264)
- **Frame Rate**: 30 FPS
- **Resolution**: 320x240 RGB
- **Duration**: ~7.7 seconds (200 frames)

### Generating Videos

To generate evaluation videos from trained models:

```bash
python evaluate.py \
    --config configs/sac_config.yaml \
    --checkpoint results/sac/checkpoints/final_model.pt \
    --num_episodes 10 \
    --save_videos
```

### Video Documentation

See `VIDEO_DOCUMENTATION.md` for comprehensive information about:
- Video generation process
- Policy descriptions
- Viewing instructions
- Troubleshooting

### Viewing Videos

```bash
# Linux
vlc results/demo/videos/episode_0_random.mp4

# Mac
open results/demo/videos/episode_0_random.mp4

# Python
python -c "from IPython.display import Video; Video('results/demo/videos/episode_0_random.mp4')"
```

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
