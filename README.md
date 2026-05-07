# RL Stack for Bipedal Robot Control

A comprehensive reinforcement learning framework for training bipedal robot controllers in PyBullet simulation.

## Overview

This project implements three state-of-the-art RL algorithms for bipedal locomotion:

1. **SAC (Soft Actor-Critic)** - Model-free baseline with n-step returns
2. **Dreamer** - World model-based learning with imagination rollouts
3. **Hierarchical** - Two-level controller with skill selection and execution

## Key Features

- **Domain Randomization**: Automated randomization of physics parameters (mass, friction, damping)
- **Safety Constraints**: Joint limits, torque limits, and early termination
- **Comprehensive Logging**: TensorBoard, JSON episodes, CSV summaries
- **Checkpointing**: Automatic periodic saves with resume functionality
- **Video Recording**: Generate videos of policy rollouts
- **Reproducibility**: Deterministic seeding with RNG state tracking

## Project Structure

```
.
├── src/
│   ├── envs/              # PyBullet environments
│   ├── algorithms/        # RL algorithm implementations
│   │   ├── sac/          # Soft Actor-Critic
│   │   ├── dreamer/      # World model
│   │   └── hierarchical/ # Hierarchical control
│   └── utils/            # Utilities (logging, checkpointing, seeding)
├── configs/              # YAML configuration files
├── tests/                # Unit tests
├── results/              # Training outputs
│   ├── sac/
│   ├── dreamer/
│   └── hierarchical/
├── docs/                 # Documentation
├── run_experiment.py     # Main training script
├── evaluate.py           # Evaluation script
├── run_all.sh           # Master orchestration script
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── Dockerfile           # Docker container
```

## Quick Start

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate rl-biped

# Using Docker
docker build -t rl-biped .
docker run -it --gpus all -v $(pwd):/workspace rl-biped
```

### Training

Train a specific algorithm:

```bash
# SAC
python run_experiment.py --config configs/sac_config.yaml

# Dreamer
python run_experiment.py --config configs/dreamer_config.yaml

# Hierarchical
python run_experiment.py --config configs/hierarchical_config.yaml
```

Or run all experiments:

```bash
./run_all.sh
```

### Evaluation

```bash
python evaluate.py \
    --config configs/sac_config.yaml \
    --checkpoint results/sac/checkpoints/final_model.pt \
    --num_episodes 10 \
    --save_videos
```

### Monitoring

View training progress:

```bash
tensorboard --logdir results/
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## Algorithm Details

### SAC (Model-Free)

- Entropy-regularized actor-critic
- Double Q-networks with target networks
- Automatic temperature tuning
- N-step returns for stability
- Optional DrQv2-style augmentation for pixel observations

### Dreamer (World Model)

- Latent dynamics model (RSSM-style with GRU)
- Reconstruction and reward prediction
- Imagination rollouts for policy learning
- Sample-efficient learning in latent space

### Hierarchical

- High-level skill manager (discrete skill selection)
- Low-level controller (continuous action execution)
- Skill embeddings for behavioral diversity
- Temporal abstraction with fixed skill duration

## Configuration

All hyperparameters are specified in YAML config files. Key settings:

- **Training**: `total_steps`, `batch_size`, `learning_rate`
- **Environment**: `max_episode_steps`, `domain_randomization`
- **Checkpointing**: `checkpoint_interval`, `eval_interval`
- **Paths**: `log_dir`, `checkpoint_dir`, `video_dir`

## Citation

If you use this code, please cite the relevant papers:

- **SAC**: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
- **Dreamer**: Hafner et al., "Mastering Diverse Domains through World Models" (2023)
- **DrQv2**: Yarats et al., "Mastering Visual Continuous Control" (2021)

See `citations.md` for complete references.

## License

This project is for research and educational purposes.

## Contact

For issues and questions, please open a GitHub issue.
