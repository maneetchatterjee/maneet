# Reproducibility Guide

Instructions for exactly reproducing the reported results.

## Random Seed Management

All experiments use deterministic seeding:

```python
from src.utils.seeding import set_seed

set_seed(42)  # Default seed for all experiments
```

This sets:
- Python `random` module
- NumPy random state
- PyTorch CPU and CUDA random states
- PyTorch deterministic mode

## RNG State Checkpointing

Checkpoints include full RNG state:

```python
checkpoint = {
    'step': step,
    'model_state': {...},
    'optimizer_state': {...},
    'rng_state': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }
}
```

## Deterministic Operations

### PyTorch

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Environment

PyBullet simulation is deterministic given same initial conditions and action sequence.

## Reproducing Results

### Exact Configuration

Use the provided config files without modifications:

```bash
python run_experiment.py --config configs/sac_config.yaml
```

Configuration is automatically saved with each run:
- `results/{algorithm}/logs/config.yaml`

### Hardware Considerations

**GPU**: Results may vary slightly between GPU models due to floating-point precision differences.

**Recommended**:
- NVIDIA GPU with CUDA 11.7+
- 16GB+ VRAM for batch training
- 32GB+ system RAM

**For exact reproduction**:
- Use same GPU model
- Same CUDA version
- Same PyTorch version

### Software Versions

Critical dependencies:

```
torch==2.0.0
pybullet==3.2.5
numpy==1.24.0
```

Full environment: `requirements.txt` and `environment.yml`

### Training from Scratch

```bash
# Remove existing results
rm -rf results/sac

# Train with default seed (42)
python run_experiment.py --config configs/sac_config.yaml

# Results will be in results/sac/
```

### Multiple Seeds

To test robustness across seeds:

```bash
# Create configs with different seeds
for seed in 42 123 456; do
    sed "s/seed: 42/seed: $seed/" configs/sac_config.yaml > configs/sac_seed_${seed}.yaml
    sed -i "s|results/sac|results/sac_seed_${seed}|g" configs/sac_seed_${seed}.yaml
done

# Train all seeds
for seed in 42 123 456; do
    python run_experiment.py --config configs/sac_seed_${seed}.yaml
done
```

## Verification

### Checkpoint Verification

Load checkpoint and verify RNG state:

```python
from src.utils.checkpointing import load_checkpoint

checkpoint = load_checkpoint('results/sac/checkpoints/checkpoint_step_100000.pt')
print(f"Step: {checkpoint['step']}")
print(f"RNG state keys: {checkpoint['rng_state'].keys()}")
```

### Result Comparison

Compare your results with reported metrics:

| Algorithm | Mean Reward | Std | Episode Length |
|-----------|-------------|-----|----------------|
| SAC       | 150-250     | 50  | 600-900       |
| Dreamer   | 100-200     | 60  | 500-800       |
| Hierarchical | 120-220  | 55  | 550-850       |

Exact values depend on randomization settings and training duration.

## Known Variability Sources

### Sources of Non-Determinism

1. **PyBullet Contact Resolution**: Slight numerical variations in contact dynamics
2. **GPU Operations**: Different GPU architectures may have minor differences
3. **Python Multiprocessing**: If used (not in default configs)

### Acceptable Variance

- **Episode Rewards**: ±10% variation is normal across seeds
- **Training Dynamics**: Learning curves should follow similar trends
- **Final Performance**: Mean over 10 eval episodes should match within ±15%

## Debugging Reproducibility Issues

### Check Seed Setting

```python
import torch
import numpy as np

# Should print same values for same seed
set_seed(42)
print(np.random.rand(3))  # [0.37454012 0.95071431 0.73199394]
print(torch.rand(3))      # tensor([0.8823, 0.9150, 0.3829])
```

### Verify Deterministic Mode

```python
print(torch.backends.cudnn.deterministic)  # Should be True
print(torch.backends.cudnn.benchmark)      # Should be False
```

### Check Environment State

```python
from src.envs.biped_env import BipedEnv

env = BipedEnv(use_gui=False)
obs1 = env.reset(seed=42)
obs2 = env.reset(seed=42)

assert np.allclose(obs1, obs2), "Environment reset not deterministic"
```

## Reporting Results

When reporting results, include:

1. **Config file** used
2. **Random seed(s)** used
3. **Hardware** specifications (GPU model, CUDA version)
4. **Software versions** (PyTorch, PyBullet)
5. **Training time** (wall-clock hours)
6. **Final checkpoint** (step number, file size)
7. **Evaluation metrics** (mean, std over multiple seeds)

## Archived Results

Reference checkpoints and logs are available:

```
results/
├── sac_seed_42/           # SAC with seed 42
├── dreamer_seed_42/       # Dreamer with seed 42
└── hierarchical_seed_42/  # Hierarchical with seed 42
```

Each contains:
- `checkpoints/final_model.pt`
- `logs/training_summary.csv`
- `logs/config.yaml`
- `videos/` sample rollouts

## Contact

For reproducibility questions:
- Open GitHub issue with "reproducibility" tag
- Include: config used, seed, hardware, observed vs expected results
