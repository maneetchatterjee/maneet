# Execution Guide

Step-by-step instructions for setting up, training, and evaluating the RL stack.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- Linux or macOS

## 1. Environment Setup

### Option A: Using pip (Recommended)

```bash
# Clone repository
cd /path/to/maneet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate rl-biped
```

### Option C: Using Docker

```bash
# Build Docker image
docker build -t rl-biped .

# Run container
docker run -it --gpus all \
    -v $(pwd):/workspace \
    -v $(pwd)/results:/workspace/results \
    rl-biped
```

## 2. Verify Installation

Run unit tests to ensure everything is working:

```bash
pytest tests/ -v
```

Expected output: All tests pass (some tests may take a few minutes).

## 3. Training Experiments

### Quick Test (Smoke Test)

Train for a short duration to verify the pipeline:

```bash
# Modify config to reduce steps
python run_experiment.py --config configs/sac_config.yaml
# Press Ctrl+C after a few minutes to stop
```

### Full Training

#### SAC (Model-Free Baseline)

```bash
python run_experiment.py --config configs/sac_config.yaml
```

**Expected Duration**: ~8-10 hours on GPU for 1M steps

**Key Hyperparameters**:
- Learning rate: 3e-4
- Batch size: 256
- Replay buffer: 1M transitions
- Domain randomization: Enabled

#### Dreamer (World Model)

```bash
python run_experiment.py --config configs/dreamer_config.yaml
```

**Expected Duration**: ~4-6 hours on GPU for 500k steps (more sample efficient)

**Key Hyperparameters**:
- Latent dimension: 64
- Imagination horizon: 15 steps
- Sequence length: 50
- Domain randomization: Disabled (simpler setting)

#### Hierarchical Controller

```bash
python run_experiment.py --config configs/hierarchical_config.yaml
```

**Expected Duration**: ~8-10 hours on GPU for 1M steps

**Key Hyperparameters**:
- Number of skills: 8
- Skill duration: 10 steps
- High-level LR: 1e-4
- Low-level LR: 3e-4

### Run All Experiments

```bash
./run_all.sh
```

This will train and evaluate all three algorithms sequentially.

**Total Expected Duration**: ~20-26 hours on GPU

## 4. Monitoring Training

### TensorBoard

```bash
# View all experiments
tensorboard --logdir results/

# View specific experiment
tensorboard --logdir results/sac/logs/
```

Open browser to `http://localhost:6006`

### Log Files

- **Scalars**: `results/{algorithm}/logs/tensorboard/`
- **Episodes**: `results/{algorithm}/logs/training_episodes.json`
- **Summary**: `results/{algorithm}/logs/training_summary.csv`

## 5. Resuming Training

If training is interrupted:

```bash
python run_experiment.py --config configs/sac_config.yaml --resume
```

The script will automatically find and load the latest checkpoint.

## 6. Evaluation

### Evaluate Single Model

```bash
python evaluate.py \
    --config configs/sac_config.yaml \
    --checkpoint results/sac/checkpoints/final_model.pt \
    --num_episodes 10 \
    --save_videos
```

Videos will be saved to `results/sac/videos/`

### Evaluate with GUI

```bash
python evaluate.py \
    --config configs/sac_config.yaml \
    --checkpoint results/sac/checkpoints/final_model.pt \
    --num_episodes 3 \
    --render
```

This will open PyBullet GUI windows to visualize the policy.

## 7. Comparing Results

### TensorBoard Comparison

```bash
tensorboard --logdir results/
```

Navigate to the "Scalars" tab and select multiple runs to compare:
- Episode rewards
- Training losses
- Evaluation metrics

### Checkpoint Selection

Checkpoints are saved every 50,000 steps:
- `checkpoint_step_50000.pt`
- `checkpoint_step_100000.pt`
- `checkpoint_latest.pt` (most recent)
- `final_model.pt` (end of training)

## 8. Customization

### Modify Hyperparameters

Edit YAML config files:

```bash
nano configs/sac_config.yaml
```

Key sections:
- `agent`: Algorithm hyperparameters
- `training`: Training loop settings
- `env`: Environment configuration
- `logging`: Log intervals and paths

### Adjust Training Duration

For faster testing:

```yaml
training:
  total_steps: 10000  # Reduced from 1000000
  checkpoint_interval: 5000
  eval_interval: 5000
```

### Enable/Disable Domain Randomization

```yaml
env:
  domain_randomization:
    randomize_mass: false  # Disable mass randomization
    randomize_friction: true
    randomize_damping: true
```

## 9. Troubleshooting

### PyBullet Display Issues

If you get OpenGL errors:

```bash
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
```

Or use DIRECT mode (no GUI):

```python
# In biped_env.py, ensure:
use_gui=False
```

### CUDA Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 128  # Reduced from 256
```

### Slow Training

- Verify GPU is being used: Check `nvidia-smi`
- Reduce evaluation frequency
- Use smaller networks (reduce `hidden_dim` in config)

## 10. Expected Results

### SAC (1M steps)

- Episode reward: 100-300 (varies with randomization)
- Episode length: 500-1000 steps
- Gait: Stable forward walking on flat terrain

### Dreamer (500k steps)

- Episode reward: 80-200
- Sample efficiency: Achieves comparable performance in 50% of steps
- Gait: Forward locomotion with occasional instability

### Hierarchical (1M steps)

- Episode reward: 80-250
- Skill diversity: 8 distinct behaviors emerge
- Gait: Variable depending on skill selection

## 11. Generating Report

After training completes, generate comparison plots and report (manual step):

1. Extract learning curves from TensorBoard
2. Create comparison plots using matplotlib
3. Record sample videos for each algorithm
4. Compile results into `report.pdf`

See `docs/report_template.md` for structure.

## 12. Next Steps

- **Ablation Studies**: Vary domain randomization settings
- **Curriculum Learning**: Implement progressive terrain difficulty
- **Sim-to-Real**: Collect data on physical robot for system ID
- **Imitation Learning**: Add DeepMimic-style reference motions

See `reproducibility.md` for instructions on exact result reproduction.
