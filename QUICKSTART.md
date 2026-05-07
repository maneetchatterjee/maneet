# Quick Start Guide

Get started with the RL bipedal robot stack in 5 minutes.

## 1. Installation (2 minutes)

```bash
# Clone repository
cd /path/to/maneet

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

Expected output: `✅ Setup complete! Ready to train.`

## 2. Quick Demo (3 minutes)

Run a quick training demo to verify everything works:

```bash
./run_demo.sh
```

This will:
- Train SAC for 5000 steps (~2-3 minutes)
- Save results to `results/demo/`
- Verify the complete pipeline

## 3. View Results

```bash
# View TensorBoard logs
tensorboard --logdir results/demo/logs/

# Check episode data
cat results/demo/logs/training_episodes.json
```

## 4. Full Training

### Train Single Algorithm

```bash
# SAC (model-free)
python run_experiment.py --config configs/sac_config.yaml

# Dreamer (world model)
python run_experiment.py --config configs/dreamer_config.yaml

# Hierarchical
python run_experiment.py --config configs/hierarchical_config.yaml
```

### Train All Algorithms

```bash
./run_all.sh
```

**Note**: Full training takes 20-26 GPU hours total.

## 5. Evaluation

```bash
# Evaluate trained model
python evaluate.py \
    --config configs/sac_config.yaml \
    --checkpoint results/sac/checkpoints/final_model.pt \
    --num_episodes 10 \
    --save_videos
```

Videos saved to `results/sac/videos/`

## 6. Generate Plots

```bash
python generate_plots.py --output_dir results/plots
```

Creates:
- `learning_curves.png` - Training progress
- `final_performance.png` - Performance comparison
- `sample_efficiency.png` - Steps to threshold

## Common Commands

```bash
# Monitor training
tensorboard --logdir results/

# Resume training
python run_experiment.py --config configs/sac_config.yaml --resume

# Run tests
pytest tests/ -v

# Verify setup
python verify_setup.py
```

## Project Structure

```
maneet/
├── run_experiment.py      # Main training script
├── evaluate.py            # Evaluation script
├── run_all.sh            # Train all algorithms
├── run_demo.sh           # Quick demo
├── configs/              # Algorithm configs
│   ├── sac_config.yaml
│   ├── dreamer_config.yaml
│   └── hierarchical_config.yaml
├── src/                  # Source code
│   ├── envs/            # PyBullet environment
│   ├── algorithms/      # RL algorithms
│   └── utils/           # Utilities
├── tests/               # Unit tests
└── results/             # Training outputs
```

## Troubleshooting

### ImportError: No module named 'pybullet'

```bash
pip install pybullet
```

### CUDA out of memory

Edit config to reduce batch size:
```yaml
training:
  batch_size: 128  # Reduced from 256
```

### PyBullet GUI errors

Use headless mode (default):
```python
env = BipedEnv(use_gui=False)
```

## Next Steps

1. **Customize**: Edit config files to adjust hyperparameters
2. **Experiment**: Try different domain randomization settings
3. **Analyze**: Generate plots and compare algorithms
4. **Document**: Write report using `docs/report_template.md`

## Documentation

- `README.md` - Project overview
- `EXECUTION_GUIDE.md` - Detailed instructions
- `reproducibility.md` - Reproducibility guide
- `citations.md` - Paper references
- `DONE.md` - Completion checklist

## Support

- GitHub Issues: Report bugs or ask questions
- Documentation: See guides above
- Tests: `pytest tests/ -v` to verify installation

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./run_demo.sh` | Quick 5-minute demo |
| `./run_all.sh` | Train all algorithms |
| `python run_experiment.py --config <config>` | Train single algorithm |
| `python evaluate.py --config <config> --checkpoint <path>` | Evaluate model |
| `tensorboard --logdir results/` | View logs |
| `python generate_plots.py` | Generate plots |
| `pytest tests/ -v` | Run tests |
| `python verify_setup.py` | Verify installation |

Happy training! 🚀🤖
