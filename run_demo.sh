#!/bin/bash
# Quick demo script - trains for a short time to verify everything works

set -e

echo "=========================================="
echo "RL Stack Quick Demo"
echo "=========================================="

# Check if dependencies are installed
echo ""
echo "Checking dependencies..."
python3 -c "import torch; import numpy; import yaml; print('✅ Core dependencies OK')" || {
    echo "❌ Dependencies not installed. Run: pip install -r requirements.txt"
    exit 1
}

# Create results directories
mkdir -p results/demo/{checkpoints,logs,videos}

# Create demo config (reduced steps)
echo ""
echo "Creating demo configuration..."
cat > configs/demo_config.yaml << 'EOF'
algorithm:
  type: sac

agent:
  lr: 0.0003
  gamma: 0.99
  tau: 0.005
  alpha: 0.2
  auto_entropy_tuning: true
  n_step: 3

training:
  total_steps: 5000  # Reduced for demo
  random_steps: 1000
  learning_starts: 1000
  batch_size: 64  # Smaller batch for demo
  buffer_size: 10000  # Smaller buffer for demo
  max_episode_steps: 200  # Shorter episodes
  checkpoint_interval: 2000
  eval_interval: 2000
  eval_episodes: 2
  update_every: 1

env:
  domain_randomization:
    randomize_mass: false  # Disable for demo
    randomize_friction: false
    randomize_damping: false

logging:
  log_interval: 100

paths:
  log_dir: results/demo/logs
  checkpoint_dir: results/demo/checkpoints
  video_dir: results/demo/videos

device: cpu  # Use CPU for demo
seed: 42
EOF

echo "✅ Demo config created"

# Run quick training
echo ""
echo "=========================================="
echo "Running quick training demo (5000 steps)..."
echo "This will take 2-5 minutes"
echo "=========================================="

python3 run_experiment.py --config configs/demo_config.yaml

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Training completed successfully. Check results:"
echo "  - Logs: results/demo/logs/"
echo "  - Checkpoints: results/demo/checkpoints/"
echo ""
echo "To view logs:"
echo "  tensorboard --logdir results/demo/logs/"
echo ""
echo "To run full training:"
echo "  python run_experiment.py --config configs/sac_config.yaml"
echo ""
echo "Or run all experiments:"
echo "  ./run_all.sh"
echo ""
