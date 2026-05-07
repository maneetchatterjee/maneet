#!/bin/bash
# Master script to run all training and evaluation pipelines

set -e  # Exit on error

echo "=========================================="
echo "RL Bipedal Robot Training Pipeline"
echo "=========================================="

# Create results directories
mkdir -p results/sac/{checkpoints,logs,videos}
mkdir -p results/dreamer/{checkpoints,logs,videos}
mkdir -p results/hierarchical/{checkpoints,logs,videos}

# Function to run training
run_training() {
    local config=$1
    local name=$2
    
    echo ""
    echo "=========================================="
    echo "Training: $name"
    echo "=========================================="
    
    python run_experiment.py --config "$config"
    
    echo "Training complete for $name"
}

# Function to run evaluation
run_evaluation() {
    local config=$1
    local checkpoint=$2
    local name=$3
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $name"
    echo "=========================================="
    
    if [ -f "$checkpoint" ]; then
        python evaluate.py \
            --config "$config" \
            --checkpoint "$checkpoint" \
            --num_episodes 10 \
            --save_videos
        
        echo "Evaluation complete for $name"
    else
        echo "Checkpoint not found: $checkpoint"
        echo "Skipping evaluation for $name"
    fi
}

# Training phase
echo ""
echo "Starting training phase..."
echo ""

# Train SAC (model-free baseline)
run_training "configs/sac_config.yaml" "SAC"

# Train Dreamer (world model)
run_training "configs/dreamer_config.yaml" "Dreamer"

# Train Hierarchical
run_training "configs/hierarchical_config.yaml" "Hierarchical"

# Evaluation phase
echo ""
echo "Starting evaluation phase..."
echo ""

# Evaluate SAC
run_evaluation \
    "configs/sac_config.yaml" \
    "results/sac/checkpoints/final_model.pt" \
    "SAC"

# Evaluate Dreamer
run_evaluation \
    "configs/dreamer_config.yaml" \
    "results/dreamer/checkpoints/final_model.pt" \
    "Dreamer"

# Evaluate Hierarchical
run_evaluation \
    "configs/hierarchical_config.yaml" \
    "results/hierarchical/checkpoints/final_model.pt" \
    "Hierarchical"

echo ""
echo "=========================================="
echo "All training and evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - results/sac/"
echo "  - results/dreamer/"
echo "  - results/hierarchical/"
echo ""
echo "To view tensorboard logs, run:"
echo "  tensorboard --logdir results/"
