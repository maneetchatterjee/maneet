#!/bin/bash

# Setup script for Change Detection System

echo "=============================================="
echo "Change Detection System - Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Setup completed successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Download and prepare your dataset (OSCD or LEVIR-CD)"
echo "  2. Validate installation: python test_system.py"
echo "  3. Train a model: python experiments/train.py --config experiments/configs/oscd_baseline.yaml"
echo ""
