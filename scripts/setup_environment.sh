#!/bin/bash
# Automated environment setup script for VLA pipeline
# Usage: bash setup_environment.sh

set -e  # Exit on any error

echo "========================================"
echo "VLA Pipeline Environment Setup"
echo "========================================"
echo ""

# Check Python version
echo "Step 1/5: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "Step 2/5: Creating virtual environment..."
if [ -d "vla_env" ]; then
    echo "⚠️  Virtual environment already exists, skipping creation"
else
    python3 -m venv vla_env
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Step 3/5: Activating virtual environment..."
source vla_env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Step 4/5: Upgrading pip..."
pip install --quiet --upgrade pip==23.3.1
echo "✓ pip upgraded to 23.3.1"
echo ""

# Install dependencies
echo "Step 5/5: Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --quiet -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy, scipy, matplotlib, cv2, pybullet, sklearn, networkx" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All core packages verified"
else
    echo "❌ Package verification failed"
    exit 1
fi
echo ""

echo "========================================"
echo "✓ Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source vla_env/bin/activate"
echo ""
echo "To run basic demo:"
echo "  python demo/demo_basic.py --seed 42"
echo ""
