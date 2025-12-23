#!/bin/bash
# Setup script for Advanced CV System

echo "🚀 Setting up Advanced Computer Vision System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p output
echo "✓ Created output directory"

# Check for CUDA
echo ""
echo "🔍 Checking for CUDA availability..."
python3 -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('  CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the system:"
echo "  source venv/bin/activate"
echo "  python src/main.py"
echo ""
echo "For help:"
echo "  python src/main.py --help"
