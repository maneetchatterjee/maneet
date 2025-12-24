#!/bin/bash
# Quick start script for F1 Car CFD Simulation

echo "F1 Car CFD Simulation - Quick Start"
echo "===================================="
echo ""

# Check if dependencies are installed
if ! python -c "import pyvista" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run simulation with xvfb for headless rendering
echo "Running CFD simulation..."
echo ""

if command -v xvfb-run &> /dev/null; then
    # With xvfb (headless)
    xvfb-run -a python run_simulation.py "$@"
else
    # Direct execution
    python run_simulation.py "$@"
fi
