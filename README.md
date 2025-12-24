# F1 Car CFD Simulation

A comprehensive 3D F1 car design with Computational Fluid Dynamics (CFD) flow simulation and realistic animation visualization.

## Features

- **3D F1 Car Model**: Detailed geometric model including:
  - Main body/monocoque
  - Front and rear wings
  - Wheels
  - Nose cone
  - Airbox and engine cover
  - Sidepods

- **CFD Flow Simulation**: Simplified potential flow solver that computes:
  - Velocity field around the car
  - Pressure distribution
  - Streamlines
  - Aerodynamic forces (drag and downforce)

- **Realistic Animations**: Multiple visualization outputs:
  - Particle-based flow animation (like real wind tunnel tests)
  - Static streamline visualization
  - Pressure field contours
  - Velocity magnitude plots
  - Detailed simulation report

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

Run the simulation with default parameters:

```bash
python run_simulation.py
```

This will:
1. Create a 3D F1 car model
2. Run CFD simulation at 30 m/s (108 km/h)
3. Generate visualizations and animations
4. Save all outputs to the `output/` directory

### Advanced Options

```bash
python run_simulation.py --velocity 40 --resolution 60 --frames 180 --particles 150
```

**Command-line Arguments:**

- `--velocity`: Freestream velocity in m/s (default: 30.0)
- `--resolution`: Grid resolution (30, 40, 50, or 60) - higher is more accurate but slower (default: 50)
- `--frames`: Number of animation frames (default: 120)
- `--particles`: Number of flow particles (default: 100)
- `--fps`: Animation frames per second (default: 30)
- `--output-dir`: Output directory (default: output)
- `--skip-animation`: Skip animation creation for faster testing

### Example Commands

**Quick test run (no animation):**
```bash
python run_simulation.py --skip-animation --resolution 40
```

**High-quality simulation:**
```bash
python run_simulation.py --velocity 50 --resolution 60 --frames 240 --particles 200
```

**Racing speed simulation (200 km/h):**
```bash
python run_simulation.py --velocity 55.6 --frames 180
```

## Output Files

After running the simulation, you'll find these files in the output directory:

1. **simulation_static.png**: Static view with streamlines showing airflow around the car
2. **pressure_field.png**: Pressure distribution contour plot
3. **velocity_field.png**: Velocity magnitude with flow vectors
4. **simulation_animation.gif**: Animated particle flow visualization
5. **simulation_report.txt**: Detailed text report with aerodynamic forces

## Technical Details

### CFD Approach

This simulation uses a simplified potential flow approach with:
- Incompressible flow assumption
- Steady-state solution
- Bernoulli equation for pressure calculation
- Particle advection for animation

**Note**: This is an educational/visualization tool. Real F1 aerodynamics requires high-fidelity CFD solvers (like OpenFOAM, ANSYS Fluent, or Star-CCM+) with:
- Navier-Stokes equations
- Turbulence modeling (RANS, LES, or DES)
- Boundary layer resolution
- Much finer meshes (millions of cells)

### Typical F1 Aerodynamics

For reference, real F1 cars:
- Generate 750-1000 kg of downforce at racing speeds
- Have drag coefficients around 0.7-1.1
- Produce downforce equivalent to 3-5x the car's weight at 250 km/h
- Use complex aerodynamic features (barge boards, diffusers, DRS, etc.)

## Project Structure

```
maneet/
├── f1_car_model.py       # 3D F1 car geometry generator
├── cfd_simulation.py     # CFD flow solver
├── visualization.py      # Visualization and animation tools
├── run_simulation.py     # Main execution script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── output/              # Generated results (created on first run)
```

## Dependencies

- **NumPy**: Numerical computations
- **PyVista**: 3D visualization and mesh handling
- **Matplotlib**: Plotting and colormaps
- **SciPy**: Interpolation for flow field
- **VTK**: Visualization toolkit (backend for PyVista)
- **imageio**: Animation export
- **Pillow**: Image processing

## Performance Notes

- **Resolution 30**: ~27,000 grid points - Fast, good for testing (~1-2 minutes)
- **Resolution 40**: ~64,000 grid points - Balanced quality/speed (~2-3 minutes)
- **Resolution 50**: ~125,000 grid points - High quality (default) (~3-5 minutes)
- **Resolution 60**: ~216,000 grid points - Maximum quality (~5-8 minutes)

Animation creation is the most time-consuming step. Each frame requires rendering the 3D scene with particles.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

## Acknowledgments

Inspired by real-world F1 aerodynamics and CFD visualization techniques used in motorsports engineering.