# F1 Car CFD Simulation - Example Outputs

This directory contains the output files from running the F1 car CFD simulation.

## Generated Files

### 1. simulation_static.png
A static 3D visualization of the F1 car with streamlines showing the airflow pattern around the vehicle. This provides a comprehensive view of the aerodynamic flow field.

### 2. pressure_field.png
A 2D contour plot showing the pressure distribution in a centerline slice (Y=0) through the car. 
- **Red regions**: High pressure (front of car, ground)
- **Blue regions**: Low pressure (over the car, wake region)
- Shows how the car creates low pressure zones that contribute to downforce

### 3. velocity_field.png
A 2D contour plot with velocity vectors showing the flow speed and direction around the car.
- **Yellow/Green**: High velocity regions (over the top of the car)
- **Blue/Dark**: Low velocity regions (wake behind the car, near stagnation points)
- White arrows show local flow direction
- Demonstrates flow acceleration over aerodynamic surfaces

### 4. simulation_animation.gif
An animated visualization showing particle-based flow simulation. Colored particles move through the flow field, providing a dynamic view similar to wind tunnel smoke visualization.
- Particle colors represent velocity magnitude
- Shows flow separation, wake formation, and aerodynamic effects in real-time
- Typical duration: 3-6 seconds depending on frame count and FPS settings

### 5. simulation_report.txt
A text report containing:
- Simulation parameters (velocity, air properties, grid resolution)
- Car geometry dimensions
- Computed aerodynamic forces (drag, downforce)
- Aerodynamic coefficients (Cd, Cl)

## Example Results

For a simulation at 30 m/s (108 km/h):
- **Drag Force**: ~570 N
- **Downforce**: ~2440 N (negative lift)
- **Drag Coefficient**: 0.70
- **Lift Coefficient**: -3.00 (downforce)

These values are representative of F1 car aerodynamics, though real F1 cars achieve even higher downforce-to-drag ratios with more complex designs.

## Recreating These Results

To generate these outputs yourself:

```bash
# Basic simulation (default settings)
python run_simulation.py

# Or use the wrapper script
./run.sh

# High-quality simulation
python run_simulation.py --velocity 50 --resolution 60 --frames 240 --particles 200

# Quick test without animation
python run_simulation.py --skip-animation --resolution 40
```

## Notes

- Output files are excluded from git via .gitignore
- Animations can be large (1-5 MB depending on settings)
- Higher resolution simulations provide more accurate results but take longer to compute
- The simulation uses simplified CFD - real F1 teams use much more sophisticated tools
