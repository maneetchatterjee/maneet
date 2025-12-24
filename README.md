# maneet

## Galaxy Collision Simulation

This repository contains a physically accurate N-body simulation of two spiral galaxies colliding with each other, implemented in Julia.

### Features

- **Accurate N-body gravitational dynamics** using direct force calculation with softened potential
- **Realistic spiral galaxy generation** with bulge and disk components
- **Energy-conserving time integration** using the leapfrog (Verlet) integrator
- **Real-time energy monitoring** to verify numerical accuracy
- **Animated visualization** showing the collision evolution in 3D

### Requirements

- Julia 1.12+ 
- Plots.jl
- LinearAlgebra.jl (standard library)

### Installation

```bash
julia -e 'using Pkg; Pkg.add("Plots")'
```

### Usage

Run the simulation:

```bash
julia galaxy_collision.jl
```

The simulation will:
1. Generate two spiral galaxies with 500 particles each
2. Set them on a collision course
3. Simulate 50 time units of gravitational evolution
4. Create an animation showing the collision from multiple viewpoints
5. Save the result as `galaxy_collision.gif`

### Physics Details

The simulation implements the following physics:

- **Gravitational Force**: F = -G·m₁·m₂·r/(r² + ε²)^(3/2)
  - Softening parameter ε prevents numerical singularities
- **Leapfrog Integration**: Symplectic integrator for excellent energy conservation
- **Spiral Structure**: Two-arm logarithmic spiral pattern
- **Differential Rotation**: Velocity profile v(r) ∝ √(r/(r+R)) mimics realistic galaxy rotation curves

### Simulation Parameters

- **N_particles**: 500 per galaxy
- **Galaxy radius**: 1.0 (normalized units)
- **Rotational velocity**: 0.5 (normalized units)
- **Time step**: 0.05
- **Total time**: 50.0
- **Softening length**: 0.1

### Performance

The simulation achieves excellent numerical accuracy with energy conservation typically better than 0.1% over the entire simulation period.

### Output

The simulation generates:
- `galaxy_collision.gif`: Animated visualization showing:
  - 3D view of both galaxies
  - Energy conservation plot
  - Time evolution from t=0 to t=50

### Author

Maneet Chatterjee