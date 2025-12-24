# maneet

## Galaxy Collision Simulation

This repository contains a physically accurate N-body simulation of two spiral galaxies colliding with each other, implemented in Julia.

### Features

- **Accurate N-body gravitational dynamics** using direct force calculation with softened potential
- **Realistic spiral galaxy generation** with prominent 3-arm logarithmic spiral structure
- **Density wave theory** for well-defined spiral arms with concentrated particle distribution
- **Flat rotation curve** characteristic of spiral galaxies with dark matter halos
- **Energy-conserving time integration** using the leapfrog (Verlet) integrator
- **Real-time energy monitoring** to verify numerical accuracy
- **High-resolution animated visualization** showing the collision evolution in 3D

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
1. Generate two spiral galaxies with 1000 particles each (prominent 3-arm spiral structure)
2. Set them on an angled collision trajectory with offset impact parameter
3. Simulate 60 time units of gravitational evolution
4. Create an animation showing the collision from multiple viewpoints
5. Save the result as `galaxy_collision.gif`

### Physics Details

The simulation implements the following physics:

- **Gravitational Force**: F = -G·m₁·m₂·r/(r² + ε²)^(3/2)
  - Softening parameter ε prevents numerical singularities
- **Leapfrog Integration**: Symplectic integrator for excellent energy conservation
- **Spiral Structure**: Three-arm logarithmic spiral pattern using density wave theory
  - 75% of disk particles follow spiral arms with tight winding
  - Logarithmic spiral: θ = θ₀ + 0.3·ln(r/R)
- **Flat Rotation Curve**: v(r) = constant beyond core radius (mimics dark matter halo)
  - Solid body rotation in core region (r < 0.3R)
  - Flat/slowly rising beyond core (characteristic of spiral galaxies)
- **Velocity Dispersion**: Radially decreasing dispersion (hotter in center)

### Simulation Parameters

- **N_particles**: 1000 per galaxy (2000 total)
- **Galaxy radius**: 1.0 (normalized units)
- **Rotational velocity**: 0.6 (normalized units)
- **Time step**: 0.04
- **Total time**: 60.0
- **Softening length**: 0.1
- **Spiral arms**: 3 arms per galaxy
- **Bulge fraction**: 15% of particles
- **Disk thickness**: 0.03R (very thin, realistic disk)

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