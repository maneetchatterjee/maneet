# maneet

## Binary Black Hole Collision Simulation

This repository contains a physically accurate simulation of two black holes colliding and merging, implemented using Post-Newtonian approximations in Julia.

### Features

- **Post-Newtonian orbital dynamics** up to 2.5PN order including:
  - Newtonian gravity (0PN)
  - 1PN relativistic corrections (v/c)²
  - 2PN corrections (v/c)⁴  
  - 2.5PN radiation reaction (gravitational wave backreaction)
- **Gravitational wave energy loss** using Peters-Mathews formula
- **Accurate black hole merger detection** based on Schwarzschild radii
- **4th-order Runge-Kutta integration** for high accuracy
- **Real-time monitoring** of separation, energy loss, and GW luminosity
- **Animated visualization** showing inspiral, merger trajectory, and physical quantities

### Requirements

- Julia 1.12+
- Plots.jl
- LinearAlgebra.jl (standard library)

### Installation

```bash
julia -e 'using Pkg; Pkg.add("Plots")'
```

### Usage

Run the black hole collision simulation:

```bash
julia black_hole_collision.jl
```

The simulation will:
1. Initialize two 30 M☉ black holes in circular orbit at 100 Schwarzschild radii separation
2. Simulate the inspiral phase using 2.5PN equations of motion
3. Track gravitational wave energy loss and orbital decay
4. Detect and simulate the merger when black holes approach within 3 combined Schwarzschild radii
5. Create an animation showing the collision dynamics
6. Save the result as `black_hole_collision.gif`

### Physics Details

The simulation implements the following physics accurately:

**Post-Newtonian Equations of Motion:**
- **0PN (Newtonian)**: a = -GM/r² n̂
- **1PN corrections**: Relativistic effects (v/c)²
- **2PN corrections**: Higher-order relativistic effects (v/c)⁴
- **2.5PN radiation reaction**: Energy and angular momentum loss due to gravitational wave emission

**Gravitational Wave Luminosity:**
```
L_GW = (32/5) × (G⁴/c⁵) × μ² × M³ / r⁵
```
Where:
- μ = m₁m₂/(m₁+m₂) is the reduced mass
- M = m₁ + m₂ is the total mass
- r is the orbital separation

**Schwarzschild Radius:**
```
R_s = 2GM/c²
```

**Merger Criterion:**
- Black holes merge when separation < 3(R_s1 + R_s2)
- This approximates the innermost stable circular orbit (ISCO)

### Simulation Parameters

Default configuration (similar to GW150914):
- **BH1 mass**: 30 M☉
- **BH2 mass**: 30 M☉
- **Initial separation**: 100 Schwarzschild radii
- **PN order**: 2.5 (includes radiation reaction)
- **Integration method**: 4th-order Runge-Kutta
- **Time step**: 0.1 R_s (adaptive based on Schwarzschild radius)

### Performance

The simulation:
- Accurately tracks energy loss through gravitational wave radiation
- Maintains numerical stability through high-order integration
- Typical inspiral simulation takes 2-5 minutes on modern hardware
- Energy conservation error typically < 0.5% before merger

### Output

The simulation generates:
- `black_hole_collision.gif`: Multi-panel animated visualization showing:
  - **Orbital trajectory**: 2D view of black hole paths in the xy-plane with Schwarzschild radius circles
  - **Separation vs time**: Logarithmic plot of orbital decay
  - **Energy loss**: Percentage of energy radiated as gravitational waves
  - **GW luminosity**: Time evolution of gravitational wave power output

### Scientific Background

This simulation is based on:
- **Peters & Mathews (1963)**: "Gravitational Radiation from Point Masses"
- **Blanchet (2014)**: "Gravitational Radiation from Post-Newtonian Sources"
- **Post-Newtonian formalism**: Systematic expansion in v/c for relativistic corrections
- **Numerical Relativity principles**: For merger detection and dynamics

The simulation accurately captures:
- Inspiral phase dynamics
- Gravitational wave driven orbital decay
- Energy and angular momentum loss
- Transition to merger

### Limitations

- Uses Post-Newtonian approximation (valid for v << c)
- Does not include full numerical relativity for final merger and ringdown
- Merger detection is approximation based on ISCO
- Assumes non-spinning black holes
- Uses geometric units (G=c=1) for simplicity

### Author

Maneet Chatterjee

---

## Galaxy Collision Simulation (Also Available)

The repository also contains a galaxy collision simulation (`galaxy_collision.jl`) with enhanced spiral structure and N-body dynamics. See `README_galaxy.md` for details on the galaxy simulation.
