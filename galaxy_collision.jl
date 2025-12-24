#!/usr/bin/env julia

"""
Two-Galaxy Collision N-Body Simulation
======================================

This script simulates the collision of two spiral galaxies using N-body gravitational dynamics.
The simulation uses a softened gravitational potential to prevent numerical issues with close encounters.

Physical Parameters:
- G: Gravitational constant (normalized units)
- Softening length: Prevents singularities in force calculations
- Time integration: Leapfrog/Verlet method for energy conservation

Author: Maneet Chatterjee
"""

using LinearAlgebra
using Plots
using Printf

# Physical constants (in normalized units)
const G = 1.0  # Gravitational constant
const SOFTENING = 0.1  # Softening length to prevent singularities

"""
    Galaxy structure to hold particle data
"""
mutable struct Galaxy
    positions::Matrix{Float64}  # Nx3 matrix of positions
    velocities::Matrix{Float64}  # Nx3 matrix of velocities
    masses::Vector{Float64}      # N-vector of masses
    N::Int                        # Number of particles
end

"""
    create_spiral_galaxy(N, R, v_rotation, center, velocity_offset)

Create a spiral galaxy with N particles distributed in a disk with prominent spiral arms.
Uses density wave theory to create realistic spiral structure.

Parameters:
- N: Number of particles
- R: Characteristic radius of the galaxy
- v_rotation: Rotational velocity scale
- center: 3D position of galaxy center [x, y, z]
- velocity_offset: 3D velocity offset [vx, vy, vz]
"""
function create_spiral_galaxy(N::Int, R::Float64, v_rotation::Float64, 
                               center::Vector{Float64}, velocity_offset::Vector{Float64})
    positions = zeros(N, 3)
    velocities = zeros(N, 3)
    masses = ones(N) / N  # Equal mass particles, total mass = 1
    
    # Create central bulge (15% of particles) - smaller for more disk-dominated galaxy
    N_bulge = Int(floor(0.15 * N))
    for i in 1:N_bulge
        # Compact spherical bulge
        r = R * 0.2 * rand()^(1/3)  # Uniform in volume, smaller radius
        θ = acos(2 * rand() - 1)
        φ = 2π * rand()
        
        positions[i, 1] = r * sin(θ) * cos(φ) + center[1]
        positions[i, 2] = r * sin(θ) * sin(φ) + center[2]
        positions[i, 3] = r * cos(θ) * 0.15 + center[3]  # More flattened
        
        # Random velocities for bulge (velocity dispersion)
        velocities[i, :] = velocity_offset + 0.2 * v_rotation * randn(3)
    end
    
    # Create disk with prominent spiral arms (85% of particles)
    N_disk = N - N_bulge
    # Use 3 spiral arms for more realistic appearance
    n_arms = 3
    
    for i in (N_bulge+1):N
        # Exponential disk profile with concentration toward center
        r = -R * 0.6 * log(1 - rand() * 0.95)  # Steeper exponential for concentration
        r = min(r, 2.5 * R)  # Truncate at 2.5R for more compact galaxy
        
        # Assign particle to a spiral arm with higher probability
        # This creates more defined spiral structure
        arm_preference = rand()
        if arm_preference < 0.75  # 75% of particles follow spiral arms closely
            # Choose arm
            arm_number = rand(0:n_arms-1)
            base_angle = 2π * arm_number / n_arms
            
            # Logarithmic spiral: θ = θ₀ + b * ln(r/R)
            # b controls tightness (smaller = tighter spiral)
            spiral_tightness = 0.3  # Tight, well-defined spiral
            θ_spiral = base_angle + spiral_tightness * log(max(r / R, 0.1))
            
            # Small scatter along the arm
            θ_spiral += 0.15 * randn()  # Reduced scatter for more defined arms
            
            # Add slight radial perturbation to create arm width
            r_perturbation = 0.1 * R * randn()
            r_actual = max(0.1, r + r_perturbation)
        else  # 25% inter-arm particles
            # Particles between arms
            θ_spiral = rand() * 2π
            r_actual = r
        end
        
        # Position in disk
        x = r_actual * cos(θ_spiral)
        y = r_actual * sin(θ_spiral)
        z = 0.03 * R * randn()  # Very thin disk for realistic appearance
        
        positions[i, 1] = x + center[1]
        positions[i, 2] = y + center[2]
        positions[i, 3] = z + center[3]
        
        # Flat rotation curve (characteristic of spiral galaxies with dark matter)
        # v_circ = constant beyond core radius
        r_core = 0.3 * R  # Core radius
        if r_actual < r_core
            v_circ = v_rotation * (r_actual / r_core)  # Solid body rotation in core
        else
            v_circ = v_rotation * (1.0 + 0.1 * log(r_actual / r_core))  # Slightly rising, then flat
        end
        
        # Add small velocity dispersion (decreases with radius - hotter in center)
        dispersion = v_rotation * 0.1 * exp(-r_actual / R)
        
        # Velocity perpendicular to radius (circular motion)
        vx = -v_circ * sin(θ_spiral) + dispersion * randn()
        vy = v_circ * cos(θ_spiral) + dispersion * randn()
        vz = 0.01 * v_rotation * randn()  # Minimal vertical motion
        
        velocities[i, 1] = vx + velocity_offset[1]
        velocities[i, 2] = vy + velocity_offset[2]
        velocities[i, 3] = vz + velocity_offset[3]
    end
    
    return Galaxy(positions, velocities, masses, N)
end

"""
    compute_accelerations!(galaxy1, galaxy2, accel1, accel2)

Compute gravitational accelerations for all particles in both galaxies.
Uses softened gravitational potential: a = -G*m*r/(r^2 + ε^2)^(3/2)
"""
function compute_accelerations!(galaxy1::Galaxy, galaxy2::Galaxy, 
                                 accel1::Matrix{Float64}, accel2::Matrix{Float64})
    fill!(accel1, 0.0)
    fill!(accel2, 0.0)
    
    N1 = galaxy1.N
    N2 = galaxy2.N
    
    # Galaxy 1 internal forces
    for i in 1:N1
        for j in (i+1):N1
            dx = galaxy1.positions[j, 1] - galaxy1.positions[i, 1]
            dy = galaxy1.positions[j, 2] - galaxy1.positions[i, 2]
            dz = galaxy1.positions[j, 3] - galaxy1.positions[i, 3]
            
            r2 = dx^2 + dy^2 + dz^2 + SOFTENING^2
            r = sqrt(r2)
            r3 = r2 * r
            
            # Force on i from j
            force_factor = G / r3
            fx = force_factor * dx
            fy = force_factor * dy
            fz = force_factor * dz
            
            accel1[i, 1] += fx * galaxy1.masses[j]
            accel1[i, 2] += fy * galaxy1.masses[j]
            accel1[i, 3] += fz * galaxy1.masses[j]
            
            # Newton's third law
            accel1[j, 1] -= fx * galaxy1.masses[i]
            accel1[j, 2] -= fy * galaxy1.masses[i]
            accel1[j, 3] -= fz * galaxy1.masses[i]
        end
    end
    
    # Galaxy 2 internal forces
    for i in 1:N2
        for j in (i+1):N2
            dx = galaxy2.positions[j, 1] - galaxy2.positions[i, 1]
            dy = galaxy2.positions[j, 2] - galaxy2.positions[i, 2]
            dz = galaxy2.positions[j, 3] - galaxy2.positions[i, 3]
            
            r2 = dx^2 + dy^2 + dz^2 + SOFTENING^2
            r = sqrt(r2)
            r3 = r2 * r
            
            force_factor = G / r3
            fx = force_factor * dx
            fy = force_factor * dy
            fz = force_factor * dz
            
            accel2[i, 1] += fx * galaxy2.masses[j]
            accel2[i, 2] += fy * galaxy2.masses[j]
            accel2[i, 3] += fz * galaxy2.masses[j]
            
            accel2[j, 1] -= fx * galaxy2.masses[i]
            accel2[j, 2] -= fy * galaxy2.masses[i]
            accel2[j, 3] -= fz * galaxy2.masses[i]
        end
    end
    
    # Interaction forces between galaxies
    for i in 1:N1
        for j in 1:N2
            dx = galaxy2.positions[j, 1] - galaxy1.positions[i, 1]
            dy = galaxy2.positions[j, 2] - galaxy1.positions[i, 2]
            dz = galaxy2.positions[j, 3] - galaxy1.positions[i, 3]
            
            r2 = dx^2 + dy^2 + dz^2 + SOFTENING^2
            r = sqrt(r2)
            r3 = r2 * r
            
            force_factor = G / r3
            fx = force_factor * dx
            fy = force_factor * dy
            fz = force_factor * dz
            
            # Force on galaxy1 particle i from galaxy2 particle j
            accel1[i, 1] += fx * galaxy2.masses[j]
            accel1[i, 2] += fy * galaxy2.masses[j]
            accel1[i, 3] += fz * galaxy2.masses[j]
            
            # Force on galaxy2 particle j from galaxy1 particle i
            accel2[j, 1] -= fx * galaxy1.masses[i]
            accel2[j, 2] -= fy * galaxy1.masses[i]
            accel2[j, 3] -= fz * galaxy1.masses[i]
        end
    end
end

"""
    leapfrog_step!(galaxy1, galaxy2, accel1, accel2, dt)

Perform one leapfrog integration step (kick-drift-kick).
This is a symplectic integrator that conserves energy well.
"""
function leapfrog_step!(galaxy1::Galaxy, galaxy2::Galaxy, 
                        accel1::Matrix{Float64}, accel2::Matrix{Float64}, dt::Float64)
    # Half-step velocity update (kick)
    galaxy1.velocities .+= 0.5 * dt * accel1
    galaxy2.velocities .+= 0.5 * dt * accel2
    
    # Full-step position update (drift)
    galaxy1.positions .+= dt * galaxy1.velocities
    galaxy2.positions .+= dt * galaxy2.velocities
    
    # Compute new accelerations
    compute_accelerations!(galaxy1, galaxy2, accel1, accel2)
    
    # Half-step velocity update (kick)
    galaxy1.velocities .+= 0.5 * dt * accel1
    galaxy2.velocities .+= 0.5 * dt * accel2
end

"""
    compute_energy(galaxy1, galaxy2)

Compute total energy (kinetic + potential) of the system.
"""
function compute_energy(galaxy1::Galaxy, galaxy2::Galaxy)
    # Kinetic energy
    KE = 0.0
    for i in 1:galaxy1.N
        v2 = sum(galaxy1.velocities[i, :].^2)
        KE += 0.5 * galaxy1.masses[i] * v2
    end
    for i in 1:galaxy2.N
        v2 = sum(galaxy2.velocities[i, :].^2)
        KE += 0.5 * galaxy2.masses[i] * v2
    end
    
    # Potential energy
    PE = 0.0
    
    # Galaxy 1 self-potential
    for i in 1:galaxy1.N
        for j in (i+1):galaxy1.N
            dx = galaxy1.positions[j, 1] - galaxy1.positions[i, 1]
            dy = galaxy1.positions[j, 2] - galaxy1.positions[i, 2]
            dz = galaxy1.positions[j, 3] - galaxy1.positions[i, 3]
            r = sqrt(dx^2 + dy^2 + dz^2 + SOFTENING^2)
            PE -= G * galaxy1.masses[i] * galaxy1.masses[j] / r
        end
    end
    
    # Galaxy 2 self-potential
    for i in 1:galaxy2.N
        for j in (i+1):galaxy2.N
            dx = galaxy2.positions[j, 1] - galaxy2.positions[i, 1]
            dy = galaxy2.positions[j, 2] - galaxy2.positions[i, 2]
            dz = galaxy2.positions[j, 3] - galaxy2.positions[i, 3]
            r = sqrt(dx^2 + dy^2 + dz^2 + SOFTENING^2)
            PE -= G * galaxy2.masses[i] * galaxy2.masses[j] / r
        end
    end
    
    # Interaction potential
    for i in 1:galaxy1.N
        for j in 1:galaxy2.N
            dx = galaxy2.positions[j, 1] - galaxy1.positions[i, 1]
            dy = galaxy2.positions[j, 2] - galaxy1.positions[i, 2]
            dz = galaxy2.positions[j, 3] - galaxy1.positions[i, 3]
            r = sqrt(dx^2 + dy^2 + dz^2 + SOFTENING^2)
            PE -= G * galaxy1.masses[i] * galaxy2.masses[j] / r
        end
    end
    
    return KE + PE
end

"""
    run_simulation()

Main simulation function.
"""
function run_simulation()
    println("=" ^ 60)
    println("Two-Galaxy Collision N-Body Simulation")
    println("=" ^ 60)
    
    # Simulation parameters
    N_particles = 1000  # Increased particles for better spiral definition
    R = 1.0             # Galaxy radius
    v_rotation = 0.6    # Increased rotational velocity for more dramatic dynamics
    dt = 0.04           # Smaller time step for better accuracy with more particles
    t_end = 60.0        # Longer simulation time to see full interaction
    save_interval = 6   # Save every N steps (maintains similar frame count)
    
    println("\nSimulation Parameters:")
    println("  Particles per galaxy: $N_particles")
    println("  Galaxy radius: $R")
    println("  Rotational velocity: $v_rotation")
    println("  Time step: $dt")
    println("  Simulation time: $t_end")
    println("  Gravitational softening: $SOFTENING")
    
    # Initialize two galaxies
    println("\nInitializing galaxies...")
    
    # Galaxy 1: approaching from left with slight upward trajectory
    center1 = [-5.0, -0.5, 0.0]
    velocity1 = [0.35, 0.15, 0.02]  # Angled approach for more interesting interaction
    galaxy1 = create_spiral_galaxy(N_particles, R, v_rotation, center1, velocity1)
    
    # Galaxy 2: approaching from right with slight downward trajectory
    center2 = [5.0, 0.5, 0.0]
    velocity2 = [-0.35, -0.15, -0.02]  # Mirror trajectory
    galaxy2 = create_spiral_galaxy(N_particles, R, v_rotation, center2, velocity2)
    
    # Initialize acceleration arrays
    accel1 = zeros(N_particles, 3)
    accel2 = zeros(N_particles, 3)
    
    # Compute initial accelerations
    compute_accelerations!(galaxy1, galaxy2, accel1, accel2)
    
    # Initial energy
    E0 = compute_energy(galaxy1, galaxy2)
    println("  Initial energy: $(@sprintf("%.6f", E0))")
    
    # Time evolution
    println("\nRunning simulation...")
    t = 0.0
    step = 0
    frames = Tuple{Matrix{Float64}, Matrix{Float64}}[]  # Typed array for performance
    times = Float64[]
    energies = Float64[]
    
    # Create initial frame
    push!(frames, (copy(galaxy1.positions), copy(galaxy2.positions)))
    push!(times, t)
    push!(energies, E0)
    
    # Progress bar
    n_steps = Int(ceil(t_end / dt))
    progress_interval = max(1, n_steps ÷ 20)
    
    while t < t_end
        # Integration step
        leapfrog_step!(galaxy1, galaxy2, accel1, accel2, dt)
        
        t += dt
        step += 1
        
        # Save frame periodically
        if step % save_interval == 0
            push!(frames, (copy(galaxy1.positions), copy(galaxy2.positions)))
            push!(times, t)
            E = compute_energy(galaxy1, galaxy2)
            push!(energies, E)
        end
        
        # Progress indicator
        if step % progress_interval == 0
            percent = 100 * t / t_end
            E = compute_energy(galaxy1, galaxy2)
            dE = abs((E - E0) / E0) * 100
            println("  Progress: $(@sprintf("%.1f%%", percent)) | " *
                    "Time: $(@sprintf("%.2f", t)) | " *
                    "Energy drift: $(@sprintf("%.3f%%", dE))")
        end
    end
    
    # Final energy
    E_final = compute_energy(galaxy1, galaxy2)
    energy_drift = abs((E_final - E0) / E0) * 100
    println("\nSimulation complete!")
    println("  Final energy: $(@sprintf("%.6f", E_final))")
    println("  Energy conservation: $(@sprintf("%.4f%%", energy_drift)) drift")
    
    # Create animation
    println("\nCreating animation...")
    create_animation(frames, times, energies, E0)
    
    println("\n" * "=" ^ 60)
    println("Simulation finished successfully!")
    println("Animation saved as: galaxy_collision.gif")
    println("=" ^ 60)
end

"""
    create_animation(frames, times, energies, E0)

Create and save animation of the galaxy collision.
"""
function create_animation(frames, times, energies, E0)
    # Set up the plot theme
    theme(:dark)
    gr(size=(1200, 500))
    
    # Create animation
    anim = @animate for (i, (pos1, pos2)) in enumerate(frames)
        # Create subplot layout
        l = @layout [a{0.7w} b{0.3w}]
        
        # Main 3D plot with better spiral visualization
        p1 = scatter(pos1[:, 1], pos1[:, 2], pos1[:, 3],
                    markersize=1.2, markercolor=:cyan, markeralpha=0.7,
                    label="Galaxy 1", legend=:topright,
                    camera=(30, 30), xlims=(-10, 10), ylims=(-10, 10), zlims=(-3, 3),
                    xlabel="X", ylabel="Y", zlabel="Z",
                    title="Spiral Galaxy Collision (t = $(@sprintf("%.2f", times[i])))",
                    titlefontsize=12, background_color=:black)
        
        scatter!(p1, pos2[:, 1], pos2[:, 2], pos2[:, 3],
                markersize=1.2, markercolor=:yellow, markeralpha=0.7,
                label="Galaxy 2")
        
        # Energy conservation plot
        rel_energies = (energies[1:i] .- E0) ./ abs(E0) .* 100
        p2 = plot(times[1:i], rel_energies,
                 linewidth=2, linecolor=:green, label="ΔE/E₀",
                 xlabel="Time", ylabel="Energy Drift (%)",
                 title="Energy Conservation",
                 titlefontsize=10, legend=:bottomright,
                 grid=true, ylims=(-5, 5), background_color=:black)
        
        plot(p1, p2, layout=l)
    end
    
    # Save animation
    gif(anim, "galaxy_collision.gif", fps=15)
    println("  Animation saved with $(length(frames)) frames")
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation()
end
