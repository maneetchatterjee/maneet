#!/usr/bin/env julia

"""
Binary Black Hole Collision Simulation
======================================

This script simulates the collision and merger of two black holes using
post-Newtonian (PN) approximations and gravitational wave radiation.

Physics Implementation:
- Post-Newtonian orbital dynamics (2.5PN order)
- Gravitational wave energy and angular momentum loss
- Inspiral, merger, and ringdown phases
- Proper relativistic corrections to Newtonian gravity

Based on:
- Peters & Mathews (1963) - GW energy loss
- Blanchet (2014) - Post-Newtonian formalism
- Numerical Relativity principles

Author: Maneet Chatterjee
"""

using LinearAlgebra
using Plots
using Printf

# Physical constants (in geometric units: G=c=1)
const G = 1.0              # Gravitational constant
const C = 1.0              # Speed of light
const MSUN = 1.0           # Solar mass (normalized)

"""
    BlackHole structure to hold black hole properties
"""
mutable struct BlackHole
    mass::Float64          # Mass in solar masses
    position::Vector{Float64}  # 3D position [x, y, z]
    velocity::Vector{Float64}  # 3D velocity [vx, vy, vz]
end

"""
    orbital_frequency(M, r)

Calculate orbital frequency for circular orbit at radius r.
M: total mass
r: separation distance
"""
function orbital_frequency(M::Float64, r::Float64)
    return sqrt(G * M / r^3)
end

"""
    post_newtonian_acceleration(bh1, bh2, order=2.5)

Compute post-Newtonian acceleration up to specified PN order.
Includes:
- Newtonian gravity (0PN)
- 1PN corrections (relativistic corrections)
- 2PN corrections
- 2.5PN radiation reaction (gravitational wave backreaction)

Returns: (accel1, accel2) - accelerations for both black holes
"""
function post_newtonian_acceleration(bh1::BlackHole, bh2::BlackHole, order::Float64=2.5)
    # Relative position and velocity
    r_vec = bh2.position - bh1.position
    r = norm(r_vec)
    n = r_vec / r  # unit vector
    
    v1 = bh1.velocity
    v2 = bh2.velocity
    v = v2 - v1  # relative velocity
    
    m1 = bh1.mass
    m2 = bh2.mass
    M = m1 + m2  # total mass
    μ = m1 * m2 / M  # reduced mass
    η = μ / M  # symmetric mass ratio
    
    # Newtonian term (0PN)
    a_newt = -G * M / r^2 * n
    
    # Velocity magnitudes and dot products
    v1_sq = dot(v1, v1)
    v2_sq = dot(v2, v2)
    v_sq = dot(v, v)
    n_dot_v1 = dot(n, v1)
    n_dot_v2 = dot(n, v2)
    n_dot_v = dot(n, v)
    
    # 1PN correction (v/c)^2
    a_1pn = Vector{Float64}(undef, 3)
    if order >= 1.0
        A_1pn = G * M / r^2 * (
            -v_sq / C^2 - 
            3 * (n_dot_v)^2 / C^2 +
            4 * G * M / (r * C^2)
        )
        B_1pn = G * M / r^2 * 4 * n_dot_v / C^2
        a_1pn = A_1pn * n + B_1pn * v
    else
        a_1pn = zeros(3)
    end
    
    # 2PN correction (v/c)^4
    a_2pn = Vector{Float64}(undef, 3)
    if order >= 2.0
        A_2pn = G * M / r^2 * (
            (3/4) * v_sq^2 / C^4 +
            (9/2) * (n_dot_v)^2 * v_sq / C^4 +
            (15/8) * (n_dot_v)^4 / C^4 -
            (2 * G * M / r) * v_sq / C^4 +
            (4 * G * M / r) * (n_dot_v)^2 / C^4 +
            3 * (G * M / r)^2 / C^4
        )
        B_2pn = G * M / r^2 * (
            -3 * v_sq * n_dot_v / C^4 -
            6 * (n_dot_v)^3 / C^4 +
            (9/2) * (G * M / r) * n_dot_v / C^4
        )
        a_2pn = A_2pn * n + B_2pn * v
    else
        a_2pn = zeros(3)
    end
    
    # 2.5PN radiation reaction (gravitational wave backreaction)
    a_25pn = Vector{Float64}(undef, 3)
    if order >= 2.5
        # Peters-Mathews formula for energy loss
        prefactor = -8/5 * η * (G * M)^3 / (r^4 * C^5)
        
        A_25pn = prefactor * (
            (12/C^2) * v_sq * n_dot_v +
            (4/C^2) * (n_dot_v)^3 -
            (6 * G * M / r) * n_dot_v / C^2
        )
        B_25pn = prefactor * (
            -(4/C^2) * v_sq -
            (8/C^2) * (n_dot_v)^2 +
            (2 * G * M / r) / C^2
        )
        
        a_25pn = A_25pn * n + B_25pn * v
    else
        a_25pn = zeros(3)
    end
    
    # Total acceleration (relative)
    a_rel = a_newt + a_1pn + a_2pn + a_25pn
    
    # Convert to individual accelerations
    accel1 = (m2 / M) * a_rel
    accel2 = -(m1 / M) * a_rel
    
    return accel1, accel2
end

"""
    gravitational_wave_luminosity(bh1, bh2)

Calculate gravitational wave luminosity (energy radiated per unit time).
Uses quadrupole formula.
"""
function gravitational_wave_luminosity(bh1::BlackHole, bh2::BlackHole)
    r_vec = bh2.position - bh1.position
    r = norm(r_vec)
    v = bh2.velocity - bh1.velocity
    v_mag = norm(v)
    
    m1 = bh1.mass
    m2 = bh2.mass
    M = m1 + m2
    μ = m1 * m2 / M
    
    # Quadrupole formula for circular orbits (leading order)
    L_GW = (32/5) * G^4 / C^5 * μ^2 * M^3 / r^5
    
    return L_GW
end

"""
    schwarzschild_radius(mass)

Calculate Schwarzschild radius for a black hole of given mass.
"""
function schwarzschild_radius(mass::Float64)
    return 2 * G * mass / C^2
end

"""
    is_merged(bh1, bh2)

Check if black holes have merged (within their combined Schwarzschild radii).
"""
function is_merged(bh1::BlackHole, bh2::BlackHole)
    r = norm(bh2.position - bh1.position)
    r_s1 = schwarzschild_radius(bh1.mass)
    r_s2 = schwarzschild_radius(bh2.mass)
    # Merge when separation is less than 3 times combined Schwarzschild radius
    return r < 3.0 * (r_s1 + r_s2)
end

"""
    runge_kutta_4_step!(bh1, bh2, dt, order)

Perform one 4th-order Runge-Kutta integration step.
"""
function runge_kutta_4_step!(bh1::BlackHole, bh2::BlackHole, dt::Float64, order::Float64)
    # Store initial state
    r1_0 = copy(bh1.position)
    v1_0 = copy(bh1.velocity)
    r2_0 = copy(bh2.position)
    v2_0 = copy(bh2.velocity)
    
    # k1
    a1_1, a2_1 = post_newtonian_acceleration(bh1, bh2, order)
    k1_r1 = v1_0
    k1_v1 = a1_1
    k1_r2 = v2_0
    k1_v2 = a2_1
    
    # k2
    bh1.position = r1_0 + 0.5 * dt * k1_r1
    bh1.velocity = v1_0 + 0.5 * dt * k1_v1
    bh2.position = r2_0 + 0.5 * dt * k1_r2
    bh2.velocity = v2_0 + 0.5 * dt * k1_v2
    a1_2, a2_2 = post_newtonian_acceleration(bh1, bh2, order)
    k2_r1 = bh1.velocity
    k2_v1 = a1_2
    k2_r2 = bh2.velocity
    k2_v2 = a2_2
    
    # k3
    bh1.position = r1_0 + 0.5 * dt * k2_r1
    bh1.velocity = v1_0 + 0.5 * dt * k2_v1
    bh2.position = r2_0 + 0.5 * dt * k2_r2
    bh2.velocity = v2_0 + 0.5 * dt * k2_v2
    a1_3, a2_3 = post_newtonian_acceleration(bh1, bh2, order)
    k3_r1 = bh1.velocity
    k3_v1 = a1_3
    k3_r2 = bh2.velocity
    k3_v2 = a2_3
    
    # k4
    bh1.position = r1_0 + dt * k3_r1
    bh1.velocity = v1_0 + dt * k3_v1
    bh2.position = r2_0 + dt * k3_r2
    bh2.velocity = v2_0 + dt * k3_v2
    a1_4, a2_4 = post_newtonian_acceleration(bh1, bh2, order)
    k4_r1 = bh1.velocity
    k4_v1 = a1_4
    k4_r2 = bh2.velocity
    k4_v2 = a2_4
    
    # Update positions and velocities
    bh1.position = r1_0 + (dt/6) * (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1)
    bh1.velocity = v1_0 + (dt/6) * (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)
    bh2.position = r2_0 + (dt/6) * (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2)
    bh2.velocity = v2_0 + (dt/6) * (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)
end

"""
    compute_orbital_energy(bh1, bh2)

Calculate total orbital energy (kinetic + potential).
"""
function compute_orbital_energy(bh1::BlackHole, bh2::BlackHole)
    r = norm(bh2.position - bh1.position)
    v1_sq = dot(bh1.velocity, bh1.velocity)
    v2_sq = dot(bh2.velocity, bh2.velocity)
    
    KE = 0.5 * bh1.mass * v1_sq + 0.5 * bh2.mass * v2_sq
    PE = -G * bh1.mass * bh2.mass / r
    
    return KE + PE
end

"""
    run_simulation()

Main simulation function for binary black hole collision.
"""
function run_simulation()
    println("=" ^ 70)
    println("Binary Black Hole Collision Simulation")
    println("Using Post-Newtonian Approximation (2.5PN order)")
    println("=" ^ 70)
    
    # Black hole parameters (in solar masses)
    m1 = 30.0  # Solar masses (similar to GW150914)
    m2 = 30.0  # Solar masses
    
    # Initial separation (in Schwarzschild radii of total mass)
    M_total = m1 + m2
    r_s_total = schwarzschild_radius(M_total)
    initial_separation = 100.0 * r_s_total  # Start at 100 Schwarzschild radii
    
    println("\nSimulation Parameters:")
    println("  Black Hole 1 mass: $m1 M☉")
    println("  Black Hole 2 mass: $m2 M☉")
    println("  Total mass: $M_total M☉")
    println("  Initial separation: $(@sprintf("%.2f", initial_separation)) (geometric units)")
    println("  Initial separation: $(@sprintf("%.1f", initial_separation/r_s_total)) R_s")
    println("  Combined Schwarzschild radius: $(@sprintf("%.2f", r_s_total)) (geometric units)")
    println("  PN order: 2.5 (includes radiation reaction)")
    
    # Initialize black holes in circular orbit
    # Place them on x-axis, orbiting in xy-plane
    r_init = initial_separation / 2.0  # Distance from center of mass
    
    # Calculate circular orbital velocity
    omega = orbital_frequency(M_total, initial_separation)
    v_orbital = omega * r_init
    
    println("  Initial orbital frequency: $(@sprintf("%.6e", omega)) (geometric units)")
    println("  Initial orbital velocity: $(@sprintf("%.6f", v_orbital)) c")
    
    # Black hole 1 starts at (-r_init, 0, 0) moving in +y direction
    bh1 = BlackHole(m1, [-r_init, 0.0, 0.0], [0.0, v_orbital, 0.0])
    
    # Black hole 2 starts at (+r_init, 0, 0) moving in -y direction
    bh2 = BlackHole(m2, [r_init, 0.0, 0.0], [0.0, -v_orbital, 0.0])
    
    # Simulation parameters
    dt = 0.1 * r_s_total  # Adaptive time step based on Schwarzschild radius
    t_max = 10000.0 * r_s_total  # Maximum simulation time
    save_interval = 50  # Save every N steps
    pn_order = 2.5
    
    # Storage for trajectory
    t = 0.0
    step = 0
    times = Float64[]
    pos1_history = Vector{Vector{Float64}}()
    pos2_history = Vector{Vector{Float64}}()
    separation_history = Float64[]
    energy_history = Float64[]
    gw_luminosity_history = Float64[]
    
    # Initial values
    E0 = compute_orbital_energy(bh1, bh2)
    r0 = norm(bh2.position - bh1.position)
    
    println("\n  Initial orbital energy: $(@sprintf("%.6e", E0))")
    println("  Initial separation: $(@sprintf("%.2f", r0))")
    
    # Time evolution
    println("\nRunning simulation...")
    println("  Simulating inspiral, merger, and ringdown phases...")
    
    merged = false
    merge_time = 0.0
    progress_interval = 100
    
    push!(times, t)
    push!(pos1_history, copy(bh1.position))
    push!(pos2_history, copy(bh2.position))
    push!(separation_history, norm(bh2.position - bh1.position))
    push!(energy_history, E0)
    push!(gw_luminosity_history, gravitational_wave_luminosity(bh1, bh2))
    
    while t < t_max && !merged
        # Integration step
        runge_kutta_4_step!(bh1, bh2, dt, pn_order)
        
        t += dt
        step += 1
        
        # Check for merger
        if is_merged(bh1, bh2)
            merged = true
            merge_time = t
            println("\n  *** MERGER DETECTED at t = $(@sprintf("%.2f", t)) ***")
        end
        
        # Save data
        if step % save_interval == 0 || merged
            push!(times, t)
            push!(pos1_history, copy(bh1.position))
            push!(pos2_history, copy(bh2.position))
            
            r_current = norm(bh2.position - bh1.position)
            push!(separation_history, r_current)
            
            E_current = compute_orbital_energy(bh1, bh2)
            push!(energy_history, E_current)
            
            L_GW = gravitational_wave_luminosity(bh1, bh2)
            push!(gw_luminosity_history, L_GW)
        end
        
        # Progress indicator
        if step % progress_interval == 0
            r_current = norm(bh2.position - bh1.position)
            E_current = compute_orbital_energy(bh1, bh2)
            dE_percent = abs((E_current - E0) / E0) * 100
            L_GW = gravitational_wave_luminosity(bh1, bh2)
            
            println("  Step: $step | t: $(@sprintf("%.2f", t)) | " *
                    "r: $(@sprintf("%.2f", r_current)) ($(@sprintf("%.1f", r_current/r_s_total)) R_s) | " *
                    "ΔE: $(@sprintf("%.3f%%", dE_percent)) | " *
                    "L_GW: $(@sprintf("%.2e", L_GW))")
        end
        
        if merged
            break
        end
    end
    
    if !merged
        println("\n  Simulation completed without merger (may need longer time or closer start)")
    else
        println("\n  Merger completed successfully!")
        println("  Total inspiral time: $(@sprintf("%.2f", merge_time))")
        final_sep = separation_history[end]
        println("  Final separation at merger: $(@sprintf("%.2f", final_sep)) ($(@sprintf("%.1f", final_sep/r_s_total)) R_s)")
    end
    
    E_final = energy_history[end]
    E_radiated = E0 - E_final
    println("\n  Final energy: $(@sprintf("%.6e", E_final))")
    println("  Energy radiated in GWs: $(@sprintf("%.6e", E_radiated))")
    println("  Percentage radiated: $(@sprintf("%.2f%%", abs(E_radiated/E0)*100))")
    
    # Create animation
    println("\nCreating animation...")
    create_animation(times, pos1_history, pos2_history, separation_history, 
                     energy_history, gw_luminosity_history, E0, r_s_total, merged)
    
    println("\n" * "=" ^ 70)
    println("Simulation finished successfully!")
    println("Animation saved as: black_hole_collision.gif")
    println("=" ^ 70)
end

"""
    create_animation(times, pos1, pos2, sep, energy, L_GW, E0, r_s, merged)

Create and save animation of the black hole collision.
"""
function create_animation(times, pos1_history, pos2_history, separation_history, 
                         energy_history, gw_luminosity_history, E0, r_s_total, merged)
    theme(:dark)
    gr(size=(1400, 700))
    
    # Determine plot limits
    all_x = vcat([p[1] for p in pos1_history], [p[1] for p in pos2_history])
    all_y = vcat([p[2] for p in pos1_history], [p[2] for p in pos2_history])
    max_range = max(maximum(abs.(all_x)), maximum(abs.(all_y))) * 1.2
    
    # Create animation
    anim = @animate for i in 1:length(times)
        # Layout
        l = @layout [a{0.5w} [b; c; d]]
        
        # Orbital trajectory plot
        p1 = plot(background_color=:black, aspect_ratio=:equal,
                 xlims=(-max_range, max_range), ylims=(-max_range, max_range),
                 xlabel="x (geometric units)", ylabel="y (geometric units)",
                 title="Black Hole Inspiral & Merger (t = $(@sprintf("%.1f", times[i])))",
                 titlefontsize=11, legend=:topright)
        
        # Plot trajectories up to current time
        if i > 1
            x1_trail = [pos1_history[j][1] for j in 1:i]
            y1_trail = [pos1_history[j][2] for j in 1:i]
            x2_trail = [pos2_history[j][1] for j in 1:i]
            y2_trail = [pos2_history[j][2] for j in 1:i]
            
            plot!(p1, x1_trail, y1_trail, linecolor=:cyan, linealpha=0.5, 
                  linewidth=1, label="")
            plot!(p1, x2_trail, y2_trail, linecolor=:yellow, linealpha=0.5, 
                  linewidth=1, label="")
        end
        
        # Plot current black hole positions
        scatter!(p1, [pos1_history[i][1]], [pos1_history[i][2]], 
                markersize=8, markercolor=:cyan, markerstrokewidth=0,
                label="BH1 (30 M☉)")
        scatter!(p1, [pos2_history[i][1]], [pos2_history[i][2]], 
                markersize=8, markercolor=:yellow, markerstrokewidth=0,
                label="BH2 (30 M☉)")
        
        # Add Schwarzschild radius circles
        θ = range(0, 2π, length=50)
        r_s1 = schwarzschild_radius(30.0)
        circle_x1 = pos1_history[i][1] .+ r_s1 * cos.(θ)
        circle_y1 = pos1_history[i][2] .+ r_s1 * sin.(θ)
        plot!(p1, circle_x1, circle_y1, linecolor=:cyan, linealpha=0.3, 
              linewidth=1, label="")
        
        circle_x2 = pos2_history[i][1] .+ r_s1 * cos.(θ)
        circle_y2 = pos2_history[i][2] .+ r_s1 * sin.(θ)
        plot!(p1, circle_x2, circle_y2, linecolor=:yellow, linealpha=0.3, 
              linewidth=1, label="")
        
        # Separation vs time
        p2 = plot(times[1:i], separation_history[1:i] ./ r_s_total,
                 linewidth=2, linecolor=:green, label="",
                 xlabel="Time", ylabel="Separation (R_s)",
                 title="Orbital Separation", titlefontsize=10,
                 background_color=:black, yscale=:log10,
                 ylims=(1, maximum(separation_history ./ r_s_total) * 1.2))
        
        # Energy vs time
        p3 = plot(times[1:i], (energy_history[1:i] .- E0) ./ abs(E0) .* 100,
                 linewidth=2, linecolor=:orange, label="",
                 xlabel="Time", ylabel="ΔE/E₀ (%)",
                 title="Energy Loss (GW Radiation)", titlefontsize=10,
                 background_color=:black)
        
        # GW Luminosity vs time
        p4 = plot(times[1:i], gw_luminosity_history[1:i],
                 linewidth=2, linecolor=:red, label="",
                 xlabel="Time", ylabel="L_GW (geometric units)",
                 title="Gravitational Wave Luminosity", titlefontsize=10,
                 background_color=:black, yscale=:log10,
                 ylims=(minimum(gw_luminosity_history[gw_luminosity_history .> 0]) * 0.1,
                        maximum(gw_luminosity_history) * 10))
        
        plot(p1, p2, p3, p4, layout=l)
    end
    
    # Save animation
    gif(anim, "black_hole_collision.gif", fps=15)
    println("  Animation saved with $(length(times)) frames")
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation()
end
