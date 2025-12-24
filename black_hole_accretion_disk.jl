#!/usr/bin/env julia

"""
Binary Black Hole Collision with Accretion Disk Visualization
============================================================

Enhanced simulation showing black holes with accretion disks,
visualized as disc-shaped structures during collision.

Physics: Post-Newtonian approximations up to 2.5PN order
Visualization: Accretion disks rendered as disc particles

Author: Maneet Chatterjee
"""

using LinearAlgebra
using Plots
using Printf

# Physical constants (geometric units: G=c=1)
const G = 1.0
const C = 1.0

"""
    BlackHole structure
"""
mutable struct BlackHole
    mass::Float64
    position::Vector{Float64}
    velocity::Vector{Float64}
end

"""
    AccretionDisk structure - represents the disk around a black hole
"""
struct AccretionDisk
    inner_radius::Float64  # Inner edge (near ISCO)
    outer_radius::Float64  # Outer edge
    n_particles::Int       # Number of particles in disk
    particles_x::Vector{Float64}  # Particle positions relative to BH
    particles_y::Vector{Float64}
    colors::Vector{Float64}  # Temperature/brightness of particles
end

function schwarzschild_radius(mass::Float64)
    return 2 * G * mass / C^2
end

function create_accretion_disk(bh_mass::Float64, n_particles::Int=200)
    r_s = schwarzschild_radius(bh_mass)
    inner_r = 3 * r_s  # ISCO for non-rotating BH
    outer_r = 20 * r_s  # Outer edge of visible disk
    
    # Generate disk particles in a spiral pattern
    particles_x = Float64[]
    particles_y = Float64[]
    colors = Float64[]
    
    for i in 1:n_particles
        # Logarithmic spiral for realistic appearance
        angle = 2π * rand() + 0.5 * log(rand())
        r = inner_r + (outer_r - inner_r) * rand()^0.5  # r^0.5 for surface density
        
        x = r * cos(angle)
        y = r * sin(angle)
        
        push!(particles_x, x)
        push!(particles_y, y)
        
        # Temperature decreases with radius (hotter near BH)
        temp = 1.0 / (r / inner_r)
        push!(colors, temp)
    end
    
    return AccretionDisk(inner_r, outer_r, n_particles, 
                         particles_x, particles_y, colors)
end

function orbital_frequency(M::Float64, r::Float64)
    return sqrt(G * M / r^3)
end

function post_newtonian_acceleration(bh1::BlackHole, bh2::BlackHole, order::Float64=2.5)
    r_vec = bh2.position - bh1.position
    r = norm(r_vec)
    n = r_vec / r
    
    v1 = bh1.velocity
    v2 = bh2.velocity
    v = v2 - v1
    
    m1 = bh1.mass
    m2 = bh2.mass
    M = m1 + m2
    μ = m1 * m2 / M
    η = μ / M
    
    # Newtonian
    a_newt = -G * M / r^2 * n
    
    v1_sq = dot(v1, v1)
    v2_sq = dot(v2, v2)
    v_sq = dot(v, v)
    n_dot_v = dot(n, v)
    
    # 1PN
    a_1pn = zeros(3)
    if order >= 1.0
        A_1pn = G * M / r^2 * (-v_sq / C^2 - 3 * (n_dot_v)^2 / C^2 + 4 * G * M / (r * C^2))
        B_1pn = G * M / r^2 * 4 * n_dot_v / C^2
        a_1pn = A_1pn * n + B_1pn * v
    end
    
    # 2PN
    a_2pn = zeros(3)
    if order >= 2.0
        A_2pn = G * M / r^2 * ((3/4) * v_sq^2 / C^4 + (9/2) * (n_dot_v)^2 * v_sq / C^4 +
                               (15/8) * (n_dot_v)^4 / C^4 - (2 * G * M / r) * v_sq / C^4 +
                               (4 * G * M / r) * (n_dot_v)^2 / C^4 + 3 * (G * M / r)^2 / C^4)
        B_2pn = G * M / r^2 * (-3 * v_sq * n_dot_v / C^4 - 6 * (n_dot_v)^3 / C^4 +
                               (9/2) * (G * M / r) * n_dot_v / C^4)
        a_2pn = A_2pn * n + B_2pn * v
    end
    
    # 2.5PN radiation reaction
    a_25pn = zeros(3)
    if order >= 2.5
        prefactor = -8/5 * η * (G * M)^3 / (r^4 * C^5)
        A_25pn = prefactor * ((12/C^2) * v_sq * n_dot_v + (4/C^2) * (n_dot_v)^3 -
                              (6 * G * M / r) * n_dot_v / C^2)
        B_25pn = prefactor * (-(4/C^2) * v_sq - (8/C^2) * (n_dot_v)^2 +
                              (2 * G * M / r) / C^2)
        a_25pn = A_25pn * n + B_25pn * v
    end
    
    a_rel = a_newt + a_1pn + a_2pn + a_25pn
    accel1 = (m2 / M) * a_rel
    accel2 = -(m1 / M) * a_rel
    
    return accel1, accel2
end

function runge_kutta_4_step!(bh1::BlackHole, bh2::BlackHole, dt::Float64, order::Float64)
    r1_0 = copy(bh1.position)
    v1_0 = copy(bh1.velocity)
    r2_0 = copy(bh2.position)
    v2_0 = copy(bh2.velocity)
    
    # k1
    a1_1, a2_1 = post_newtonian_acceleration(bh1, bh2, order)
    k1_r1, k1_v1 = v1_0, a1_1
    k1_r2, k1_v2 = v2_0, a2_1
    
    # k2
    bh1.position = r1_0 + 0.5 * dt * k1_r1
    bh1.velocity = v1_0 + 0.5 * dt * k1_v1
    bh2.position = r2_0 + 0.5 * dt * k1_r2
    bh2.velocity = v2_0 + 0.5 * dt * k1_v2
    a1_2, a2_2 = post_newtonian_acceleration(bh1, bh2, order)
    k2_r1, k2_v1 = bh1.velocity, a1_2
    k2_r2, k2_v2 = bh2.velocity, a2_2
    
    # k3
    bh1.position = r1_0 + 0.5 * dt * k2_r1
    bh1.velocity = v1_0 + 0.5 * dt * k2_v1
    bh2.position = r2_0 + 0.5 * dt * k2_r2
    bh2.velocity = v2_0 + 0.5 * dt * k2_v2
    a1_3, a2_3 = post_newtonian_acceleration(bh1, bh2, order)
    k3_r1, k3_v1 = bh1.velocity, a1_3
    k3_r2, k3_v2 = bh2.velocity, a2_3
    
    # k4
    bh1.position = r1_0 + dt * k3_r1
    bh1.velocity = v1_0 + dt * k3_v1
    bh2.position = r2_0 + dt * k3_r2
    bh2.velocity = v2_0 + dt * k3_v2
    a1_4, a2_4 = post_newtonian_acceleration(bh1, bh2, order)
    k4_r1, k4_v1 = bh1.velocity, a1_4
    k4_r2, k4_v2 = bh2.velocity, a2_4
    
    # Update
    bh1.position = r1_0 + (dt/6) * (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1)
    bh1.velocity = v1_0 + (dt/6) * (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)
    bh2.position = r2_0 + (dt/6) * (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2)
    bh2.velocity = v2_0 + (dt/6) * (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)
end

function compute_orbital_energy(bh1::BlackHole, bh2::BlackHole)
    r = norm(bh2.position - bh1.position)
    v1_sq = dot(bh1.velocity, bh1.velocity)
    v2_sq = dot(bh2.velocity, bh2.velocity)
    KE = 0.5 * bh1.mass * v1_sq + 0.5 * bh2.mass * v2_sq
    PE = -G * bh1.mass * bh2.mass / r
    return KE + PE
end

function gravitational_wave_luminosity(bh1::BlackHole, bh2::BlackHole)
    r = norm(bh2.position - bh1.position)
    m1, m2 = bh1.mass, bh2.mass
    M = m1 + m2
    μ = m1 * m2 / M
    return (32/5) * G^4 / C^5 * μ^2 * M^3 / r^5
end

function is_merged(bh1::BlackHole, bh2::BlackHole)
    r = norm(bh2.position - bh1.position)
    r_s1 = schwarzschild_radius(bh1.mass)
    r_s2 = schwarzschild_radius(bh2.mass)
    return r < 3.0 * (r_s1 + r_s2)
end

function run_simulation()
    println("=" ^ 70)
    println("Black Hole Collision with Accretion Disk Visualization")
    println("=" ^ 70)
    
    # Parameters - closer start for visible inspiral
    m1 = 30.0
    m2 = 30.0
    M_total = m1 + m2
    r_s_total = schwarzschild_radius(M_total)
    initial_separation = 15.0 * r_s_total  # Closer start: 15 R_s
    
    println("\nSimulation Parameters:")
    println("  Black Hole masses: $m1 M☉, $m2 M☉")
    println("  Initial separation: $(@sprintf("%.1f", initial_separation/r_s_total)) R_s")
    println("  Combined Schwarzschild radius: $(@sprintf("%.2f", r_s_total))")
    
    # Circular orbit
    r_init = initial_separation / 2.0
    omega = orbital_frequency(M_total, initial_separation)
    v_orbital = omega * r_init
    
    println("  Orbital velocity: $(@sprintf("%.4f", v_orbital)) c")
    
    bh1 = BlackHole(m1, [-r_init, 0.0, 0.0], [0.0, v_orbital, 0.0])
    bh2 = BlackHole(m2, [r_init, 0.0, 0.0], [0.0, -v_orbital, 0.0])
    
    # Create accretion disks
    disk1 = create_accretion_disk(m1, 150)
    disk2 = create_accretion_disk(m2, 150)
    
    # Simulation parameters
    dt = 2.0  # Larger time step for faster evolution
    t_max = 50000.0
    save_interval = 20
    pn_order = 2.5
    
    t = 0.0
    step = 0
    times = Float64[]
    pos1_history = Vector{Vector{Float64}}()
    pos2_history = Vector{Vector{Float64}}()
    separation_history = Float64[]
    energy_history = Float64[]
    gw_luminosity_history = Float64[]
    
    E0 = compute_orbital_energy(bh1, bh2)
    
    println("\nRunning simulation...")
    
    merged = false
    progress_interval = 50
    
    push!(times, t)
    push!(pos1_history, copy(bh1.position))
    push!(pos2_history, copy(bh2.position))
    push!(separation_history, norm(bh2.position - bh1.position))
    push!(energy_history, E0)
    push!(gw_luminosity_history, gravitational_wave_luminosity(bh1, bh2))
    
    while t < t_max && !merged && length(times) < 500  # Limit frames
        runge_kutta_4_step!(bh1, bh2, dt, pn_order)
        t += dt
        step += 1
        
        if is_merged(bh1, bh2)
            merged = true
            println("\n  *** MERGER at t = $(@sprintf("%.1f", t)) ***")
        end
        
        if step % save_interval == 0 || merged
            push!(times, t)
            push!(pos1_history, copy(bh1.position))
            push!(pos2_history, copy(bh2.position))
            push!(separation_history, norm(bh2.position - bh1.position))
            push!(energy_history, compute_orbital_energy(bh1, bh2))
            push!(gw_luminosity_history, gravitational_wave_luminosity(bh1, bh2))
        end
        
        if step % progress_interval == 0
            r_current = norm(bh2.position - bh1.position)
            println("  t: $(@sprintf("%.1f", t)) | r: $(@sprintf("%.1f", r_current/r_s_total)) R_s")
        end
        
        if merged
            break
        end
    end
    
    if merged
        println("  Merger completed at t = $(@sprintf("%.1f", t))")
    end
    
    println("\nCreating animation...")
    create_animation(times, pos1_history, pos2_history, separation_history,
                     energy_history, gw_luminosity_history, E0, r_s_total,
                     disk1, disk2, merged)
    
    println("\n" * "=" ^ 70)
    println("Animation saved as: black_hole_accretion_disk.gif")
    println("=" ^ 70)
end

function create_animation(times, pos1_history, pos2_history, separation_history,
                         energy_history, gw_luminosity_history, E0, r_s_total,
                         disk1, disk2, merged)
    theme(:dark)
    gr(size=(1600, 800))
    
    all_x = vcat([p[1] for p in pos1_history], [p[1] for p in pos2_history])
    all_y = vcat([p[2] for p in pos1_history], [p[2] for p in pos2_history])
    max_range = max(maximum(abs.(all_x)), maximum(abs.(all_y))) * 1.3
    
    anim = @animate for i in 1:length(times)
        l = @layout [a{0.55w} [b; c; d]]
        
        # Main plot with accretion disks
        p1 = plot(background_color=:black, aspect_ratio=:equal,
                 xlims=(-max_range, max_range), ylims=(-max_range, max_range),
                 xlabel="x (geometric units)", ylabel="y (geometric units)",
                 title="Black Hole Collision (t = $(@sprintf("%.0f", times[i])))",
                 titlefontsize=12, legend=:topright, grid=false)
        
        # Orbital trails
        if i > 1
            x1_trail = [pos1_history[j][1] for j in max(1, i-50):i]
            y1_trail = [pos1_history[j][2] for j in max(1, i-50):i]
            x2_trail = [pos2_history[j][1] for j in max(1, i-50):i]
            y2_trail = [pos2_history[j][2] for j in max(1, i-50):i]
            
            plot!(p1, x1_trail, y1_trail, linecolor=:cyan, linealpha=0.3,
                  linewidth=2, label="")
            plot!(p1, x2_trail, y2_trail, linecolor=:orange, linealpha=0.3,
                  linewidth=2, label="")
        end
        
        # Accretion disk 1 (cyan)
        pos1 = pos1_history[i]
        disk1_x = disk1.particles_x .+ pos1[1]
        disk1_y = disk1.particles_y .+ pos1[2]
        scatter!(p1, disk1_x, disk1_y,
                markersize=2.5, markercolor=:cyan, markeralpha=0.6,
                markerstrokewidth=0, label="")
        
        # Accretion disk 2 (orange)
        pos2 = pos2_history[i]
        disk2_x = disk2.particles_x .+ pos2[1]
        disk2_y = disk2.particles_y .+ pos2[2]
        scatter!(p1, disk2_x, disk2_y,
                markersize=2.5, markercolor=:orange, markeralpha=0.6,
                markerstrokewidth=0, label="")
        
        # Black hole centers
        scatter!(p1, [pos1[1]], [pos1[2]],
                markersize=10, markercolor=:white, markerstrokewidth=2,
                markerstrokecolor=:cyan, label="BH1 (30 M☉)")
        scatter!(p1, [pos2[1]], [pos2[2]],
                markersize=10, markercolor=:white, markerstrokewidth=2,
                markerstrokecolor=:orange, label="BH2 (30 M☉)")
        
        # Schwarzschild radii
        θ = range(0, 2π, length=50)
        r_s1 = schwarzschild_radius(30.0)
        circle_x1 = pos1[1] .+ r_s1 * cos.(θ)
        circle_y1 = pos1[2] .+ r_s1 * sin.(θ)
        plot!(p1, circle_x1, circle_y1, linecolor=:cyan, linealpha=0.4,
              linewidth=1, linestyle=:dash, label="")
        
        circle_x2 = pos2[1] .+ r_s1 * cos.(θ)
        circle_y2 = pos2[2] .+ r_s1 * sin.(θ)
        plot!(p1, circle_x2, circle_y2, linecolor=:orange, linealpha=0.4,
              linewidth=1, linestyle=:dash, label="")
        
        # Metrics panels
        p2 = plot(times[1:i], separation_history[1:i] ./ r_s_total,
                 linewidth=2.5, linecolor=:lime, label="",
                 xlabel="Time", ylabel="Separation (R_s)",
                 title="Orbital Decay", titlefontsize=10,
                 background_color=:black, yscale=:log10,
                 ylims=(1, maximum(separation_history ./ r_s_total) * 1.2),
                 grid=true, gridcolor=:gray, gridalpha=0.3)
        
        p3 = plot(times[1:i], (energy_history[1:i] .- E0) ./ abs(E0) .* 100,
                 linewidth=2.5, linecolor=:yellow, label="",
                 xlabel="Time", ylabel="ΔE/E₀ (%)",
                 title="Energy Loss (GW Radiation)", titlefontsize=10,
                 background_color=:black, grid=true, gridcolor=:gray, gridalpha=0.3)
        
        p4 = plot(times[1:i], gw_luminosity_history[1:i],
                 linewidth=2.5, linecolor=:red, label="",
                 xlabel="Time", ylabel="L_GW",
                 title="GW Luminosity", titlefontsize=10,
                 background_color=:black, yscale=:log10,
                 ylims=(minimum(gw_luminosity_history[gw_luminosity_history .> 0]) * 0.1,
                        maximum(gw_luminosity_history) * 10),
                 grid=true, gridcolor=:gray, gridalpha=0.3)
        
        plot(p1, p2, p3, p4, layout=l)
    end
    
    gif(anim, "black_hole_accretion_disk.gif", fps=20)
    println("  Animation saved with $(length(times)) frames")
end

# Run simulation
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation()
end
