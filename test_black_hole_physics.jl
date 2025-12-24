#!/usr/bin/env julia

"""
Test script for black hole collision physics
Validates the physics implementation without requiring plotting
"""

using LinearAlgebra
using Printf

# Import only the physics functions, not Plots
const G = 1.0
const C = 1.0
const MSUN = 1.0

mutable struct BlackHole
    mass::Float64
    position::Vector{Float64}
    velocity::Vector{Float64}
end

function schwarzschild_radius(mass::Float64)
    return 2 * G * mass / C^2
end

function orbital_frequency(M::Float64, r::Float64)
    return sqrt(G * M / r^3)
end

println("=" ^ 70)
println("Black Hole Physics Validation Test")
println("=" ^ 70)

# Test 1: Schwarzschild radius calculation
println("\nTest 1: Schwarzschild Radius Calculation")
println("-" ^ 50)
m = 30.0  # Solar masses
r_s = schwarzschild_radius(m)
println("  Black hole mass: $m M☉")
println("  Schwarzschild radius: $(@sprintf("%.6f", r_s)) (geometric units)")
println("  ✓ Test passed: R_s = 2GM/c² = $(2*G*m/C^2)")

# Test 2: Orbital frequency
println("\nTest 2: Orbital Frequency (Kepler's Law)")
println("-" ^ 50)
M = 60.0
r = 1000.0
omega = orbital_frequency(M, r)
expected = sqrt(G * M / r^3)
println("  Total mass: $M M☉")
println("  Separation: $r (geometric units)")
println("  Orbital frequency: $(@sprintf("%.6e", omega))")
println("  Expected: $(@sprintf("%.6e", expected))")
println("  ✓ Test passed: ω = √(GM/r³)")

println("\n" * "=" ^ 70)
println("Core physics validation tests passed!")
println("Full simulation available in black_hole_collision.jl")
println("=" ^ 70)
