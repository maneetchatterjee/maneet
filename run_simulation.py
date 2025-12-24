#!/usr/bin/env python3
"""
Main script to run F1 Car CFD Simulation
Creates 3D model, runs flow simulation, and generates animations
"""

import os
import sys
import argparse
from f1_car_model import F1CarModel
from cfd_simulation import CFDSimulation
from visualization import SimulationVisualizer


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='F1 Car CFD Simulation - Design, Simulate, and Animate'
    )
    parser.add_argument(
        '--velocity',
        type=float,
        default=30.0,
        help='Freestream velocity in m/s (default: 30.0 m/s ≈ 108 km/h)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=50,
        choices=[30, 40, 50, 60],
        help='Grid resolution for CFD (default: 50, higher = more accurate but slower)'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=120,
        help='Number of animation frames (default: 120)'
    )
    parser.add_argument(
        '--particles',
        type=int,
        default=100,
        help='Number of flow particles in animation (default: 100)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Animation frames per second (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--skip-animation',
        action='store_true',
        help='Skip creating the animation (faster for testing)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("F1 CAR CFD SIMULATION")
    print("=" * 70)
    print()
    
    # Step 1: Create F1 car model
    print("Step 1: Creating 3D F1 car model...")
    print("-" * 70)
    car_model = F1CarModel(scale=1.0)
    car_mesh = car_model.create_car()
    print(f"✓ F1 car model created successfully!")
    print(f"  Car bounds: {car_mesh.bounds}")
    print(f"  Number of cells: {car_mesh.n_cells}")
    print()
    
    # Step 2: Setup and run CFD simulation
    print("Step 2: Running CFD simulation...")
    print("-" * 70)
    print(f"  Freestream velocity: {args.velocity:.1f} m/s ({args.velocity * 3.6:.1f} km/h)")
    print(f"  Grid resolution: {args.resolution}³ = {args.resolution**3:,} points")
    
    cfd = CFDSimulation(car_mesh, freestream_velocity=args.velocity)
    cfd.grid_resolution = args.resolution
    cfd.setup_domain(padding=2.0)
    
    # Compute flow field
    velocity_field, pressure_field = cfd.compute_flow_field()
    
    # Compute aerodynamic forces
    forces = cfd.compute_forces()
    print(f"✓ CFD simulation complete!")
    print(f"  Drag: {forces['drag']:.1f} N")
    print(f"  Downforce: {forces['downforce']:.1f} N")
    print()
    
    # Step 3: Create visualizations
    print("Step 3: Creating visualizations...")
    print("-" * 70)
    
    viz = SimulationVisualizer(car_mesh, cfd)
    
    # Static visualization with streamlines
    static_path = os.path.join(args.output_dir, 'simulation_static.png')
    viz.create_static_visualization(save_path=static_path)
    print(f"✓ Static visualization created: {static_path}")
    
    # Pressure field
    pressure_path = os.path.join(args.output_dir, 'pressure_field.png')
    viz.create_pressure_visualization(save_path=pressure_path)
    print(f"✓ Pressure field created: {pressure_path}")
    
    # Velocity field
    velocity_path = os.path.join(args.output_dir, 'velocity_field.png')
    viz.create_velocity_magnitude_plot(save_path=velocity_path)
    print(f"✓ Velocity field created: {velocity_path}")
    
    # Simulation report
    report_path = os.path.join(args.output_dir, 'simulation_report.txt')
    viz.create_summary_report(forces, save_path=report_path)
    print(f"✓ Simulation report created: {report_path}")
    print()
    
    # Step 4: Create animation
    if not args.skip_animation:
        print("Step 4: Creating flow animation...")
        print("-" * 70)
        print(f"  Frames: {args.frames}")
        print(f"  Particles: {args.particles}")
        print(f"  FPS: {args.fps}")
        print("  This may take several minutes...")
        
        animation_path = os.path.join(args.output_dir, 'simulation_animation.gif')
        viz.create_particle_animation(
            output_path=animation_path,
            n_particles=args.particles,
            n_frames=args.frames,
            fps=args.fps
        )
        print(f"✓ Animation created: {animation_path}")
        print()
    else:
        print("Step 4: Skipping animation (--skip-animation flag set)")
        print()
    
    # Summary
    print("=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"All results saved to: {os.path.abspath(args.output_dir)}/")
    print()
    print("Output files:")
    print(f"  • {static_path}")
    print(f"  • {pressure_path}")
    print(f"  • {velocity_path}")
    print(f"  • {report_path}")
    if not args.skip_animation:
        print(f"  • {animation_path}")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
