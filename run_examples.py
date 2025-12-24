#!/usr/bin/env python3
"""
Example script demonstrating different F1 car CFD simulation scenarios
"""

import os
import subprocess
import sys


def run_simulation(name, args):
    """Run a simulation with given parameters"""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}\n")
    
    cmd = ["xvfb-run", "-a", "python", "run_simulation.py"] + args
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {name} failed with error code {e.returncode}")
        return False


def main():
    """Run multiple simulation scenarios"""
    
    print("F1 Car CFD Simulation - Multiple Scenarios")
    print("=" * 70)
    print("\nThis script will run several simulations with different settings.")
    print("Each simulation will be saved to a separate output directory.\n")
    
    scenarios = [
        {
            "name": "Scenario 1: City Speed (50 km/h)",
            "args": [
                "--velocity", "13.9",  # 50 km/h
                "--resolution", "40",
                "--frames", "60",
                "--particles", "80",
                "--output-dir", "output_city_speed",
                "--fps", "20"
            ]
        },
        {
            "name": "Scenario 2: Highway Speed (120 km/h)",
            "args": [
                "--velocity", "33.3",  # 120 km/h
                "--resolution", "40",
                "--frames", "60",
                "--particles", "80",
                "--output-dir", "output_highway_speed",
                "--fps", "20"
            ]
        },
        {
            "name": "Scenario 3: Racing Speed (200 km/h)",
            "args": [
                "--velocity", "55.6",  # 200 km/h
                "--resolution", "50",
                "--frames", "90",
                "--particles", "120",
                "--output-dir", "output_racing_speed",
                "--fps", "30"
            ]
        },
        {
            "name": "Scenario 4: Max Speed (350 km/h)",
            "args": [
                "--velocity", "97.2",  # 350 km/h
                "--resolution", "50",
                "--frames", "90",
                "--particles", "120",
                "--output-dir", "output_max_speed",
                "--fps", "30"
            ]
        }
    ]
    
    # Ask user which scenarios to run
    print("Available scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario['name']}")
    print(f"  {len(scenarios) + 1}. Run all scenarios")
    print(f"  0. Exit")
    
    try:
        choice = input("\nSelect scenario to run (0-5): ").strip()
        choice_num = int(choice)
        
        if choice_num == 0:
            print("Exiting...")
            return
        elif choice_num == len(scenarios) + 1:
            # Run all scenarios
            for scenario in scenarios:
                success = run_simulation(scenario["name"], scenario["args"])
                if not success:
                    print(f"\nStopping due to failure.")
                    return
        elif 1 <= choice_num <= len(scenarios):
            # Run selected scenario
            scenario = scenarios[choice_num - 1]
            run_simulation(scenario["name"], scenario["args"])
        else:
            print("Invalid choice.")
            return
            
    except (ValueError, KeyboardInterrupt):
        print("\nOperation cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("All simulations complete!")
    print("=" * 70)
    print("\nResults are in the respective output_* directories.")


if __name__ == "__main__":
    main()
