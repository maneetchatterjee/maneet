#!/usr/bin/env python3
"""
Demo: Basic Pick and Place

Demonstrates the VLA pipeline with simple pick-and-place tasks.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vla_pipeline import VLAPipeline


def main():
    """Run basic pick-and-place demo."""
    print("\n" + "="*60)
    print("VLA Pipeline Demo: Basic Pick and Place")
    print("="*60)
    
    # Initialize pipeline
    print("\nInitializing VLA pipeline...")
    pipeline = VLAPipeline(use_gui=True, log_metrics=True)
    
    # Setup scene
    scene_config = {
        'objects': [
            {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
            {'shape': 'cube', 'color': 'blue', 'position': [0.3, 0.15, 0.05], 'size': 0.05},
            {'shape': 'sphere', 'color': 'green', 'position': [0.3, -0.15, 0.05], 'size': 0.05},
        ]
    }
    
    pipeline.setup_scene(scene_config)
    
    # Test commands
    commands = [
        "Pick the red cube and place it left of the blue cube",
        "Pick the green sphere and place it right of the red cube",
        "Pick the blue cube and place it on the green sphere",
    ]
    
    # Execute commands
    for i, command in enumerate(commands):
        print(f"\n\nCommand {i+1}/{len(commands)}")
        success = pipeline.execute_command(command)
        
        if not success:
            print(f"Command failed: {command}")
    
    # Print metrics
    pipeline.print_metrics_summary()
    pipeline.save_metrics("demo_metrics.json")
    
    # Get final scene
    scene = pipeline.get_scene_representation()
    print("\nFinal Scene Representation:")
    print(f"Objects detected: {scene['num_objects']}")
    
    # Keep simulation open for inspection
    input("\nPress Enter to close simulation...")
    
    # Cleanup
    pipeline.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
