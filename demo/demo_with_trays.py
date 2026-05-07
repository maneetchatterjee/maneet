#!/usr/bin/env python3
"""
Demo: Sorting with Trays/Placement Zones

Demonstrates the VLA pipeline with visual trays that mark placement zones.
Shows that trays sit flush on the table surface without floating.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vla_pipeline import VLAPipeline


def main():
    """Run sorting demo with placement zone trays."""
    print("\n" + "="*60)
    print("VLA Pipeline Demo: Sorting with Placement Zones")
    print("="*60)
    
    # Initialize pipeline
    print("\nInitializing VLA pipeline...")
    pipeline = VLAPipeline(use_gui=True, log_metrics=True)
    
    # Setup scene with trays and objects
    # Trays are positioned flush on the table surface (z=0.0)
    scene_config = {
        'trays': [
            # Red sorting zone on the left
            {
                'position': [-0.1, -0.2, 0.0],
                'size': (0.15, 0.15),
                'color': 'red',
                'label': 'Red Zone'
            },
            # Blue sorting zone in the middle
            {
                'position': [-0.1, 0.0, 0.0],
                'size': (0.15, 0.15),
                'color': 'blue',
                'label': 'Blue Zone'
            },
            # Green sorting zone on the right
            {
                'position': [-0.1, 0.2, 0.0],
                'size': (0.15, 0.15),
                'color': 'green',
                'label': 'Green Zone'
            },
        ],
        'objects': [
            # Mixed colored blocks to be sorted
            {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
            {'shape': 'cube', 'color': 'blue', 'position': [0.3, 0.15, 0.05], 'size': 0.05},
            {'shape': 'cube', 'color': 'green', 'position': [0.3, -0.15, 0.05], 'size': 0.05},
            {'shape': 'sphere', 'color': 'red', 'position': [0.4, 0.1, 0.05], 'size': 0.04},
            {'shape': 'sphere', 'color': 'blue', 'position': [0.4, -0.1, 0.05], 'size': 0.04},
        ]
    }
    
    pipeline.setup_scene(scene_config)
    
    print("\n" + "="*60)
    print("VISUAL VERIFICATION:")
    print("  - Check that trays (colored zones) are visible on the table")
    print("  - Trays should be flush with the table surface (not floating)")
    print("  - Trays are semi-transparent for visibility")
    print("="*60)
    
    # Test sorting commands
    commands = [
        "Pick the red cube and place it on the red zone",
        "Pick the blue cube and place it on the blue zone", 
        "Pick the green cube and place it on the green zone",
    ]
    
    # Execute commands
    for i, command in enumerate(commands):
        print(f"\n\nCommand {i+1}/{len(commands)}: {command}")
        success = pipeline.execute_command(command)
        
        if not success:
            print(f"Command failed: {command}")
    
    # Print metrics
    pipeline.print_metrics_summary()
    pipeline.save_metrics("demo_tray_metrics.json")
    
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
