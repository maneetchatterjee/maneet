#!/usr/bin/env python3
"""
Demo: Alternate Object Shapes (Cup and Bottle)

Demonstrates the VLA pipeline with cup and bottle objects instead of just blocks.
Shows how to configure alternate object models in scene setup.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vla_pipeline import VLAPipeline


def main():
    """Run demo with cups and bottles."""
    print("\n" + "="*60)
    print("VLA Pipeline Demo: Alternate Object Shapes")
    print("="*60)
    
    # Initialize pipeline
    print("\nInitializing VLA pipeline...")
    pipeline = VLAPipeline(use_gui=True, log_metrics=True)
    
    # Setup scene with different object types
    scene_config = {
        'trays': [
            # Cup placement zone
            {
                'position': [-0.1, -0.15, 0.0],
                'size': (0.12, 0.12),
                'color': 'blue',
                'label': 'Cup Zone'
            },
            # Bottle placement zone
            {
                'position': [-0.1, 0.15, 0.0],
                'size': (0.12, 0.12),
                'color': 'green',
                'label': 'Bottle Zone'
            },
        ],
        'objects': [
            # Traditional blocks
            {'shape': 'cube', 'color': 'red', 'position': [0.3, -0.2, 0.05], 'size': 0.05},
            
            # Cup objects (taller cylinders)
            {'shape': 'cup', 'color': 'blue', 'position': [0.3, 0.0, 0.075], 'size': 0.05},
            {'shape': 'cup', 'color': 'yellow', 'position': [0.35, 0.05, 0.075], 'size': 0.045},
            
            # Bottle objects (narrow, tall cylinders)
            {'shape': 'bottle', 'color': 'green', 'position': [0.3, 0.2, 0.0625], 'size': 0.05},
            {'shape': 'bottle', 'color': 'orange', 'position': [0.35, 0.15, 0.0625], 'size': 0.045},
            
            # Mixed traditional shapes
            {'shape': 'sphere', 'color': 'purple', 'position': [0.4, -0.15, 0.05], 'size': 0.04},
        ]
    }
    
    pipeline.setup_scene(scene_config)
    
    print("\n" + "="*60)
    print("OBJECT TYPES IN SCENE:")
    print("  - Cubes/Blocks: Traditional geometric objects")
    print("  - Cups: Wider cylinders (height = 1.5x size)")
    print("  - Bottles: Narrow cylinders (height = 2.5x size, radius = size/3)")
    print("  - Spheres: Round objects")
    print("")
    print("VISUAL VERIFICATION:")
    print("  - All objects should rest properly on the table")
    print("  - Trays should be flush with table surface")
    print("  - Different shapes should be distinguishable")
    print("="*60)
    
    # Test commands with different object types
    commands = [
        "Pick the blue cup and place it on the cup zone",
        "Pick the green bottle and place it on the bottle zone",
        "Pick the red cube and place it next to the purple sphere",
    ]
    
    # Execute commands
    for i, command in enumerate(commands):
        print(f"\n\nCommand {i+1}/{len(commands)}: {command}")
        success = pipeline.execute_command(command)
        
        if not success:
            print(f"Command failed: {command}")
        
        # Small delay between commands for visualization
        pipeline.simulation.step(num_steps=50)
    
    # Print metrics
    pipeline.print_metrics_summary()
    pipeline.save_metrics("demo_alternate_objects_metrics.json")
    
    # Get final scene
    scene = pipeline.get_scene_representation()
    print("\nFinal Scene Representation:")
    print(f"Objects detected: {scene['num_objects']}")
    
    print("\n" + "="*60)
    print("OBJECT SUBSTITUTION NOTES:")
    print("  - See OBJECT_SUBSTITUTION_GUIDE.md for details")
    print("  - Any primitive shape can be added by extending add_object()")
    print("  - Custom meshes/URDF files are also supported")
    print("  - Z-positioning: z = object_height/2 for table-resting objects")
    print("="*60)
    
    # Keep simulation open for inspection
    input("\nPress Enter to close simulation...")
    
    # Cleanup
    pipeline.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
