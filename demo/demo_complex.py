#!/usr/bin/env python3
"""
Demo: Complex Scenarios

Tests the VLA pipeline with more complex manipulation scenarios.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vla_pipeline import VLAPipeline


def run_stacking_scenario(pipeline):
    """Test stacking objects."""
    print("\n" + "="*60)
    print("Scenario 1: Stacking Objects")
    print("="*60)
    
    scene_config = {
        'objects': [
            {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
            {'shape': 'cube', 'color': 'blue', 'position': [0.3, 0.2, 0.05], 'size': 0.05},
            {'shape': 'cube', 'color': 'yellow', 'position': [0.3, -0.2, 0.05], 'size': 0.05},
        ]
    }
    
    pipeline.setup_scene(scene_config)
    
    commands = [
        "Pick the red cube",
        "Place it on the blue cube",
        "Pick the yellow cube",
        "Place it on the red cube",
    ]
    
    for command in commands:
        pipeline.execute_command(command)


def run_sorting_scenario(pipeline):
    """Test sorting objects by color."""
    print("\n" + "="*60)
    print("Scenario 2: Sorting by Color")
    print("="*60)
    
    scene_config = {
        'objects': [
            {'shape': 'sphere', 'color': 'red', 'position': [0.2, 0.0, 0.05], 'size': 0.04},
            {'shape': 'sphere', 'color': 'blue', 'position': [0.25, 0.05, 0.05], 'size': 0.04},
            {'shape': 'sphere', 'color': 'red', 'position': [0.3, -0.05, 0.05], 'size': 0.04},
            {'shape': 'sphere', 'color': 'blue', 'position': [0.35, 0.1, 0.05], 'size': 0.04},
        ]
    }
    
    pipeline.setup_scene(scene_config)
    
    # Note: This is a simplified sorting - full sorting would need multiple steps
    commands = [
        "Pick the red sphere and place it left of the blue sphere",
    ]
    
    for command in commands:
        pipeline.execute_command(command)


def run_spatial_relations_scenario(pipeline):
    """Test spatial relation understanding."""
    print("\n" + "="*60)
    print("Scenario 3: Spatial Relations")
    print("="*60)
    
    scene_config = {
        'objects': [
            {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
            {'shape': 'cube', 'color': 'blue', 'position': [0.3, 0.15, 0.05], 'size': 0.05},
            {'shape': 'cylinder', 'color': 'green', 'position': [0.3, -0.15, 0.05], 'size': 0.05},
        ]
    }
    
    pipeline.setup_scene(scene_config)
    
    commands = [
        "Pick the green cylinder and place it right of the red cube",
        "Pick the blue cube and place it next to the green cylinder",
    ]
    
    for command in commands:
        pipeline.execute_command(command)


def main():
    """Run complex demonstration scenarios."""
    print("\n" + "="*60)
    print("VLA Pipeline Demo: Complex Scenarios")
    print("="*60)
    
    # Initialize pipeline
    pipeline = VLAPipeline(use_gui=True, log_metrics=True)
    
    # Run scenarios
    scenarios = [
        ("Stacking", run_stacking_scenario),
        ("Sorting", run_sorting_scenario),
        ("Spatial Relations", run_spatial_relations_scenario),
    ]
    
    for name, scenario_func in scenarios:
        try:
            scenario_func(pipeline)
        except Exception as e:
            print(f"\nScenario '{name}' failed: {str(e)}")
    
    # Print final metrics
    pipeline.print_metrics_summary()
    pipeline.save_metrics("complex_demo_metrics.json")
    
    input("\nPress Enter to close simulation...")
    
    pipeline.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
