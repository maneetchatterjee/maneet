#!/usr/bin/env python3
"""
Demo: VLA Pipeline with Franka Panda

Demonstrates the full VLA (Vision-Language-Action) pipeline using
a Franka Panda robotic arm for pick and place tasks.

This shows how the Franka Panda can be integrated with:
- Computer vision perception
- Natural language understanding
- Symbolic action planning
- Inverse kinematics control
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vla_pipeline.simulation import SimulationEnvironment


def main():
    """Run VLA pipeline demo with Franka Panda."""
    print("\n" + "="*60)
    print("VLA Pipeline with Franka Panda Demo")
    print("="*60)
    
    # Initialize simulation environment
    print("\nInitializing simulation environment...")
    sim = SimulationEnvironment(use_gui=True)
    
    # Load Franka Panda robot explicitly
    print("Loading Franka Panda robot...")
    robot_id = sim.load_franka_panda()
    print(f"  ✓ Loaded robot with ID: {robot_id}")
    print(f"  ✓ Number of joints: {sim.num_joints}")
    print(f"  ✓ Arm joints: {sim.arm_joint_indices}")
    print(f"  ✓ Gripper joints: {sim.gripper_joint_indices}")
    
    # Setup scene with objects
    print("\nSetting up scene...")
    objects = []
    
    # Add red cube
    red_cube = sim.add_object(
        shape='cube',
        color='red',
        position=[0.5, 0.2, 0.05],
        size=0.05
    )
    objects.append(('red_cube', red_cube))
    print("  ✓ Added red cube")
    
    # Add blue cube
    blue_cube = sim.add_object(
        shape='cube',
        color='blue',
        position=[0.5, -0.2, 0.05],
        size=0.05
    )
    objects.append(('blue_cube', blue_cube))
    print("  ✓ Added blue cube")
    
    # Add green sphere
    green_sphere = sim.add_object(
        shape='sphere',
        color='green',
        position=[0.5, 0.0, 0.025],
        size=0.05
    )
    objects.append(('green_sphere', green_sphere))
    print("  ✓ Added green sphere")
    
    # Let objects settle
    print("\nLetting objects settle...")
    sim.step(240)
    
    # Get camera image
    print("\nCapturing scene...")
    rgb_image, depth_image = sim.get_camera_image()
    print(f"  ✓ RGB image shape: {rgb_image.shape}")
    print(f"  ✓ Depth image shape: {depth_image.shape}")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nThe Franka Panda robot is ready for:")
    print("  - Computer vision perception (object detection)")
    print("  - Natural language command parsing")
    print("  - Symbolic action planning")
    print("  - Inverse kinematics control")
    print("\nFor full VLA pipeline demo, use:")
    print("  python demo/demo_basic.py")
    print("\nFor standalone Panda pick-place demo, use:")
    print("  python demo/demo_franka_panda_pick_place.py")
    
    print("\nPress Enter to exit...")
    input()
    
    # Cleanup
    sim.close()
    print("Demo complete!")


if __name__ == "__main__":
    main()
