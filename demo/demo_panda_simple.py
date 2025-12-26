#!/usr/bin/env python3
"""
Simple Franka Panda Demo

A minimal, standalone example of the Franka Panda robot performing
a single pick and place task. Perfect for getting started!
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


def main():
    """Run a simple pick and place demo."""
    print("Starting Franka Panda Simple Demo...")
    
    # Connect to PyBullet with GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240.0)
    
    # Setup camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0, 0.2]
    )
    
    # Load ground plane
    plane = p.loadURDF("plane.urdf")
    
    # Load Franka Panda robot
    print("Loading Franka Panda robot...")
    panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # Reset to home position
    print("Moving to home position...")
    home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i in range(7):
        p.resetJointState(panda, i, home_joints[i])
    
    # Open gripper
    for i in [9, 10]:
        p.setJointMotorControl2(panda, i, p.POSITION_CONTROL, 
                               targetPosition=0.04, force=20)
    
    # Create a red cube to pick up
    print("Adding cube...")
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025]*3, 
                                rgbaColor=[1, 0, 0, 1])
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025]*3)
    cube = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision,
                            baseVisualShapeIndex=visual, 
                            basePosition=[0.5, 0.0, 0.05])
    
    # Let everything settle
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1/240.0)
    
    print("\nStarting pick and place...")
    
    # Helper function to move robot
    def move_to(pos, orn=None):
        if orn is None:
            orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        joint_poses = p.calculateInverseKinematics(panda, 8, pos, orn,
                                                  maxNumIterations=100)
        for i in range(7):
            p.setJointMotorControl2(panda, i, p.POSITION_CONTROL,
                                   targetPosition=joint_poses[i], force=500)
        
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    # Helper function to control gripper
    def set_gripper(open_amount):
        position = open_amount * 0.04
        for i in [9, 10]:
            p.setJointMotorControl2(panda, i, p.POSITION_CONTROL,
                                   targetPosition=position, force=20)
        time.sleep(0.5)
    
    # Step 1: Move above the cube
    print("1. Moving above cube...")
    move_to([0.5, 0.0, 0.20])
    
    # Step 2: Move down to cube
    print("2. Moving down to cube...")
    move_to([0.5, 0.0, 0.05])
    
    # Step 3: Close gripper
    print("3. Grasping cube...")
    set_gripper(0.0)
    
    # Step 4: Lift cube
    print("4. Lifting cube...")
    move_to([0.5, 0.0, 0.30])
    
    # Step 5: Move to new position
    print("5. Moving to target position...")
    move_to([0.3, 0.2, 0.30])
    
    # Step 6: Lower cube
    print("6. Lowering cube...")
    move_to([0.3, 0.2, 0.05])
    
    # Step 7: Open gripper
    print("7. Releasing cube...")
    set_gripper(1.0)
    
    # Step 8: Move up
    print("8. Moving up...")
    move_to([0.3, 0.2, 0.20])
    
    print("\n✓ Pick and place complete!")
    print("\nClose the window or press Ctrl+C to exit...")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240.0)
    except KeyboardInterrupt:
        pass
    
    p.disconnect()
    print("Goodbye!")


if __name__ == "__main__":
    main()
