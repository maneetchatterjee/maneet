#!/usr/bin/env python3
"""
Demo: Franka Panda Pick and Place

Demonstrates PyBullet simulation of a Franka Panda robotic arm
performing pick and place tasks.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


class FrankaPandaSimulation:
    """Franka Panda robot simulation for pick and place tasks."""
    
    def __init__(self, use_gui=True):
        """
        Initialize the Franka Panda simulation.
        
        Args:
            use_gui: Whether to use GUI or headless mode
        """
        # Connect to PyBullet
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Setup environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240.0)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load Franka Panda robot
        self.panda_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.panda_id)
        self.arm_joint_indices = list(range(7))  # First 7 joints are arm joints
        self.gripper_joint_indices = [9, 10]  # Finger joints
        
        # Setup camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.4, 0, 0.2]
        )
        
        # Home position for the arm
        self.home_position = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Objects dictionary
        self.objects = {}
        
    def reset_to_home(self):
        """Reset robot to home position."""
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.panda_id, joint_idx, self.home_position[i])
        
        # Open gripper
        self.set_gripper(1.0)
        
    def set_gripper(self, state):
        """
        Set gripper state.
        
        Args:
            state: 0.0 = closed, 1.0 = open
        """
        # Panda gripper: 0.0 (closed) to 0.04 (open)
        target_position = state * 0.04
        
        for joint_idx in self.gripper_joint_indices:
            p.setJointMotorControl2(
                self.panda_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=20
            )
    
    def move_to_position(self, target_position, target_orientation=None):
        """
        Move end-effector to target position using inverse kinematics.
        
        Args:
            target_position: [x, y, z] position
            target_orientation: [x, y, z, w] quaternion (optional)
        """
        if target_orientation is None:
            # Default downward orientation
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Get end-effector link index (link 8 is the hand)
        end_effector_index = 8
        
        # Compute inverse kinematics
        joint_positions = p.calculateInverseKinematics(
            self.panda_id,
            end_effector_index,
            target_position,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Set joint positions for arm joints only
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                self.panda_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )
        
        # Wait for robot to reach position
        for _ in range(240):  # 1 second at 240Hz
            p.stepSimulation()
            time.sleep(1/240.0)
    
    def add_object(self, shape, color, position, size=0.05):
        """
        Add object to simulation.
        
        Args:
            shape: 'cube' or 'sphere'
            color: RGB color as [r, g, b, a]
            position: [x, y, z] position
            size: Object size
            
        Returns:
            Object ID
        """
        # Create shape
        if shape == 'cube':
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2],
                rgbaColor=color
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2]
            )
        else:  # sphere
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size/2,
                rgbaColor=color
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=size/2
            )
        
        # Create object
        object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        return object_id
    
    def pick_and_place(self, pick_pos, place_pos, lift_height=0.3):
        """
        Perform pick and place task.
        
        Args:
            pick_pos: [x, y, z] position to pick from
            place_pos: [x, y, z] position to place at
            lift_height: Height to lift object to during transport
        """
        print(f"\n--- Pick and Place Task ---")
        print(f"Pick position: {pick_pos}")
        print(f"Place position: {place_pos}")
        
        # 1. Move above pick position
        print("Step 1: Moving above pick position...")
        above_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.15]
        self.move_to_position(above_pick)
        
        # 2. Open gripper
        print("Step 2: Opening gripper...")
        self.set_gripper(1.0)
        time.sleep(0.5)
        
        # 3. Move down to pick position
        print("Step 3: Moving down to pick position...")
        self.move_to_position(pick_pos)
        
        # 4. Close gripper
        print("Step 4: Closing gripper...")
        self.set_gripper(0.0)
        time.sleep(0.5)
        
        # 5. Lift object
        print("Step 5: Lifting object...")
        lift_pos = [pick_pos[0], pick_pos[1], lift_height]
        self.move_to_position(lift_pos)
        
        # 6. Move to above place position
        print("Step 6: Moving to place position...")
        above_place = [place_pos[0], place_pos[1], lift_height]
        self.move_to_position(above_place)
        
        # 7. Move down to place position
        print("Step 7: Lowering to place position...")
        self.move_to_position(place_pos)
        
        # 8. Open gripper
        print("Step 8: Opening gripper...")
        self.set_gripper(1.0)
        time.sleep(0.5)
        
        # 9. Move up
        print("Step 9: Moving up...")
        above_place = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
        self.move_to_position(above_place)
        
        print("Pick and place complete!")
    
    def run_demo(self):
        """Run demonstration with multiple pick and place tasks."""
        print("\n" + "="*60)
        print("Franka Panda Pick and Place Simulation Demo")
        print("="*60)
        
        # Reset to home position
        print("\nResetting to home position...")
        self.reset_to_home()
        time.sleep(1.0)
        
        # Add objects to the scene
        print("\nAdding objects to scene...")
        
        # Red cube
        red_cube = self.add_object(
            'cube',
            [1, 0, 0, 1],
            [0.5, 0.2, 0.05],
            size=0.05
        )
        self.objects['red_cube'] = red_cube
        
        # Blue cube
        blue_cube = self.add_object(
            'cube',
            [0, 0, 1, 1],
            [0.5, -0.2, 0.05],
            size=0.05
        )
        self.objects['blue_cube'] = blue_cube
        
        # Green sphere
        green_sphere = self.add_object(
            'sphere',
            [0, 1, 0, 1],
            [0.5, 0, 0.025],
            size=0.05
        )
        self.objects['green_sphere'] = green_sphere
        
        print(f"Added {len(self.objects)} objects")
        
        # Wait for objects to settle
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1/240.0)
        
        # Task 1: Pick red cube and place it on the left
        print("\n\n=== Task 1: Pick red cube and place left ===")
        self.pick_and_place(
            pick_pos=[0.5, 0.2, 0.05],
            place_pos=[0.3, 0.2, 0.05]
        )
        time.sleep(1.0)
        
        # Task 2: Pick blue cube and place it on the right
        print("\n\n=== Task 2: Pick blue cube and place right ===")
        self.pick_and_place(
            pick_pos=[0.5, -0.2, 0.05],
            place_pos=[0.3, -0.2, 0.05]
        )
        time.sleep(1.0)
        
        # Task 3: Pick green sphere and place it in center
        print("\n\n=== Task 3: Pick green sphere and place center ===")
        self.pick_and_place(
            pick_pos=[0.5, 0, 0.025],
            place_pos=[0.3, 0, 0.025]
        )
        time.sleep(1.0)
        
        # Task 4: Stack red cube on blue cube
        print("\n\n=== Task 4: Stack red cube on blue cube ===")
        self.pick_and_place(
            pick_pos=[0.3, 0.2, 0.05],
            place_pos=[0.3, -0.2, 0.10]
        )
        
        print("\n\nAll tasks complete!")
        print("\nPress Ctrl+C to exit or close the window...")
        
        # Keep simulation running
        try:
            while True:
                p.stepSimulation()
                time.sleep(1/240.0)
        except KeyboardInterrupt:
            print("\nShutting down simulation...")
    
    def close(self):
        """Close the simulation."""
        p.disconnect()


def main():
    """Main function to run the demo."""
    # Create and run simulation
    sim = FrankaPandaSimulation(use_gui=True)
    
    try:
        sim.run_demo()
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
