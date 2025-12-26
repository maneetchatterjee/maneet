#!/usr/bin/env python3
"""
Record Franka Panda Pick and Place Demo

Creates a video/GIF of the Franka Panda simulation performing pick and place tasks.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import imageio


class PandaRecorder:
    """Records Franka Panda simulation to video/GIF."""
    
    def __init__(self, width=640, height=480):
        """Initialize recorder."""
        self.width = width
        self.height = height
        self.frames = []
        
        # Connect to PyBullet in DIRECT mode (headless)
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240.0)
        
        # Load plane and Panda
        self.plane = p.loadURDF("plane.urdf")
        self.panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        
        # Home position
        self.home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Camera parameters
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, -0.5, 0.8],
            cameraTargetPosition=[0.4, 0, 0.1],
            cameraUpVector=[0, 0, 1]
        )
        
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=5.0
        )
    
    def capture_frame(self):
        """Capture current frame."""
        # Get camera image
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy array
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)
        # Remove alpha channel
        rgb_array = rgb_array[:, :, :3]
        
        self.frames.append(rgb_array)
    
    def reset_to_home(self):
        """Reset robot to home position."""
        for i in range(7):
            p.resetJointState(self.panda, i, self.home_joints[i])
        
        # Open gripper
        for i in [9, 10]:
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, 
                                   targetPosition=0.04, force=20)
    
    def add_object(self, color, position, size=0.05):
        """Add object to scene."""
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, 
                                    rgbaColor=color + [1])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3)
        obj = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision,
                               baseVisualShapeIndex=visual, basePosition=position)
        return obj
    
    def move_to(self, pos, steps=120):
        """Move robot to position."""
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_poses = p.calculateInverseKinematics(self.panda, 8, pos, orn,
                                                  maxNumIterations=100)
        
        for _ in range(steps):
            for i in range(7):
                p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL,
                                       targetPosition=joint_poses[i], force=500)
            p.stepSimulation()
            
            # Capture every 4th frame (60 fps from 240 fps simulation)
            if _ % 4 == 0:
                self.capture_frame()
    
    def set_gripper(self, state, steps=30):
        """Set gripper state."""
        position = state * 0.04
        
        for _ in range(steps):
            for i in [9, 10]:
                p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL,
                                       targetPosition=position, force=20)
            p.stepSimulation()
            
            # Capture every 4th frame
            if _ % 4 == 0:
                self.capture_frame()
    
    def pick_and_place(self, pick_pos, place_pos):
        """Perform pick and place."""
        # Move above pick
        above_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.15]
        self.move_to(above_pick, steps=80)
        
        # Open gripper
        self.set_gripper(1.0, steps=20)
        
        # Move down
        self.move_to(pick_pos, steps=60)
        
        # Close gripper
        self.set_gripper(0.0, steps=20)
        
        # Lift
        lift_pos = [pick_pos[0], pick_pos[1], 0.25]
        self.move_to(lift_pos, steps=60)
        
        # Move to place
        above_place = [place_pos[0], place_pos[1], 0.25]
        self.move_to(above_place, steps=80)
        
        # Lower
        self.move_to(place_pos, steps=60)
        
        # Open gripper
        self.set_gripper(1.0, steps=20)
        
        # Move up
        final_up = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
        self.move_to(final_up, steps=60)
    
    def run_demo(self):
        """Run the demo and record it."""
        print("Starting recording...")
        
        # Reset to home
        print("  Resetting to home...")
        self.reset_to_home()
        
        # Let objects settle
        for _ in range(60):
            p.stepSimulation()
            if _ % 4 == 0:
                self.capture_frame()
        
        # Add objects
        print("  Adding objects...")
        red_cube = self.add_object([1, 0, 0], [0.5, 0.2, 0.05])
        blue_cube = self.add_object([0, 0, 1], [0.5, -0.2, 0.05])
        
        # Let objects settle
        for _ in range(120):
            p.stepSimulation()
            if _ % 4 == 0:
                self.capture_frame()
        
        # Task 1: Pick red cube
        print("  Task 1: Pick red cube...")
        self.pick_and_place([0.5, 0.2, 0.05], [0.3, 0.2, 0.05])
        
        # Pause
        for _ in range(60):
            p.stepSimulation()
            if _ % 4 == 0:
                self.capture_frame()
        
        # Task 2: Pick blue cube
        print("  Task 2: Pick blue cube...")
        self.pick_and_place([0.5, -0.2, 0.05], [0.3, -0.2, 0.05])
        
        # Pause
        for _ in range(60):
            p.stepSimulation()
            if _ % 4 == 0:
                self.capture_frame()
        
        # Task 3: Stack red on blue
        print("  Task 3: Stack red cube on blue...")
        self.pick_and_place([0.3, 0.2, 0.05], [0.3, -0.2, 0.10])
        
        # Final pause
        for _ in range(120):
            p.stepSimulation()
            if _ % 4 == 0:
                self.capture_frame()
        
        print(f"Recording complete! Captured {len(self.frames)} frames")
    
    def save_video(self, filename):
        """Save as MP4 video."""
        print(f"Saving video to {filename}...")
        imageio.mimsave(filename, self.frames, fps=30, codec='libx264')
        print(f"✓ Video saved: {filename}")
    
    def save_gif(self, filename):
        """Save as GIF."""
        print(f"Saving GIF to {filename}...")
        # Downsample frames for smaller GIF
        gif_frames = self.frames[::2]  # Every other frame
        imageio.mimsave(filename, gif_frames, fps=15, loop=0)
        print(f"✓ GIF saved: {filename}")
    
    def close(self):
        """Close simulation."""
        p.disconnect()


def main():
    """Main function."""
    print("="*60)
    print("Franka Panda Demo Recorder")
    print("="*60)
    
    # Create recorder
    recorder = PandaRecorder(width=800, height=600)
    
    # Run demo and record
    recorder.run_demo()
    
    # Save outputs
    recorder.save_video('demo/panda_demo.mp4')
    recorder.save_gif('demo/panda_demo.gif')
    
    # Cleanup
    recorder.close()
    
    print("\n" + "="*60)
    print("Recording complete!")
    print("Files created:")
    print("  - demo/panda_demo.mp4 (video)")
    print("  - demo/panda_demo.gif (animated GIF)")
    print("="*60)


if __name__ == "__main__":
    main()
