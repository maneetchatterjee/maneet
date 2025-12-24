"""
Control Module

Implements custom inverse kinematics (IK) and forward kinematics (FK)
for trajectory execution in simulation.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation

from ..planning.planner import Waypoint


class KinematicsController:
    """
    Custom IK/FK controller for robotic manipulator.
    
    Implements analytical IK for simple geometries and numerical IK
    for more complex configurations.
    """
    
    def __init__(
        self,
        link_lengths: Optional[List[float]] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize kinematics controller.
        
        Args:
            link_lengths: Lengths of robot links (default: 6-DOF arm)
            joint_limits: Joint angle limits in radians
        """
        # Default 6-DOF manipulator parameters
        self.link_lengths = link_lengths or [0.1, 0.2, 0.2, 0.15, 0.1, 0.05]
        self.num_joints = len(self.link_lengths)
        
        # Default joint limits (-π to π)
        self.joint_limits = joint_limits or [
            (-np.pi, np.pi) for _ in range(self.num_joints)
        ]
    
    def inverse_kinematics(
        self,
        target_position: Tuple[float, float, float],
        target_orientation: Tuple[float, float, float, float],
        current_joints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute joint angles to reach target end-effector pose.
        
        Args:
            target_position: Desired position (x, y, z)
            target_orientation: Desired orientation (quaternion)
            current_joints: Current joint configuration for initialization
            
        Returns:
            Joint angles as numpy array
        """
        # Use numerical IK (Jacobian-based)
        if current_joints is None:
            current_joints = np.zeros(self.num_joints)
        
        target_pose = np.array(target_position)
        max_iterations = 100
        tolerance = 0.01
        
        joint_angles = current_joints.copy()
        
        for iteration in range(max_iterations):
            # Compute current end-effector position
            current_pos, _ = self.forward_kinematics(joint_angles)
            
            # Position error
            error = target_pose - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                break
            
            # Compute Jacobian
            jacobian = self._compute_jacobian(joint_angles)
            
            # Pseudo-inverse for joint velocity
            try:
                jacobian_pinv = np.linalg.pinv(jacobian)
                delta_joints = jacobian_pinv @ error
            except np.linalg.LinAlgError:
                # Fallback if Jacobian is singular
                delta_joints = np.zeros(self.num_joints)
            
            # Update joint angles with damping
            alpha = 0.1  # Step size
            joint_angles += alpha * delta_joints
            
            # Enforce joint limits
            joint_angles = self._enforce_joint_limits(joint_angles)
        
        return joint_angles
    
    def forward_kinematics(
        self,
        joint_angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector pose from joint angles.
        
        Args:
            joint_angles: Joint configuration
            
        Returns:
            Tuple of (position, orientation_matrix)
        """
        # Simple forward kinematics using DH parameters
        # For a planar-like arm in 3D space
        
        x = 0.0
        y = 0.0
        z = 0.0
        
        # Cumulative rotation
        cumulative_angle = 0.0
        
        for i in range(min(3, len(joint_angles))):  # Consider first 3 joints for position
            cumulative_angle += joint_angles[i]
            x += self.link_lengths[i] * np.cos(cumulative_angle)
            z += self.link_lengths[i] * np.sin(cumulative_angle)
        
        position = np.array([x, y, z])
        
        # Simplified orientation (rotation about y-axis)
        orientation = Rotation.from_euler('y', cumulative_angle).as_matrix()
        
        return position, orientation
    
    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian matrix.
        
        Args:
            joint_angles: Current joint configuration
            
        Returns:
            3xN Jacobian matrix (position only)
        """
        epsilon = 1e-5
        jacobian = np.zeros((3, self.num_joints))
        
        # Current position
        pos, _ = self.forward_kinematics(joint_angles)
        
        # Numerical differentiation
        for i in range(self.num_joints):
            joint_angles_plus = joint_angles.copy()
            joint_angles_plus[i] += epsilon
            
            pos_plus, _ = self.forward_kinematics(joint_angles_plus)
            
            jacobian[:, i] = (pos_plus - pos) / epsilon
        
        return jacobian
    
    def _enforce_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Enforce joint limits on configuration."""
        limited_angles = joint_angles.copy()
        
        for i in range(len(joint_angles)):
            lower, upper = self.joint_limits[i]
            limited_angles[i] = np.clip(joint_angles[i], lower, upper)
        
        return limited_angles
    
    def interpolate_trajectory(
        self,
        waypoints: List[Waypoint],
        current_joints: np.ndarray,
        num_steps: int = 50
    ) -> List[np.ndarray]:
        """
        Generate smooth trajectory through waypoints.
        
        Args:
            waypoints: List of target waypoints
            current_joints: Current joint configuration
            num_steps: Number of interpolation steps between waypoints
            
        Returns:
            List of joint configurations forming trajectory
        """
        trajectory = [current_joints.copy()]
        
        for waypoint in waypoints:
            # Compute IK for waypoint
            target_joints = self.inverse_kinematics(
                waypoint.position,
                waypoint.orientation,
                trajectory[-1]
            )
            
            # Linear interpolation between current and target
            for t in np.linspace(0, 1, num_steps):
                interpolated_joints = (1 - t) * trajectory[-1] + t * target_joints
                trajectory.append(interpolated_joints)
            
            # Add final target configuration
            trajectory.append(target_joints)
        
        return trajectory


class TrajectoryExecutor:
    """
    Executes planned trajectories using kinematics controller.
    Interfaces with simulation environment.
    """
    
    def __init__(self, controller: KinematicsController):
        """
        Initialize trajectory executor.
        
        Args:
            controller: Kinematics controller instance
        """
        self.controller = controller
        self.current_joint_state = np.zeros(controller.num_joints)
    
    def execute_waypoints(
        self,
        waypoints: List[Waypoint],
        simulation_interface
    ) -> bool:
        """
        Execute sequence of waypoints in simulation.
        
        Args:
            waypoints: List of waypoints to execute
            simulation_interface: Interface to simulation environment
            
        Returns:
            True if execution successful
        """
        for waypoint in waypoints:
            # Compute target joint configuration
            target_joints = self.controller.inverse_kinematics(
                waypoint.position,
                waypoint.orientation,
                self.current_joint_state
            )
            
            # Generate smooth trajectory
            trajectory = self.controller.interpolate_trajectory(
                [waypoint],
                self.current_joint_state,
                num_steps=30
            )
            
            # Execute trajectory in simulation
            for joint_config in trajectory:
                success = simulation_interface.set_joint_positions(joint_config)
                if not success:
                    return False
                simulation_interface.step()
            
            # Update gripper state
            simulation_interface.set_gripper_state(waypoint.gripper_state)
            simulation_interface.step()
            
            # Update current state
            self.current_joint_state = target_joints
        
        return True
    
    def get_current_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose from joint state."""
        return self.controller.forward_kinematics(self.current_joint_state)
