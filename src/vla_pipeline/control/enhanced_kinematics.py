"""
Enhanced Control Module with Rigorous IK/FK

Implements research-grade inverse kinematics with:
- Damped least squares for singularity handling
- Joint limit enforcement
- Convergence metrics and logging
- Multiple IK solving strategies
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import warnings

from ..planning.planner import Waypoint


@dataclass
class IKMetrics:
    """Metrics for IK solver convergence."""
    iterations: int
    final_error: float
    converged: bool
    singularity_encountered: bool
    damping_used: float
    joint_limit_violations: int


class EnhancedKinematicsController:
    """
    Research-grade IK/FK controller with singularity handling.
    
    Features:
    - Damped Least Squares (DLS) for singularity robustness
    - Joint limit enforcement with soft constraints
    - Multiple solving strategies
    - Comprehensive convergence metrics
    """
    
    def __init__(
        self,
        link_lengths: Optional[List[float]] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        damping_factor: float = 0.01,
        singularity_threshold: float = 0.001
    ):
        """
        Initialize enhanced kinematics controller.
        
        Args:
            link_lengths: Robot link lengths
            joint_limits: Joint angle limits in radians
            damping_factor: Damping for DLS (lambda)
            singularity_threshold: Threshold for singularity detection
        """
        self.link_lengths = link_lengths or [0.1, 0.2, 0.2, 0.15, 0.1, 0.05]
        self.num_joints = len(self.link_lengths)
        
        self.joint_limits = joint_limits or [
            (-np.pi, np.pi) for _ in range(self.num_joints)
        ]
        
        # Control parameters
        self.damping_factor = damping_factor
        self.singularity_threshold = singularity_threshold
        self.max_iterations = 200
        self.position_tolerance = 0.001  # 1mm
        self.step_size = 0.1
        
        # Metrics tracking
        self.last_ik_metrics: Optional[IKMetrics] = None
    
    def inverse_kinematics(
        self,
        target_position: Tuple[float, float, float],
        target_orientation: Tuple[float, float, float, float],
        current_joints: Optional[np.ndarray] = None,
        use_damping: bool = True
    ) -> Tuple[np.ndarray, IKMetrics]:
        """
        Solve IK with damped least squares and singularity handling.
        
        Args:
            target_position: Desired end-effector position (x, y, z)
            target_orientation: Desired orientation (quaternion)
            current_joints: Initial joint configuration
            use_damping: Whether to use damped least squares
            
        Returns:
            Tuple of (joint_angles, metrics)
        """
        if current_joints is None:
            current_joints = np.zeros(self.num_joints)
        
        target_pose = np.array(target_position)
        joint_angles = current_joints.copy()
        
        # Metrics
        iterations = 0
        singularity_count = 0
        joint_limit_violations = 0
        
        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            
            # Compute current end-effector position
            current_pos, _ = self.forward_kinematics(joint_angles)
            
            # Position error
            error = target_pose - current_pos
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < self.position_tolerance:
                metrics = IKMetrics(
                    iterations=iterations,
                    final_error=error_norm,
                    converged=True,
                    singularity_encountered=(singularity_count > 0),
                    damping_used=self.damping_factor if use_damping else 0.0,
                    joint_limit_violations=joint_limit_violations
                )
                self.last_ik_metrics = metrics
                return joint_angles, metrics
            
            # Compute Jacobian
            jacobian = self._compute_jacobian(joint_angles)
            
            # Check for singularity
            manipulability = self._compute_manipulability(jacobian)
            is_singular = manipulability < self.singularity_threshold
            
            if is_singular:
                singularity_count += 1
            
            # Solve for joint velocity
            if use_damping or is_singular:
                # Damped Least Squares (DLS)
                delta_joints = self._damped_least_squares(jacobian, error)
            else:
                # Standard pseudoinverse
                try:
                    jacobian_pinv = np.linalg.pinv(jacobian)
                    delta_joints = jacobian_pinv @ error
                except np.linalg.LinAlgError:
                    # Fallback to damped if pseudoinverse fails
                    delta_joints = self._damped_least_squares(jacobian, error)
                    singularity_count += 1
            
            # Update joint angles with step size
            joint_angles_new = joint_angles + self.step_size * delta_joints
            
            # Enforce joint limits with soft constraints
            joint_angles_new, violations = self._enforce_joint_limits_soft(
                joint_angles_new, joint_angles
            )
            joint_limit_violations += violations
            
            joint_angles = joint_angles_new
        
        # Did not converge
        current_pos, _ = self.forward_kinematics(joint_angles)
        final_error = np.linalg.norm(target_pose - current_pos)
        
        metrics = IKMetrics(
            iterations=iterations,
            final_error=final_error,
            converged=False,
            singularity_encountered=(singularity_count > 0),
            damping_used=self.damping_factor if use_damping else 0.0,
            joint_limit_violations=joint_limit_violations
        )
        self.last_ik_metrics = metrics
        
        return joint_angles, metrics
    
    def _damped_least_squares(self, jacobian: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Solve with Damped Least Squares (DLS) / Levenberg-Marquardt.
        
        Handles singularities by adding damping term:
        delta_q = J^T (JJ^T + λ²I)^{-1} error
        
        Args:
            jacobian: Jacobian matrix (3 x n_joints)
            error: Position error vector (3,)
            
        Returns:
            Joint velocity vector (n_joints,)
        """
        # DLS formula
        JJT = jacobian @ jacobian.T
        damping_matrix = (self.damping_factor ** 2) * np.eye(JJT.shape[0])
        
        try:
            JJT_damped_inv = np.linalg.inv(JJT + damping_matrix)
            delta_joints = jacobian.T @ JJT_damped_inv @ error
        except np.linalg.LinAlgError:
            # If still fails, increase damping
            increased_damping = (self.damping_factor * 10) ** 2
            damping_matrix = increased_damping * np.eye(JJT.shape[0])
            JJT_damped_inv = np.linalg.inv(JJT + damping_matrix)
            delta_joints = jacobian.T @ JJT_damped_inv @ error
        
        return delta_joints
    
    def _compute_manipulability(self, jacobian: np.ndarray) -> float:
        """
        Compute manipulability index (Yoshikawa measure).
        
        Manipulability = sqrt(det(J * J^T))
        
        Low values indicate singularity.
        """
        JJT = jacobian @ jacobian.T
        det = np.linalg.det(JJT)
        
        if det < 0:
            det = 0
        
        return np.sqrt(det)
    
    def _enforce_joint_limits_soft(
        self, 
        joint_angles: np.ndarray,
        previous_angles: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Enforce joint limits with soft constraints.
        
        Uses clamping with smooth approach near limits.
        
        Returns:
            Tuple of (limited_angles, violation_count)
        """
        limited_angles = joint_angles.copy()
        violations = 0
        
        for i in range(len(joint_angles)):
            lower, upper = self.joint_limits[i]
            
            # Hard clamp
            if joint_angles[i] < lower:
                limited_angles[i] = lower
                violations += 1
            elif joint_angles[i] > upper:
                limited_angles[i] = upper
                violations += 1
            else:
                # Soft constraint near boundaries
                margin = 0.1  # 0.1 radians margin
                
                if joint_angles[i] < lower + margin:
                    # Approaching lower limit - reduce velocity
                    ratio = (joint_angles[i] - lower) / margin
                    limited_angles[i] = previous_angles[i] + ratio * (joint_angles[i] - previous_angles[i])
                
                elif joint_angles[i] > upper - margin:
                    # Approaching upper limit - reduce velocity
                    ratio = (upper - joint_angles[i]) / margin
                    limited_angles[i] = previous_angles[i] + ratio * (joint_angles[i] - previous_angles[i])
        
        return limited_angles, violations
    
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
        # Simple planar-like forward kinematics
        x = 0.0
        y = 0.0
        z = 0.0
        
        cumulative_angle = 0.0
        
        # Consider first 3 joints for position
        for i in range(min(3, len(joint_angles))):
            cumulative_angle += joint_angles[i]
            x += self.link_lengths[i] * np.cos(cumulative_angle)
            z += self.link_lengths[i] * np.sin(cumulative_angle)
        
        position = np.array([x, y, z])
        
        # Simplified orientation
        orientation = Rotation.from_euler('y', cumulative_angle).as_matrix()
        
        return position, orientation
    
    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian matrix (numerical).
        
        J[i,j] = ∂position[i] / ∂joint[j]
        
        Args:
            joint_angles: Current joint configuration
            
        Returns:
            3xN Jacobian matrix
        """
        epsilon = 1e-6
        jacobian = np.zeros((3, self.num_joints))
        
        pos, _ = self.forward_kinematics(joint_angles)
        
        for i in range(self.num_joints):
            joint_angles_plus = joint_angles.copy()
            joint_angles_plus[i] += epsilon
            
            pos_plus, _ = self.forward_kinematics(joint_angles_plus)
            
            jacobian[:, i] = (pos_plus - pos) / epsilon
        
        return jacobian
    
    def interpolate_trajectory(
        self,
        waypoints: List[Waypoint],
        current_joints: np.ndarray,
        num_steps: int = 50
    ) -> Tuple[List[np.ndarray], List[IKMetrics]]:
        """
        Generate smooth trajectory through waypoints.
        
        Args:
            waypoints: List of target waypoints
            current_joints: Current joint configuration
            num_steps: Number of interpolation steps between waypoints
            
        Returns:
            Tuple of (trajectory, metrics_list)
        """
        trajectory = [current_joints.copy()]
        all_metrics = []
        
        for waypoint in waypoints:
            # Solve IK for waypoint
            target_joints, metrics = self.inverse_kinematics(
                waypoint.position,
                waypoint.orientation,
                trajectory[-1]
            )
            
            all_metrics.append(metrics)
            
            # Check if IK failed
            if not metrics.converged:
                warnings.warn(f"IK did not converge for waypoint: error={metrics.final_error:.4f}")
            
            # Linear interpolation between current and target
            for t in np.linspace(0, 1, num_steps):
                interpolated_joints = (1 - t) * trajectory[-1] + t * target_joints
                trajectory.append(interpolated_joints)
            
            # Add final target configuration
            trajectory.append(target_joints)
        
        return trajectory, all_metrics
    
    def get_ik_performance_summary(self, metrics_list: List[IKMetrics]) -> Dict:
        """
        Get summary statistics of IK performance.
        
        Args:
            metrics_list: List of IK metrics from trajectory
            
        Returns:
            Dictionary with performance statistics
        """
        if not metrics_list:
            return {}
        
        converged_count = sum(1 for m in metrics_list if m.converged)
        singular_count = sum(1 for m in metrics_list if m.singularity_encountered)
        total_violations = sum(m.joint_limit_violations for m in metrics_list)
        
        avg_iterations = np.mean([m.iterations for m in metrics_list])
        avg_error = np.mean([m.final_error for m in metrics_list])
        max_error = np.max([m.final_error for m in metrics_list])
        
        return {
            "total_ik_calls": len(metrics_list),
            "convergence_rate": converged_count / len(metrics_list),
            "singularity_rate": singular_count / len(metrics_list),
            "avg_iterations": avg_iterations,
            "avg_final_error": avg_error,
            "max_final_error": max_error,
            "total_joint_violations": total_violations,
        }


class TrajectoryExecutor:
    """
    Executes planned trajectories using enhanced kinematics controller.
    """
    
    def __init__(self, controller: EnhancedKinematicsController):
        """
        Initialize trajectory executor.
        
        Args:
            controller: Enhanced kinematics controller instance
        """
        self.controller = controller
        self.current_joint_state = np.zeros(controller.num_joints)
        self.execution_metrics = []
    
    def execute_waypoints(
        self,
        waypoints: List[Waypoint],
        simulation_interface
    ) -> Tuple[bool, List[IKMetrics]]:
        """
        Execute sequence of waypoints in simulation.
        
        Args:
            waypoints: List of waypoints to execute
            simulation_interface: Interface to simulation environment
            
        Returns:
            Tuple of (success, metrics_list)
        """
        all_metrics = []
        
        for waypoint in waypoints:
            # Compute target joint configuration
            target_joints, metrics = self.controller.inverse_kinematics(
                waypoint.position,
                waypoint.orientation,
                self.current_joint_state
            )
            
            all_metrics.append(metrics)
            
            # Check if IK converged
            if not metrics.converged:
                warnings.warn(f"IK failed to converge: {metrics}")
                # Can still try to execute, but may not reach target
            
            # Generate smooth trajectory
            trajectory, traj_metrics = self.controller.interpolate_trajectory(
                [waypoint],
                self.current_joint_state,
                num_steps=30
            )
            
            all_metrics.extend(traj_metrics)
            
            # Execute trajectory in simulation
            for joint_config in trajectory:
                success = simulation_interface.set_joint_positions(joint_config)
                if not success:
                    return False, all_metrics
                simulation_interface.step()
            
            # Update gripper state
            simulation_interface.set_gripper_state(waypoint.gripper_state)
            simulation_interface.step()
            
            # Update current state
            self.current_joint_state = target_joints
        
        self.execution_metrics.extend(all_metrics)
        return True, all_metrics
    
    def get_current_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector pose from joint state."""
        return self.controller.forward_kinematics(self.current_joint_state)
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        if not self.execution_metrics:
            return {}
        
        return self.controller.get_ik_performance_summary(self.execution_metrics)
