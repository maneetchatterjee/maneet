# Custom Gymnasium environment for bipedal walking with PyBullet
"""Bipedal robot environment with domain randomization and safety constraints."""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import time


class BipedEnv(gym.Env):
    """Custom bipedal robot environment using PyBullet."""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        render_mode: str = 'rgb_array',
        use_gui: bool = False,
        control_frequency: int = 50,
        max_episode_steps: int = 1000,
        forward_reward_weight: float = 1.0,
        survive_reward: float = 0.1,
        energy_penalty_weight: float = 0.01,
        joint_limit_penalty_weight: float = 0.1,
        target_velocity: float = 1.0,
        domain_randomization: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.control_frequency = control_frequency
        self.max_episode_steps = max_episode_steps
        self.forward_reward_weight = forward_reward_weight
        self.survive_reward = survive_reward
        self.energy_penalty_weight = energy_penalty_weight
        self.joint_limit_penalty_weight = joint_limit_penalty_weight
        self.target_velocity = target_velocity
        
        # Domain randomization config
        self.domain_randomization = domain_randomization or {}
        
        # Connect to PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Robot configuration
        self.robot_id = None
        self.num_joints = 0
        self.joint_indices = []
        self.joint_limits = []
        self.torque_limits = []
        
        # State tracking
        self.step_count = 0
        self.last_pos = None
        self.footstep_count = 0
        self.energy_consumption = 0.0
        
        # Load robot once to determine correct dimensions for spaces
        # This ensures observation_space and action_space are correct from start
        self._initial_setup()
        
        # Define action and observation spaces based on actual robot
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
    
    def _initial_setup(self):
        """Initial setup to determine correct dimensions."""
        # Temporarily load robot to get dimensions
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.loadURDF("plane.urdf")
        
        start_pos = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        temp_robot = p.loadURDF(
            "humanoid/humanoid.urdf",
            start_pos,
            start_orientation,
            useFixedBase=False,
        )
        
        # Get joint information to determine action_dim
        num_joints = p.getNumJoints(temp_robot)
        joint_indices = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(temp_robot, i)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                joint_indices.append(i)
        
        self.action_dim = len(joint_indices)
        self.obs_dim = 37  # Fixed based on observation structure
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load humanoid robot (using pybullet data)
        # We'll use the built-in humanoid for now
        start_pos = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robot_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            start_pos,
            start_orientation,
            useFixedBase=False,
        )
        
        # Get joint information
        self._setup_joints()
        
        # Apply domain randomization
        self._apply_domain_randomization()
        
        # Reset state tracking
        self.step_count = 0
        self.last_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        self.footstep_count = 0
        self.energy_consumption = 0.0
        
        # Let robot settle
        for _ in range(10):
            p.stepSimulation()
        
        obs = self._get_observation()
        return obs
    
    def _setup_joints(self):
        """Setup joint indices and limits."""
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_limits = []
        self.torque_limits = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            
            # Only consider revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                max_force = joint_info[10]
                
                self.joint_limits.append((lower_limit, upper_limit))
                self.torque_limits.append(max_force if max_force > 0 else 100.0)
        
        # Verify action dimension matches (should be consistent)
        assert len(self.joint_indices) == self.action_dim, \
            f"Joint count mismatch: {len(self.joint_indices)} != {self.action_dim}"
    
    def _apply_domain_randomization(self):
        """Apply domain randomization to physics parameters."""
        if not self.domain_randomization:
            return
        
        # Randomize mass
        if self.domain_randomization.get('randomize_mass', False):
            mass_scale = np.random.uniform(
                self.domain_randomization.get('mass_range', [0.8, 1.2])[0],
                self.domain_randomization.get('mass_range', [0.8, 1.2])[1],
            )
            for i in range(-1, self.num_joints):
                p.changeDynamics(self.robot_id, i, mass=mass_scale)
        
        # Randomize friction
        if self.domain_randomization.get('randomize_friction', False):
            friction = np.random.uniform(
                self.domain_randomization.get('friction_range', [0.5, 1.5])[0],
                self.domain_randomization.get('friction_range', [0.5, 1.5])[1],
            )
            for i in range(-1, self.num_joints):
                p.changeDynamics(self.robot_id, i, lateralFriction=friction)
        
        # Randomize joint damping
        if self.domain_randomization.get('randomize_damping', False):
            damping = np.random.uniform(
                self.domain_randomization.get('damping_range', [0.1, 1.0])[0],
                self.domain_randomization.get('damping_range', [0.1, 1.0])[1],
            )
            for i in self.joint_indices:
                p.changeDynamics(self.robot_id, i, jointDamping=damping)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to torque limits
        scaled_action = []
        for i, act in enumerate(action):
            torque = act * self.torque_limits[i] * 0.5  # Scale down for safety
            scaled_action.append(torque)
        
        # Apply actions to joints
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=scaled_action[i],
            )
        
        # Step simulation (multiple substeps for control frequency)
        for _ in range(240 // self.control_frequency):
            p.stepSimulation()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward, info = self._compute_reward(action)
        
        # Check termination
        done = self._check_termination()
        
        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            done = True
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Base position and orientation
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Convert orientation to euler
        base_euler = p.getEulerFromQuaternion(base_orn)
        
        # Joint states
        joint_states = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_states.extend([joint_state[0], joint_state[1]])  # position, velocity
        
        # Construct observation vector
        obs = np.concatenate([
            [base_pos[2]],  # Height
            base_euler,  # Roll, pitch, yaw
            base_vel,  # Linear velocity
            base_ang_vel,  # Angular velocity
            joint_states,  # Joint positions and velocities
        ])
        
        # Pad or truncate to fixed size
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)), mode='constant')
        else:
            obs = obs[:self.obs_dim]
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Compute reward and info dict."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Forward progress reward
        current_pos = np.array(base_pos)
        forward_vel = base_vel[0]  # x-direction velocity
        forward_reward = self.forward_reward_weight * forward_vel
        
        # Target velocity tracking
        velocity_error = abs(forward_vel - self.target_velocity)
        velocity_reward = -0.5 * velocity_error
        
        # Survival reward (being upright)
        height = base_pos[2]
        euler = p.getEulerFromQuaternion(base_orn)
        upright_reward = self.survive_reward if height > 0.5 and abs(euler[0]) < 1.0 else 0.0
        
        # Energy penalty
        energy = np.sum(np.abs(action))
        energy_penalty = -self.energy_penalty_weight * energy
        self.energy_consumption += energy
        
        # Joint limit penalty
        joint_penalty = 0.0
        for i, joint_idx in enumerate(self.joint_indices):
            joint_state = p.getJointState(self.robot_id, joint_idx)
            pos = joint_state[0]
            lower, upper = self.joint_limits[i]
            if pos < lower or pos > upper:
                joint_penalty -= self.joint_limit_penalty_weight
        
        # Total reward
        reward = forward_reward + velocity_reward + upright_reward + energy_penalty + joint_penalty
        
        # Info dict
        info = {
            'forward_reward': float(forward_reward),
            'velocity_reward': float(velocity_reward),
            'upright_reward': float(upright_reward),
            'energy_penalty': float(energy_penalty),
            'joint_penalty': float(joint_penalty),
            'forward_velocity': float(forward_vel),
            'height': float(height),
            'energy_consumption': float(self.energy_consumption),
        }
        
        self.last_pos = current_pos
        
        # Ensure reward is Python float, not numpy scalar
        return float(reward), info
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Terminate if fallen
        if base_pos[2] < 0.3:
            return True
        
        # Terminate if flipped over
        euler = p.getEulerFromQuaternion(base_orn)
        if abs(euler[0]) > 1.57 or abs(euler[1]) > 1.57:  # ~90 degrees
            return True
        
        return False
    
    def render(self, mode: str = 'rgb_array'):
        """Render environment."""
        if mode == 'rgb_array':
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 1],
                distance=3.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (240, 320, 4))
            rgb_array = rgb_array[:, :, :3]
            
            return rgb_array
        
        return None
    
    def close(self):
        """Clean up environment."""
        try:
            if self.physics_client >= 0:
                p.disconnect(self.physics_client)
        except:
            pass  # Already disconnected or invalid
