"""
Simulation Module

PyBullet-based simulation environment for robotic manipulation.
Provides interface for robot control and scene management.
"""

import pybullet as p
try:
    import pybullet_data
    HAS_PYBULLET_DATA = True
except ImportError:
    HAS_PYBULLET_DATA = False
import numpy as np
from typing import List, Tuple, Optional, Dict
import time


class SimulationEnvironment:
    """
    PyBullet simulation environment for robotic manipulation.
    
    Manages robot, objects, and physics simulation.
    """
    
    def __init__(
        self,
        use_gui: bool = True,
        time_step: float = 1/240.0
    ):
        """
        Initialize simulation environment.
        
        Args:
            use_gui: Whether to use GUI or headless mode
            time_step: Physics simulation time step
        """
        self.use_gui = use_gui
        self.time_step = time_step
        
        # Connect to PyBullet
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        if HAS_PYBULLET_DATA:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = None
        self.objects = {}
        self.object_counter = 0
        
        # Camera parameters
        self.camera_params = {
            'width': 640,
            'height': 480,
            'fov': 60,
            'aspect': 640 / 480,
            'near': 0.1,
            'far': 5.0,
            'fx': 500,
            'fy': 500,
            'cx': 320,
            'cy': 240
        }
        
        # Setup camera
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup default camera view."""
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
    
    def load_robot(self, urdf_path: Optional[str] = None) -> int:
        """
        Load robot into simulation.
        
        Args:
            urdf_path: Path to robot URDF file
            
        Returns:
            Robot body ID
        """
        if urdf_path is None:
            # Use default Panda arm or simple arm
            # For demonstration, create a simple kinematic chain
            self.robot_id = self._create_simple_robot()
        else:
            self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        
        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))
        
        return self.robot_id
    
    def _create_simple_robot(self) -> int:
        """Create a simple robot arm for demonstration."""
        # Create multi-body with simple arm
        base_position = [0, 0, 0]
        base_orientation = [0, 0, 0, 1]
        
        # For simplicity, load Kuka arm if available
        try:
            robot_id = p.loadURDF("kuka_iiwa/model.urdf", base_position, base_orientation, useFixedBase=True)
        except:
            # Create simple arm from primitives
            robot_id = self._create_primitive_arm()
        
        return robot_id
    
    def _create_primitive_arm(self) -> int:
        """Create simple arm from primitive shapes."""
        # Create a simple 3-link arm
        base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.1)
        base_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.1)
        
        link_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.2)
        link_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.2)
        
        # Create multi-body
        base_mass = 1.0
        link_masses = [0.5, 0.5, 0.3]
        
        robot_id = p.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0, 0, 0.05],
            linkMasses=link_masses,
            linkCollisionShapeIndices=[link_collision] * 3,
            linkVisualShapeIndices=[link_visual] * 3,
            linkPositions=[[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]],
            linkOrientations=[[0, 0, 0, 1]] * 3,
            linkInertialFramePositions=[[0, 0, 0]] * 3,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 3,
            linkParentIndices=[0, 1, 2],
            linkJointTypes=[p.JOINT_REVOLUTE] * 3,
            linkJointAxis=[[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        )
        
        return robot_id
    
    def add_object(
        self,
        shape: str,
        color: str,
        position: Tuple[float, float, float],
        size: float = 0.05
    ) -> int:
        """
        Add object to simulation.
        
        Args:
            shape: Object shape (cube, sphere, cylinder)
            color: Object color name
            position: Initial position (x, y, z)
            size: Object size
            
        Returns:
            Object body ID
        """
        # Map color names to RGB
        color_map = {
            'red': [1, 0, 0, 1],
            'blue': [0, 0, 1, 1],
            'green': [0, 1, 0, 1],
            'yellow': [1, 1, 0, 1],
            'orange': [1, 0.5, 0, 1],
            'purple': [0.5, 0, 0.5, 1],
        }
        rgba = color_map.get(color, [0.5, 0.5, 0.5, 1])
        
        # Create shape
        if shape in ['cube', 'box', 'block']:
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2],
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2]
            )
        elif shape in ['sphere', 'ball']:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size/2,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=size/2
            )
        elif shape in ['cylinder', 'can']:
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size/2,
                length=size,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=size/2,
                height=size
            )
        elif shape in ['cup', 'mug']:
            # Cup: wider cylinder (taller than wide)
            cup_radius = size / 2
            cup_height = size * 1.5
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=cup_radius,
                length=cup_height,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=cup_radius,
                height=cup_height
            )
        elif shape in ['bottle']:
            # Bottle: narrow cylinder (much taller than wide)
            bottle_radius = size / 3
            bottle_height = size * 2.5
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=bottle_radius,
                length=bottle_height,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=bottle_radius,
                height=bottle_height
            )
        else:
            # Default to cube
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2],
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2]
            )
        
        # Create multi-body
        object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        # Store object info
        self.objects[object_id] = {
            'name': f"{color}_{shape}",
            'color': color,
            'shape': shape,
            'size': size
        }
        self.object_counter += 1
        
        return object_id
    
    def add_tray(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float] = (0.2, 0.2),
        height: float = 0.005,
        color: str = 'gray',
        label: Optional[str] = None
    ) -> int:
        """
        Add a tray/placement zone to simulation.
        
        Trays are thin boxes that sit flush on the table surface to mark placement zones.
        
        Args:
            position: Center position (x, y, z) - z should be very small (e.g., 0.001) to sit on table
            size: Tray dimensions (width, depth) in meters
            height: Tray thickness (default 0.005m = 5mm)
            color: Tray color name
            label: Optional label for the tray
            
        Returns:
            Tray body ID
        """
        # Map color names to RGB with semi-transparency for trays
        color_map = {
            'red': [1, 0, 0, 0.5],
            'blue': [0, 0, 1, 0.5],
            'green': [0, 1, 0, 0.5],
            'yellow': [1, 1, 0, 0.5],
            'orange': [1, 0.5, 0, 0.5],
            'purple': [0.5, 0, 0.5, 0.5],
            'gray': [0.7, 0.7, 0.7, 0.5],
            'white': [0.9, 0.9, 0.9, 0.5],
        }
        rgba = color_map.get(color, [0.7, 0.7, 0.7, 0.5])
        
        # Create thin box for tray - sits flush on table
        width, depth = size
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[width/2, depth/2, height/2],
            rgbaColor=rgba
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[width/2, depth/2, height/2]
        )
        
        # Adjust z-position to sit flush on table surface
        # Use small offset (height/2) to avoid z-fighting with table plane
        adjusted_position = (position[0], position[1], height/2)
        
        # Create multi-body with very low mass (essentially static)
        tray_id = p.createMultiBody(
            baseMass=0.001,  # Very light so objects don't disturb it
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=adjusted_position
        )
        
        # Store tray info
        tray_name = label if label else f"{color}_tray"
        self.objects[tray_id] = {
            'name': tray_name,
            'color': color,
            'shape': 'tray',
            'size': size,
            'is_tray': True
        }
        self.object_counter += 1
        
        return tray_id
    
    def set_joint_positions(self, joint_positions: np.ndarray) -> bool:
        """
        Set robot joint positions.
        
        Args:
            joint_positions: Target joint angles
            
        Returns:
            True if successful
        """
        if self.robot_id is None:
            return False
        
        num_joints = min(len(joint_positions), self.num_joints)
        
        for i in range(num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )
        
        return True
    
    def set_gripper_state(self, state: float):
        """
        Set gripper state.
        
        Args:
            state: 0.0 = closed, 1.0 = open
        """
        # Gripper control would go here
        # For simple simulation, we can skip or use last joints
        pass
    
    def step(self, num_steps: int = 1):
        """
        Step simulation forward.
        
        Args:
            num_steps: Number of simulation steps
        """
        for _ in range(num_steps):
            p.stepSimulation()
            if self.use_gui:
                time.sleep(self.time_step)
    
    def get_camera_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get RGB and depth images from camera.
        
        Returns:
            Tuple of (rgb_image, depth_image)
        """
        # Camera position and orientation
        camera_pos = [0.5, 0, 0.5]
        target_pos = [0, 0, 0]
        up_vector = [0, 0, 1]
        
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_params['fov'],
            aspect=self.camera_params['aspect'],
            nearVal=self.camera_params['near'],
            farVal=self.camera_params['far']
        )
        
        # Capture image
        width = self.camera_params['width']
        height = self.camera_params['height']
        
        img_data = p.getCameraImage(
            width,
            height,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER
        )
        
        # Extract RGB
        rgb_array = np.array(img_data[2], dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))[:, :, :3]
        
        # Extract depth
        depth_array = np.array(img_data[3], dtype=np.float32)
        depth_array = depth_array.reshape((height, width))
        
        # Convert depth buffer to actual depth values
        far = self.camera_params['far']
        near = self.camera_params['near']
        depth_array = far * near / (far - (far - near) * depth_array)
        
        return rgb_array, depth_array
    
    def get_object_pose(self, object_id: int) -> Tuple[Tuple, Tuple]:
        """
        Get object pose.
        
        Args:
            object_id: Object body ID
            
        Returns:
            Tuple of (position, orientation)
        """
        pos, orn = p.getBasePositionAndOrientation(object_id)
        return pos, orn
    
    def reset(self):
        """Reset simulation to initial state."""
        # Remove all objects
        for obj_id in list(self.objects.keys()):
            p.removeBody(obj_id)
        self.objects.clear()
        self.object_counter = 0
        
        # Reset robot if exists
        if self.robot_id is not None:
            for i in range(self.num_joints):
                p.resetJointState(self.robot_id, i, 0)
    
    def close(self):
        """Close simulation."""
        p.disconnect(self.client)
