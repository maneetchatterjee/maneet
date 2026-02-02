"""
Planning Module

Converts parsed language commands into ordered symbolic actions
and generates collision-safe pick-and-place waypoints.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..language.parser import ParsedCommand, ActionType, SpatialRelation
from ..perception.detector import Object3D


class MotionPhase(Enum):
    """Phases of a manipulation motion."""
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"


@dataclass
class Waypoint:
    """Represents a waypoint in 3D space."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    gripper_state: float  # 0.0 = closed, 1.0 = open
    phase: MotionPhase
    velocity_scale: float = 0.5  # Scale factor for motion speed


@dataclass
class SymbolicAction:
    """High-level symbolic action."""
    action_type: ActionType
    target_object_id: Optional[int]
    destination_position: Optional[Tuple[float, float, float]]
    spatial_relation: SpatialRelation
    waypoints: List[Waypoint]


class PlanningModule:
    """
    Symbolic action planner for robotic manipulation.
    
    Generates collision-safe waypoints for pick-and-place tasks.
    """
    
    def __init__(self, safety_margin: float = 0.05, lift_height: float = 0.15):
        """
        Initialize planning module.
        
        Args:
            safety_margin: Safety distance for collision avoidance (meters)
            lift_height: Height to lift objects during transport (meters)
        """
        self.safety_margin = safety_margin
        self.lift_height = lift_height
        self.workspace_bounds = {
            'x': (-0.5, 0.5),
            'y': (-0.5, 0.5),
            'z': (0.0, 0.5)
        }
    
    def plan_action(
        self, 
        parsed_command: ParsedCommand,
        scene_objects: List[Object3D],
        current_ee_pose: Optional[Tuple] = None
    ) -> SymbolicAction:
        """
        Plan symbolic action from parsed command and scene.
        
        Args:
            parsed_command: Parsed language command
            scene_objects: List of detected objects in scene
            current_ee_pose: Current end-effector pose (optional)
            
        Returns:
            SymbolicAction with waypoints
        """
        # Find target object
        target_obj = self._find_object(
            parsed_command.target_color,
            parsed_command.target_shape,
            scene_objects
        )
        
        if target_obj is None:
            return self._create_null_action(parsed_command)
        
        # Generate waypoints based on action type
        if parsed_command.action == ActionType.PICK:
            waypoints = self._generate_pick_waypoints(target_obj, current_ee_pose)
            return SymbolicAction(
                action_type=ActionType.PICK,
                target_object_id=target_obj.id,
                destination_position=None,
                spatial_relation=SpatialRelation.NONE,
                waypoints=waypoints
            )
        
        elif parsed_command.action == ActionType.PLACE:
            # Find destination
            dest_position = self._compute_destination_position(
                parsed_command,
                scene_objects,
                target_obj
            )
            
            waypoints = self._generate_place_waypoints(
                target_obj,
                dest_position,
                current_ee_pose
            )
            
            return SymbolicAction(
                action_type=ActionType.PLACE,
                target_object_id=target_obj.id,
                destination_position=dest_position,
                spatial_relation=parsed_command.spatial_relation,
                waypoints=waypoints
            )
        
        else:
            return self._create_null_action(parsed_command)
    
    def plan_pick_and_place(
        self,
        parsed_command: ParsedCommand,
        scene_objects: List[Object3D],
        current_ee_pose: Optional[Tuple] = None
    ) -> List[SymbolicAction]:
        """
        Plan complete pick-and-place sequence.
        
        Args:
            parsed_command: Parsed command with pick and place intent
            scene_objects: Scene objects
            current_ee_pose: Current end-effector pose
            
        Returns:
            List of symbolic actions (pick, place)
        """
        actions = []
        
        # Find target object
        target_obj = self._find_object(
            parsed_command.target_color,
            parsed_command.target_shape,
            scene_objects
        )
        
        if target_obj is None:
            return [self._create_null_action(parsed_command)]
        
        # Generate pick action
        pick_waypoints = self._generate_pick_waypoints(target_obj, current_ee_pose)
        actions.append(SymbolicAction(
            action_type=ActionType.PICK,
            target_object_id=target_obj.id,
            destination_position=None,
            spatial_relation=SpatialRelation.NONE,
            waypoints=pick_waypoints
        ))
        
        # Compute destination
        dest_position = self._compute_destination_position(
            parsed_command,
            scene_objects,
            target_obj
        )
        
        # Generate place action
        # Current pose after pick is the last waypoint of pick
        current_pose = (pick_waypoints[-1].position, pick_waypoints[-1].orientation)
        place_waypoints = self._generate_place_waypoints(
            target_obj,
            dest_position,
            current_pose
        )
        
        actions.append(SymbolicAction(
            action_type=ActionType.PLACE,
            target_object_id=target_obj.id,
            destination_position=dest_position,
            spatial_relation=parsed_command.spatial_relation,
            waypoints=place_waypoints
        ))
        
        return actions
    
    def _find_object(
        self,
        color: Optional[str],
        shape: Optional[str],
        scene_objects: List[Object3D]
    ) -> Optional[Object3D]:
        """Find object in scene by color and/or shape."""
        for obj in scene_objects:
            if color and shape:
                if obj.color == color and obj.shape == shape:
                    return obj
            elif color and obj.color == color:
                return obj
            elif shape and obj.shape == shape:
                return obj
        return None
    
    def _compute_destination_position(
        self,
        parsed_command: ParsedCommand,
        scene_objects: List[Object3D],
        target_obj: Object3D
    ) -> Tuple[float, float, float]:
        """Compute destination position based on spatial relation."""
        # Find reference object
        ref_obj = self._find_object(
            parsed_command.destination_color,
            parsed_command.destination_shape,
            scene_objects
        )
        
        if ref_obj is None:
            # Default position if no reference
            return (0.2, 0.0, 0.0)
        
        # Apply spatial relation
        offset = self._get_spatial_offset(
            parsed_command.spatial_relation,
            target_obj.size
        )
        
        dest_pos = (
            ref_obj.position[0] + offset[0],
            ref_obj.position[1] + offset[1],
            ref_obj.position[2] + offset[2]
        )
        
        return self._clamp_to_workspace(dest_pos)
    
    def _get_spatial_offset(
        self,
        relation: SpatialRelation,
        object_size: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Get position offset for spatial relation."""
        offset_distance = max(object_size) + self.safety_margin
        
        offsets = {
            SpatialRelation.LEFT_OF: (-offset_distance, 0, 0),
            SpatialRelation.RIGHT_OF: (offset_distance, 0, 0),
            SpatialRelation.IN_FRONT_OF: (0, offset_distance, 0),
            SpatialRelation.BEHIND: (0, -offset_distance, 0),
            SpatialRelation.ABOVE: (0, 0, offset_distance),
            SpatialRelation.ON: (0, 0, object_size[2]),
            SpatialRelation.NEXT_TO: (offset_distance, 0, 0),
            SpatialRelation.NONE: (0, 0, 0)
        }
        
        return offsets.get(relation, (0, 0, 0))
    
    def _generate_pick_waypoints(
        self,
        target_obj: Object3D,
        current_ee_pose: Optional[Tuple]
    ) -> List[Waypoint]:
        """Generate waypoints for pick action."""
        waypoints = []
        obj_pos = target_obj.position
        
        # Default orientation (gripper pointing down)
        default_orn = self._euler_to_quaternion(np.pi, 0, 0)
        
        # 1. Approach waypoint (above object)
        approach_pos = (obj_pos[0], obj_pos[1], obj_pos[2] + 0.1)
        waypoints.append(Waypoint(
            position=approach_pos,
            orientation=default_orn,
            gripper_state=1.0,  # Open
            phase=MotionPhase.APPROACH,
            velocity_scale=0.5
        ))
        
        # 2. Grasp waypoint (at object)
        grasp_pos = obj_pos
        waypoints.append(Waypoint(
            position=grasp_pos,
            orientation=default_orn,
            gripper_state=1.0,  # Still open
            phase=MotionPhase.GRASP,
            velocity_scale=0.3
        ))
        
        # 3. Close gripper (same position)
        waypoints.append(Waypoint(
            position=grasp_pos,
            orientation=default_orn,
            gripper_state=0.0,  # Closed
            phase=MotionPhase.GRASP,
            velocity_scale=0.0
        ))
        
        # 4. Lift waypoint
        lift_pos = (obj_pos[0], obj_pos[1], obj_pos[2] + self.lift_height)
        waypoints.append(Waypoint(
            position=lift_pos,
            orientation=default_orn,
            gripper_state=0.0,  # Closed
            phase=MotionPhase.LIFT,
            velocity_scale=0.4
        ))
        
        return waypoints
    
    def _generate_place_waypoints(
        self,
        target_obj: Object3D,
        destination: Tuple[float, float, float],
        current_ee_pose: Optional[Tuple]
    ) -> List[Waypoint]:
        """Generate waypoints for place action."""
        waypoints = []
        
        # Default orientation
        default_orn = self._euler_to_quaternion(np.pi, 0, 0)
        
        # 1. Transport waypoint (above destination)
        transport_pos = (destination[0], destination[1], destination[2] + 0.1)
        waypoints.append(Waypoint(
            position=transport_pos,
            orientation=default_orn,
            gripper_state=0.0,  # Closed
            phase=MotionPhase.TRANSPORT,
            velocity_scale=0.5
        ))
        
        # 2. Lower waypoint (at destination)
        lower_pos = destination
        waypoints.append(Waypoint(
            position=lower_pos,
            orientation=default_orn,
            gripper_state=0.0,  # Closed
            phase=MotionPhase.LOWER,
            velocity_scale=0.3
        ))
        
        # 3. Release (open gripper)
        waypoints.append(Waypoint(
            position=lower_pos,
            orientation=default_orn,
            gripper_state=1.0,  # Open
            phase=MotionPhase.RELEASE,
            velocity_scale=0.0
        ))
        
        # 4. Retreat waypoint
        retreat_pos = (destination[0], destination[1], destination[2] + 0.1)
        waypoints.append(Waypoint(
            position=retreat_pos,
            orientation=default_orn,
            gripper_state=1.0,  # Open
            phase=MotionPhase.RETREAT,
            velocity_scale=0.4
        ))
        
        return waypoints
    
    def _create_null_action(self, parsed_command: ParsedCommand) -> SymbolicAction:
        """Create a null action when planning fails."""
        return SymbolicAction(
            action_type=ActionType.NONE,
            target_object_id=None,
            destination_position=None,
            spatial_relation=SpatialRelation.NONE,
            waypoints=[]
        )
    
    def _clamp_to_workspace(
        self,
        position: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Clamp position to workspace bounds."""
        x = np.clip(position[0], *self.workspace_bounds['x'])
        y = np.clip(position[1], *self.workspace_bounds['y'])
        z = np.clip(position[2], *self.workspace_bounds['z'])
        return (x, y, z)
    
    def _euler_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float
    ) -> Tuple[float, float, float, float]:
        """Convert Euler angles to quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return (qx, qy, qz, qw)
