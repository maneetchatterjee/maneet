"""
Language Reasoning Module

Parses natural language commands and extracts intent, target objects,
and spatial relations.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of robotic actions."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    PUSH = "push"
    NONE = "none"


class SpatialRelation(Enum):
    """Spatial relationships between objects."""
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    ON = "on"
    ABOVE = "above"
    NEXT_TO = "next_to"
    NONE = "none"


@dataclass
class ParsedCommand:
    """Represents a parsed natural language command."""
    raw_command: str
    action: ActionType
    target_object: Optional[str] = None
    target_color: Optional[str] = None
    target_shape: Optional[str] = None
    destination_object: Optional[str] = None
    destination_color: Optional[str] = None
    destination_shape: Optional[str] = None
    spatial_relation: SpatialRelation = SpatialRelation.NONE
    confidence: float = 1.0


class LanguageReasoningModule:
    """
    Natural language command parser for robotic manipulation.
    
    Decoupled from control logic - only parses intent and extracts
    semantic information from commands.
    """
    
    def __init__(self):
        """Initialize language reasoning module."""
        self.action_keywords = {
            'pick': [r'pick', r'grab', r'grasp', r'take'],
            'place': [r'place', r'put', r'set', r'drop'],
            'move': [r'move', r'bring', r'carry'],
            'push': [r'push', r'slide'],
        }
        
        self.color_keywords = [
            'red', 'blue', 'green', 'yellow', 'orange', 
            'purple', 'black', 'white', 'brown'
        ]
        
        self.shape_keywords = [
            'cube', 'box', 'block', 
            'sphere', 'ball', 
            'cylinder', 'can'
        ]
        
        self.spatial_keywords = {
            'left_of': [r'left of', r'to the left of'],
            'right_of': [r'right of', r'to the right of'],
            'in_front_of': [r'in front of', r'before'],
            'behind': [r'behind', r'back of'],
            'on': [r'\bon\b', r'on top of', r'onto'],
            'above': [r'above', r'over'],
            'next_to': [r'next to', r'beside', r'near'],
        }
    
    def parse_command(self, command: str) -> ParsedCommand:
        """
        Parse natural language command into structured format.
        
        Args:
            command: Natural language command string
            
        Returns:
            ParsedCommand with extracted intent and objects
        """
        command_lower = command.lower()
        
        # Extract action
        action = self._extract_action(command_lower)
        
        # Extract objects and their properties
        target_color, target_shape = self._extract_object_properties(
            command_lower, is_target=True
        )
        
        # Extract spatial relation
        spatial_relation = self._extract_spatial_relation(command_lower)
        
        # Extract destination object for place/move actions
        dest_color, dest_shape = None, None
        if action in [ActionType.PLACE, ActionType.MOVE]:
            dest_color, dest_shape = self._extract_object_properties(
                command_lower, is_target=False
            )
        
        # Build target object name
        target_object = self._build_object_name(target_color, target_shape)
        destination_object = self._build_object_name(dest_color, dest_shape)
        
        return ParsedCommand(
            raw_command=command,
            action=action,
            target_object=target_object,
            target_color=target_color,
            target_shape=target_shape,
            destination_object=destination_object,
            destination_color=dest_color,
            destination_shape=dest_shape,
            spatial_relation=spatial_relation,
            confidence=0.9  # Can be improved with ML models
        )
    
    def _extract_action(self, command: str) -> ActionType:
        """Extract action type from command."""
        for action_name, patterns in self.action_keywords.items():
            for pattern in patterns:
                if re.search(pattern, command):
                    return ActionType(action_name)
        return ActionType.NONE
    
    def _extract_object_properties(
        self, 
        command: str, 
        is_target: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract color and shape from command.
        
        Args:
            command: Command string
            is_target: If True, extract first object; otherwise second
        """
        # Find all color mentions
        colors_found = [c for c in self.color_keywords if c in command]
        
        # Find all shape mentions
        shapes_found = [s for s in self.shape_keywords if s in command]
        
        # Return first or second match based on is_target
        if is_target:
            color = colors_found[0] if colors_found else None
            shape = shapes_found[0] if shapes_found else None
        else:
            color = colors_found[1] if len(colors_found) > 1 else (colors_found[0] if colors_found else None)
            shape = shapes_found[1] if len(shapes_found) > 1 else (shapes_found[0] if shapes_found else None)
        
        return color, shape
    
    def _extract_spatial_relation(self, command: str) -> SpatialRelation:
        """Extract spatial relation from command."""
        for relation_name, patterns in self.spatial_keywords.items():
            for pattern in patterns:
                if re.search(pattern, command):
                    return SpatialRelation(relation_name)
        return SpatialRelation.NONE
    
    def _build_object_name(
        self, 
        color: Optional[str], 
        shape: Optional[str]
    ) -> Optional[str]:
        """Build object name from color and shape."""
        if color and shape:
            return f"{color}_{shape}"
        elif color:
            return color
        elif shape:
            return shape
        return None
    
    def get_action_sequence(self, command: str) -> List[ParsedCommand]:
        """
        Parse command into a sequence of actions.
        Handles compound commands with 'and', 'then', etc.
        
        Args:
            command: Natural language command
            
        Returns:
            List of ParsedCommand objects
        """
        # Split on conjunctions
        sub_commands = re.split(r'\band\b|\bthen\b', command.lower())
        
        parsed_commands = []
        for sub_cmd in sub_commands:
            sub_cmd = sub_cmd.strip()
            if sub_cmd:
                parsed_commands.append(self.parse_command(sub_cmd))
        
        return parsed_commands
