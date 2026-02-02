"""
Semantic Parser for Compositional Command Understanding

Converts natural language commands into formal semantic representations
as structured programs following RT-2/PaLM-E style semantic parsing.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import json


class GoalType(Enum):
    """High-level goal types."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    STACK = "stack"
    ARRANGE = "arrange"
    NONE = "none"


class RelationType(Enum):
    """Spatial and semantic relations."""
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    ON = "on"
    ABOVE = "above"
    BELOW = "below"
    NEXT_TO = "next_to"
    NEAR = "near"
    FAR_FROM = "far_from"
    BETWEEN = "between"
    NONE = "none"


@dataclass
class ObjectDescriptor:
    """Formal object description with properties."""
    color: Optional[str] = None
    shape: Optional[str] = None
    size: Optional[str] = None  # small, medium, large
    material: Optional[str] = None
    id: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def matches(self, other: 'ObjectDescriptor') -> bool:
        """Check if this descriptor matches another."""
        if self.color and other.color and self.color != other.color:
            return False
        if self.shape and other.shape and self.shape != other.shape:
            return False
        if self.size and other.size and self.size != other.size:
            return False
        if self.id is not None and other.id is not None and self.id != other.id:
            return False
        return True


@dataclass
class SpatialRelation:
    """Formal spatial relation specification."""
    type: RelationType
    reference: Optional[ObjectDescriptor] = None
    reference2: Optional[ObjectDescriptor] = None  # For 'between'
    distance: Optional[float] = None  # Metric distance if specified
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {"type": self.type.value}
        if self.reference:
            result["reference"] = self.reference.to_dict()
        if self.reference2:
            result["reference2"] = self.reference2.to_dict()
        if self.distance:
            result["distance"] = self.distance
        if self.parameters:
            result["parameters"] = self.parameters
        return result


@dataclass
class SemanticProgram:
    """
    Structured semantic representation of a command.
    
    This is the formal output of the semantic parser, representing
    the command as a composable program.
    """
    goal: GoalType
    object: ObjectDescriptor
    relation: Optional[SpatialRelation] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    subgoals: List['SemanticProgram'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {
            "goal": self.goal.value,
            "object": self.object.to_dict()
        }
        if self.relation:
            result["relation"] = self.relation.to_dict()
        if self.constraints:
            result["constraints"] = self.constraints
        if self.subgoals:
            result["subgoals"] = [sg.to_dict() for sg in self.subgoals]
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class SemanticParser:
    """
    Compositional semantic parser for robotic commands.
    
    Converts natural language to formal semantic programs that can be
    composed and executed systematically.
    """
    
    def __init__(self):
        """Initialize semantic parser."""
        # Goal keywords
        self.goal_keywords = {
            'pick': [r'\bpick\b', r'\bgrab\b', r'\bgrasp\b', r'\btake\b', r'\bget\b'],
            'place': [r'\bplace\b', r'\bput\b', r'\bset\b', r'\bdrop\b', r'\bposition\b'],
            'move': [r'\bmove\b', r'\bbring\b', r'\bcarry\b', r'\btransfer\b'],
            'stack': [r'\bstack\b', r'\bpile\b'],
            'arrange': [r'\barrange\b', r'\borganize\b', r'\bsort\b'],
        }
        
        # Object properties
        self.color_keywords = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple',
            'black', 'white', 'brown', 'pink', 'gray', 'grey'
        ]
        
        self.shape_keywords = [
            'cube', 'box', 'block',
            'sphere', 'ball',
            'cylinder', 'can', 'tube',
            'pyramid', 'cone'
        ]
        
        self.size_keywords = ['small', 'medium', 'large', 'tiny', 'big', 'huge']
        
        # Relation keywords with enhanced patterns
        self.relation_patterns = {
            'left_of': [r'(?:to the )?left of', r'left side of'],
            'right_of': [r'(?:to the )?right of', r'right side of'],
            'in_front_of': [r'in front of', r'before', r'ahead of'],
            'behind': [r'behind', r'back of'],
            'on': [r'\bon\b', r'on top of', r'onto', r'atop'],
            'above': [r'above', r'over'],
            'below': [r'below', r'under', r'beneath'],
            'next_to': [r'next to', r'beside', r'adjacent to'],
            'near': [r'near', r'close to', r'around'],
            'far_from': [r'far from', r'away from'],
            'between': [r'between'],
        }
    
    def parse(self, command: str) -> SemanticProgram:
        """
        Parse natural language command into semantic program.
        
        Args:
            command: Natural language command
            
        Returns:
            SemanticProgram with formal structure
        """
        command_lower = command.lower().strip()
        
        # Check for compound commands (multi-step)
        if self._is_compound(command_lower):
            return self._parse_compound(command_lower)
        
        # Extract goal
        goal = self._extract_goal(command_lower)
        
        # Extract object descriptor
        obj_desc = self._extract_object(command_lower, is_primary=True)
        
        # Extract spatial relation (if place/move goal)
        relation = None
        if goal in [GoalType.PLACE, GoalType.MOVE, GoalType.STACK]:
            relation = self._extract_relation(command_lower)
        
        # Extract constraints
        constraints = self._extract_constraints(command_lower)
        
        # Build semantic program
        program = SemanticProgram(
            goal=goal,
            object=obj_desc,
            relation=relation,
            constraints=constraints,
            metadata={"raw_command": command}
        )
        
        return program
    
    def _is_compound(self, command: str) -> bool:
        """Check if command is compound (multiple steps)."""
        compound_markers = [r'\band\b', r'\bthen\b', r'\bafter that\b', r'\bnext\b']
        for marker in compound_markers:
            if re.search(marker, command):
                return True
        return False
    
    def _parse_compound(self, command: str) -> SemanticProgram:
        """Parse compound command into nested semantic program."""
        # Split on compound markers
        parts = re.split(r'\band\b|\bthen\b|\bafter that\b', command)
        
        # Parse each part as a subgoal
        subgoals = []
        for part in parts:
            part = part.strip()
            if part:
                subgoal = self.parse(part)
                subgoals.append(subgoal)
        
        # If we have subgoals, wrap in a high-level program
        if len(subgoals) > 1:
            # Infer overall goal from subgoals
            overall_goal = GoalType.ARRANGE  # Default for multi-step
            
            # Create wrapper program
            return SemanticProgram(
                goal=overall_goal,
                object=subgoals[0].object,  # Primary object
                subgoals=subgoals,
                metadata={"raw_command": command, "is_compound": True}
            )
        elif subgoals:
            return subgoals[0]
        else:
            return self._create_null_program(command)
    
    def _extract_goal(self, command: str) -> GoalType:
        """Extract goal from command."""
        for goal_name, patterns in self.goal_keywords.items():
            for pattern in patterns:
                if re.search(pattern, command):
                    return GoalType(goal_name)
        return GoalType.NONE
    
    def _extract_object(self, command: str, is_primary: bool = True) -> ObjectDescriptor:
        """
        Extract object descriptor from command.
        
        Args:
            command: Command string
            is_primary: If True, extract first object; else second
        """
        # Find all colors
        colors_found = []
        for color in self.color_keywords:
            if re.search(rf'\b{color}\b', command):
                colors_found.append(color)
        
        # Find all shapes
        shapes_found = []
        for shape in self.shape_keywords:
            if re.search(rf'\b{shape}\b', command):
                shapes_found.append(shape)
        
        # Find sizes
        sizes_found = []
        for size in self.size_keywords:
            if re.search(rf'\b{size}\b', command):
                sizes_found.append(size)
        
        # Select appropriate match based on is_primary
        if is_primary:
            color = colors_found[0] if colors_found else None
            shape = shapes_found[0] if shapes_found else None
            size = sizes_found[0] if sizes_found else None
        else:
            color = colors_found[1] if len(colors_found) > 1 else (colors_found[0] if colors_found else None)
            shape = shapes_found[1] if len(shapes_found) > 1 else (shapes_found[0] if shapes_found else None)
            size = sizes_found[1] if len(sizes_found) > 1 else (sizes_found[0] if sizes_found else None)
        
        return ObjectDescriptor(color=color, shape=shape, size=size)
    
    def _extract_relation(self, command: str) -> Optional[SpatialRelation]:
        """Extract spatial relation from command."""
        for relation_name, patterns in self.relation_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command)
                if match:
                    # Extract reference object (comes after the relation)
                    # Find text after the relation keyword
                    remaining = command[match.end():].strip()
                    
                    # Handle 'between' specially
                    if relation_name == 'between':
                        return self._extract_between_relation(remaining)
                    
                    # Extract reference object
                    ref_obj = self._extract_object(remaining, is_primary=True)
                    
                    return SpatialRelation(
                        type=RelationType(relation_name),
                        reference=ref_obj
                    )
        
        return None
    
    def _extract_between_relation(self, text: str) -> SpatialRelation:
        """Extract 'between X and Y' relation."""
        # Look for 'and' to split references
        if ' and ' in text:
            parts = text.split(' and ', 1)
            ref1 = self._extract_object(parts[0], is_primary=True)
            ref2 = self._extract_object(parts[1], is_primary=True)
            return SpatialRelation(
                type=RelationType.BETWEEN,
                reference=ref1,
                reference2=ref2
            )
        else:
            # Fallback
            ref = self._extract_object(text, is_primary=True)
            return SpatialRelation(type=RelationType.BETWEEN, reference=ref)
    
    def _extract_constraints(self, command: str) -> List[Dict[str, Any]]:
        """Extract additional constraints from command."""
        constraints = []
        
        # Check for careful/gentle handling
        if re.search(r'\bcarefully\b|\bgently\b|\bslowly\b', command):
            constraints.append({"type": "velocity", "max_speed": 0.3})
        
        # Check for precision requirements
        if re.search(r'\bprecisely\b|\bexactly\b|\bcarefully\b', command):
            constraints.append({"type": "precision", "tolerance": 0.001})
        
        # Check for force constraints
        if re.search(r'\bsoftly\b|\blightly\b', command):
            constraints.append({"type": "force", "max_force": 5.0})
        
        return constraints
    
    def _create_null_program(self, command: str) -> SemanticProgram:
        """Create a null semantic program for unparseable commands."""
        return SemanticProgram(
            goal=GoalType.NONE,
            object=ObjectDescriptor(),
            metadata={"raw_command": command, "parse_failed": True}
        )
    
    def validate_program(self, program: SemanticProgram) -> tuple[bool, List[str]]:
        """
        Validate semantic program completeness and correctness.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check goal is specified
        if program.goal == GoalType.NONE:
            errors.append("No valid goal specified")
        
        # Check object has at least one property
        if not any([program.object.color, program.object.shape, program.object.id]):
            errors.append("Object descriptor is incomplete")
        
        # Check relation consistency
        if program.relation and program.goal not in [GoalType.PLACE, GoalType.MOVE, GoalType.STACK]:
            errors.append(f"Spatial relation specified but goal is {program.goal.value}")
        
        # Check subgoals consistency
        if program.subgoals:
            for i, subgoal in enumerate(program.subgoals):
                is_valid, sub_errors = self.validate_program(subgoal)
                if not is_valid:
                    errors.append(f"Subgoal {i} invalid: {sub_errors}")
        
        return (len(errors) == 0, errors)
