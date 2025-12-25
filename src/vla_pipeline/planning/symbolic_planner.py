"""
State-Based Symbolic Planner

Implements STRIPS-style planning with preconditions, effects, and world state.
Supports replanning, failure handling, and occlusion reasoning.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
import numpy as np

from ..language.semantic_parser import SemanticProgram, GoalType, ObjectDescriptor
from ..perception.detector import Object3D


class PredicateType(Enum):
    """Types of world state predicates."""
    AT = "at"  # Object at position
    HOLDING = "holding"  # Robot holding object
    ON = "on"  # Object on another object
    CLEAR = "clear"  # Object top is clear
    GRASPABLE = "graspable"  # Object can be grasped
    REACHABLE = "reachable"  # Position is reachable
    OCCLUDED = "occluded"  # Object is occluded
    EMPTY_HAND = "empty_hand"  # Gripper is empty


@dataclass
class Predicate:
    """A logical predicate in the world state."""
    type: PredicateType
    args: Tuple[Any, ...]  # Arguments (object_id, position, etc.)
    
    def __hash__(self):
        return hash((self.type, self.args))
    
    def __eq__(self, other):
        return isinstance(other, Predicate) and self.type == other.type and self.args == other.args
    
    def __repr__(self):
        return f"{self.type.value}({', '.join(map(str, self.args))})"


@dataclass
class WorldState:
    """
    Current state of the world.
    
    Maintains predicates about objects, positions, and robot state.
    """
    predicates: Set[Predicate] = field(default_factory=set)
    objects: Dict[int, Object3D] = field(default_factory=dict)
    robot_holding: Optional[int] = None  # Object ID if holding something
    
    def add_predicate(self, pred: Predicate):
        """Add a predicate to the state."""
        self.predicates.add(pred)
    
    def remove_predicate(self, pred: Predicate):
        """Remove a predicate from the state."""
        self.predicates.discard(pred)
    
    def has_predicate(self, pred: Predicate) -> bool:
        """Check if predicate holds in current state."""
        return pred in self.predicates
    
    def query(self, pred_type: PredicateType, args: Optional[Tuple] = None) -> List[Predicate]:
        """Query predicates of a given type."""
        results = []
        for pred in self.predicates:
            if pred.type == pred_type:
                if args is None or pred.args == args:
                    results.append(pred)
        return results
    
    def copy(self) -> 'WorldState':
        """Create a deep copy of the state."""
        new_state = WorldState()
        new_state.predicates = copy.copy(self.predicates)
        new_state.objects = copy.copy(self.objects)
        new_state.robot_holding = self.robot_holding
        return new_state


@dataclass
class Action:
    """
    A symbolic action with preconditions and effects.
    
    STRIPS-style action definition.
    """
    name: str
    parameters: Dict[str, Any]
    preconditions: List[Predicate]
    add_effects: List[Predicate]  # Predicates to add
    del_effects: List[Predicate]  # Predicates to delete
    cost: float = 1.0
    
    def is_applicable(self, state: WorldState) -> bool:
        """Check if action preconditions are satisfied."""
        for precond in self.preconditions:
            if not state.has_predicate(precond):
                return False
        return True
    
    def apply(self, state: WorldState) -> WorldState:
        """Apply action effects to state."""
        new_state = state.copy()
        
        # Remove delete effects
        for pred in self.del_effects:
            new_state.remove_predicate(pred)
        
        # Add add effects
        for pred in self.add_effects:
            new_state.add_predicate(pred)
        
        # Update robot holding state if applicable
        if self.name == "pick":
            new_state.robot_holding = self.parameters.get("object_id")
        elif self.name == "place":
            new_state.robot_holding = None
        
        return new_state


class StateBasedPlanner:
    """
    State-based symbolic planner with STRIPS-style reasoning.
    
    Handles:
    - Precondition checking
    - Effect application
    - Replanning on failure
    - Occlusion reasoning
    """
    
    def __init__(self):
        """Initialize planner."""
        self.max_plan_length = 20
        self.replan_attempts = 3
    
    def initialize_state(self, objects: List[Object3D]) -> WorldState:
        """
        Initialize world state from perceived objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Initial WorldState
        """
        state = WorldState()
        
        # Add objects to state
        for obj in objects:
            state.objects[obj.id] = obj
            
            # Add predicates
            state.add_predicate(Predicate(PredicateType.AT, (obj.id, obj.position)))
            state.add_predicate(Predicate(PredicateType.CLEAR, (obj.id,)))
            state.add_predicate(Predicate(PredicateType.GRASPABLE, (obj.id,)))
            
            # Check if reachable (simple heuristic)
            if self._is_position_reachable(obj.position):
                state.add_predicate(Predicate(PredicateType.REACHABLE, (obj.position,)))
        
        # Robot starts with empty hand
        state.add_predicate(Predicate(PredicateType.EMPTY_HAND, ()))
        
        # Check for occlusions
        self._update_occlusions(state)
        
        return state
    
    def plan(self, 
             semantic_program: SemanticProgram, 
             initial_state: WorldState) -> Optional[List[Action]]:
        """
        Generate action sequence to achieve goal.
        
        Args:
            semantic_program: Parsed semantic program
            initial_state: Current world state
            
        Returns:
            List of Actions or None if planning fails
        """
        # Handle compound programs with subgoals
        if semantic_program.subgoals:
            return self._plan_compound(semantic_program, initial_state)
        
        # Single goal planning
        goal_predicates = self._goal_to_predicates(semantic_program, initial_state)
        
        if not goal_predicates:
            return None
        
        # Forward search planning
        plan = self._forward_search(initial_state, goal_predicates)
        
        return plan
    
    def _plan_compound(self, 
                       program: SemanticProgram, 
                       state: WorldState) -> Optional[List[Action]]:
        """Plan for compound command with multiple subgoals."""
        all_actions = []
        current_state = state.copy()
        
        for subgoal in program.subgoals:
            # Plan for this subgoal
            subplan = self.plan(subgoal, current_state)
            
            if subplan is None:
                return None  # Planning failed
            
            # Add to overall plan
            all_actions.extend(subplan)
            
            # Update state by applying actions
            for action in subplan:
                current_state = action.apply(current_state)
        
        return all_actions
    
    def _goal_to_predicates(self, 
                            program: SemanticProgram, 
                            state: WorldState) -> List[Predicate]:
        """Convert semantic program goal to goal predicates."""
        goal_preds = []
        
        # Find target object
        target_obj = self._find_object(program.object, state)
        if target_obj is None:
            return []
        
        if program.goal == GoalType.PICK:
            # Goal: Robot holding object
            goal_preds.append(Predicate(PredicateType.HOLDING, (target_obj.id,)))
        
        elif program.goal == GoalType.PLACE:
            # Goal: Object at destination
            if program.relation and program.relation.reference:
                ref_obj = self._find_object(program.relation.reference, state)
                if ref_obj:
                    # Compute target position based on relation
                    target_pos = self._compute_relation_position(
                        ref_obj.position, 
                        program.relation.type.value
                    )
                    goal_preds.append(Predicate(PredicateType.AT, (target_obj.id, target_pos)))
                    goal_preds.append(Predicate(PredicateType.EMPTY_HAND, ()))
        
        elif program.goal == GoalType.MOVE:
            # Similar to place
            if program.relation and program.relation.reference:
                ref_obj = self._find_object(program.relation.reference, state)
                if ref_obj:
                    target_pos = self._compute_relation_position(
                        ref_obj.position,
                        program.relation.type.value
                    )
                    goal_preds.append(Predicate(PredicateType.AT, (target_obj.id, target_pos)))
        
        return goal_preds
    
    def _forward_search(self, 
                        initial_state: WorldState, 
                        goal_predicates: List[Predicate]) -> Optional[List[Action]]:
        """
        Forward search from initial state to goal.
        
        Simple breadth-first search for now.
        Could be enhanced with A* or other heuristics.
        """
        from collections import deque
        
        # Queue of (state, plan) tuples
        queue = deque([(initial_state, [])])
        visited = set()
        
        while queue and len(queue) < 1000:  # Limit search
            current_state, current_plan = queue.popleft()
            
            # Check if goal reached
            if self._goal_satisfied(current_state, goal_predicates):
                return current_plan
            
            # State hash for visited check
            state_hash = self._state_hash(current_state)
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # If plan too long, skip
            if len(current_plan) >= self.max_plan_length:
                continue
            
            # Generate applicable actions
            applicable_actions = self._get_applicable_actions(current_state)
            
            # Expand successors
            for action in applicable_actions:
                new_state = action.apply(current_state)
                new_plan = current_plan + [action]
                queue.append((new_state, new_plan))
        
        return None  # Planning failed
    
    def _goal_satisfied(self, state: WorldState, goals: List[Predicate]) -> bool:
        """Check if all goal predicates are satisfied."""
        for goal in goals:
            if not state.has_predicate(goal):
                return False
        return True
    
    def _state_hash(self, state: WorldState) -> int:
        """Compute hash for state (for visited set)."""
        pred_tuple = tuple(sorted([str(p) for p in state.predicates]))
        return hash((pred_tuple, state.robot_holding))
    
    def _get_applicable_actions(self, state: WorldState) -> List[Action]:
        """Generate all applicable actions in current state."""
        actions = []
        
        # Pick actions
        if state.robot_holding is None:  # Empty hand
            for obj_id, obj in state.objects.items():
                pick_action = self._create_pick_action(obj_id, obj, state)
                if pick_action and pick_action.is_applicable(state):
                    actions.append(pick_action)
        
        # Place actions
        if state.robot_holding is not None:
            # Can place at various positions
            for position in self._get_valid_positions(state):
                place_action = self._create_place_action(state.robot_holding, position, state)
                if place_action and place_action.is_applicable(state):
                    actions.append(place_action)
        
        return actions
    
    def _create_pick_action(self, obj_id: int, obj: Object3D, state: WorldState) -> Optional[Action]:
        """Create a pick action for an object."""
        # Preconditions
        preconditions = [
            Predicate(PredicateType.GRASPABLE, (obj_id,)),
            Predicate(PredicateType.CLEAR, (obj_id,)),
            Predicate(PredicateType.EMPTY_HAND, ()),
        ]
        
        # Effects
        add_effects = [
            Predicate(PredicateType.HOLDING, (obj_id,)),
        ]
        
        del_effects = [
            Predicate(PredicateType.AT, (obj_id, obj.position)),
            Predicate(PredicateType.EMPTY_HAND, ()),
        ]
        
        return Action(
            name="pick",
            parameters={"object_id": obj_id},
            preconditions=preconditions,
            add_effects=add_effects,
            del_effects=del_effects,
            cost=1.0
        )
    
    def _create_place_action(self, obj_id: int, position: Tuple, state: WorldState) -> Optional[Action]:
        """Create a place action."""
        # Preconditions
        preconditions = [
            Predicate(PredicateType.HOLDING, (obj_id,)),
            Predicate(PredicateType.REACHABLE, (position,)),
        ]
        
        # Effects
        add_effects = [
            Predicate(PredicateType.AT, (obj_id, position)),
            Predicate(PredicateType.EMPTY_HAND, ()),
        ]
        
        del_effects = [
            Predicate(PredicateType.HOLDING, (obj_id,)),
        ]
        
        return Action(
            name="place",
            parameters={"object_id": obj_id, "position": position},
            preconditions=preconditions,
            add_effects=add_effects,
            del_effects=del_effects,
            cost=1.0
        )
    
    def replan(self, 
               failed_action: Action, 
               current_state: WorldState,
               original_goal: List[Predicate]) -> Optional[List[Action]]:
        """
        Replan after action failure.
        
        Args:
            failed_action: The action that failed
            current_state: Current world state
            original_goal: Original goal predicates
            
        Returns:
            New plan or None
        """
        # Try multiple replanning strategies
        for attempt in range(self.replan_attempts):
            # Strategy 1: Update state and replan
            if attempt == 0:
                # Assume object might be occluded or moved
                self._update_occlusions(current_state)
                new_plan = self._forward_search(current_state, original_goal)
                if new_plan:
                    return new_plan
            
            # Strategy 2: Try alternative actions
            elif attempt == 1:
                # Modify search to avoid similar failures
                # (simplified - in practice would track failure reasons)
                new_plan = self._forward_search(current_state, original_goal)
                if new_plan:
                    return new_plan
        
        return None  # Replanning failed
    
    def _find_object(self, descriptor: ObjectDescriptor, state: WorldState) -> Optional[Object3D]:
        """Find object in state matching descriptor."""
        for obj_id, obj in state.objects.items():
            if descriptor.color and obj.color != descriptor.color:
                continue
            if descriptor.shape and obj.shape != descriptor.shape:
                continue
            if descriptor.id is not None and obj.id != descriptor.id:
                continue
            return obj
        return None
    
    def _compute_relation_position(self, ref_pos: Tuple, relation: str) -> Tuple:
        """Compute target position based on spatial relation."""
        offset_map = {
            'left_of': (-0.1, 0, 0),
            'right_of': (0.1, 0, 0),
            'in_front_of': (0, 0.1, 0),
            'behind': (0, -0.1, 0),
            'on': (0, 0, 0.05),
            'above': (0, 0, 0.1),
            'next_to': (0.1, 0, 0),
        }
        
        offset = offset_map.get(relation, (0, 0, 0))
        return tuple(ref_pos[i] + offset[i] for i in range(3))
    
    def _is_position_reachable(self, position: Tuple) -> bool:
        """Check if position is within robot workspace."""
        # Simple workspace bounds
        x, y, z = position
        return (-0.5 <= x <= 0.5 and 
                -0.5 <= y <= 0.5 and 
                0.0 <= z <= 0.5)
    
    def _get_valid_positions(self, state: WorldState) -> List[Tuple]:
        """Get list of valid placement positions."""
        positions = []
        
        # Sample positions in workspace
        for x in np.linspace(-0.3, 0.3, 5):
            for y in np.linspace(-0.3, 0.3, 5):
                pos = (x, y, 0.05)
                if self._is_position_reachable(pos):
                    positions.append(pos)
        
        return positions
    
    def _update_occlusions(self, state: WorldState):
        """Update occlusion predicates based on object positions."""
        # Simple occlusion check - objects at same x,y
        for obj1_id, obj1 in state.objects.items():
            for obj2_id, obj2 in state.objects.items():
                if obj1_id != obj2_id:
                    # Check if obj2 occludes obj1
                    if (abs(obj1.position[0] - obj2.position[0]) < 0.05 and
                        abs(obj1.position[1] - obj2.position[1]) < 0.05 and
                        obj2.position[2] > obj1.position[2]):
                        state.add_predicate(Predicate(PredicateType.OCCLUDED, (obj1_id,)))
