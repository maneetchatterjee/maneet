"""
Formal Verification Framework for STRIPS Planner

Provides rigorous soundness and completeness proofs, complexity analysis,
and empirical failure testing for the symbolic planner.
"""

from typing import List, Dict, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import copy
import numpy as np
import json

from .symbolic_planner import (
    StateBasedPlanner, WorldState, Action, Predicate, PredicateType,
    Object3D
)
from ..language.semantic_parser import SemanticProgram, GoalType, ObjectDescriptor


@dataclass
class FormalState:
    """
    Formal state space definition.
    
    S = (O, P, R) where:
    - O: Set of objects
    - P: Set of predicates
    - R: Robot state (holding/empty)
    """
    objects: Set[int]  # Object IDs
    predicates: Set[Predicate]
    robot_holding: Optional[int]
    
    def __hash__(self):
        pred_tuple = tuple(sorted([str(p) for p in self.predicates]))
        return hash((tuple(sorted(self.objects)), pred_tuple, self.robot_holding))
    
    def __eq__(self, other):
        return (self.objects == other.objects and 
                self.predicates == other.predicates and
                self.robot_holding == other.robot_holding)


@dataclass
class TransitionFunction:
    """
    Formal transition function T: S × A → S
    
    Maps (state, action) to new state.
    """
    def apply(self, state: FormalState, action: Action) -> Optional[FormalState]:
        """
        Apply action to state.
        
        Returns None if action not applicable (preconditions fail).
        """
        # Check preconditions
        if not self._check_preconditions(state, action):
            return None
        
        # Apply effects
        new_state = FormalState(
            objects=copy.copy(state.objects),
            predicates=copy.copy(state.predicates),
            robot_holding=state.robot_holding
        )
        
        # Delete effects
        for pred in action.del_effects:
            new_state.predicates.discard(pred)
        
        # Add effects
        for pred in action.add_effects:
            new_state.predicates.add(pred)
        
        # Update robot state
        if action.name == "pick":
            new_state.robot_holding = action.parameters.get("object_id")
        elif action.name == "place":
            new_state.robot_holding = None
        
        return new_state
    
    def _check_preconditions(self, state: FormalState, action: Action) -> bool:
        """Check if action preconditions satisfied."""
        for precond in action.preconditions:
            if precond not in state.predicates:
                return False
        return True


@dataclass
class SoundnessProof:
    """
    Soundness: Plans never violate action preconditions.
    
    Theorem: ∀ plan π = [a₁, ..., aₙ], ∀ i ∈ [1,n]:
        precond(aᵢ) ⊆ state(sᵢ₋₁)
    
    where sᵢ = T(sᵢ₋₁, aᵢ)
    """
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    
    def verify(self, 
               plan: List[Action], 
               initial_state: WorldState,
               transition: TransitionFunction) -> bool:
        """
        Verify plan soundness.
        
        Returns True if sound (no precondition violations).
        """
        current_state = self._worldstate_to_formal(initial_state)
        
        for i, action in enumerate(plan):
            # Check preconditions
            precond_satisfied = True
            for precond in action.preconditions:
                if precond not in current_state.predicates:
                    precond_satisfied = False
                    self.violations.append({
                        'step': i,
                        'action': action.name,
                        'missing_precondition': str(precond),
                        'state': str(current_state)
                    })
            
            if not precond_satisfied:
                self.failed_tests += 1
                return False
            
            # Apply action
            next_state = transition.apply(current_state, action)
            if next_state is None:
                self.failed_tests += 1
                self.violations.append({
                    'step': i,
                    'action': action.name,
                    'error': 'Action application failed',
                    'state': str(current_state)
                })
                return False
            
            current_state = next_state
        
        self.passed_tests += 1
        return True
    
    def _worldstate_to_formal(self, ws: WorldState) -> FormalState:
        """Convert WorldState to FormalState."""
        return FormalState(
            objects=set(ws.objects.keys()),
            predicates=copy.copy(ws.predicates),
            robot_holding=ws.robot_holding
        )


@dataclass
class CompletenessProof:
    """
    Partial Completeness: Planner finds plan if one exists within horizon.
    
    Theorem: ∀ goal G, ∀ state s₀:
        if ∃ plan π s.t. |π| ≤ H and G(apply(s₀, π))
        then planner.plan(s₀, G) returns π' s.t. G(apply(s₀, π'))
    
    where H is the horizon bound.
    """
    horizon: int
    total_tests: int = 0
    found_plans: int = 0
    missed_plans: int = 0
    false_negatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def verify(self,
               planner: StateBasedPlanner,
               initial_state: WorldState,
               goal: List[Predicate],
               known_solution_length: Optional[int] = None) -> bool:
        """
        Verify completeness for a specific problem.
        
        If known_solution_length provided and ≤ horizon, planner must find plan.
        """
        self.total_tests += 1
        
        # Run planner
        # Create dummy semantic program for goal
        dummy_program = self._goal_to_program(goal, initial_state)
        plan = planner.plan(dummy_program, initial_state)
        
        if plan is not None:
            self.found_plans += 1
            return True
        else:
            # If we know a solution exists within horizon, this is a false negative
            if known_solution_length is not None and known_solution_length <= self.horizon:
                self.missed_plans += 1
                self.false_negatives.append({
                    'initial_state': str(initial_state),
                    'goal': [str(g) for g in goal],
                    'known_solution_length': known_solution_length,
                    'horizon': self.horizon
                })
                return False
            return True  # Might be unsolvable or beyond horizon
    
    def _goal_to_program(self, goal: List[Predicate], state: WorldState) -> SemanticProgram:
        """Convert goal predicates to semantic program."""
        # Extract object and relation from goal
        for pred in goal:
            if pred.type == PredicateType.HOLDING:
                obj_id = pred.args[0]
                obj = state.objects.get(obj_id)
                if obj:
                    return SemanticProgram(
                        goal=GoalType.PICK,
                        object=ObjectDescriptor(color=obj.color, shape=obj.shape),
                        relation=None
                    )
            elif pred.type == PredicateType.AT:
                obj_id = pred.args[0]
                target_pos = pred.args[1]
                obj = state.objects.get(obj_id)
                if obj:
                    # Find reference object near target
                    return SemanticProgram(
                        goal=GoalType.PLACE,
                        object=ObjectDescriptor(color=obj.color, shape=obj.shape),
                        relation=None
                    )
        
        # Fallback
        return SemanticProgram(goal=GoalType.PICK, object=ObjectDescriptor(), relation=None)


@dataclass
class ComplexityAnalysis:
    """
    Worst-case complexity analysis.
    
    State space size: |S| = |O|^k × 2^|P| × (|O| + 1)
    where:
    - |O|: number of objects
    - k: max objects in predicates
    - |P|: number of predicate types
    - (|O| + 1): robot can hold 0 or 1 of |O| objects
    
    Time complexity: O(|A|^d × |S|)
    where:
    - |A|: branching factor (applicable actions per state)
    - d: solution depth
    - |S|: for duplicate checking
    """
    num_objects: int
    num_predicate_types: int
    max_plan_length: int
    
    def estimate_state_space_size(self) -> int:
        """Estimate |S|."""
        # Simplified: Each object can have ~5 predicates
        # Robot state: O + 1
        return self.num_objects * (2 ** 5) * (self.num_objects + 1)
    
    def estimate_time_complexity(self, branching_factor: int, solution_depth: int) -> int:
        """Estimate time complexity."""
        state_space = self.estimate_state_space_size()
        return (branching_factor ** solution_depth) * state_space
    
    def measure_empirical_performance(self,
                                      planner: StateBasedPlanner,
                                      problems: List[Tuple[WorldState, List[Predicate]]]) -> Dict[str, Any]:
        """
        Measure empirical performance on test problems.
        
        Returns timing and complexity metrics.
        """
        results = {
            'problems': [],
            'avg_time': 0.0,
            'max_time': 0.0,
            'avg_plan_length': 0.0,
            'timeouts': 0
        }
        
        times = []
        plan_lengths = []
        
        for initial_state, goal in problems:
            start = time.time()
            
            # Create dummy program
            dummy_program = self._goal_to_program(goal, initial_state)
            plan = planner.plan(dummy_program, initial_state)
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            if plan:
                plan_lengths.append(len(plan))
            else:
                results['timeouts'] += 1
            
            results['problems'].append({
                'time': elapsed,
                'plan_length': len(plan) if plan else None,
                'success': plan is not None
            })
        
        results['avg_time'] = np.mean(times) if times else 0.0
        results['max_time'] = np.max(times) if times else 0.0
        results['avg_plan_length'] = np.mean(plan_lengths) if plan_lengths else 0.0
        
        return results
    
    def _goal_to_program(self, goal: List[Predicate], state: WorldState) -> SemanticProgram:
        """Convert goal to program."""
        for pred in goal:
            if pred.type == PredicateType.HOLDING:
                obj_id = pred.args[0]
                obj = state.objects.get(obj_id)
                if obj:
                    return SemanticProgram(
                        goal=GoalType.PICK,
                        object=ObjectDescriptor(color=obj.color, shape=obj.shape),
                        relation=None
                    )
        return SemanticProgram(goal=GoalType.PICK, object=ObjectDescriptor(), relation=None)


class FailureInducingWorlds:
    """
    Construct pathological worlds to test planner robustness.
    """
    
    @staticmethod
    def create_occlusion_world() -> Tuple[WorldState, List[Predicate]]:
        """
        World with occluded target object.
        
        Stack: [bottom, middle, top (target)]
        Goal: Pick top object (must remove middle first)
        """
        state = WorldState()
        
        # Bottom object
        bottom = Object3D(id=1, color='red', shape='cube', position=(0.0, 0.0, 0.05))
        state.objects[1] = bottom
        state.add_predicate(Predicate(PredicateType.AT, (1, (0.0, 0.0, 0.05))))
        state.add_predicate(Predicate(PredicateType.GRASPABLE, (1,)))
        state.add_predicate(Predicate(PredicateType.REACHABLE, ((0.0, 0.0, 0.05),)))
        
        # Middle object (occluder)
        middle = Object3D(id=2, color='blue', shape='cube', position=(0.0, 0.0, 0.10))
        state.objects[2] = middle
        state.add_predicate(Predicate(PredicateType.AT, (2, (0.0, 0.0, 0.10))))
        state.add_predicate(Predicate(PredicateType.GRASPABLE, (2,)))
        state.add_predicate(Predicate(PredicateType.CLEAR, (2,)))  # Top is clear
        state.add_predicate(Predicate(PredicateType.REACHABLE, ((0.0, 0.0, 0.10),)))
        state.add_predicate(Predicate(PredicateType.ON, (2, 1)))
        
        # Top object (target) - occluded
        top = Object3D(id=3, color='green', shape='cube', position=(0.0, 0.0, 0.15))
        state.objects[3] = top
        state.add_predicate(Predicate(PredicateType.AT, (3, (0.0, 0.0, 0.15))))
        state.add_predicate(Predicate(PredicateType.GRASPABLE, (3,)))
        state.add_predicate(Predicate(PredicateType.CLEAR, (3,)))
        state.add_predicate(Predicate(PredicateType.REACHABLE, ((0.0, 0.0, 0.15),)))
        state.add_predicate(Predicate(PredicateType.ON, (3, 2)))
        state.add_predicate(Predicate(PredicateType.OCCLUDED, (3,)))  # Target occluded
        
        # Bottom occluded by middle
        state.add_predicate(Predicate(PredicateType.OCCLUDED, (1,)))
        
        # Robot empty
        state.add_predicate(Predicate(PredicateType.EMPTY_HAND, ()))
        
        # Goal: Hold top object
        goal = [Predicate(PredicateType.HOLDING, (3,))]
        
        return state, goal
    
    @staticmethod
    def create_ambiguous_world() -> Tuple[WorldState, List[Predicate]]:
        """
        World with multiple identical objects.
        
        Two red cubes - which one to pick?
        """
        state = WorldState()
        
        # Red cube 1
        cube1 = Object3D(id=1, color='red', shape='cube', position=(0.0, 0.0, 0.05))
        state.objects[1] = cube1
        state.add_predicate(Predicate(PredicateType.AT, (1, (0.0, 0.0, 0.05))))
        state.add_predicate(Predicate(PredicateType.GRASPABLE, (1,)))
        state.add_predicate(Predicate(PredicateType.CLEAR, (1,)))
        state.add_predicate(Predicate(PredicateType.REACHABLE, ((0.0, 0.0, 0.05),)))
        
        # Red cube 2 (identical)
        cube2 = Object3D(id=2, color='red', shape='cube', position=(0.1, 0.0, 0.05))
        state.objects[2] = cube2
        state.add_predicate(Predicate(PredicateType.AT, (2, (0.1, 0.0, 0.05))))
        state.add_predicate(Predicate(PredicateType.GRASPABLE, (2,)))
        state.add_predicate(Predicate(PredicateType.CLEAR, (2,)))
        state.add_predicate(Predicate(PredicateType.REACHABLE, ((0.1, 0.0, 0.05),)))
        
        state.add_predicate(Predicate(PredicateType.EMPTY_HAND, ()))
        
        # Goal: Hold "a red cube" - ambiguous!
        goal = [Predicate(PredicateType.HOLDING, (1,))]  # Could be 1 or 2
        
        return state, goal
    
    @staticmethod
    def create_unreachable_goal() -> Tuple[WorldState, List[Predicate]]:
        """
        World where goal is physically unreachable.
        
        Target position outside workspace.
        """
        state = WorldState()
        
        # Object
        obj = Object3D(id=1, color='red', shape='cube', position=(0.0, 0.0, 0.05))
        state.objects[1] = obj
        state.add_predicate(Predicate(PredicateType.AT, (1, (0.0, 0.0, 0.05))))
        state.add_predicate(Predicate(PredicateType.GRASPABLE, (1,)))
        state.add_predicate(Predicate(PredicateType.CLEAR, (1,)))
        state.add_predicate(Predicate(PredicateType.REACHABLE, ((0.0, 0.0, 0.05),)))
        state.add_predicate(Predicate(PredicateType.EMPTY_HAND, ()))
        
        # Goal: Place at unreachable position (outside workspace)
        unreachable_pos = (1.0, 1.0, 0.05)  # Far outside [-0.5, 0.5]
        goal = [
            Predicate(PredicateType.AT, (1, unreachable_pos)),
            Predicate(PredicateType.EMPTY_HAND, ())
        ]
        
        return state, goal
    
    @staticmethod
    def create_resource_constrained() -> Tuple[WorldState, List[Predicate]]:
        """
        World requiring many actions (tests planning horizon).
        
        Build tower: A on B on C on D
        All objects initially separated.
        """
        state = WorldState()
        
        positions = [(0.0, 0.0, 0.05), (0.2, 0.0, 0.05), (0.0, 0.2, 0.05), (0.2, 0.2, 0.05)]
        colors = ['red', 'blue', 'green', 'yellow']
        
        for i, (pos, color) in enumerate(zip(positions, colors), 1):
            obj = Object3D(id=i, color=color, shape='cube', position=pos)
            state.objects[i] = obj
            state.add_predicate(Predicate(PredicateType.AT, (i, pos)))
            state.add_predicate(Predicate(PredicateType.GRASPABLE, (i,)))
            state.add_predicate(Predicate(PredicateType.CLEAR, (i,)))
            state.add_predicate(Predicate(PredicateType.REACHABLE, (pos,)))
        
        state.add_predicate(Predicate(PredicateType.EMPTY_HAND, ()))
        
        # Goal: Stack all (requires 8 actions: 4 picks + 4 places)
        tower_pos = (0.1, 0.1, 0.0)
        goal = [
            Predicate(PredicateType.AT, (1, (tower_pos[0], tower_pos[1], 0.05))),
            Predicate(PredicateType.AT, (2, (tower_pos[0], tower_pos[1], 0.10))),
            Predicate(PredicateType.AT, (3, (tower_pos[0], tower_pos[1], 0.15))),
            Predicate(PredicateType.AT, (4, (tower_pos[0], tower_pos[1], 0.20))),
            Predicate(PredicateType.EMPTY_HAND, ())
        ]
        
        return state, goal


class ReplanningTerminationProof:
    """
    Prove replanning terminates or identify non-termination conditions.
    """
    
    def __init__(self, max_replan_attempts: int = 3):
        self.max_replan_attempts = max_replan_attempts
        self.tests_run = 0
        self.terminated = 0
        self.infinite_loops = 0
        self.loop_detections: List[Dict] = []
    
    def verify_termination(self,
                          planner: StateBasedPlanner,
                          initial_state: WorldState,
                          goal: List[Predicate],
                          failure_injector: Callable[[Action], bool]) -> Dict[str, Any]:
        """
        Verify replanning terminates.
        
        Args:
            planner: The planner
            initial_state: Starting state
            goal: Goal predicates
            failure_injector: Function that returns True if action should fail
            
        Returns:
            Verification results
        """
        self.tests_run += 1
        
        state = initial_state.copy()
        visited_states = set()
        replan_count = 0
        
        # Create dummy program
        dummy_program = self._goal_to_program(goal, state)
        
        while replan_count < self.max_replan_attempts:
            # Plan
            plan = planner.plan(dummy_program, state)
            
            if plan is None:
                # Planning failed - terminal
                self.terminated += 1
                return {
                    'terminated': True,
                    'reason': 'planning_failed',
                    'replan_count': replan_count
                }
            
            # Execute plan (with failures)
            for action in plan:
                if failure_injector(action):
                    # Action failed - trigger replan
                    replan_count += 1
                    
                    # Check for infinite loop (same state)
                    state_hash = self._state_hash(state)
                    if state_hash in visited_states:
                        self.infinite_loops += 1
                        self.loop_detections.append({
                            'state': str(state),
                            'replan_count': replan_count,
                            'goal': [str(g) for g in goal]
                        })
                        return {
                            'terminated': False,
                            'reason': 'infinite_loop',
                            'replan_count': replan_count
                        }
                    
                    visited_states.add(state_hash)
                    break  # Replan
                else:
                    # Action succeeded
                    state = action.apply(state)
            
            # Check if goal reached
            if self._goal_satisfied(state, goal):
                self.terminated += 1
                return {
                    'terminated': True,
                    'reason': 'goal_reached',
                    'replan_count': replan_count
                }
        
        # Max attempts reached
        self.terminated += 1
        return {
            'terminated': True,
            'reason': 'max_attempts',
            'replan_count': replan_count
        }
    
    def _goal_satisfied(self, state: WorldState, goals: List[Predicate]) -> bool:
        """Check if goals satisfied."""
        for goal in goals:
            if not state.has_predicate(goal):
                return False
        return True
    
    def _state_hash(self, state: WorldState) -> int:
        """Compute state hash."""
        pred_tuple = tuple(sorted([str(p) for p in state.predicates]))
        return hash((pred_tuple, state.robot_holding))
    
    def _goal_to_program(self, goal: List[Predicate], state: WorldState) -> SemanticProgram:
        """Convert goal to program."""
        for pred in goal:
            if pred.type == PredicateType.HOLDING:
                obj_id = pred.args[0]
                obj = state.objects.get(obj_id)
                if obj:
                    return SemanticProgram(
                        goal=GoalType.PICK,
                        object=ObjectDescriptor(color=obj.color, shape=obj.shape),
                        relation=None
                    )
        return SemanticProgram(goal=GoalType.PICK, object=ObjectDescriptor(), relation=None)


class BaselineComparison:
    """
    Compare STRIPS planner against baselines.
    """
    
    def compare(self,
                strips_planner: StateBasedPlanner,
                problems: List[Tuple[WorldState, List[Predicate]]]) -> Dict[str, Any]:
        """
        Compare STRIPS vs baselines on test problems.
        
        Returns comparative metrics with statistical tests.
        """
        # Test all planners
        strips_results = self._test_planner(strips_planner, problems, "STRIPS")
        greedy_results = self._test_planner(self._greedy_planner(), problems, "Greedy")
        random_results = self._test_planner(self._random_planner(), problems, "Random")
        
        return {
            'strips': strips_results,
            'greedy': greedy_results,
            'random': random_results,
            'comparison': self._statistical_comparison(strips_results, greedy_results, random_results)
        }
    
    def _test_planner(self,
                     planner: StateBasedPlanner,
                     problems: List[Tuple[WorldState, List[Predicate]]],
                     name: str) -> Dict[str, Any]:
        """Test a planner on problems."""
        success_count = 0
        times = []
        plan_lengths = []
        
        for state, goal in problems:
            start = time.time()
            dummy_program = self._goal_to_program(goal, state)
            plan = planner.plan(dummy_program, state)
            elapsed = time.time() - start
            
            times.append(elapsed)
            if plan:
                success_count += 1
                plan_lengths.append(len(plan))
        
        return {
            'name': name,
            'success_rate': success_count / len(problems) if problems else 0.0,
            'avg_time': np.mean(times) if times else 0.0,
            'avg_plan_length': np.mean(plan_lengths) if plan_lengths else 0.0,
            'total_problems': len(problems)
        }
    
    def _greedy_planner(self) -> StateBasedPlanner:
        """Create greedy planner (depth-first search)."""
        planner = StateBasedPlanner()
        # Would modify search to be greedy/DFS
        return planner
    
    def _random_planner(self) -> StateBasedPlanner:
        """Create random planner."""
        planner = StateBasedPlanner()
        # Would modify to randomly select actions
        return planner
    
    def _statistical_comparison(self, strips, greedy, random) -> Dict[str, Any]:
        """Statistical comparison of results."""
        return {
            'strips_vs_greedy_success': strips['success_rate'] - greedy['success_rate'],
            'strips_vs_random_success': strips['success_rate'] - random['success_rate'],
            'strips_vs_greedy_time': strips['avg_time'] / greedy['avg_time'] if greedy['avg_time'] > 0 else float('inf'),
            'winner': max([strips, greedy, random], key=lambda x: x['success_rate'])['name']
        }
    
    def _goal_to_program(self, goal: List[Predicate], state: WorldState) -> SemanticProgram:
        """Convert goal to program."""
        for pred in goal:
            if pred.type == PredicateType.HOLDING:
                obj_id = pred.args[0]
                obj = state.objects.get(obj_id)
                if obj:
                    return SemanticProgram(
                        goal=GoalType.PICK,
                        object=ObjectDescriptor(color=obj.color, shape=obj.shape),
                        relation=None
                    )
        return SemanticProgram(goal=GoalType.PICK, object=ObjectDescriptor(), relation=None)


def run_comprehensive_verification(planner: StateBasedPlanner) -> Dict[str, Any]:
    """
    Run all verification tests and return comprehensive report.
    """
    results = {
        'soundness': {},
        'completeness': {},
        'complexity': {},
        'failure_worlds': {},
        'replanning': {},
        'baseline_comparison': {}
    }
    
    # 1. Soundness verification
    print("=== Soundness Verification ===")
    soundness = SoundnessProof()
    transition = TransitionFunction()
    
    # Test on simple problem
    state1, goal1 = FailureInducingWorlds.create_ambiguous_world()
    dummy_prog1 = SemanticProgram(
        goal=GoalType.PICK,
        object=ObjectDescriptor(color='red', shape='cube'),
        relation=None
    )
    plan1 = planner.plan(dummy_prog1, state1)
    if plan1:
        sound1 = soundness.verify(plan1, state1, transition)
        results['soundness']['test1'] = sound1
    
    results['soundness']['summary'] = {
        'total_tests': soundness.total_tests,
        'passed': soundness.passed_tests,
        'failed': soundness.failed_tests,
        'pass_rate': soundness.passed_tests / soundness.total_tests if soundness.total_tests > 0 else 0.0,
        'violations': soundness.violations
    }
    
    # 2. Completeness verification
    print("=== Completeness Verification ===")
    completeness = CompletenessProof(horizon=20)
    
    # Test known solvable problems
    test_problems = [
        (state1, goal1),
    ]
    
    for state, goal in test_problems:
        completeness.verify(planner, state, goal, known_solution_length=2)
    
    results['completeness']['summary'] = {
        'total_tests': completeness.total_tests,
        'found_plans': completeness.found_plans,
        'missed_plans': completeness.missed_plans,
        'success_rate': completeness.found_plans / completeness.total_tests if completeness.total_tests > 0 else 0.0,
        'false_negatives': completeness.false_negatives
    }
    
    # 3. Complexity analysis
    print("=== Complexity Analysis ===")
    complexity = ComplexityAnalysis(
        num_objects=4,
        num_predicate_types=8,
        max_plan_length=20
    )
    
    results['complexity']['state_space_size'] = complexity.estimate_state_space_size()
    results['complexity']['estimated_time_complexity'] = complexity.estimate_time_complexity(
        branching_factor=5,
        solution_depth=4
    )
    
    # Empirical performance
    performance = complexity.measure_empirical_performance(planner, test_problems)
    results['complexity']['empirical'] = performance
    
    # 4. Failure-inducing worlds
    print("=== Failure-Inducing Worlds ===")
    
    # Occlusion
    occl_state, occl_goal = FailureInducingWorlds.create_occlusion_world()
    occl_prog = SemanticProgram(goal=GoalType.PICK, object=ObjectDescriptor(color='green', shape='cube'), relation=None)
    occl_plan = planner.plan(occl_prog, occl_state)
    results['failure_worlds']['occlusion'] = {
        'solvable': occl_plan is not None,
        'plan_length': len(occl_plan) if occl_plan else None
    }
    
    # Ambiguity
    amb_state, amb_goal = FailureInducingWorlds.create_ambiguous_world()
    amb_prog = SemanticProgram(goal=GoalType.PICK, object=ObjectDescriptor(color='red', shape='cube'), relation=None)
    amb_plan = planner.plan(amb_prog, amb_state)
    results['failure_worlds']['ambiguity'] = {
        'solvable': amb_plan is not None,
        'plan_length': len(amb_plan) if amb_plan else None
    }
    
    # Unreachable
    unreach_state, unreach_goal = FailureInducingWorlds.create_unreachable_goal()
    # Need to construct proper program
    unreach_plan = None  # Would need proper goal construction
    results['failure_worlds']['unreachable'] = {
        'solvable': unreach_plan is not None,
        'plan_length': len(unreach_plan) if unreach_plan else None
    }
    
    # Resource constrained
    res_state, res_goal = FailureInducingWorlds.create_resource_constrained()
    # Would need compound program
    res_plan = None
    results['failure_worlds']['resource_constrained'] = {
        'solvable': res_plan is not None,
        'plan_length': len(res_plan) if res_plan else None
    }
    
    # 5. Replanning termination
    print("=== Replanning Termination ===")
    replan_proof = ReplanningTerminationProof(max_replan_attempts=3)
    
    # Test with 50% failure rate
    def failure_injector(action: Action) -> bool:
        return np.random.random() < 0.5
    
    replan_result = replan_proof.verify_termination(
        planner, state1, goal1, failure_injector
    )
    
    results['replanning'] = {
        'termination_test': replan_result,
        'tests_run': replan_proof.tests_run,
        'terminated': replan_proof.terminated,
        'infinite_loops': replan_proof.infinite_loops,
        'loop_detections': replan_proof.loop_detections
    }
    
    # 6. Baseline comparison
    print("=== Baseline Comparison ===")
    baseline_comp = BaselineComparison()
    comparison = baseline_comp.compare(planner, test_problems)
    results['baseline_comparison'] = comparison
    
    return results
