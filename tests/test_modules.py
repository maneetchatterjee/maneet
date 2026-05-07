"""
Unit tests for VLA Pipeline modules.

Run with: python -m pytest tests/ -v
Or simply: python tests/test_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vla_pipeline.perception import PerceptionModule, Object3D
from vla_pipeline.language import LanguageReasoningModule, ActionType, SpatialRelation
from vla_pipeline.planning import PlanningModule
from vla_pipeline.control import KinematicsController
from vla_pipeline.utils import MetricsLogger, Timer


def test_language_parsing():
    """Test language command parsing."""
    print("\nTest: Language Parsing")
    language = LanguageReasoningModule()
    
    # Test pick command
    parsed = language.parse_command("Pick the red cube")
    assert parsed.action == ActionType.PICK
    assert parsed.target_color == "red"
    assert parsed.target_shape == "cube"
    print("  ✓ Pick command parsed correctly")
    
    # Test place command with spatial relation
    parsed = language.parse_command("Place the blue sphere left of the green cube")
    assert parsed.action == ActionType.PLACE
    assert parsed.target_color == "blue"
    assert parsed.destination_color == "green"
    assert parsed.spatial_relation == SpatialRelation.LEFT_OF
    print("  ✓ Place command with spatial relation parsed correctly")
    
    # Test compound command
    commands = language.get_action_sequence("Pick the red cube and place it on the blue cube")
    assert len(commands) >= 1
    print("  ✓ Compound command parsed correctly")


def test_perception():
    """Test perception module."""
    print("\nTest: Perception")
    perception = PerceptionModule()
    
    # Create test image with colored regions
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_image[100:200, 100:200] = [255, 0, 0]  # Red region
    rgb_image[300:400, 300:400] = [0, 0, 255]  # Blue region
    
    objects = perception.detect_objects(rgb_image)
    assert len(objects) == 2
    assert any(obj.color == "red" for obj in objects)
    assert any(obj.color == "blue" for obj in objects)
    print(f"  ✓ Detected {len(objects)} objects correctly")
    
    # Test JSON export
    json_str = perception.to_json(objects)
    assert "red" in json_str
    print("  ✓ JSON export works")


def test_planning():
    """Test planning module."""
    print("\nTest: Planning")
    planning = PlanningModule()
    
    # Create test objects
    objects = [
        Object3D(0, "red_cube", "red", "cube", (0.3, 0.0, 0.05), (0, 0, 0, 1), (0.05, 0.05, 0.05)),
        Object3D(1, "blue_cube", "blue", "cube", (0.3, 0.15, 0.05), (0, 0, 0, 1), (0.05, 0.05, 0.05)),
    ]
    
    # Test action planning
    language = LanguageReasoningModule()
    parsed = language.parse_command("Pick the red cube")
    
    action = planning.plan_action(parsed, objects)
    assert action.action_type == ActionType.PICK
    assert action.target_object_id == 0
    assert len(action.waypoints) > 0
    print(f"  ✓ Generated {len(action.waypoints)} waypoints for pick action")
    
    # Test pick and place
    parsed = language.parse_command("Pick the red cube and place it left of the blue cube")
    actions = planning.plan_pick_and_place(parsed, objects)
    assert len(actions) == 2
    assert actions[0].action_type == ActionType.PICK
    assert actions[1].action_type == ActionType.PLACE
    print("  ✓ Pick-and-place sequence generated correctly")


def test_control():
    """Test control module."""
    print("\nTest: Control")
    controller = KinematicsController()
    
    # Test forward kinematics
    joint_angles = np.zeros(6)
    pos, orn = controller.forward_kinematics(joint_angles)
    assert pos.shape == (3,)
    print("  ✓ Forward kinematics works")
    
    # Test inverse kinematics
    target_pos = (0.3, 0.0, 0.1)
    target_orn = (0, 0, 0, 1)
    joint_solution = controller.inverse_kinematics(target_pos, target_orn)
    assert joint_solution.shape == (6,)
    print("  ✓ Inverse kinematics works")
    
    # Test trajectory interpolation
    from vla_pipeline.planning import Waypoint, MotionPhase
    waypoints = [
        Waypoint((0.3, 0.0, 0.1), (0, 0, 0, 1), 1.0, MotionPhase.APPROACH),
        Waypoint((0.3, 0.0, 0.05), (0, 0, 0, 1), 1.0, MotionPhase.GRASP),
    ]
    trajectory = controller.interpolate_trajectory(waypoints, np.zeros(6), num_steps=10)
    assert len(trajectory) > 0
    print(f"  ✓ Trajectory interpolation generated {len(trajectory)} steps")


def test_metrics():
    """Test metrics logging."""
    print("\nTest: Metrics")
    logger = MetricsLogger()
    
    # Log some executions
    logger.log_execution("task_1", "Pick cube", True, 1.5, 10, 2)
    logger.log_execution("task_2", "Place cube", False, 2.0, 8, 1, "ik_failed")
    logger.log_execution("task_3", "Move cube", True, 1.2, 12, 2)
    
    # Check metrics
    success_rate = logger.get_success_rate()
    assert 0.6 <= success_rate <= 0.7  # 2/3 successes
    print(f"  ✓ Success rate: {success_rate*100:.1f}%")
    
    avg_time = logger.get_average_execution_time()
    assert avg_time > 0
    print(f"  ✓ Average execution time: {avg_time:.2f}s")
    
    failure_modes = logger.get_failure_modes()
    assert "ik_failed" in failure_modes
    print("  ✓ Failure modes tracked correctly")


def test_timer():
    """Test timer utility."""
    print("\nTest: Timer")
    import time
    
    with Timer() as t:
        time.sleep(0.01)
    
    assert t.elapsed >= 0.01
    print(f"  ✓ Timer measured {t.elapsed:.3f}s")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("VLA Pipeline Unit Tests")
    print("="*60)
    
    tests = [
        test_language_parsing,
        test_perception,
        test_planning,
        test_control,
        test_metrics,
        test_timer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
