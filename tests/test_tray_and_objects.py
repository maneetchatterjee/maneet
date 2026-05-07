#!/usr/bin/env python3
"""
Test: Tray Placement and Object Support

Validates that trays sit flush on the table surface and alternate object shapes work.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vla_pipeline.simulation.environment import SimulationEnvironment


def test_tray_positioning():
    """Test that trays are positioned correctly on the table surface."""
    print("\n" + "="*60)
    print("TEST: Tray Positioning")
    print("="*60)
    
    # Create simulation environment (headless for testing)
    env = SimulationEnvironment(use_gui=False)
    
    # Add a tray
    tray_id = env.add_tray(
        position=(0.0, 0.0, 0.0),
        size=(0.2, 0.2),
        height=0.005,
        color='red',
        label='Test Tray'
    )
    
    # Step simulation to settle
    env.step(num_steps=100)
    
    # Get tray position
    tray_pos, _ = env.get_object_pose(tray_id)
    
    print(f"  Tray position: {tray_pos}")
    print(f"  Expected z-coordinate: ~0.0025 (height/2)")
    print(f"  Actual z-coordinate: {tray_pos[2]:.6f}")
    
    # Verify tray is close to table surface
    expected_z = 0.005 / 2  # height/2
    tolerance = 0.001  # 1mm tolerance
    
    if abs(tray_pos[2] - expected_z) < tolerance:
        print("  ✓ PASS: Tray is positioned flush on table surface")
        result = True
    else:
        print(f"  ✗ FAIL: Tray z-position is off by {abs(tray_pos[2] - expected_z)*1000:.2f}mm")
        result = False
    
    # Step additional simulation cycles to verify tray remains stable
    env.step(num_steps=200)
    tray_pos_after, _ = env.get_object_pose(tray_id)
    
    # Verify tray hasn't moved significantly (within 0.5mm)
    position_drift = abs(tray_pos_after[2] - tray_pos[2])
    if position_drift < 0.0005:
        print(f"  ✓ PASS: Tray remains stable after physics simulation (drift: {position_drift*1000:.2f}mm)")
    else:
        print(f"  ✗ FAIL: Tray moved significantly after simulation (drift: {position_drift*1000:.2f}mm)")
        result = False
    
    # Check tray is in objects dict
    if tray_id in env.objects:
        print("  ✓ PASS: Tray is registered in objects dictionary")
        print(f"    Tray info: {env.objects[tray_id]}")
    else:
        print("  ✗ FAIL: Tray not found in objects dictionary")
        result = False
    
    env.close()
    return result


def test_alternate_shapes():
    """Test that alternate object shapes (cup, bottle) can be created."""
    print("\n" + "="*60)
    print("TEST: Alternate Object Shapes")
    print("="*60)
    
    # Create simulation environment (headless for testing)
    env = SimulationEnvironment(use_gui=False)
    
    results = []
    
    # Test cup shape
    try:
        cup_id = env.add_object(
            shape='cup',
            color='blue',
            position=(0.2, 0.0, 0.075),
            size=0.05
        )
        env.step(num_steps=50)
        cup_pos, _ = env.get_object_pose(cup_id)
        print(f"  ✓ Cup created successfully at position {cup_pos}")
        results.append(True)
    except Exception as e:
        print(f"  ✗ Cup creation failed: {e}")
        results.append(False)
    
    # Test bottle shape
    try:
        bottle_id = env.add_object(
            shape='bottle',
            color='green',
            position=(0.3, 0.0, 0.0625),
            size=0.05
        )
        env.step(num_steps=50)
        bottle_pos, _ = env.get_object_pose(bottle_id)
        print(f"  ✓ Bottle created successfully at position {bottle_pos}")
        results.append(True)
    except Exception as e:
        print(f"  ✗ Bottle creation failed: {e}")
        results.append(False)
    
    # Test traditional shapes still work
    try:
        cube_id = env.add_object(
            shape='cube',
            color='red',
            position=(0.4, 0.0, 0.05),
            size=0.05
        )
        env.step(num_steps=50)
        cube_pos, _ = env.get_object_pose(cube_id)
        print(f"  ✓ Cube created successfully at position {cube_pos}")
        results.append(True)
    except Exception as e:
        print(f"  ✗ Cube creation failed: {e}")
        results.append(False)
    
    env.close()
    return all(results)


def test_mixed_scene():
    """Test a scene with both trays and various object types."""
    print("\n" + "="*60)
    print("TEST: Mixed Scene with Trays and Objects")
    print("="*60)
    
    # Create simulation environment (headless for testing)
    env = SimulationEnvironment(use_gui=False)
    
    # Add trays
    tray1_id = env.add_tray(
        position=(-0.1, -0.2, 0.0),
        size=(0.15, 0.15),
        color='red'
    )
    tray2_id = env.add_tray(
        position=(-0.1, 0.2, 0.0),
        size=(0.15, 0.15),
        color='blue'
    )
    
    # Add various objects with calculated z-positions
    # For objects resting on table, z = object_height / 2
    cube_size = 0.05
    cube_id = env.add_object('cube', 'red', (0.3, 0.0, cube_size/2), cube_size)
    
    sphere_size = 0.05
    sphere_id = env.add_object('sphere', 'blue', (0.3, 0.15, sphere_size/2), sphere_size)
    
    cup_size = 0.05
    cup_height = cup_size * 1.5  # Cups are 1.5x taller
    cup_id = env.add_object('cup', 'yellow', (0.3, -0.15, cup_height/2), cup_size)
    
    bottle_size = 0.05
    bottle_height = bottle_size * 2.5  # Bottles are 2.5x taller
    bottle_id = env.add_object('bottle', 'green', (0.4, 0.0, bottle_height/2), bottle_size)
    
    # Step simulation
    env.step(num_steps=100)
    
    # Verify all objects exist and have reasonable positions
    all_ids = [tray1_id, tray2_id, cube_id, sphere_id, cup_id, bottle_id]
    success = True
    
    for obj_id in all_ids:
        if obj_id not in env.objects:
            print(f"  ✗ FAIL: Object {obj_id} not found in scene")
            success = False
            continue
        
        pos, _ = env.get_object_pose(obj_id)
        obj_info = env.objects[obj_id]
        
        # Check z-coordinate is positive (above table)
        if pos[2] < 0:
            print(f"  ✗ FAIL: {obj_info['name']} has negative z-coordinate: {pos[2]}")
            success = False
        else:
            print(f"  ✓ {obj_info['name']} at position {pos}")
    
    if success:
        print("\n  ✓ PASS: All objects positioned correctly")
    
    env.close()
    return success


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRAY AND ALTERNATE OBJECT TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Tray Positioning", test_tray_positioning()))
    results.append(("Alternate Shapes", test_alternate_shapes()))
    results.append(("Mixed Scene", test_mixed_scene()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
