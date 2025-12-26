# Franka Panda Pick and Place Simulation

This demo showcases a PyBullet simulation of a Franka Panda robotic arm performing pick and place tasks.

## Overview

The `demo_franka_panda_pick_place.py` script demonstrates:
- Loading the Franka Panda robot model in PyBullet
- Inverse kinematics for position control
- Gripper control for grasping objects
- Multiple pick and place tasks including stacking

## Features

- **Robot**: Franka Emika Panda (7-DOF arm + 2-DOF gripper)
- **Tasks**: 
  1. Pick red cube and move it left
  2. Pick blue cube and move it right
  3. Pick green sphere and move it to center
  4. Stack red cube on top of blue cube
- **Physics**: Full PyBullet physics simulation with gravity and collisions
- **Control**: Inverse kinematics-based position control

## Requirements

- Python >= 3.8
- PyBullet >= 3.2.5
- NumPy >= 1.21.0

## Usage

### Run with GUI (Visual Simulation)

```bash
python demo/demo_franka_panda_pick_place.py
```

This will open a PyBullet GUI window showing the simulation in real-time.

### Run Headless (No GUI)

Modify the script to use headless mode:
```python
sim = FrankaPandaSimulation(use_gui=False)
```

## How It Works

### 1. Initialization
- Connects to PyBullet physics engine
- Loads ground plane and Franka Panda robot
- Sets up camera view and gravity

### 2. Robot Control
- **Home Position**: Predefined safe starting configuration
- **Inverse Kinematics**: Converts target end-effector positions to joint angles
- **Gripper Control**: Opens (1.0) and closes (0.0) gripper fingers

### 3. Pick and Place Sequence
For each object:
1. Move above pick position
2. Open gripper
3. Move down to object
4. Close gripper (grasp)
5. Lift object
6. Move to place position
7. Lower object
8. Open gripper (release)
9. Move up

### 4. Physics Simulation
- Time step: 1/240 seconds
- Gravity: -9.81 m/s²
- Object mass: 0.1 kg each
- Realistic collision detection

## Franka Panda Robot Details

- **Arm Joints**: 7 revolute joints (panda_joint1 through panda_joint7)
- **Gripper**: 2 prismatic finger joints (panda_finger_joint1, panda_finger_joint2)
- **Gripper Range**: 0.0 m (closed) to 0.04 m (open) per finger
- **End-Effector**: Link 8 (panda_hand)

## Customization

### Add More Objects

```python
# Add a yellow cube
yellow_cube = sim.add_object(
    'cube',
    [1, 1, 0, 1],  # Yellow color (RGBA)
    [0.4, 0.1, 0.05],  # Position
    size=0.05
)
```

### Modify Pick and Place

```python
# Custom pick and place
sim.pick_and_place(
    pick_pos=[x1, y1, z1],
    place_pos=[x2, y2, z2],
    lift_height=0.25  # Height during transport
)
```

### Change Robot Speed

Modify the simulation steps in `move_to_position()`:
```python
for _ in range(120):  # Faster (0.5 seconds)
    p.stepSimulation()
    time.sleep(1/240.0)
```

## Troubleshooting

### Robot Not Moving
- Check that joint indices are correct (0-6 for arm)
- Verify IK solution is valid
- Ensure target position is within workspace

### Objects Falling Through Floor
- Increase simulation settling time
- Check object mass and collision shapes
- Verify gravity is set correctly

### Gripper Not Grasping
- Adjust gripper force (default: 20)
- Check object size matches gripper width
- Ensure gripper is properly aligned

## Technical Notes

### Workspace Limits
- X: 0.2 to 0.8 m (forward/back)
- Y: -0.4 to 0.4 m (left/right)
- Z: 0.0 to 0.6 m (up/down)

### IK Parameters
- Max iterations: 100
- Residual threshold: 1e-5
- Default orientation: Downward (π, 0, 0)

### Performance
- Simulation rate: 240 Hz
- Real-time factor: ~1.0x (depends on hardware)
- Typical pick-place time: 8-10 seconds

## Future Enhancements

- [ ] Add obstacle avoidance
- [ ] Implement trajectory smoothing
- [ ] Add force/torque sensing
- [ ] Support for dual-arm manipulation
- [ ] Integration with computer vision
- [ ] Real-to-sim transfer learning

## References

- [Franka Emika Panda Documentation](https://frankaemika.github.io/)
- [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)
- [Inverse Kinematics in PyBullet](https://pybullet.org/Bullet/BulletFull/classbtIKSolver.html)

## License

MIT License - See main repository LICENSE for details.
