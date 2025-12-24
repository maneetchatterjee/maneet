# VLA Pipeline - Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Test

Test that all modules work:

```bash
python tests/test_modules.py
```

Expected output: All 6 tests should pass.

## Running Demos

### Basic Demo (Headless Mode)

The basic demo can run without GUI:

```python
from vla_pipeline import VLAPipeline

# Initialize pipeline (use_gui=False for headless)
pipeline = VLAPipeline(use_gui=False, log_metrics=True)

# Setup scene
scene_config = {
    'objects': [
        {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
        {'shape': 'cube', 'color': 'blue', 'position': [0.3, 0.15, 0.05], 'size': 0.05},
    ]
}
pipeline.setup_scene(scene_config)

# Execute command
pipeline.execute_command("Pick the red cube and place it left of the blue cube")

# View results
pipeline.print_metrics_summary()
pipeline.close()
```

### Interactive Demo with GUI

```bash
# Note: Requires display/X server
python demo/demo_basic.py
```

## Architecture Overview

The VLA Pipeline consists of 5 main modules:

1. **Perception** - Detects objects using computer vision
2. **Language** - Parses natural language commands
3. **Planning** - Generates collision-safe waypoints
4. **Control** - Executes trajectories using IK/FK
5. **Simulation** - PyBullet physics environment

## Module Examples

### Language Module

```python
from vla_pipeline.language import LanguageReasoningModule

language = LanguageReasoningModule()
parsed = language.parse_command("Pick the red cube and place it on the blue cube")

print(f"Action: {parsed.action.value}")
print(f"Target: {parsed.target_object}")
print(f"Spatial Relation: {parsed.spatial_relation.value}")
```

### Perception Module

```python
from vla_pipeline.perception import PerceptionModule
import numpy as np

perception = PerceptionModule()

# Get image from simulation or camera
rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
# ... populate image ...

objects = perception.detect_objects(rgb_image)
for obj in objects:
    print(f"{obj.color} {obj.shape} at {obj.position}")
```

### Planning Module

```python
from vla_pipeline.planning import PlanningModule

planner = PlanningModule(
    safety_margin=0.05,  # Collision avoidance
    lift_height=0.15     # Transport height
)

actions = planner.plan_pick_and_place(parsed_command, scene_objects)
```

### Control Module

```python
from vla_pipeline.control import KinematicsController

controller = KinematicsController()

# Inverse kinematics
joint_angles = controller.inverse_kinematics(
    target_position=(0.3, 0.0, 0.1),
    target_orientation=(0, 0, 0, 1)
)

# Forward kinematics
position, orientation = controller.forward_kinematics(joint_angles)
```

## Supported Commands

### Actions
- Pick: "pick", "grab", "grasp", "take"
- Place: "place", "put", "set", "drop"
- Move: "move", "bring", "carry"

### Object Properties
- Colors: red, blue, green, yellow, orange, purple
- Shapes: cube, box, block, sphere, ball, cylinder, can

### Spatial Relations
- "left of", "right of"
- "in front of", "behind"
- "on", "above", "on top of"
- "next to", "beside", "near"

### Example Commands

```
"Pick the red cube"
"Place the blue sphere left of the green cube"
"Grab the yellow cylinder and put it on the red box"
"Take the green ball and place it next to the blue cube"
```

## Metrics and Logging

The pipeline automatically tracks:

- Success rate
- Execution time per task
- Number of waypoints/actions
- Failure modes

Access metrics:

```python
pipeline.print_metrics_summary()
pipeline.save_metrics("my_metrics.json")
```

## Configuration

### Workspace Bounds

```python
planner = PlanningModule()
planner.workspace_bounds = {
    'x': (-0.5, 0.5),
    'y': (-0.5, 0.5),
    'z': (0.0, 0.5)
}
```

### Custom Robot

```python
controller = KinematicsController(
    link_lengths=[0.1, 0.2, 0.2, 0.15, 0.1, 0.05],
    joint_limits=[(-3.14, 3.14)] * 6
)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install numpy scipy opencv-python-headless pybullet matplotlib pyyaml
```

### Display Issues

For headless environments (servers, containers):

```python
# Use headless mode
pipeline = VLAPipeline(use_gui=False)
```

### PyBullet Issues

If PyBullet has issues, try:

```bash
pip install --upgrade pybullet
```

## Development

### Running Tests

```bash
python tests/test_modules.py
```

### Adding New Objects

Edit color ranges in `perception/detector.py`:

```python
self.color_ranges = {
    'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
    'purple': {'lower': np.array([140, 100, 100]), 'upper': np.array([160, 255, 255])},
}
```

### Adding New Actions

1. Add to `ActionType` enum in `language/parser.py`
2. Add parsing rules in `LanguageReasoningModule`
3. Add planning logic in `PlanningModule`

## Documentation

- [README.md](../README.md) - Full documentation
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Detailed architecture
- [architecture_diagram.png](../docs/architecture_diagram.png) - Visual diagram

## Support

For issues or questions, please file an issue on GitHub.

## License

MIT License - see LICENSE file for details.
