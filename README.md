# Vision-Language-Action (VLA) Pipeline for Robotic Manipulation

A modular, production-grade pipeline for robotic manipulation in simulation that integrates computer vision, natural language understanding, symbolic planning, and control.

## 🎯 Overview

This VLA pipeline enables robots to understand and execute natural language commands for pick-and-place tasks in simulation. The system is designed with a clean, modular architecture that separates perception, reasoning, planning, and control.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VLA Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐    ┌──────────┐    ┌──────────┐            │
│  │ Perception │───▶│ Language │───▶│ Planning │            │
│  │  Module    │    │  Module  │    │  Module  │            │
│  └────────────┘    └──────────┘    └──────────┘            │
│        │                                  │                  │
│        │                                  ▼                  │
│        │                          ┌──────────┐              │
│        │                          │ Control  │              │
│        │                          │  Module  │              │
│        │                          └──────────┘              │
│        │                                  │                  │
│        └──────────────────────────────────┼─────────────┐  │
│                                           ▼              │  │
│                                   ┌──────────────┐      │  │
│                                   │ Simulation   │      │  │
│                                   │ Environment  │      │  │
│                                   └──────────────┘      │  │
│                                           │              │  │
│                                           └──────────────┘  │
│                                      (Feedback Loop)        │
└─────────────────────────────────────────────────────────────┘
```

### Module Descriptions

1. **Perception Module** (`perception/`)
   - Computer vision-based object detection
   - Color and shape recognition
   - 3D pose estimation from RGB-D images
   - Structured scene representation (JSON output)

2. **Language Module** (`language/`)
   - Natural language command parsing
   - Intent extraction (pick, place, move)
   - Object and spatial relation identification
   - Decoupled from control logic

3. **Planning Module** (`planning/`)
   - Symbolic action planning
   - Collision-safe waypoint generation
   - Spatial relation resolution
   - Pick-and-place sequence generation

4. **Control Module** (`control/`)
   - Custom inverse kinematics (IK)
   - Forward kinematics (FK)
   - Smooth trajectory interpolation
   - Joint-level control

5. **Simulation Module** (`simulation/`)
   - PyBullet-based physics simulation
   - Robot and scene management
   - Camera interface for perception
   - No hardware dependencies

6. **Utils Module** (`utils/`)
   - Metrics logging (success rate, execution time)
   - Performance benchmarking
   - Failure mode tracking

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PyBullet >= 3.2.5
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0

## 📖 Usage

### Basic Example

```python
from vla_pipeline import VLAPipeline

# Initialize pipeline
pipeline = VLAPipeline(use_gui=True, log_metrics=True)

# Setup scene with objects
scene_config = {
    'objects': [
        {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
        {'shape': 'cube', 'color': 'blue', 'position': [0.3, 0.15, 0.05], 'size': 0.05},
    ]
}
pipeline.setup_scene(scene_config)

# Execute natural language command
pipeline.execute_command("Pick the red cube and place it left of the blue cube")

# View metrics
pipeline.print_metrics_summary()

# Cleanup
pipeline.close()
```

### Running Demos

```bash
# Basic pick-and-place demo
python demo/demo_basic.py

# Complex scenarios (stacking, sorting, spatial relations)
python demo/demo_complex.py

# Franka Panda pick-and-place simulation (PyBullet)
python demo/demo_franka_panda_pick_place.py
```

See [Franka Panda Demo README](demo/README_FRANKA_PANDA.md) for details on the Panda robot simulation.

## 🎮 Supported Commands

The system understands natural language commands with the following patterns:

### Actions
- **Pick**: "pick", "grab", "grasp", "take"
- **Place**: "place", "put", "set", "drop"
- **Move**: "move", "bring", "carry"

### Object Properties
- **Colors**: red, blue, green, yellow, orange, purple
- **Shapes**: cube, box, block, sphere, ball, cylinder, can

### Spatial Relations
- **Positional**: "left of", "right of", "in front of", "behind"
- **Vertical**: "on", "above", "on top of"
- **Proximity**: "next to", "beside", "near"

### Example Commands
```
"Pick the red cube and place it left of the blue cube"
"Grab the green sphere and put it next to the yellow block"
"Take the blue cylinder and place it on the red cube"
```

## 📊 Metrics and Benchmarking

The pipeline automatically logs execution metrics:

- **Success Rate**: Percentage of successfully completed tasks
- **Execution Time**: Average time per task
- **Failure Modes**: Categorized failure reasons
- **Waypoint Count**: Number of waypoints per action

Metrics are saved to `metrics.json` and can be analyzed for performance evaluation.

## 🔧 Configuration

### Workspace Bounds
```python
workspace_bounds = {
    'x': (-0.5, 0.5),
    'y': (-0.5, 0.5),
    'z': (0.0, 0.5)
}
```

### Planning Parameters
```python
planner = PlanningModule(
    safety_margin=0.05,  # Collision avoidance distance
    lift_height=0.15     # Height for object transport
)
```

### Kinematics Configuration
```python
controller = KinematicsController(
    link_lengths=[0.1, 0.2, 0.2, 0.15, 0.1, 0.05],  # Robot geometry
    joint_limits=[(-np.pi, np.pi)] * 6               # Joint limits
)
```

## 🏭 Design Principles

### Modularity
- Each module has a well-defined interface
- Modules can be replaced or upgraded independently
- No circular dependencies

### Extensibility
- Easy to add new object types
- Pluggable perception models
- Extensible command grammar

### Simulation-First
- No hardware dependencies
- Reproducible experiments
- Fast iteration cycles

### Production-Ready
- Comprehensive error handling
- Logging and metrics
- Clean code structure
- Type hints and documentation

## 📐 Architecture Decisions

### Why Separate Modules?

1. **Perception Module**: Decoupling CV from control allows easy upgrade to deep learning models
2. **Language Module**: Rule-based parser can be replaced with LLM without changing downstream
3. **Planning Module**: Symbolic planning enables explainability and debugging
4. **Control Module**: Custom IK/FK allows fine-tuning for specific robots

### Why Not End-to-End?

End-to-end models lack:
- Interpretability (can't debug failures)
- Modularity (can't upgrade parts independently)
- Sample efficiency (need huge datasets)
- Generalization (brittle to new scenarios)

Our approach provides:
- Clear failure modes
- Modular testing
- Explainable decisions
- Easy extension to new tasks

## 🔬 Limitations and Future Work

### Current Limitations
- Color-based perception (not robust to lighting)
- Rule-based language parsing (limited grammar)
- Simplified collision checking
- Single-arm manipulation only

### Future Enhancements
- Deep learning perception (YOLO, PointNet)
- LLM-based language understanding
- Advanced motion planning (RRT, optimization)
- Dual-arm coordination
- Dynamic object manipulation
- Real-world transfer learning

## 📝 File Structure

```
vla_pipeline/
├── src/vla_pipeline/
│   ├── __init__.py
│   ├── pipeline.py          # Main orchestrator
│   ├── perception/          # CV-based object detection
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── language/            # NLP command parsing
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── planning/            # Symbolic action planning
│   │   ├── __init__.py
│   │   └── planner.py
│   ├── control/             # IK/FK controller
│   │   ├── __init__.py
│   │   └── kinematics.py
│   ├── simulation/          # PyBullet environment
│   │   ├── __init__.py
│   │   └── environment.py
│   └── utils/               # Metrics and utilities
│       ├── __init__.py
│       └── metrics.py
├── demo/
│   ├── demo_basic.py        # Basic demonstrations
│   └── demo_complex.py      # Complex scenarios
├── docs/
│   └── ARCHITECTURE.md      # Detailed architecture
├── requirements.txt
├── setup.py
└── README.md
```

## 🤝 Contributing

This is a research-grade implementation. Contributions are welcome for:
- Additional perception models
- Extended language understanding
- New planning algorithms
- Additional demo scenarios
- Performance optimizations

## 📄 License

MIT License - see LICENSE file for details

## 📚 References

- PyBullet: Physics simulation
- OpenCV: Computer vision
- NumPy/SciPy: Numerical computing

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{vla_pipeline,
  title={Vision-Language-Action Pipeline for Robotic Manipulation},
  author={Maneet Chatterjee},
  year={2025},
  url={https://github.com/maneetchatterjee/maneet}
}
```

---

**Built with ❤️ for the robotics research community**