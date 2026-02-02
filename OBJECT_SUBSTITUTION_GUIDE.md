# Object Substitution Guide

This guide explains how to substitute current block objects with alternative shapes and models (e.g., cup, bottle, custom meshes) in the VLA pipeline simulation.

## Overview

The VLA pipeline simulation is built on PyBullet and supports multiple object types through the `SimulationEnvironment.add_object()` method. Objects can be basic geometric primitives or custom URDF/mesh files.

## Current Object Support

### Built-in Shapes

The simulation currently supports these primitive shapes:

| Shape Type | Aliases | Description |
|------------|---------|-------------|
| **Cube** | `'cube'`, `'box'`, `'block'` | Rectangular box with equal sides |
| **Sphere** | `'sphere'`, `'ball'` | Perfect sphere |
| **Cylinder** | `'cylinder'`, `'can'` | Cylindrical shape |

### Available Colors

- Red, Blue, Green, Yellow, Orange, Purple

### Default Parameters

- **Size**: 0.05 meters (5cm)
- **Mass**: 0.1 kg
- **Position**: `[x, y, z]` where z=0.05 for table-resting objects

## Adding New Primitive Shapes

### Step 1: Extend the add_object() method

Edit `/src/vla_pipeline/simulation/environment.py` and add your shape to the `add_object()` method:

```python
def add_object(
    self,
    shape: str,
    color: str,
    position: Tuple[float, float, float],
    size: float = 0.05
) -> int:
    # ... existing color mapping ...
    
    # Add your new shape here
    if shape in ['cube', 'box', 'block']:
        # existing cube code...
    elif shape in ['sphere', 'ball']:
        # existing sphere code...
    elif shape in ['cylinder', 'can']:
        # existing cylinder code...
    elif shape in ['cone', 'pyramid']:
        # NEW: Add cone shape
        visual_shape = p.createVisualShape(
            p.GEOM_MESH,
            fileName="path/to/cone.obj",  # or use createVisualShape with GEOM parameters
            rgbaColor=rgba,
            meshScale=[size, size, size]
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_MESH,
            fileName="path/to/cone.obj",
            meshScale=[size, size, size]
        )
    # ... rest of method
```

### Example: Adding a Cup Shape

```python
elif shape in ['cup', 'mug']:
    # Cup as a cylinder with a handle (simplified)
    cup_radius = size / 2
    cup_height = size * 1.5
    
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=cup_radius,
        length=cup_height,
        rgbaColor=rgba
    )
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=cup_radius,
        height=cup_height
    )
```

### Example: Adding a Bottle Shape

```python
elif shape in ['bottle']:
    # Bottle as a scaled cylinder (taller and thinner)
    bottle_radius = size / 3
    bottle_height = size * 2.5
    
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=bottle_radius,
        length=bottle_height,
        rgbaColor=rgba
    )
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=bottle_radius,
        height=bottle_height
    )
```

## Using Custom 3D Models

### Method 1: URDF Files

For complex objects, use URDF format:

1. **Create a URDF file** (e.g., `assets/cup.urdf`):

```xml
<?xml version="1.0"?>
<robot name="cup">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="cup.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="cup_collision.obj" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>
```

2. **Add a method to load URDF objects**:

```python
def add_urdf_object(
    self,
    urdf_path: str,
    position: Tuple[float, float, float],
    orientation: Optional[Tuple[float, float, float, float]] = None,
    scale: float = 1.0
) -> int:
    """Load object from URDF file."""
    if orientation is None:
        orientation = [0, 0, 0, 1]
    
    object_id = p.loadURDF(
        urdf_path,
        basePosition=position,
        baseOrientation=orientation,
        globalScaling=scale
    )
    
    return object_id
```

3. **Use in scene configuration**:

```python
scene_config = {
    'objects': [
        {
            'type': 'urdf',
            'path': 'assets/cup.urdf',
            'position': [0.3, 0.0, 0.05],
            'scale': 1.0
        }
    ]
}
```

### Method 2: Mesh Files (OBJ, STL)

Load custom meshes directly:

```python
def add_mesh_object(
    self,
    mesh_path: str,
    color: str,
    position: Tuple[float, float, float],
    scale: float = 1.0
) -> int:
    """Add object from mesh file."""
    color_map = {
        'red': [1, 0, 0, 1],
        'blue': [0, 0, 1, 1],
        # ... other colors
    }
    rgba = color_map.get(color, [0.5, 0.5, 0.5, 1])
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        rgbaColor=rgba,
        meshScale=[scale, scale, scale]
    )
    
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[scale, scale, scale]
    )
    
    object_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    return object_id
```

## Scene Configuration Examples

### Example 1: Mixed Primitives and Custom Objects

```python
scene_config = {
    'objects': [
        # Standard primitives
        {'shape': 'cube', 'color': 'red', 'position': [0.3, 0.0, 0.05], 'size': 0.05},
        {'shape': 'sphere', 'color': 'blue', 'position': [0.3, 0.15, 0.05], 'size': 0.05},
        
        # Custom shapes (after implementing them)
        {'shape': 'cup', 'color': 'white', 'position': [0.3, -0.15, 0.05], 'size': 0.06},
        {'shape': 'bottle', 'color': 'green', 'position': [0.4, 0.0, 0.05], 'size': 0.05},
    ]
}
```

### Example 2: Using URDF Models

After implementing `add_urdf_object()` and updating the pipeline:

```python
scene_config = {
    'objects': [
        {
            'type': 'urdf',
            'path': 'assets/models/coffee_cup.urdf',
            'position': [0.3, 0.0, 0.05],
            'scale': 1.0,
            'name': 'coffee cup'
        },
        {
            'type': 'urdf', 
            'path': 'assets/models/water_bottle.urdf',
            'position': [0.3, 0.15, 0.05],
            'scale': 0.8,
            'name': 'water bottle'
        }
    ]
}
```

## Updating Language Understanding

When adding new object types, update the language parser to recognize them:

Edit `/src/vla_pipeline/language/parser.py`:

```python
class LanguageParser:
    def __init__(self):
        # Add new object types to recognition
        self.object_shapes = [
            'cube', 'box', 'block',
            'sphere', 'ball',
            'cylinder', 'can',
            'cup', 'mug',        # NEW
            'bottle',            # NEW
            # ... add more
        ]
```

## Best Practices

### 1. Object Placement Z-Coordinates

- **Table surface**: z = 0.0
- **Objects resting on table**: z = object_height/2 (e.g., 0.05 for 0.1m tall objects)
- **Trays on table**: z = 0.0 (trays auto-adjust to sit flush)

### 2. Object Sizing

- Keep objects between 0.03m - 0.1m for realistic manipulation
- Ensure objects aren't too heavy (0.05 - 0.5 kg) for the robot arm
- Scale custom meshes appropriately (typically 0.001 for mm-scale models)

### 3. Collision Shapes

- Use simplified collision shapes for performance
- Separate visual and collision meshes when possible
- Convex hulls work well for most objects

### 4. Mass and Inertia

- Objects that are too light will slide/bounce excessively
- Objects that are too heavy may be hard to grasp
- Default mass of 0.1 kg works well for small objects

## Testing New Objects

1. **Visual Test**: Run simulation with GUI to verify appearance
2. **Physics Test**: Drop object from height to check stability
3. **Grasp Test**: Attempt pick-and-place with the object
4. **Perception Test**: Ensure object is detected correctly

### Quick Test Script

```python
from vla_pipeline import VLAPipeline

pipeline = VLAPipeline(use_gui=True)
scene_config = {
    'objects': [
        {'shape': 'YOUR_NEW_SHAPE', 'color': 'red', 'position': [0.3, 0.0, 0.05]}
    ]
}
pipeline.setup_scene(scene_config)
input("Check object appearance. Press Enter to continue...")
pipeline.close()
```

## Common Issues and Solutions

### Issue: Object floats above table
**Solution**: Set z-position to half the object's height

### Issue: Object falls through table
**Solution**: Ensure collision shape is defined properly

### Issue: Object is invisible
**Solution**: Check that visual shape is created with valid parameters

### Issue: Object is too large/small
**Solution**: Adjust `size` parameter or mesh `scale` factor

### Issue: Language parser doesn't recognize new object
**Solution**: Add object name to parser's recognized shapes list

## Resources

- **PyBullet Documentation**: https://pybullet.org/
- **URDF Format**: http://wiki.ros.org/urdf/XML
- **3D Model Repositories**:
  - Thingiverse: https://www.thingiverse.com/
  - GrabCAD: https://grabcad.com/
  - Shapenet: https://shapenet.org/

## Summary

To add new objects to the simulation:

1. Extend `add_object()` method for primitives, or add `add_urdf_object()` for complex models
2. Update scene configuration format to support new object types
3. Update language parser to recognize new object names
4. Test thoroughly with visual and physics validation
5. Document any special handling requirements

The existing VLA pipeline logic (perception, planning, control) will continue to work without modification as long as objects are properly added to the simulation and have valid poses.
