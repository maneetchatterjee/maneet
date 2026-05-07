# Z-Coordinate Positioning Guide

## Visual Reference for Object Placement

This guide shows how objects are positioned on the table surface in the simulation.

## The Problem We Solved

**Before**: Trays were floating above the table due to incorrect z-coordinates.

**After**: Trays sit flush on the table surface by positioning at z = height/2.

## Z-Coordinate System

```
     Z-axis (vertical)
         ↑
         |
    0.15 |                    [Lifted object during manipulation]
         |
    0.10 |        [Tall object]
         |            |
    0.05 |   [Cube]  |  [Sphere]
         |     |      |     O
         |_____|______|_____|________________ Table Surface (z=0)
         |
   -0.01 |
```

## Positioning Formula

For objects resting on the table:
```
z_position = object_height / 2
```

This works because PyBullet positions objects by their **center point**.

## Object Height Reference

| Object Type | Size Parameter | Actual Height | Z-Position |
|-------------|---------------|---------------|------------|
| **Table** | N/A | Plane | 0.0 m |
| **Tray** | 0.005 | 0.005 m | 0.0025 m |
| **Cube** | 0.05 | 0.05 m | 0.025 m |
| **Sphere** | 0.05 | 0.05 m | 0.025 m |
| **Cylinder** | 0.05 | 0.05 m | 0.025 m |
| **Cup** | 0.05 | 0.075 m (1.5×) | 0.0375 m |
| **Bottle** | 0.05 | 0.125 m (2.5×) | 0.0625 m |

## Detailed Visual

```
Side View:
                                    
     0.13 ┤                    Bottle
          │                      ║
     0.10 ┤         Cup          ║
          │          ║           ║
     0.08 ┤          ║           ║
          │    ┌─────┐           ║
     0.05 ┤    │Cube │     O     ║
          │    │     │  Sphere   ║
     0.03 ┤    └─────┘           ║
          │                      ║
     0.00 ┼══[Tray]══════════════════════ Table
          │
    -0.01 ┤

Legend:
  ══ Tray (5mm thick, semi-transparent)
  │  Cube edges
  O  Sphere
  ║  Cylinder (cup/bottle)
  ── Table surface (z=0)
```

## Top View (with Trays)

```
     Y-axis
       ↑
       │
   0.2 │    [Blue Zone]
       │    ┌─────────┐
   0.1 │    │         │    ⊕ Cup
       │    └─────────┘
   0.0 ├────[Red Zone]────────► X-axis
       │    ┌─────────┐
  -0.1 │    │    ■    │
       │    └─────────┘
  -0.2 │        Cube
       │

Legend:
  ┌─┐  Tray outline (15cm × 15cm)
  ■    Cube (top view)
  ⊕    Cup (top view)
```

## Code Examples

### Correct Positioning

```python
# Tray: User specifies z=0.0, code auto-adjusts
tray = {'position': [0.0, 0.0, 0.0], 'height': 0.005}
# Internally adjusted to: z = 0.005/2 = 0.0025

# Cube: Position at half its height
cube = {'shape': 'cube', 'position': [0.3, 0.0, 0.025], 'size': 0.05}
# z = 0.05/2 = 0.025

# Cup: Position at half its height (1.5× taller)
cup = {'shape': 'cup', 'position': [0.3, 0.0, 0.0375], 'size': 0.05}
# z = (0.05 * 1.5)/2 = 0.0375

# Bottle: Position at half its height (2.5× taller)
bottle = {'shape': 'bottle', 'position': [0.3, 0.0, 0.0625], 'size': 0.05}
# z = (0.05 * 2.5)/2 = 0.0625
```

### Common Mistakes (Avoided)

```python
# ❌ WRONG: Tray floating above table
tray = {'position': [0.0, 0.0, 0.05]}  # Too high!

# ❌ WRONG: Object below table
cube = {'position': [0.3, 0.0, 0.0]}  # Will sink!

# ❌ WRONG: Object too high
sphere = {'position': [0.3, 0.0, 0.1]}  # Floating!

# ✅ CORRECT: Objects at height/2
tray = {'position': [0.0, 0.0, 0.0]}     # Auto-adjusted to 0.0025
cube = {'position': [0.3, 0.0, 0.025]}   # Half of 0.05
sphere = {'position': [0.3, 0.0, 0.025]} # Half of 0.05
```

## Why height/2?

PyBullet (and most physics engines) use the **center of mass** for positioning:

```
   Top of cube: z = 0.05
        ↑
        │
  Center (specified position): z = 0.025
        │
        ↓
   Bottom: z = 0.0 (on table)
```

If you specify z=0.0 for a cube, half of it would be below the table!

## Special Case: Trays

Trays are special because:
1. They should appear flush on the table
2. We want to avoid z-fighting (visual glitches)

Solution: Position at a tiny offset (height/2 = 0.0025m):
- Bottom at z ≈ 0.0
- Top at z ≈ 0.005
- Visually appears flush
- No z-fighting with table plane

## Stacking Objects

For stacked objects, add the heights:

```python
# Stack cube on tray
tray_height = 0.005
cube_size = 0.05
cube_z = tray_height + (cube_size / 2)  # 0.005 + 0.025 = 0.03

# Stack second cube on first
second_cube_z = cube_z + cube_size  # 0.03 + 0.05 = 0.08
```

## Quick Reference Table

| Scenario | Formula | Example |
|----------|---------|---------|
| Object on table | z = height/2 | z = 0.05/2 = 0.025 |
| Tray on table | z = height/2 | z = 0.005/2 = 0.0025 |
| Object on tray | z = tray_height + height/2 | z = 0.005 + 0.025 = 0.03 |
| Stacked objects | z = base_z + height | z = 0.05 + 0.05 = 0.10 |

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Object floating | z too high | Set z = height/2 |
| Object sinking | z too low or negative | Ensure z ≥ height/2 |
| Tray floating | z > 0.01 | Set z = 0.0 (auto-adjusted) |
| Z-fighting | Surfaces at same z | Slight offset (0.001m) |

## Related Documentation

- **OBJECT_SUBSTITUTION_GUIDE.md**: Detailed guide on adding objects
- **IMPLEMENTATION_NOTES.md**: Technical implementation details
- **README.md**: Usage examples
- **demo_with_trays.py**: Working example code

---

**Key Takeaway**: Always position objects at z = object_height/2 for them to rest properly on the table surface.
