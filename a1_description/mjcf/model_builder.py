"""
MuJoCo MJCF model builder for the Unitree A1 quadruped.

Physical constants are sourced from the official Unitree A1 URDF/datasheet and
the stair geometry is taken from the stairs.xacro file in the a1_description
package.  The generated XML string is consumed by MuJoCo at runtime so no
files need to be written to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Robot physical constants (from Unitree A1 URDF / datasheet)
# ---------------------------------------------------------------------------
BODY_MASS: float = 6.0     # kg  (torso link)
THIGH_MASS: float = 1.013
CALF_MASS: float = 0.166
HIP_MASS: float = 0.696

# PD servo gains (from robot_control.yaml)
HIP_KP: float = 100.0
HIP_KD: float = 5.0
THIGH_KP: float = 300.0
THIGH_KD: float = 8.0
CALF_KP: float = 300.0
CALF_KD: float = 8.0

# Joint limits (radians)
HIP_LIMIT: tuple[float, float] = (-0.802, 0.802)
THIGH_LIMIT: tuple[float, float] = (-1.047, 4.189)
CALF_LIMIT: tuple[float, float] = (-2.697, -0.916)

# Nominal standing pose per leg: [hip, thigh, calf]
DEFAULT_POSE_PER_LEG: list[float] = [0.0, 0.8, -1.6]


@dataclass
class A1ModelConfig:
    """All tuneable parameters for the MJCF model."""

    # Physics
    timestep: float = 0.002        # seconds (500 Hz physics)
    integrator: str = "RK4"

    # Robot geometry
    body_mass: float = BODY_MASS
    trunk_half_size: tuple[float, float, float] = (0.1805, 0.047, 0.057)
    trunk_start_z: float = 0.42   # metres above ground at reset

    # Hip offset positions in trunk frame
    hip_offsets: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "FL": (0.183,  0.047, 0.0),
            "FR": (0.183, -0.047, 0.0),
            "RL": (-0.183,  0.047, 0.0),
            "RR": (-0.183, -0.047, 0.0),
        }
    )

    # Contact / friction
    floor_friction: tuple[float, float, float] = (0.8, 0.02, 0.001)
    stair_friction: tuple[float, float, float] = (0.9, 0.02, 0.001)

    # Rendering
    skybox_rgb1: tuple[float, float, float] = (0.3, 0.5, 0.7)
    skybox_rgb2: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _build_leg_xml(
    prefix: str,
    hip_xyz: tuple[float, float, float],
    hip_side: int = 1,
) -> str:
    """Return MJCF XML for one leg of the A1 robot.

    Args:
        prefix:   One of ``FL``, ``FR``, ``RL``, ``RR``.
        hip_xyz:  Position of the hip joint in the trunk body frame.
        hip_side: +1 for left legs, -1 for right legs (flips axis sign).
    """
    hx, hy, hz = hip_xyz
    return f"""
        <!-- {prefix} leg -->
        <body name="{prefix}_hip" pos="{hx:.4f} {hy:.4f} {hz:.4f}">
          <joint name="{prefix}_hip_joint"   type="hinge" axis="1 0 0"
                 range="{HIP_LIMIT[0]} {HIP_LIMIT[1]}"
                 damping="{HIP_KD}" stiffness="0" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0  0 {0.08 * hip_side:.3f} 0"
                size="0.046" mass="{HIP_MASS}" rgba="0.5 0.5 0.5 1"/>
          <body name="{prefix}_thigh" pos="0 {0.083 * hip_side:.4f} 0">
            <joint name="{prefix}_thigh_joint" type="hinge" axis="0 1 0"
                   range="{THIGH_LIMIT[0]} {THIGH_LIMIT[1]}"
                   damping="{THIGH_KD}" stiffness="0" armature="0.01"/>
            <geom type="capsule" fromto="0 0 0  0 0 -0.2"
                  size="0.0265" mass="{THIGH_MASS}" rgba="0.4 0.4 0.7 1"/>
            <body name="{prefix}_calf" pos="0 0 -0.2">
              <joint name="{prefix}_calf_joint" type="hinge" axis="0 1 0"
                     range="{CALF_LIMIT[0]} {CALF_LIMIT[1]}"
                     damping="{CALF_KD}" stiffness="0" armature="0.01"/>
              <geom type="capsule" fromto="0 0 0  0 0 -0.2"
                    size="0.0265" mass="{CALF_MASS}" rgba="0.3 0.3 0.6 1"/>
              <!-- Foot site used for contact-force sensing -->
              <site name="{prefix}_foot" pos="0 0 -0.2" size="0.025" rgba="1 0.3 0.3 1"/>
            </body>
          </body>
        </body>"""


def build_a1_xml(
    terrain_xml: str = "",
    cfg: A1ModelConfig | None = None,
) -> str:
    """Build a complete MuJoCo MJCF XML string for the A1 quadruped.

    Args:
        terrain_xml: Extra ``<geom>`` or ``<body>`` elements injected into the
                     world body (e.g. stairs, ramps).  Pass an empty string for
                     a flat ground.
        cfg:         Optional :class:`A1ModelConfig` for tuning geometry and
                     physics parameters.  Defaults to the standard A1 values.

    Returns:
        A complete MJCF XML string ready to be loaded with
        ``mujoco.MjModel.from_xml_string()``.
    """
    if cfg is None:
        cfg = A1ModelConfig()

    # Build all four legs
    legs = "".join(
        _build_leg_xml(
            prefix=p,
            hip_xyz=cfg.hip_offsets[p],
            hip_side=(1 if p in ("FL", "RL") else -1),
        )
        for p in ("FL", "FR", "RL", "RR")
    )

    # Position-servo actuators (one per joint, kp from robot_control.yaml)
    actuator_lines = []
    for p in ("FL", "FR", "RL", "RR"):
        actuator_lines += [
            f'<position name="{p}_hip_act"   joint="{p}_hip_joint"   '
            f'kp="{HIP_KP}"   forcelimited="true" forcerange="-33.5 33.5"/>',
            f'<position name="{p}_thigh_act" joint="{p}_thigh_joint" '
            f'kp="{THIGH_KP}" forcelimited="true" forcerange="-33.5 33.5"/>',
            f'<position name="{p}_calf_act"  joint="{p}_calf_joint"  '
            f'kp="{CALF_KP}"  forcelimited="true" forcerange="-33.5 33.5"/>',
        ]
    actuators = "\n        ".join(actuator_lines)

    # IMU-like sensors + contact force sensors at each foot
    contact_sensors = "\n        ".join(
        f'<touch name="{p}_foot_touch" site="{p}_foot"/>'
        for p in ("FL", "FR", "RL", "RR")
    )

    tx, ty, tz = cfg.trunk_half_size
    sx1, sx2, sx3 = cfg.skybox_rgb1
    sx4, sx5, sx6 = cfg.skybox_rgb2

    xml = f"""<?xml version="1.0"?>
<mujoco model="a1_stair_climbing">

  <!-- ===== Compiler & simulation options ===== -->
  <compiler angle="radian" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81"
          timestep="{cfg.timestep}"
          integrator="{cfg.integrator}"
          cone="elliptic"
          noslip_iterations="4"/>

  <!-- ===== Solver settings ===== -->
  <size njmax="500" nconmax="100"/>

  <!-- ===== Defaults ===== -->
  <default>
    <joint limited="true" armature="0.01"/>
    <geom contype="1" conaffinity="1" condim="3"
          friction="{cfg.floor_friction[0]} {cfg.floor_friction[1]} {cfg.floor_friction[2]}"
          rgba="0.8 0.6 0.4 1"/>
  </default>

  <!-- ===== Assets ===== -->
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="{sx1} {sx2} {sx3}" rgb2="{sx4} {sx5} {sx6}"
             width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"
             width="300" height="300"/>
    <material name="grid"  texture="grid"   texrepeat="8 8"  reflectance="0.2"/>
    <material name="stair" rgba="0.6 0.5 0.4 1" reflectance="0.1"/>
  </asset>

  <!-- ===== World body ===== -->
  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="20 20 0.1"
          material="grid" condim="3"
          friction="{cfg.floor_friction[0]} {cfg.floor_friction[1]} {cfg.floor_friction[2]}"/>

    <!-- Lighting -->
    <light directional="true" diffuse="0.9 0.9 0.9" pos="0 0 6"
           dir="0 0 -1" castshadow="true"/>
    <light directional="false" diffuse="0.4 0.4 0.4" pos="-2 -2 4"
           dir="1 1 -1" castshadow="false"/>

    <!-- ===== Injected terrain (stairs / slope / etc.) ===== -->
    {terrain_xml}

    <!-- ===== A1 trunk (free-floating base) ===== -->
    <body name="trunk" pos="0 0 {cfg.trunk_start_z}">
      <freejoint name="trunk_free"/>
      <site name="imu_site" pos="0 0 0" size="0.01" rgba="0 1 0 0.3"/>
      <geom type="box" size="{tx} {ty} {tz}"
            mass="{cfg.body_mass}" rgba="0.2 0.4 0.8 1"
            condim="1"/>
      {legs}
    </body>
  </worldbody>

  <!-- ===== Actuators ===== -->
  <actuator>
    {actuators}
  </actuator>

  <!-- ===== Sensors ===== -->
  <sensor>
    <!-- IMU -->
    <gyro        name="imu_gyro"     site="imu_site"/>
    <accelerometer name="imu_accel"  site="imu_site"/>
    <framequat   name="trunk_quat"   objtype="body" objname="trunk"/>
    <framelinvel name="trunk_linvel" objtype="body" objname="trunk"/>
    <frameangvel name="trunk_angvel" objtype="body" objname="trunk"/>
    <!-- Foot contact (binary touch sensors) -->
    {contact_sensors}
  </sensor>

</mujoco>"""
    return xml
