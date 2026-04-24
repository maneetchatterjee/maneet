"""
Terrain generators for the A1 stair-climbing environment.

Each function returns a string of MJCF XML geom/body elements that are
injected into the world body.  Terrain parameters match the stair geometry
specified in ``stairs.xacro`` from the a1_description package:

    stair_height = 0.170 m
    stair_width  = 0.310 m   (step depth / tread depth)
    stair_length = 0.640 m   (step width across the robot)

Curriculum:
    The :class:`CurriculumTerrainScheduler` progressively increases terrain
    difficulty as the agent's success rate rises, starting from flat ground
    and ultimately reaching full-height stairs.
"""
from __future__ import annotations

import math
from typing import Callable

# ---------------------------------------------------------------------------
# Stair geometry from stairs.xacro
# ---------------------------------------------------------------------------
STAIR_HEIGHT: float = 0.170   # metres
STAIR_WIDTH: float = 0.310    # metres (tread depth)
STAIR_LENGTH: float = 0.640   # metres (staircase width)

# Number of steps in the full staircase
DEFAULT_N_STEPS: int = 6

# How far in front of the robot (x=0) the first step begins
STAIR_START_X: float = 0.7


# ---------------------------------------------------------------------------
# Terrain builders
# ---------------------------------------------------------------------------

def terrain_flat() -> str:
    """Return empty terrain XML – robot walks on the flat ground plane only."""
    return ""


def terrain_stairs(
    n_steps: int = DEFAULT_N_STEPS,
    step_height: float = STAIR_HEIGHT,
    step_depth: float = STAIR_WIDTH,
    step_width: float = STAIR_LENGTH,
    start_x: float = STAIR_START_X,
    friction: tuple[float, float, float] = (0.9, 0.02, 0.001),
) -> str:
    """Return MJCF XML for a staircase ascending in the +x direction.

    Each step is modelled as a solid box geom.  The geometry matches the
    ``stairs.xacro`` definition:

    * Step i has its top surface at height ``(i + 1) * step_height``.
    * Step i extends from ``start_x + i * step_depth`` to
      ``start_x + (i + 1) * step_depth`` along x.

    Args:
        n_steps:     Total number of steps.
        step_height: Height of each individual step (m).
        step_depth:  Tread depth of each step (m).
        step_width:  Width of the staircase across the robot (m).
        start_x:     x-coordinate of the front face of the first step (m).
        friction:    (sliding, torsional, rolling) friction coefficients.

    Returns:
        MJCF XML fragment (geom elements) for the staircase.
    """
    fr = f"{friction[0]} {friction[1]} {friction[2]}"
    geoms: list[str] = []
    for i in range(n_steps):
        # The box is described by its centre and half-extents.
        half_h = (i + 1) * step_height / 2.0
        centre_x = start_x + i * step_depth + step_depth / 2.0
        centre_z = half_h
        half_x = step_depth / 2.0
        half_y = step_width / 2.0

        geoms.append(
            f'<geom name="stair_{i}" type="box" '
            f'pos="{centre_x:.4f} 0 {centre_z:.4f}" '
            f'size="{half_x:.4f} {half_y:.4f} {half_h:.4f}" '
            f'friction="{fr}" '
            f'material="stair" condim="3"/>'
        )

    return "\n    ".join(geoms)


def terrain_slope(
    length: float = 2.0,
    width: float = STAIR_LENGTH,
    angle_deg: float = 15.0,
    start_x: float = STAIR_START_X,
    friction: tuple[float, float, float] = (0.8, 0.02, 0.001),
) -> str:
    """Return MJCF XML for an inclined ramp.

    The ramp is a box geom tilted about the y-axis.

    Args:
        length:    Horizontal length of the ramp (m).
        width:     Ramp width (m).
        angle_deg: Inclination angle in degrees.
        start_x:   x-coordinate of the ramp's near edge.
        friction:  Friction coefficients.

    Returns:
        MJCF XML fragment for the slope.
    """
    angle_rad = math.radians(angle_deg)
    # Euler angles for tilt about y: "0 angle 0"
    centre_x = start_x + length / 2.0
    thickness = 0.05
    half_l = length / 2.0
    half_w = width / 2.0
    half_t = thickness / 2.0

    # Rise of the ramp midpoint
    rise_at_mid = math.tan(angle_rad) * (length / 2.0)
    fr = f"{friction[0]} {friction[1]} {friction[2]}"

    return (
        f'<geom name="slope" type="box" '
        f'pos="{centre_x:.4f} 0 {rise_at_mid:.4f}" '
        f'size="{half_l:.4f} {half_w:.4f} {half_t:.4f}" '
        f'euler="0 {angle_rad:.4f} 0" '
        f'friction="{fr}" '
        f'material="stair" condim="3"/>'
    )


def terrain_stairs_partial(difficulty: float) -> str:
    """Return a staircase terrain scaled by *difficulty* ∈ [0, 1].

    * At ``difficulty == 0`` → flat terrain (no stairs).
    * At ``difficulty == 1`` → full staircase with default geometry.

    Intermediate values linearly scale the step height between
    ``0.04 m`` (very small) and :data:`STAIR_HEIGHT`.

    Args:
        difficulty: Curriculum progress in [0, 1].

    Returns:
        MJCF XML fragment.
    """
    if difficulty <= 0.0:
        return terrain_flat()
    min_h = 0.04
    h = min_h + (STAIR_HEIGHT - min_h) * difficulty
    n = max(1, round(DEFAULT_N_STEPS * difficulty))
    return terrain_stairs(n_steps=n, step_height=h)


# ---------------------------------------------------------------------------
# Terrain registry for the environment
# ---------------------------------------------------------------------------

#: Ordered list of terrain builder callables.  Index 0 is always flat so
#: curriculum learning can start from a clean baseline.
TERRAIN_BUILDERS: list[Callable[[], str]] = [
    terrain_flat,
    lambda: terrain_stairs(),
    lambda: terrain_slope(angle_deg=10.0),
    lambda: terrain_slope(angle_deg=20.0),
]

TERRAIN_NAMES: list[str] = ["Flat", "Stairs", "Slope10", "Slope20"]


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------

class CurriculumTerrainScheduler:
    """Gradually increases stair difficulty as the agent improves.

    The scheduler exposes a single callable :meth:`get_terrain_xml` which
    returns the terrain XML fragment to use for the next episode.  Call
    :meth:`update` after each evaluation to adjust the curriculum.

    Curriculum stages:
        0 – flat ground (no stairs)
        1 – stairs at 25 % height (0.04 → 0.08 m)
        2 – stairs at 50 % height
        3 – stairs at 75 % height
        4 – full stairs (0.170 m step height)

    Args:
        success_threshold: Episode success rate required to advance a stage.
        window:            Number of recent episodes used to compute success
                           rate.
    """

    N_STAGES: int = 5

    def __init__(
        self,
        success_threshold: float = 0.7,
        window: int = 50,
    ) -> None:
        self.success_threshold = success_threshold
        self.window = window
        self._stage: int = 0
        self._history: list[bool] = []

    @property
    def stage(self) -> int:
        """Current curriculum stage (0 = easiest, N_STAGES-1 = hardest)."""
        return self._stage

    @property
    def difficulty(self) -> float:
        """Normalised difficulty ∈ [0, 1]."""
        return self._stage / (self.N_STAGES - 1)

    def update(self, episode_success: bool) -> None:
        """Record one episode outcome and potentially advance the curriculum.

        Args:
            episode_success: ``True`` if the episode was successful
                             (robot climbed the stairs without falling).
        """
        self._history.append(episode_success)
        if len(self._history) > self.window:
            self._history.pop(0)

        if len(self._history) >= self.window:
            rate = sum(self._history) / len(self._history)
            if rate >= self.success_threshold and self._stage < self.N_STAGES - 1:
                self._stage += 1
                self._history.clear()

    def get_terrain_xml(self) -> str:
        """Return the terrain XML appropriate for the current stage.

        Returns:
            MJCF XML fragment string.
        """
        return terrain_stairs_partial(self.difficulty)

    def __repr__(self) -> str:
        return (
            f"CurriculumTerrainScheduler("
            f"stage={self._stage}/{self.N_STAGES - 1}, "
            f"difficulty={self.difficulty:.2f})"
        )
