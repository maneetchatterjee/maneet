"""
Gymnasium environment for Unitree A1 stair climbing via PPO.

Observation space (49-dimensional float32 vector):
    [0:3]   Base linear velocity in world frame (vx, vy, vz)
    [3:6]   Base angular velocity in body frame (ωx, ωy, ωz)
    [6:9]   Projected gravity vector in body frame
    [9:21]  Joint positions relative to default standing pose (12)
    [21:33] Joint velocities (12)
    [33:45] Previous actions (12)
    [45:49] Binary foot-contact states (FL, FR, RL, RR)

Action space (12-dimensional, continuous ∈ [-1, 1]):
    Normalised delta joint position targets.  Scaled by ACTION_SCALE and added
    to DEFAULT_POSE to obtain the actual position reference sent to the PD
    servos.

Reward components:
    forward_vel      Incentivises forward motion (stair climbing direction).
    height_gain      Rewards increase in trunk height (ascending steps).
    upright          Penalises tilt (roll / pitch deviation from upright).
    gait_symmetry    Rewards alternating diagonal gait pattern.
    foot_contact     Rewards stable multi-foot ground contact.
    action_smooth    Penalises high-frequency action changes.
    torque_smooth    Penalises large actuator torques (energy efficiency).
    alive            Small per-step alive bonus to encourage long episodes.
    fall             Large penalty for catastrophic falls (episode ends).

Early termination:
    Trunk height < 0.22 m (robot has fallen or collapsed).
    Trunk roll or pitch > 60° (extreme tilt).
    Episode length exceeds MAX_EPISODE_STEPS.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Allow importing sibling packages when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required. Install it with:  pip install gymnasium"
    ) from exc

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mujoco is required. Install it with:  pip install mujoco>=3.1.0"
    ) from exc

from mjcf.model_builder import (
    build_a1_xml,
    A1ModelConfig,
    DEFAULT_POSE_PER_LEG,
)
from envs.terrain import terrain_flat, terrain_stairs, TERRAIN_BUILDERS, TERRAIN_NAMES


# ---------------------------------------------------------------------------
# Robot constants
# ---------------------------------------------------------------------------

#: Default standing joint positions: [hip, thigh, calf] × 4 legs.
DEFAULT_POSE: npt.NDArray[np.float32] = np.array(
    DEFAULT_POSE_PER_LEG * 4, dtype=np.float32
)

#: Scale applied to normalised actions before adding to DEFAULT_POSE.
ACTION_SCALE: float = 0.3  # radians

#: Number of physics sub-steps per control step (≈ 50 Hz control @ 2 ms physics).
SIM_STEPS: int = 10

#: Observation dimension.
OBS_DIM: int = 49

#: Observation lower / upper bounds.
OBS_LOW: float = -np.inf
OBS_HIGH: float = np.inf

#: Maximum episode length (steps, each = 20 ms → 20 s per episode).
MAX_EPISODE_STEPS: int = 1000

#: Minimum trunk height before the episode is terminated (robot fell).
FALL_HEIGHT: float = 0.22  # metres

#: Maximum allowed roll / pitch (radians) before episode termination.
FALL_ANGLE: float = np.deg2rad(60.0)

#: Prefixes for the four legs in order.
LEG_PREFIXES: list[str] = ["FL", "FR", "RL", "RR"]


# ---------------------------------------------------------------------------
# Reward shaping weights
# ---------------------------------------------------------------------------

class RewardConfig:
    """Weights for each reward component.  Adjust to tune behaviour."""

    forward_vel: float = 2.0       # per (m/s) of forward velocity
    height_gain: float = 8.0       # per metre of trunk height gained
    upright: float = 1.0           # penalty multiplier for tilt
    gait_symmetry: float = 0.3     # reward for diagonal gait
    foot_contact: float = 0.2      # reward for ≥2 feet contacting ground
    action_smooth: float = 0.05    # penalty for large action deltas
    torque_smooth: float = 0.002   # penalty for large actuator torques
    alive: float = 0.5             # per-step alive bonus
    fall: float = -20.0            # one-off penalty when episode ends by fall

    # Target forward velocity for the velocity reward.
    # At full stairs this should be moderate (~0.4 m/s); tuned for curriculum.
    target_forward_vel: float = 0.4  # m/s


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class A1StairClimbEnv(gym.Env):
    """Gymnasium environment for Unitree A1 stair climbing.

    Args:
        terrain_type: Index into :data:`~envs.terrain.TERRAIN_BUILDERS`.
                      ``None`` means terrain is randomly sampled each episode.
        terrain_xml:  Raw MJCF XML fragment to use as terrain.  When provided,
                      *terrain_type* is ignored.
        render_mode:  ``'rgb_array'`` enables frame rendering; ``None`` disables.
        seed:         Master random seed for reproducibility.
        reward_cfg:   :class:`RewardConfig` instance for weight overrides.
        model_cfg:    :class:`A1ModelConfig` instance for robot model tweaks.
        use_domain_randomization: Apply domain randomization on each reset.
        curriculum_difficulty: Stair difficulty ∈ [0, 1] for curriculum use.
                               Overrides *terrain_type* and *terrain_xml* when
                               the :class:`~envs.terrain.CurriculumTerrainScheduler`
                               is driving training.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        terrain_type: int | None = None,
        terrain_xml: str | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        reward_cfg: RewardConfig | None = None,
        model_cfg: A1ModelConfig | None = None,
        use_domain_randomization: bool = True,
        curriculum_difficulty: float = 1.0,
    ) -> None:
        super().__init__()

        self.terrain_type = terrain_type
        self._fixed_terrain_xml = terrain_xml
        self.render_mode = render_mode
        self.reward_cfg = reward_cfg or RewardConfig()
        self.model_cfg = model_cfg or A1ModelConfig()
        self.use_domain_randomization = use_domain_randomization
        self.curriculum_difficulty = float(np.clip(curriculum_difficulty, 0.0, 1.0))

        self._rng = np.random.default_rng(seed)

        # MuJoCo model and data (initialised on first reset)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

        # Internal state
        self._prev_action: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
        self._step_count: int = 0
        self._prev_trunk_z: float = 0.0

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, terrain_xml: str) -> None:
        """(Re-)compile the MuJoCo model with the given terrain."""
        xml_str = build_a1_xml(terrain_xml=terrain_xml, cfg=self.model_cfg)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        # Cache frequently used body / joint indices
        self._trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        # First joint after the free joint is at qpos index 7
        self._jpos_start: int = 7
        self._jvel_start: int = 6

    def _choose_terrain(self) -> str:
        """Pick and return the terrain XML for this episode."""
        if self._fixed_terrain_xml is not None:
            return self._fixed_terrain_xml
        if self.terrain_type is not None:
            return TERRAIN_BUILDERS[self.terrain_type]()
        # Random terrain
        idx = int(self._rng.integers(len(TERRAIN_BUILDERS)))
        return TERRAIN_BUILDERS[idx]()

    def _apply_domain_randomization(self) -> None:
        """Perturb physics parameters to improve policy robustness."""
        rng = self._rng

        # 1. Floor friction (uniform ±30 %)
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and "floor" in name:
                scale = rng.uniform(0.7, 1.3)
                self.model.geom_friction[geom_id, 0] *= scale

        # 2. Trunk mass ±20 %
        self.model.body_mass[self._trunk_body_id] = (
            self.model_cfg.body_mass * rng.uniform(0.8, 1.2)
        )

        # 3. Random initial trunk orientation (small roll/pitch perturbation)
        roll = rng.uniform(-0.2, 0.2)
        pitch = rng.uniform(-0.15, 0.15)
        cy = np.cos(0.0); sy = np.sin(0.0)
        cp = np.cos(pitch / 2); sp = np.sin(pitch / 2)
        cr = np.cos(roll / 2);  sr = np.sin(roll / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        self.data.qpos[3:7] = [qw, qx, qy, qz]

        # 4. Random initial joint positions (±0.2 rad from default)
        noise = rng.uniform(-0.2, 0.2, size=12).astype(np.float32)
        self.data.qpos[self._jpos_start : self._jpos_start + 12] = DEFAULT_POSE + noise

        # 5. Random initial velocity push (small, ±0.2 m/s)
        self.data.qvel[0:3] = rng.uniform(-0.2, 0.2, size=3)

    def _get_obs(self) -> npt.NDArray[np.float32]:
        """Compute and return the 49-D observation vector."""
        # Trunk quaternion → rotation matrix
        quat = self.data.qpos[3:7]           # (w, x, y, z)
        rot = np.zeros((3, 3), dtype=np.float64)
        mujoco.mju_quat2Mat(rot.ravel(), quat)

        # [0:3] Base linear velocity (world frame from sensor)
        trunk_linvel = self.data.sensor("trunk_linvel").data.astype(np.float32)

        # [3:6] Base angular velocity (body frame from IMU gyro)
        imu_gyro = self.data.sensor("imu_gyro").data.astype(np.float32)

        # [6:9] Projected gravity vector in body frame
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gravity_body = (rot.T @ gravity_world).astype(np.float32)

        # [9:21] Joint positions relative to default pose
        joint_pos = (
            self.data.qpos[self._jpos_start : self._jpos_start + 12].astype(np.float32)
            - DEFAULT_POSE
        )

        # [21:33] Joint velocities
        joint_vel = self.data.qvel[
            self._jvel_start : self._jvel_start + 12
        ].astype(np.float32)

        # [33:45] Previous actions
        prev_action = self._prev_action.astype(np.float32)

        # [45:49] Foot contact (binary)
        foot_contact = np.array(
            [
                float(self.data.sensor(f"{p}_foot_touch").data[0] > 0.1)
                for p in LEG_PREFIXES
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([
            trunk_linvel,   # 3
            imu_gyro,       # 3
            gravity_body,   # 3
            joint_pos,      # 12
            joint_vel,      # 12
            prev_action,    # 12
            foot_contact,   # 4
        ])
        assert obs.shape == (OBS_DIM,), f"OBS_DIM mismatch: {obs.shape}"
        return obs.astype(np.float32)

    def _compute_reward(
        self,
        action: npt.NDArray[np.float32],
    ) -> tuple[float, dict[str, Any]]:
        """Compute the shaped reward and info dict for the current state.

        Args:
            action: The action applied this step (normalised, ∈ [-1, 1]).

        Returns:
            (total_reward, info_dict)
        """
        cfg = self.reward_cfg
        data = self.data

        # --- State quantities ---
        trunk_pos = data.qpos[:3]             # (x, y, z)
        trunk_z = float(trunk_pos[2])
        quat = data.qpos[3:7]
        rot = np.zeros((3, 3), dtype=np.float64)
        mujoco.mju_quat2Mat(rot.ravel(), quat)

        # Roll and pitch extracted from rotation matrix
        # R = Rz * Ry * Rx →  pitch = asin(-R[2,0]),  roll = atan2(R[2,1], R[2,2])
        pitch = float(np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0)))
        roll  = float(np.arctan2(rot[2, 1], rot[2, 2]))

        # Forward (x) linear velocity in world frame
        linvel = data.sensor("trunk_linvel").data
        vx = float(linvel[0])

        # --- Fall detection ---
        fell = (
            trunk_z < FALL_HEIGHT
            or abs(roll)  > FALL_ANGLE
            or abs(pitch) > FALL_ANGLE
        )

        # --- Reward components ---

        # 1. Forward velocity reward (clamped to avoid negative reward for reversal)
        vel_err = abs(vx - cfg.target_forward_vel)
        r_forward = cfg.forward_vel * np.exp(-2.0 * vel_err ** 2)

        # 2. Height gain reward
        delta_z = trunk_z - self._prev_trunk_z
        r_height = cfg.height_gain * max(0.0, delta_z)

        # 3. Upright bonus (penalise tilt)
        r_upright = -cfg.upright * (roll ** 2 + pitch ** 2)

        # 4. Foot-contact reward (reward if ≥ 2 feet are on the ground)
        contacts = np.array(
            [
                float(data.sensor(f"{p}_foot_touch").data[0] > 0.1)
                for p in LEG_PREFIXES
            ]
        )
        n_contacts = contacts.sum()
        r_contact = cfg.foot_contact * min(n_contacts / 2.0, 1.0)

        # 5. Diagonal gait symmetry (FL+RR vs FR+RL should alternate)
        fl, fr, rl, rr = contacts
        diag1 = fl * rr   # front-left + rear-right (trot diagonal)
        diag2 = fr * rl   # front-right + rear-left
        r_gait = cfg.gait_symmetry * float(diag1 + diag2) / 2.0

        # 6. Action smoothness penalty
        action_delta = action - self._prev_action
        r_action = -cfg.action_smooth * float(np.sum(action_delta ** 2))

        # 7. Torque smoothness penalty
        torques = data.actuator_force
        r_torque = -cfg.torque_smooth * float(np.sum(torques ** 2))

        # 8. Alive bonus
        r_alive = cfg.alive

        # 9. Fall penalty (applied once, only if this step caused a fall)
        r_fall = cfg.fall if fell else 0.0

        total = (
            r_forward
            + r_height
            + r_upright
            + r_contact
            + r_gait
            + r_action
            + r_torque
            + r_alive
            + r_fall
        )

        info = {
            "trunk_z": trunk_z,
            "roll": roll,
            "pitch": pitch,
            "vx": vx,
            "fell": fell,
            "n_contacts": int(n_contacts),
            "r_forward": r_forward,
            "r_height": r_height,
            "r_upright": r_upright,
            "r_contact": r_contact,
            "r_gait": r_gait,
            "r_action": r_action,
            "r_torque": r_torque,
            "r_alive": r_alive,
            "r_fall": r_fall,
        }
        return float(total), info

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Select and load terrain
        terrain_xml = self._choose_terrain()
        self._load_model(terrain_xml)

        # Reset to default pose
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.model_cfg.trunk_start_z
        self.data.qpos[self._jpos_start : self._jpos_start + 12] = DEFAULT_POSE

        # Domain randomization
        if self.use_domain_randomization:
            self._apply_domain_randomization()

        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0
        self._prev_trunk_z = float(self.data.qpos[2])

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Add small actuator noise (domain randomization)
        action_noisy = action + self._rng.normal(0, 0.01, size=12).astype(np.float32)

        # Map normalised action → joint position targets (PD setpoints)
        target_pos = DEFAULT_POSE + action_noisy * ACTION_SCALE
        self.data.ctrl[:12] = target_pos

        # Physics sub-stepping
        for _ in range(SIM_STEPS):
            mujoco.mj_step(self.model, self.data)

        reward, info = self._compute_reward(action)

        self._prev_trunk_z = info["trunk_z"]
        self._prev_action = action.copy()
        self._step_count += 1

        terminated = bool(info["fell"])
        truncated = self._step_count >= MAX_EPISODE_STEPS

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> npt.NDArray[np.uint8] | None:
        """Render the current frame as an RGB numpy array."""
        if self.render_mode != "rgb_array":
            return None
        if self.model is None or self.data is None:
            return None
        try:
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data, camera=-1)
            frame = renderer.render()
            renderer.close()
            return frame
        except Exception:
            return None

    def close(self) -> None:
        self.model = None
        self.data = None
