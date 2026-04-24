"""
Gymnasium environment wrappers for A1 stair climbing.

Wrappers provided:
    :class:`StairClimbingWrapper`  – Adds stair-specific episode metrics and
                                     curriculum difficulty injection.
    :class:`StairSuccessMonitor`   – Records per-episode stair climbing
                                     success/failure for curriculum scheduling.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install gymnasium") from exc


class StairClimbingWrapper(gym.Wrapper):
    """Adds stair-climbing specific metrics and curriculum hooks.

    Wraps :class:`~envs.a1_stair_env.A1StairClimbEnv` to track:

    * Maximum trunk height achieved this episode (proxy for stair steps
      successfully negotiated).
    * Episode cumulative reward.
    * Whether the robot fell.
    * Stair success flag (trunk_z > *success_height_threshold*).

    The wrapper forwards a ``curriculum_difficulty`` option via
    ``env.reset(options={"difficulty": float})`` to allow the
    :class:`~envs.terrain.CurriculumTerrainScheduler` to update the terrain
    each episode.

    Args:
        env:                      The base :class:`~envs.a1_stair_env.A1StairClimbEnv`.
        success_height_threshold: Minimum trunk height (m) to count as a
                                  successful stair ascent (default: 0.55 m,
                                  i.e. one full step negotiated).
    """

    def __init__(
        self,
        env: gym.Env,
        success_height_threshold: float = 0.55,
    ) -> None:
        super().__init__(env)
        self.success_height_threshold = success_height_threshold
        self._episode_reward: float = 0.0
        self._max_trunk_z: float = 0.0
        self._fell: bool = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        options = options or {}

        # Curriculum difficulty may be injected by the trainer.
        difficulty = options.pop("difficulty", None)
        if difficulty is not None and hasattr(self.env.unwrapped, "curriculum_difficulty"):
            self.env.unwrapped.curriculum_difficulty = float(difficulty)

        obs, info = self.env.reset(seed=seed, options=options)

        self._episode_reward = 0.0
        self._max_trunk_z = float(obs[0])   # trunk_z is not directly in obs,
        # but we track it via info at step; init to current trunk_z
        if hasattr(self.env.unwrapped, "data") and self.env.unwrapped.data is not None:
            self._max_trunk_z = float(self.env.unwrapped.data.qpos[2])
        self._fell = False

        return obs, info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._episode_reward += reward
        self._max_trunk_z = max(self._max_trunk_z, info.get("trunk_z", 0.0))
        if info.get("fell", False):
            self._fell = True

        success = self._max_trunk_z >= self.success_height_threshold and not self._fell

        if terminated or truncated:
            info["episode"] = {
                "reward":      self._episode_reward,
                "max_trunk_z": self._max_trunk_z,
                "fell":        self._fell,
                "success":     success,
            }

        return obs, reward, terminated, truncated, info


class StairSuccessMonitor(gym.Wrapper):
    """Lightweight monitor that records per-episode success flags.

    Intended for use with :class:`~envs.terrain.CurriculumTerrainScheduler`
    so the scheduler can be updated after each episode.

    Access the most recent success flag via :attr:`last_success`.

    Args:
        env:                      The environment to wrap.
        success_height_threshold: Same semantics as in :class:`StairClimbingWrapper`.
    """

    def __init__(
        self,
        env: gym.Env,
        success_height_threshold: float = 0.55,
    ) -> None:
        super().__init__(env)
        self.success_height_threshold = success_height_threshold
        self.last_success: bool = False
        self._max_trunk_z: float = 0.0
        self._fell: bool = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict]:
        self._max_trunk_z = 0.0
        self._fell = False
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._max_trunk_z = max(self._max_trunk_z, info.get("trunk_z", 0.0))
        if info.get("fell", False):
            self._fell = True

        if terminated or truncated:
            self.last_success = (
                self._max_trunk_z >= self.success_height_threshold and not self._fell
            )

        return obs, reward, terminated, truncated, info
