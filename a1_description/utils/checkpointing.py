"""
Model checkpointing utilities for PPO stair-climbing training.

Wraps Stable Baselines3's save/load methods with additional metadata
(hyperparameters, curriculum stage, timestep) stored alongside each
checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required: pip install stable-baselines3"
    ) from exc


def save_checkpoint(
    model: PPO,
    vec_normalize: VecNormalize | None,
    checkpoint_dir: str | Path,
    timestep: int,
    metadata: dict[str, Any] | None = None,
    name_prefix: str = "a1_ppo",
) -> Path:
    """Save a PPO model checkpoint with associated VecNormalize statistics.

    Creates three files inside *checkpoint_dir*:

    * ``{name_prefix}_{timestep}.zip``      – SB3 model weights.
    * ``{name_prefix}_{timestep}_vecnorm.pkl`` – VecNormalize running stats.
    * ``{name_prefix}_{timestep}_meta.json`` – Training metadata.

    Args:
        model:          Trained :class:`stable_baselines3.PPO` model.
        vec_normalize:  :class:`VecNormalize` wrapper (may be ``None``).
        checkpoint_dir: Directory where files will be written.
        timestep:       Current training timestep (used in filenames).
        metadata:       Arbitrary key-value dict stored as JSON.
        name_prefix:    Filename prefix.

    Returns:
        Path to the saved model ``.zip`` file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{name_prefix}_{timestep}"
    model_path = ckpt_dir / stem
    model.save(str(model_path))

    if vec_normalize is not None:
        norm_path = ckpt_dir / f"{stem}_vecnorm.pkl"
        vec_normalize.save(str(norm_path))

    meta_path = ckpt_dir / f"{stem}_meta.json"
    meta: dict[str, Any] = metadata or {}
    meta["timestep"] = timestep
    meta["name_prefix"] = name_prefix
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return model_path.with_suffix(".zip")


def load_checkpoint(
    checkpoint_path: str | Path,
    env: Any | None = None,
) -> tuple[PPO, VecNormalize | None, dict[str, Any]]:
    """Load a PPO checkpoint saved by :func:`save_checkpoint`.

    Automatically searches for companion ``_vecnorm.pkl`` and
    ``_meta.json`` files alongside the model ``.zip``.

    Args:
        checkpoint_path: Path to the ``.zip`` model file (or without suffix).
        env:             Optional :class:`VecNormalize`-wrapped env to attach
                         the loaded statistics to.

    Returns:
        (model, vec_normalize_or_None, metadata_dict)
    """
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.suffix != ".zip":
        ckpt_path = ckpt_path.with_suffix(".zip")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    stem = ckpt_path.stem  # e.g. "a1_ppo_500000"
    ckpt_dir = ckpt_path.parent

    model = PPO.load(str(ckpt_path), env=env)

    vec_normalize: VecNormalize | None = None
    norm_path = ckpt_dir / f"{stem}_vecnorm.pkl"
    if norm_path.exists() and env is not None:
        vec_normalize = VecNormalize.load(str(norm_path), env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    metadata: dict[str, Any] = {}
    meta_path = ckpt_dir / f"{stem}_meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            metadata = json.load(f)

    return model, vec_normalize, metadata
