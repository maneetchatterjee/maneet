"""
Video recording and visualization script for the A1 stair-climbing policy.

Usage
-----
    # Record one video per terrain type:
    python evaluation/visualize.py --model-dir checkpoints/

    # Record stair terrain video only, 30 fps:
    python evaluation/visualize.py --terrain stairs --fps 30

    # Record from a specific checkpoint:
    python evaluation/visualize.py --model checkpoints/a1_ppo_final_5000000.zip

    # Save videos to a custom directory:
    python evaluation/visualize.py --video-dir videos/run1

Outputs
-------
    MP4 files written to *--video-dir* (default: ``./videos``).
    Requires ``imageio[ffmpeg]``:  pip install imageio[ffmpeg]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    import imageio
except ImportError as exc:
    sys.exit(
        "imageio[ffmpeg] is required for video generation: "
        "pip install imageio[ffmpeg]\n" + str(exc)
    )

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:
    sys.exit(f"stable-baselines3 required: pip install stable-baselines3\n{exc}")

from envs.a1_stair_env import A1StairClimbEnv, MAX_EPISODE_STEPS
from envs.terrain import TERRAIN_BUILDERS, TERRAIN_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_model(model_dir: Path) -> Path:
    best = model_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    candidates = sorted(model_dir.glob("a1_ppo_final*.zip"))
    if not candidates:
        candidates = sorted(model_dir.glob("a1_ppo_*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}.")
    return candidates[-1]


def record_episode(
    model: PPO,
    terrain_idx: int,
    max_steps: int = MAX_EPISODE_STEPS,
    fps: int = 25,
    seed: int = 0,
    video_dir: Path = Path("./videos"),
) -> Path:
    """Record one episode to an MP4 file.

    Args:
        model:       Loaded SB3 PPO model.
        terrain_idx: Index into :data:`~envs.terrain.TERRAIN_BUILDERS`.
        max_steps:   Maximum episode length.
        fps:         Video frames per second.
        seed:        Environment seed.
        video_dir:   Output directory for the MP4 file.

    Returns:
        Path to the saved MP4 file.
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    terrain_name = TERRAIN_NAMES[terrain_idx]

    env = A1StairClimbEnv(
        terrain_type=terrain_idx,
        render_mode="rgb_array",
        seed=seed,
        use_domain_randomization=False,
    )

    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    fell = False

    for step in range(max_steps):
        action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
        action = action[0]
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if info.get("fell", False):
            fell = True

        if terminated or truncated:
            break

    env.close()

    out_path = video_dir / f"a1_stair_{terrain_name.lower()}.mp4"
    if frames:
        imageio.mimsave(str(out_path), frames, fps=fps, quality=5)
        print(
            f"  [{terrain_name:8s}] {len(frames):4d} frames  "
            f"reward={total_reward:8.1f}  fell={int(fell)}  → {out_path}"
        )
    else:
        print(
            f"  [{terrain_name:8s}] No frames rendered "
            f"(headless environment without osmesa/egl driver)."
        )
        print(
            "  Tip: Set MUJOCO_GL=osmesa and install libosmesa6-dev for "
            "offscreen rendering."
        )

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Record evaluation videos for the A1 stair-climbing policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      default=None,          help="Path to .zip checkpoint.")
    p.add_argument("--model-dir",  default="./checkpoints", help="Dir containing checkpoints.")
    p.add_argument("--terrain",    default="all",
                   choices=["all", "flat", "stairs", "slope10", "slope20"],
                   help="Terrain to record.")
    p.add_argument("--max-steps",  type=int, default=500,   help="Max steps per episode.")
    p.add_argument("--fps",        type=int, default=25,    help="Video frame rate.")
    p.add_argument("--seed",       type=int, default=0,     help="Random seed.")
    p.add_argument("--video-dir",  default="./videos",      help="Output directory.")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = _find_best_model(model_dir)

    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path))
    video_dir = Path(args.video_dir)

    if args.terrain == "all":
        terrain_indices = list(range(len(TERRAIN_NAMES)))
    else:
        name_to_idx = {n.lower(): i for i, n in enumerate(TERRAIN_NAMES)}
        terrain_indices = [name_to_idx[args.terrain.lower()]]

    print("\nRecording evaluation videos...")
    for t_idx in terrain_indices:
        record_episode(
            model=model,
            terrain_idx=t_idx,
            max_steps=args.max_steps,
            fps=args.fps,
            seed=args.seed + t_idx,
            video_dir=video_dir,
        )

    print(f"\nAll videos saved to: {video_dir.resolve()}")


if __name__ == "__main__":
    main()
