"""
Evaluation script for the trained A1 stair-climbing PPO policy.

Usage
-----
    # Evaluate the best saved checkpoint across all terrain types:
    python evaluation/evaluate.py --model-dir checkpoints/

    # Evaluate a specific checkpoint:
    python evaluation/evaluate.py --model checkpoints/a1_ppo_final_5000000.zip

    # Evaluate only on the stair terrain with 20 episodes:
    python evaluation/evaluate.py --terrain stairs --n-episodes 20

    # Save a summary CSV:
    python evaluation/evaluate.py --output results/eval_summary.csv

Outputs
-------
    * Per-terrain evaluation table printed to stdout.
    * Optional CSV summary at --output path.
    * Console log of each episode's reward, height, and success flag.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:
    sys.exit(f"stable-baselines3 required: pip install stable-baselines3\n{exc}")

from envs.a1_stair_env import A1StairClimbEnv, MAX_EPISODE_STEPS
from envs.terrain import TERRAIN_BUILDERS, TERRAIN_NAMES
from utils.checkpointing import load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_model(model_dir: Path) -> Path:
    """Return the best_model.zip inside *model_dir* or the most recent checkpoint."""
    best = model_dir / "best" / "best_model.zip"
    if best.exists():
        return best
    # Fall back to most recent *_final*.zip, then newest *.zip
    candidates = sorted(model_dir.glob("a1_ppo_final*.zip"))
    if not candidates:
        candidates = sorted(model_dir.glob("a1_ppo_*.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No model checkpoints found in {model_dir}. "
            "Run training/train.py first."
        )
    return candidates[-1]


def _make_eval_env(
    terrain_idx: int,
    norm_path: Path | None,
    seed: int = 0,
) -> tuple["DummyVecEnv | VecNormalize", bool]:
    """Create a (possibly VecNormalize-wrapped) evaluation environment."""
    env = DummyVecEnv([
        lambda: A1StairClimbEnv(
            terrain_type=terrain_idx,
            render_mode=None,
            seed=seed,
            use_domain_randomization=False,
        )
    ])
    has_norm = False
    if norm_path is not None and norm_path.exists():
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
        has_norm = True
    return env, has_norm


# ---------------------------------------------------------------------------
# Per-terrain evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    model: PPO,
    terrain_idx: int,
    terrain_name: str,
    norm_path: Path | None,
    n_episodes: int = 10,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run *n_episodes* evaluation episodes on a single terrain.

    Args:
        model:        Loaded SB3 PPO model.
        terrain_idx:  Index into :data:`~envs.terrain.TERRAIN_BUILDERS`.
        terrain_name: Human-readable terrain name for display.
        norm_path:    Path to VecNormalize ``.pkl`` file (may be ``None``).
        n_episodes:   Number of evaluation episodes.
        seed:         Base random seed.
        verbose:      Print per-episode results.

    Returns:
        Dict with keys: terrain, mean_reward, std_reward, mean_height,
        mean_length, success_rate, fall_rate.
    """
    ep_rewards:  list[float] = []
    ep_heights:  list[float] = []
    ep_lengths:  list[int]   = []
    ep_success:  list[bool]  = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 13
        env, _ = _make_eval_env(terrain_idx, norm_path, seed=ep_seed)

        obs = env.reset()
        done = False
        ep_rew = 0.0
        ep_len = 0
        max_z = 0.0
        fell = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done_arr, info_arr = env.step(action)
            done = bool(done_arr[0])
            ep_rew += float(rew[0])
            ep_len += 1
            trunk_z = info_arr[0].get("trunk_z", 0.0)
            max_z = max(max_z, trunk_z)
            if info_arr[0].get("fell", False):
                fell = True

        env.close()
        success = max_z >= 0.55 and not fell
        ep_rewards.append(ep_rew)
        ep_heights.append(max_z)
        ep_lengths.append(ep_len)
        ep_success.append(success)

        if verbose:
            print(
                f"  [{terrain_name}] ep={ep+1:3d}  "
                f"reward={ep_rew:8.1f}  z_max={max_z:.3f}m  "
                f"len={ep_len:4d}  success={int(success)}",
                flush=True,
            )

    results = {
        "terrain":      terrain_name,
        "mean_reward":  float(np.mean(ep_rewards)),
        "std_reward":   float(np.std(ep_rewards)),
        "mean_height":  float(np.mean(ep_heights)),
        "mean_length":  float(np.mean(ep_lengths)),
        "success_rate": float(np.mean(ep_success)),
        "fall_rate":    1.0 - float(np.mean(ep_success)),
        "n_episodes":   n_episodes,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a trained A1 stair-climbing PPO policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      default=None,          help="Path to .zip checkpoint.")
    p.add_argument("--model-dir",  default="./checkpoints", help="Dir containing checkpoints.")
    p.add_argument("--terrain",    default="all",
                   choices=["all", "flat", "stairs", "slope10", "slope20"],
                   help="Which terrain to evaluate on.")
    p.add_argument("--n-episodes", type=int, default=10,  help="Episodes per terrain.")
    p.add_argument("--seed",       type=int, default=0,   help="Random seed.")
    p.add_argument("--output",     default=None,          help="Optional CSV output path.")
    p.add_argument("--verbose",    action="store_true",   help="Print per-episode results.")
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # Locate model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = _find_best_model(model_dir)
    print(f"Loading model: {model_path}")

    # Look for companion VecNorm stats
    stem = model_path.stem
    norm_path = model_dir / f"{stem}_vecnorm.pkl"
    if not norm_path.exists():
        # Try final model norm file
        norm_candidates = list(model_dir.glob("a1_ppo_final*_vecnorm.pkl"))
        norm_path = norm_candidates[-1] if norm_candidates else None  # type: ignore[assignment]

    model = PPO.load(str(model_path))

    # Select terrains
    if args.terrain == "all":
        terrain_indices = list(range(len(TERRAIN_NAMES)))
    else:
        name_to_idx = {n.lower(): i for i, n in enumerate(TERRAIN_NAMES)}
        terrain_indices = [name_to_idx[args.terrain.lower()]]

    all_results: list[dict[str, Any]] = []
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for t_idx in terrain_indices:
        t_name = TERRAIN_NAMES[t_idx]
        print(f"\n--- Terrain: {t_name} ---")
        res = evaluate_policy(
            model=model,
            terrain_idx=t_idx,
            terrain_name=t_name,
            norm_path=norm_path,
            n_episodes=args.n_episodes,
            seed=args.seed,
            verbose=args.verbose,
        )
        all_results.append(res)

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Terrain':<10} {'MeanRew':>10} {'StdRew':>8} {'MaxZ(m)':>9} "
          f"{'EpLen':>7} {'Success%':>9} {'Fall%':>7}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['terrain']:<10} {r['mean_reward']:>10.1f} {r['std_reward']:>8.1f} "
            f"{r['mean_height']:>9.3f} {r['mean_length']:>7.0f} "
            f"{r['success_rate']:>9.1%} {r['fall_rate']:>7.1%}"
        )
    print("=" * 60)

    # Optional CSV save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
