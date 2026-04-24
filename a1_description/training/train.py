"""
PPO training script for Unitree A1 stair climbing.

Usage
-----
    # Train with default config:
    python training/train.py

    # Train with custom config:
    python training/train.py --config configs/ppo_stair.yaml

    # Override a specific hyperparameter:
    python training/train.py --total-timesteps 2000000 --n-envs 4

    # Resume from a checkpoint:
    python training/train.py --resume checkpoints/a1_ppo_500000.zip

    # Evaluate immediately after training (no separate evaluate.py call):
    python training/train.py --eval-after-training

All outputs (TensorBoard logs, CSV logs, model checkpoints) are written to
the directories specified in the config (default: ./logs, ./checkpoints).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from the a1_description root or from within training/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
except ImportError as exc:
    sys.exit(
        "Missing dependencies.  Run:\n"
        "  pip install stable-baselines3[extra] gymnasium mujoco\n"
        f"Original error: {exc}"
    )

from envs.a1_stair_env import A1StairClimbEnv, RewardConfig
from envs.terrain import CurriculumTerrainScheduler, terrain_stairs_partial
from envs.wrappers import StairClimbingWrapper, StairSuccessMonitor
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging import TrainingLogger


# ---------------------------------------------------------------------------
# Default hyperparameters (mirrors configs/ppo_stair.yaml)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "total_timesteps": 5_000_000,
    "n_envs": 8,
    "seed": 42,
    # PPO
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # Policy
    "net_arch": [256, 256],
    # Env
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    # Curriculum
    "curriculum_enabled": True,
    "curriculum_threshold": 0.70,
    "curriculum_window": 50,
    # Logging
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "checkpoint_freq": 250_000,
    "eval_freq": 100_000,
    "n_eval_episodes": 20,
    "verbose": 1,
}


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """SB3 callback that updates the curriculum after each episode.

    Reads episode info from the ``infos`` dict returned by
    :class:`~stable_baselines3.common.vec_env.VecEnv` and forwards success
    flags to the :class:`~envs.terrain.CurriculumTerrainScheduler`.

    When the scheduler advances a stage the new difficulty is propagated to
    the underlying environments via the ``env_method`` call.

    Args:
        scheduler:  The curriculum scheduler instance to update.
        verbose:    Verbosity level (0 = silent).
    """

    def __init__(
        self,
        scheduler: CurriculumTerrainScheduler,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.scheduler = scheduler
        self._prev_stage: int = scheduler.stage

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                success = bool(episode.get("success", False))
                prev = self.scheduler.stage
                self.scheduler.update(success)
                if self.scheduler.stage != prev and self.verbose > 0:
                    print(
                        f"[Curriculum] Stage advanced: {prev} → {self.scheduler.stage} "
                        f"(difficulty={self.scheduler.difficulty:.2f}) "
                        f"at timestep {self.num_timesteps}",
                        flush=True,
                    )
                    # Propagate new difficulty to all envs
                    self.training_env.env_method(
                        "set_attr",
                        "curriculum_difficulty",
                        self.scheduler.difficulty,
                    )
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env_fn(
    seed: int,
    reward_cfg: RewardConfig,
    terrain_type: int | None,
    use_domain_randomization: bool,
    success_height_threshold: float = 0.55,
) -> "callable[[], gym.Env]":  # noqa: F821
    """Return a no-argument callable that creates one training environment."""

    def _factory():
        env = A1StairClimbEnv(
            terrain_type=terrain_type,
            render_mode=None,
            seed=seed,
            reward_cfg=reward_cfg,
            use_domain_randomization=use_domain_randomization,
        )
        env = StairClimbingWrapper(
            env, success_height_threshold=success_height_threshold
        )
        return env

    return _factory


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train(cfg: dict[str, Any], resume_path: str | None = None) -> PPO:
    """Run PPO training for A1 stair climbing.

    Args:
        cfg:         Hyperparameter configuration dict (see :data:`DEFAULTS`).
        resume_path: Optional path to a checkpoint ``.zip`` to resume from.

    Returns:
        The trained :class:`~stable_baselines3.PPO` model.
    """
    seed = cfg["seed"]
    n_envs = cfg["n_envs"]
    log_dir = Path(cfg["log_dir"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # --- Reward config ---
    reward_cfg = RewardConfig()

    # --- Curriculum scheduler ---
    scheduler = CurriculumTerrainScheduler(
        success_threshold=cfg["curriculum_threshold"],
        window=cfg["curriculum_window"],
    ) if cfg["curriculum_enabled"] else None

    # --- Training environments ---
    env_fns = [
        make_env_fn(
            seed=seed + i,
            reward_cfg=reward_cfg,
            terrain_type=None,  # random; curriculum injects difficulty separately
            use_domain_randomization=True,
        )
        for i in range(n_envs)
    ]
    # SubprocVecEnv gives true parallelism; fall back to DummyVecEnv for
    # debugging by setting n_envs=1.
    if n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv(env_fns)

    train_env = VecNormalize(
        train_env,
        norm_obs=cfg["norm_obs"],
        norm_reward=cfg["norm_reward"],
        clip_obs=cfg["clip_obs"],
    )

    # --- Evaluation environment (flat terrain, deterministic) ---
    eval_env = make_vec_env(
        lambda: A1StairClimbEnv(
            terrain_type=1,  # stairs
            render_mode=None,
            seed=seed + 9999,
            use_domain_randomization=False,
        ),
        n_envs=1,
        seed=seed + 9999,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=cfg["norm_obs"],
        norm_reward=False,
        training=False,
        clip_obs=cfg["clip_obs"],
    )

    # --- PPO model ---
    policy_kwargs = dict(
        net_arch=cfg["net_arch"],
        ortho_init=True,
    )

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        model, saved_norm, meta = load_checkpoint(resume_path, env=train_env)
        model.set_env(train_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir) if cfg.get("tensorboard", True) else None,
            verbose=cfg["verbose"],
            seed=seed,
        )

    # --- Callbacks ---
    eval_freq_per_env = max(1, cfg["eval_freq"] // n_envs)
    ckpt_freq_per_env = max(1, cfg["checkpoint_freq"] // n_envs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir / "best"),
        log_path=str(log_dir),
        eval_freq=eval_freq_per_env,
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=cfg["verbose"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_freq_per_env,
        save_path=str(ckpt_dir),
        name_prefix="a1_ppo",
        verbose=0,
    )

    callbacks = [eval_callback, checkpoint_callback]
    if scheduler is not None:
        callbacks.append(CurriculumCallback(scheduler, verbose=cfg["verbose"]))

    # --- Train ---
    print("=" * 60)
    print("A1 Stair Climbing PPO Training")
    print("=" * 60)
    print(f"  total_timesteps : {cfg['total_timesteps']:,}")
    print(f"  n_envs          : {n_envs}")
    print(f"  seed            : {seed}")
    print(f"  log_dir         : {log_dir}")
    print(f"  checkpoint_dir  : {ckpt_dir}")
    if scheduler:
        print(f"  curriculum      : enabled (threshold={cfg['curriculum_threshold']})")
    print("=" * 60)

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=CallbackList(callbacks),
        progress_bar=True,
        reset_num_timesteps=resume_path is None,
    )
    elapsed = time.time() - t0

    # --- Save final model ---
    final_path = save_checkpoint(
        model=model,
        vec_normalize=train_env,
        checkpoint_dir=ckpt_dir,
        timestep=cfg["total_timesteps"],
        metadata={
            "config": cfg,
            "training_time_s": elapsed,
        },
        name_prefix="a1_ppo_final",
    )
    print(f"\nTraining complete in {elapsed / 60:.1f} min.")
    print(f"Final model saved to: {final_path}")

    train_env.close()
    eval_env.close()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO policy for Unitree A1 stair climbing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: use built-in defaults).",
    )
    p.add_argument("--resume",        default=None, help="Checkpoint .zip to resume from.")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-envs",          type=int, default=None)
    p.add_argument("--seed",            type=int, default=None)
    p.add_argument("--learning-rate",   type=float, default=None)
    p.add_argument("--log-dir",         default=None)
    p.add_argument("--checkpoint-dir",  default=None)
    p.add_argument("--no-curriculum",   action="store_true",
                   help="Disable curriculum learning.")
    p.add_argument("--eval-after-training", action="store_true",
                   help="Run evaluation script after training completes.")
    return p.parse_args()


def _load_yaml_config(path: str) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load a config file: pip install pyyaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Flatten nested YAML into the flat dict structure used by DEFAULTS
    cfg: dict[str, Any] = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for k, v in values.items():
                cfg[k] = v
        else:
            cfg[section] = values
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = dict(DEFAULTS)

    # Overlay YAML config
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)
        cfg.update(yaml_cfg)

    # Overlay CLI flags
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        cfg["n_envs"] = args.n_envs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.log_dir is not None:
        cfg["log_dir"] = args.log_dir
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.no_curriculum:
        cfg["curriculum_enabled"] = False

    model = train(cfg, resume_path=args.resume)

    if args.eval_after_training:
        eval_script = Path(__file__).resolve().parent.parent / "evaluation" / "evaluate.py"
        import subprocess
        subprocess.run(
            [sys.executable, str(eval_script), "--model-dir", cfg["checkpoint_dir"]],
            check=False,
        )


if __name__ == "__main__":
    main()
