"""
Utilities for structured logging during PPO training.

Logs episode-level metrics to CSV and optionally mirrors them to
``stdout``.  Designed to be used alongside SB3 TensorBoard logging so
that both human-readable CSV files and TensorBoard scalars are available.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Import curriculum constant to keep the max-stage display in sync.
# Imported lazily inside the print call to avoid a circular import at module load.
_CURRICULUM_MAX_STAGE: int | None = None


def _get_curriculum_max_stage() -> int:
    """Return ``CurriculumTerrainScheduler.N_STAGES - 1`` (cached)."""
    global _CURRICULUM_MAX_STAGE
    if _CURRICULUM_MAX_STAGE is None:
        try:
            from envs.terrain import CurriculumTerrainScheduler
            _CURRICULUM_MAX_STAGE = CurriculumTerrainScheduler.N_STAGES - 1
        except ImportError:
            _CURRICULUM_MAX_STAGE = 4  # fallback default
    return _CURRICULUM_MAX_STAGE


class TrainingLogger:
    """CSV + stdout logger for stair-climbing training metrics.

    Args:
        log_dir:  Directory where ``training_log.csv`` will be written.
        verbose:  Whether to also print metrics to stdout.
        run_name: Optional run identifier prepended to the filename.
    """

    FIELDS = [
        "timestep",
        "episode",
        "reward",
        "max_trunk_z",
        "fell",
        "success",
        "curriculum_stage",
        "curriculum_difficulty",
    ]

    def __init__(
        self,
        log_dir: str | Path,
        verbose: bool = True,
        run_name: str = "",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{run_name}_" if run_name else ""
        self._csv_path = self.log_dir / f"{prefix}training_log_{ts}.csv"

        self._file = self._csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

        self._episode: int = 0

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def log_episode(
        self,
        timestep: int,
        reward: float,
        max_trunk_z: float,
        fell: bool,
        success: bool,
        curriculum_stage: int = 0,
        curriculum_difficulty: float = 0.0,
    ) -> None:
        """Write one row to the CSV log.

        Args:
            timestep:              Current global training timestep.
            reward:                Episode cumulative reward.
            max_trunk_z:           Maximum trunk height reached (m).
            fell:                  Whether the robot fell.
            success:               Whether the episode was a stair success.
            curriculum_stage:      Current curriculum stage index.
            curriculum_difficulty: Current curriculum difficulty (0-1).
        """
        self._episode += 1
        row: dict[str, Any] = {
            "timestep":              timestep,
            "episode":               self._episode,
            "reward":                f"{reward:.4f}",
            "max_trunk_z":           f"{max_trunk_z:.4f}",
            "fell":                  int(fell),
            "success":               int(success),
            "curriculum_stage":      curriculum_stage,
            "curriculum_difficulty": f"{curriculum_difficulty:.4f}",
        }
        self._writer.writerow(row)
        self._file.flush()

        if self.verbose:
            print(
                f"[ep {self._episode:6d} | t={timestep:9d}] "
                f"rew={reward:8.2f}  z_max={max_trunk_z:.3f}m  "
                f"fell={int(fell)}  success={int(success)}  "
                f"curriculum={curriculum_stage}/{_get_curriculum_max_stage()}",
                file=sys.stdout,
                flush=True,
            )

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
