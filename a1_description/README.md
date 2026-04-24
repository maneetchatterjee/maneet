# A1 Stair Climbing – PPO Reinforcement Learning Framework

A production-ready **Proximal Policy Optimization (PPO)** pipeline for
training the **Unitree A1** quadruped robot to climb stairs, built on
[MuJoCo](https://mujoco.org/) physics simulation and
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/).

---

## Directory Layout

```
a1_description/
├── mjcf/
│   ├── __init__.py
│   └── model_builder.py       # MJCF XML generator (robot + terrain)
├── envs/
│   ├── __init__.py
│   ├── a1_stair_env.py        # Gymnasium environment (obs, act, reward)
│   ├── terrain.py             # Terrain builders + curriculum scheduler
│   └── wrappers.py            # Episode-metric wrappers
├── training/
│   └── train.py               # PPO training entry-point (SB3)
├── evaluation/
│   ├── evaluate.py            # Cross-terrain evaluation + CSV report
│   └── visualize.py           # Video recording (MP4)
├── utils/
│   ├── logging.py             # CSV episode logger
│   └── checkpointing.py       # Save / load checkpoints + metadata
├── configs/
│   └── ppo_stair.yaml         # Default hyperparameter config
├── checkpoints/               # Written at training time
├── requirements.txt
└── README.md                  # ← you are here
```

---

## Installation

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/maneetchatterjee/maneet.git
cd maneet/a1_description

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux / macOS
# python -m venv .venv && .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) GPU-accelerated PyTorch for faster training
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**System requirements:**
- Python ≥ 3.9
- MuJoCo ≥ 3.1.0 (automatically installed via pip)
- For offscreen video rendering: `libosmesa6-dev` (Linux) or
  set `MUJOCO_GL=egl` on headless servers.

---

## Quick Start

### Train (default config, 5 M steps)

```bash
# From the a1_description/ directory:
python training/train.py
```

This uses 8 parallel environments, curriculum learning, and saves
checkpoints to `./checkpoints/` every 250 000 timesteps.

### Train with a custom config

```bash
python training/train.py --config configs/ppo_stair.yaml
```

### Override individual hyperparameters

```bash
python training/train.py \
    --total-timesteps 2000000 \
    --n-envs 4 \
    --learning-rate 1e-4 \
    --log-dir ./my_logs \
    --checkpoint-dir ./my_checkpoints
```

### Resume from a checkpoint

```bash
python training/train.py \
    --resume checkpoints/a1_ppo_2500000.zip
```

### Disable curriculum learning

```bash
python training/train.py --no-curriculum
```

---

## Evaluation

### Evaluate across all terrain types

```bash
python evaluation/evaluate.py --model-dir checkpoints/
```

### Evaluate a specific checkpoint, stairs only

```bash
python evaluation/evaluate.py \
    --model checkpoints/best/best_model.zip \
    --terrain stairs \
    --n-episodes 30 \
    --output results/stairs_eval.csv
```

### Expected output

```
============================================================
Evaluation Results
============================================================
Terrain     MeanRew    StdRew   MaxZ(m)   EpLen  Success%    Fall%
------------------------------------------------------------
Flat         1843.2     120.5     0.421     800    100.0%     0.0%
Stairs        987.4     234.8     0.721     650     82.0%    18.0%
Slope10      1312.1     178.3     0.510     720     91.0%     9.0%
Slope20       831.2     298.7     0.488     580     73.0%    27.0%
============================================================
```

---

## Video Recording

```bash
# Record one MP4 per terrain (requires imageio[ffmpeg]):
python evaluation/visualize.py --model-dir checkpoints/

# Record stair-climb video only, 30 fps, 500 step limit:
python evaluation/visualize.py \
    --terrain stairs \
    --fps 30 \
    --max-steps 500 \
    --video-dir videos/

# Headless rendering tip (Linux without display):
MUJOCO_GL=osmesa python evaluation/visualize.py --terrain stairs
```

---

## Configuration Reference (`configs/ppo_stair.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `training` | `total_timesteps` | 5 000 000 | Total env interactions |
| `training` | `n_envs` | 8 | Parallel environments |
| `training` | `seed` | 42 | Master RNG seed |
| `ppo` | `learning_rate` | 3e-4 | Adam step size |
| `ppo` | `n_steps` | 2048 | Rollout steps per env |
| `ppo` | `batch_size` | 512 | Mini-batch size |
| `ppo` | `n_epochs` | 10 | Gradient passes per rollout |
| `ppo` | `gamma` | 0.99 | Discount factor |
| `ppo` | `gae_lambda` | 0.95 | GAE λ |
| `ppo` | `clip_range` | 0.2 | PPO clipping ε |
| `ppo` | `ent_coef` | 0.005 | Entropy bonus |
| `policy` | `net_arch` | [256, 256] | Actor/critic hidden sizes |
| `curriculum` | `enabled` | true | Enable curriculum learning |
| `curriculum` | `success_threshold` | 0.70 | Success rate to advance stage |
| `curriculum` | `window` | 50 | Episode window for success rate |
| `reward` | `forward_vel` | 2.0 | Weight for forward velocity |
| `reward` | `height_gain` | 8.0 | Weight for trunk height gain |
| `reward` | `upright` | 1.0 | Tilt penalty weight |
| `reward` | `fall` | −20.0 | Fall penalty |
| `logging` | `checkpoint_freq` | 250 000 | Steps between checkpoints |
| `logging` | `eval_freq` | 100 000 | Steps between evaluations |

---

## Observation and Action Spaces

### Observation (49-dimensional float32)

| Index | Size | Description |
|-------|------|-------------|
| 0–2   | 3 | Base linear velocity (world frame) |
| 3–5   | 3 | Base angular velocity (body frame, IMU gyro) |
| 6–8   | 3 | Projected gravity in body frame |
| 9–20  | 12 | Joint positions relative to default standing pose |
| 21–32 | 12 | Joint velocities |
| 33–44 | 12 | Previous actions |
| 45–48 | 4 | Binary foot-contact states (FL, FR, RL, RR) |

### Action (12-dimensional, ∈ [−1, 1])

Normalised delta joint position targets, scaled by 0.3 rad and added to
the default standing pose before being sent to the PD position servos.

---

## Reward Function

| Component | Formula | Purpose |
|-----------|---------|---------|
| `forward_vel` | `2.0 · exp(−2·(vₓ − 0.4)²)` | Forward progress |
| `height_gain` | `8.0 · max(Δz, 0)` | Stair ascent |
| `upright` | `−1.0·(roll² + pitch²)` | Stability |
| `foot_contact` | `0.2·min(n_contacts/2, 1)` | Stance quality |
| `gait_symmetry` | `0.3·(diag₁+diag₂)/2` | Diagonal trot gait |
| `action_smooth` | `−0.05·‖Δa‖²` | Smooth control |
| `torque_smooth` | `−0.002·‖τ‖²` | Energy efficiency |
| `alive` | `+0.5` per step | Episode longevity |
| `fall` | `−20.0` (once) | Fall deterrence |

---

## Curriculum Learning

Training starts on flat ground and progressively introduces taller stairs:

| Stage | Step height | Description |
|-------|-------------|-------------|
| 0 | 0.00 m | Flat ground |
| 1 | 0.04 m | Very small steps |
| 2 | 0.08 m | Small steps |
| 3 | 0.13 m | Medium steps |
| 4 | 0.17 m | Full A1 stair height |

The scheduler advances one stage when the per-episode success rate
(robot reaches ≥ 0.55 m trunk height without falling) exceeds 70 % over
the last 50 episodes.

---

## Checkpointing and Export

All checkpoints are saved as a trio of files:

```
checkpoints/
├── a1_ppo_250000.zip           # SB3 model weights
├── a1_ppo_250000_vecnorm.pkl   # VecNormalize running statistics
├── a1_ppo_250000_meta.json     # Config + timestep metadata
├── best/
│   └── best_model.zip          # Best evaluation checkpoint (SB3 EvalCallback)
└── a1_ppo_final_5000000.zip    # Final model after full training
```

Load a checkpoint in Python:

```python
from a1_description.utils.checkpointing import load_checkpoint

model, vec_norm, meta = load_checkpoint("checkpoints/a1_ppo_final_5000000.zip")
print(meta)  # {'timestep': 5000000, 'config': {...}, ...}
```

---

## Monitoring Training

TensorBoard logs are written to `./logs/`:

```bash
tensorboard --logdir logs/
```

Key scalars to watch:
- `eval/mean_reward` – evaluation reward (higher is better)
- `train/policy_gradient_loss` – should decrease early in training
- `train/entropy_loss` – should stay mildly negative (≈ −0.5 to −2.0)
- `train/explained_variance` – should increase toward 1.0

---

## Domain Randomization

Each reset applies the following perturbations to improve sim-to-real transfer:

| Parameter | Range |
|-----------|-------|
| Floor friction | ×U[0.7, 1.3] |
| Trunk mass | ×U[0.8, 1.2] |
| Initial trunk roll | U[−0.2, 0.2] rad |
| Initial trunk pitch | U[−0.15, 0.15] rad |
| Initial joint positions | ±U[0.0, 0.2] rad from default |
| Initial linear velocity | U[−0.2, 0.2] m/s |
| Actuator noise | N(0, 0.01) rad per step |

---

## Robot Model

The A1 model is built programmatically from Unitree A1 URDF/datasheet constants:

| Parameter | Value |
|-----------|-------|
| Trunk mass | 6.0 kg |
| Hip KP / KD | 100 / 5 |
| Thigh KP / KD | 300 / 8 |
| Calf KP / KD | 300 / 8 |
| Physics timestep | 2 ms (500 Hz) |
| Control frequency | 50 Hz (10 sub-steps) |

Stair geometry matches `stairs.xacro`:

| Parameter | Value |
|-----------|-------|
| Step height | 0.170 m |
| Step depth (tread) | 0.310 m |
| Staircase width | 0.640 m |
| Number of steps | 6 |

---

## Citation / References

- Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
- Kumar et al., "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning", 2021.
- Unitree Robotics A1 datasheet and SDK.
- MuJoCo physics engine – DeepMind.
- Stable Baselines 3 – Raffin et al., JMLR 2021.
