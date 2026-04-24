"""
a1_description – PPO Stair Climbing Framework for Unitree A1 Quadruped.

Package layout::

    a1_description/
    ├── mjcf/           MJCF XML model builder (robot + terrain geometry)
    ├── envs/           Gymnasium environment and terrain generators
    ├── training/       PPO training script (Stable Baselines 3)
    ├── evaluation/     Evaluation and video recording scripts
    ├── utils/          Logging and checkpointing helpers
    ├── configs/        YAML hyperparameter configs
    └── checkpoints/    Saved model checkpoints (written at runtime)
"""
