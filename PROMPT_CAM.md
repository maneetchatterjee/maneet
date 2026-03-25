# Prompt_CAM Repository Notes

Source repository: https://github.com/Imageomics/Prompt_CAM

## Overview
Prompt-CAM is the official implementation of **"PROMPT-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis" (CVPR 2025)**.

The project introduces class-specific prompts for Vision Transformers (ViTs) to improve interpretability without changing the base architecture.

## Key repository structure
- `data/`
- `engine/`
- `experiment/`
- `model/`
- `samples/`
- `utils/`
- `main.py`
- `visualize.py`
- `demo.ipynb`

## Quick start (from upstream README)
1. Create a conda environment with Python 3.10.
2. Activate the environment and run `source env_setup.sh`.
3. Prepare data under a dataset directory passed via `--data_path`.
4. Download checkpoints and place them in `checkpoints/{model}/{dataset}/`.
5. Run visualization with `visualize.py` using config/checkpoint paths.

## Note about this workspace
Direct `git pull` from GitHub is blocked in this environment (HTTP CONNECT 403), so this workspace contains notes instead of a full mirror clone.

## Automated runner notebook in this repo
Use `run_prompt_cam_workflow.ipynb` to execute the README workflow end-to-end (clone/update, environment setup, demo execution, training command template, visualization command template).
