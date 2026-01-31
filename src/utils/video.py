# Video recording utilities for environment rollouts
"""Video recording for policy evaluation."""

import numpy as np
import imageio
from pathlib import Path
from typing import List


def save_video(frames: List[np.ndarray], save_path: str, fps: int = 30):
    """Save frames as video file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert frames to uint8 if needed
    if frames[0].dtype != np.uint8:
        frames = [(frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8) 
                  for frame in frames]
    
    imageio.mimsave(save_path, frames, fps=fps)


def save_gif(frames: List[np.ndarray], save_path: str, fps: int = 30):
    """Save frames as GIF file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert frames to uint8 if needed
    if frames[0].dtype != np.uint8:
        frames = [(frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8) 
                  for frame in frames]
    
    imageio.mimsave(save_path, frames, fps=fps)
