# Checkpointing utilities for saving and loading model states
"""Comprehensive checkpointing with model, optimizer, and RNG states."""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from .seeding import get_rng_state, set_rng_state


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
):
    """Save checkpoint with model, optimizer, and RNG states."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_step_{step}.pt"
    
    checkpoint = {
        'step': step,
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'rng_state': get_rng_state(),
        'metadata': metadata or {},
    }
    
    save_path = checkpoint_dir / filename
    torch.save(checkpoint, save_path)
    
    # Also save as "latest" for easy resume
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    return str(save_path)


def load_checkpoint(checkpoint_path: str, restore_rng: bool = True) -> Dict[str, Any]:
    """Load checkpoint and optionally restore RNG states."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if restore_rng and 'rng_state' in checkpoint:
        set_rng_state(checkpoint['rng_state'])
    
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    
    if latest_path.exists():
        return str(latest_path)
    
    # Fallback: find highest step number
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    
    # Extract step numbers and find max
    steps = []
    for cp in checkpoints:
        try:
            step = int(cp.stem.split('_')[-1])
            steps.append((step, cp))
        except ValueError:
            continue
    
    if not steps:
        return None
    
    return str(max(steps, key=lambda x: x[0])[1])
