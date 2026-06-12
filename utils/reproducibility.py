"""
Reproducibility Utilities

Ensures deterministic behavior for reproducible experiments.
"""

import torch
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_cudnn_benchmark():
    """
    Enable cuDNN benchmark for faster training.
    
    Note: This may reduce reproducibility slightly but can
    significantly improve performance.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_device(
    device: Optional[str] = None,
    verbose: bool = True
) -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', or specific GPU like 'cuda:0')
        verbose: Print device information
        
    Returns:
        torch.device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    if verbose:
        if device.type == 'cuda':
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        else:
            print(f"Using device: {device}")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    total_params = count_parameters(model)
    print(f"Trainable parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB (assuming float32)")
    print("=" * 70)


class DeterministicContext:
    """
    Context manager for deterministic operations.
    
    Usage:
        with DeterministicContext(seed=42):
            # Your code here
            pass
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.old_benchmark = None
        self.old_deterministic = None
    
    def __enter__(self):
        # Save old settings
        if torch.cuda.is_available():
            self.old_benchmark = torch.backends.cudnn.benchmark
            self.old_deterministic = torch.backends.cudnn.deterministic
        
        # Set seed and deterministic behavior
        set_seed(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old settings
        if torch.cuda.is_available() and self.old_benchmark is not None:
            torch.backends.cudnn.benchmark = self.old_benchmark
            torch.backends.cudnn.deterministic = self.old_deterministic


def setup_reproducibility(
    seed: int = 42,
    benchmark: bool = False
):
    """
    Setup reproducibility for entire experiment.
    
    Args:
        seed: Random seed
        benchmark: Whether to enable cuDNN benchmark (faster but less reproducible)
    """
    # Set seeds
    set_seed(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure cuDNN
    if benchmark:
        enable_cudnn_benchmark()
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Set number of threads (helps with reproducibility)
    torch.set_num_threads(1)
    
    print(f"Reproducibility setup complete (seed={seed}, benchmark={benchmark})")


if __name__ == "__main__":
    print("Testing Reproducibility Utilities:")
    
    # Test seed setting
    set_seed(42)
    print(f"Random number (NumPy): {np.random.rand()}")
    print(f"Random number (PyTorch): {torch.rand(1).item()}")
    
    # Reset and test again
    set_seed(42)
    print(f"Random number (NumPy, after reset): {np.random.rand()}")
    print(f"Random number (PyTorch, after reset): {torch.rand(1).item()}")
    
    # Test device
    device = get_device(verbose=True)
    print(f"\nDevice type: {device.type}")
    
    # Test with deterministic context
    print("\nTesting DeterministicContext:")
    with DeterministicContext(seed=123):
        print(f"Inside context: {np.random.rand()}")
    
    print(f"Outside context: {np.random.rand()}")
