"""
Reproducibility utilities for deterministic execution and external reproduction.

This module provides:
- Deterministic seeding for all random operations
- Environment validation and setup
- Output comparison and validation
- Variance analysis across multiple runs
- Reproducibility audit framework
"""

import numpy as np
import random
import json
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
import subprocess


class DeterministicSeeding:
    """Manages deterministic seeding across all random operations."""
    
    MASTER_SEED = 42
    
    # Per-module seeds (derived deterministically from master)
    SEEDS = {
        'numpy': 42,
        'random': 43,
        'torch': 44,  # If using PyTorch
        'pybullet': 45,
        'perception': 100,
        'planning': 200,
        'control': 300,
        'validation': 400,
        'generalization': 500,
        'ablation': 600
    }
    
    @classmethod
    def seed_all(cls, master_seed: Optional[int] = None):
        """Seed all random number generators for reproducibility."""
        if master_seed is not None:
            cls.MASTER_SEED = master_seed
            # Regenerate module seeds
            cls.SEEDS = {k: master_seed + i for i, k in enumerate(cls.SEEDS.keys())}
        
        # Seed Python's random
        random.seed(cls.SEEDS['random'])
        
        # Seed NumPy
        np.random.seed(cls.SEEDS['numpy'])
        
        # Seed PyTorch if available
        try:
            import torch
            torch.manual_seed(cls.SEEDS['torch'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cls.SEEDS['torch'])
        except ImportError:
            pass
        
        # Set environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(cls.MASTER_SEED)
        
        return cls.SEEDS
    
    @classmethod
    def get_module_seed(cls, module_name: str) -> int:
        """Get seed for specific module."""
        return cls.SEEDS.get(module_name, cls.MASTER_SEED)


class EnvironmentValidator:
    """Validates environment setup and dependencies."""
    
    REQUIRED_PACKAGES = {
        'numpy': '1.24.3',
        'scipy': '1.11.2',
        'matplotlib': '3.7.2',
        'opencv-python': '4.8.0.76',
        'pybullet': '3.2.5',
        'scikit-learn': '1.3.0',
        'networkx': '3.1'
    }
    
    MIN_PYTHON = (3, 8)
    REC_PYTHON = (3, 10)
    
    @classmethod
    def validate_python_version(cls) -> Dict[str, Any]:
        """Validate Python version."""
        version = sys.version_info
        result = {
            'version': f"{version.major}.{version.minor}.{version.micro}",
            'meets_minimum': version >= cls.MIN_PYTHON,
            'is_recommended': version >= cls.REC_PYTHON
        }
        return result
    
    @classmethod
    def validate_packages(cls) -> Dict[str, Any]:
        """Validate installed packages."""
        results = {}
        for package, required_version in cls.REQUIRED_PACKAGES.items():
            try:
                pkg_name = package.replace('-', '_')
                module = __import__(pkg_name)
                installed_version = getattr(module, '__version__', 'unknown')
                results[package] = {
                    'installed': installed_version,
                    'required': required_version,
                    'matches': installed_version == required_version
                }
            except ImportError:
                results[package] = {
                    'installed': None,
                    'required': required_version,
                    'matches': False
                }
        return results
    
    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_implementation': platform.python_implementation()
        }
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """Complete environment validation."""
        return {
            'python': cls.validate_python_version(),
            'packages': cls.validate_packages(),
            'system': cls.get_system_info()
        }


class OutputComparator:
    """Compares outputs for reproducibility validation."""
    
    @staticmethod
    def compare_numerical(expected: Any, actual: Any, tolerance: float = 1e-6) -> bool:
        """Compare numerical values with tolerance."""
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(expected - actual) <= tolerance
        elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            return np.allclose(expected, actual, atol=tolerance)
        elif isinstance(expected, dict) and isinstance(actual, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False
            return all(OutputComparator.compare_numerical(expected[k], actual[k], tolerance)
                      for k in expected.keys())
        elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(expected) != len(actual):
                return False
            return all(OutputComparator.compare_numerical(e, a, tolerance)
                      for e, a in zip(expected, actual))
        else:
            return expected == actual
    
    @staticmethod
    def compare_outputs(expected_path: str, actual_path: str, 
                       tolerance: float = 1e-6) -> Dict[str, Any]:
        """Compare output files."""
        with open(expected_path) as f:
            expected = json.load(f)
        with open(actual_path) as f:
            actual = json.load(f)
        
        matches = OutputComparator.compare_numerical(expected, actual, tolerance)
        
        return {
            'matches': matches,
            'tolerance': tolerance,
            'expected_path': expected_path,
            'actual_path': actual_path
        }


class VarianceAnalyzer:
    """Analyzes variance across multiple runs."""
    
    @staticmethod
    def measure_variance(metric_values: List[float]) -> Dict[str, float]:
        """Compute variance statistics."""
        values = np.array(metric_values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'cv_percent': float(100 * np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0,
            'ci_95_lower': float(np.percentile(values, 2.5)),
            'ci_95_upper': float(np.percentile(values, 97.5))
        }
    
    @staticmethod
    def analyze_multiple_runs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze variance across multiple runs."""
        metrics = {}
        
        # Extract all metric keys from first result
        if not results:
            return {}
        
        for key in results[0].keys():
            values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
            if values:
                metrics[key] = VarianceAnalyzer.measure_variance(values)
        
        return metrics


class ReproducibilityAudit:
    """Main orchestrator for reproducibility audit."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results = {}
    
    def run_audit(self) -> Dict[str, Any]:
        """Run complete reproducibility audit."""
        # Seed everything
        seeds = DeterministicSeeding.seed_all(self.seed)
        
        # Validate environment
        env_validation = EnvironmentValidator.validate_environment()
        
        # Compile results
        self.results = {
            'seed': self.seed,
            'seeds': seeds,
            'environment': env_validation,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        return self.results
    
    def save_report(self, filepath: str):
        """Save audit report to file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    @staticmethod
    def verify_determinism(func, n_runs: int = 10, seed: int = 42) -> bool:
        """Verify function produces deterministic results."""
        results = []
        for i in range(n_runs):
            DeterministicSeeding.seed_all(seed)
            result = func()
            results.append(result)
        
        # Check all results are identical
        first = results[0]
        return all(OutputComparator.compare_numerical(first, r) for r in results[1:])


# Module-level seed initialization
DeterministicSeeding.seed_all()
