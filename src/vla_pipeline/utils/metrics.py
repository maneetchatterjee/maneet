"""
Utility Module

Logging, metrics, and helper functions.
"""

import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExecutionMetrics:
    """Metrics for task execution."""
    task_id: str
    command: str
    success: bool
    execution_time: float
    num_waypoints: int
    num_actions: int
    failure_mode: str = "none"
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsLogger:
    """
    Logs execution metrics and performance data.
    """
    
    def __init__(self, log_file: str = "metrics.json"):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Path to metrics log file
        """
        self.log_file = log_file
        self.metrics_history: List[ExecutionMetrics] = []
    
    def log_execution(
        self,
        task_id: str,
        command: str,
        success: bool,
        execution_time: float,
        num_waypoints: int,
        num_actions: int,
        failure_mode: str = "none"
    ):
        """Log execution metrics."""
        metrics = ExecutionMetrics(
            task_id=task_id,
            command=command,
            success=success,
            execution_time=execution_time,
            num_waypoints=num_waypoints,
            num_actions=num_actions,
            failure_mode=failure_mode,
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics_history.append(metrics)
    
    def save_metrics(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            data = [m.to_dict() for m in self.metrics_history]
            json.dump(data, f, indent=2)
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.metrics_history:
            return 0.0
        
        successes = sum(1 for m in self.metrics_history if m.success)
        return successes / len(self.metrics_history)
    
    def get_average_execution_time(self) -> float:
        """Calculate average execution time."""
        if not self.metrics_history:
            return 0.0
        
        total_time = sum(m.execution_time for m in self.metrics_history)
        return total_time / len(self.metrics_history)
    
    def get_failure_modes(self) -> Dict[str, int]:
        """Get counts of different failure modes."""
        failure_counts = {}
        for m in self.metrics_history:
            if not m.success:
                failure_counts[m.failure_mode] = failure_counts.get(m.failure_mode, 0) + 1
        return failure_counts
    
    def print_summary(self):
        """Print metrics summary."""
        print("\n" + "="*60)
        print("EXECUTION METRICS SUMMARY")
        print("="*60)
        print(f"Total Tasks: {len(self.metrics_history)}")
        print(f"Success Rate: {self.get_success_rate()*100:.1f}%")
        print(f"Average Execution Time: {self.get_average_execution_time():.3f}s")
        
        failure_modes = self.get_failure_modes()
        if failure_modes:
            print("\nFailure Modes:")
            for mode, count in failure_modes.items():
                print(f"  - {mode}: {count}")
        print("="*60 + "\n")


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time


def validate_workspace_bounds(
    position: tuple,
    bounds: Dict[str, tuple]
) -> bool:
    """
    Validate if position is within workspace bounds.
    
    Args:
        position: (x, y, z) position
        bounds: Dictionary with 'x', 'y', 'z' keys mapping to (min, max) tuples
        
    Returns:
        True if position is valid
    """
    x, y, z = position
    
    if not (bounds['x'][0] <= x <= bounds['x'][1]):
        return False
    if not (bounds['y'][0] <= y <= bounds['y'][1]):
        return False
    if not (bounds['z'][0] <= z <= bounds['z'][1]):
        return False
    
    return True
