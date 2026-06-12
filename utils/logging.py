"""
Logging Utilities

Provides logging and experiment tracking utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logger(
    name: str = 'change_detection',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    Logger for experiment tracking.
    
    Tracks hyperparameters, metrics, and artifacts.
    
    Args:
        experiment_name: Name of experiment
        output_dir: Directory to save logs
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = 'experiments/runs'
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        log_file = self.exp_dir / 'experiment.log'
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file)
        )
        
        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'output_dir': str(self.exp_dir)
        }
        
        # Metrics storage
        self.metrics = {}
        self.hyperparameters = {}
        
        self.logger.info(f"Experiment started: {experiment_name}")
        self.logger.info(f"Output directory: {self.exp_dir}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.hyperparameters.update(hparams)
        
        self.logger.info("Hyperparameters:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save to file
        hparams_file = self.exp_dir / 'hyperparameters.json'
        with open(hparams_file, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ''
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step or epoch number
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
        
        # Log to console
        step_str = f"Step {step}: " if step is not None else ""
        self.logger.info(f"{step_str}{metrics}")
    
    def log_artifact(self, artifact_name: str, artifact_path: str):
        """
        Log artifact (file) location.
        
        Args:
            artifact_name: Name of artifact
            artifact_path: Path to artifact
        """
        self.logger.info(f"Artifact saved: {artifact_name} -> {artifact_path}")
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_file = self.exp_dir / 'metrics.json'
        
        # Convert to serializable format
        metrics_dict = {}
        for key, values in self.metrics.items():
            metrics_dict[key] = [
                {'step': step, 'value': value}
                for step, value in values
            ]
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def finish(self):
        """Finish experiment and save all data."""
        self.save_metrics()
        
        # Save metadata
        metadata_file = self.exp_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info("Experiment finished")
        self.logger.info(f"All files saved to: {self.exp_dir}")


class MetricsTracker:
    """
    Simple metrics tracker for training/validation.
    
    Tracks moving averages and best values.
    """
    
    def __init__(self):
        self.metrics = {}
        self.best_values = {}
    
    def update(self, metric_name: str, value: float):
        """
        Update metric with new value.
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        # Update best value (assuming lower is better for losses)
        if metric_name not in self.best_values:
            self.best_values[metric_name] = value
        else:
            if 'loss' in metric_name.lower():
                self.best_values[metric_name] = min(self.best_values[metric_name], value)
            else:
                self.best_values[metric_name] = max(self.best_values[metric_name], value)
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """
        Get average of metric.
        
        Args:
            metric_name: Name of metric
            last_n: Average over last N values (None for all)
            
        Returns:
            Average value
        """
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values) if len(values) > 0 else 0.0
    
    def get_best(self, metric_name: str) -> float:
        """Get best value for metric."""
        return self.best_values.get(metric_name, 0.0)
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.best_values.clear()


if __name__ == "__main__":
    print("Testing Logging Utilities:")
    
    # Test basic logger
    logger = setup_logger('test_logger', log_file='/tmp/test.log')
    logger.info("This is an info message")
    logger.warning("This is a warning")
    
    # Test experiment logger
    print("\nTesting ExperimentLogger:")
    exp_logger = ExperimentLogger(
        experiment_name='test_experiment',
        output_dir='/tmp/experiments'
    )
    
    exp_logger.log_hyperparameters({
        'learning_rate': 1e-4,
        'batch_size': 16,
        'epochs': 100
    })
    
    exp_logger.log_metrics({'loss': 0.5, 'accuracy': 0.85}, step=1, prefix='train')
    exp_logger.log_metrics({'loss': 0.4, 'accuracy': 0.90}, step=2, prefix='train')
    
    exp_logger.finish()
    
    # Test metrics tracker
    print("\nTesting MetricsTracker:")
    tracker = MetricsTracker()
    tracker.update('train_loss', 0.5)
    tracker.update('train_loss', 0.4)
    tracker.update('train_loss', 0.3)
    
    print(f"Average train loss: {tracker.get_average('train_loss'):.4f}")
    print(f"Best train loss: {tracker.get_best('train_loss'):.4f}")
