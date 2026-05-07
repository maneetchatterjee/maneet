# Logging utilities for TensorBoard, JSON, and CSV
"""Multi-format logging for training metrics."""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Multi-format logger for training metrics."""
    
    def __init__(self, log_dir: str, log_name: str = "run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # JSON log file
        self.json_path = self.log_dir / f"{log_name}_episodes.json"
        self.json_data = []
        
        # CSV log file
        self.csv_path = self.log_dir / f"{log_name}_summary.csv"
        self.csv_file = None
        self.csv_writer = None
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar to TensorBoard."""
        self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars to TensorBoard."""
        self.tb_writer.add_scalars(tag, values, step)
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """Log episode data to JSON."""
        self.json_data.append(episode_data)
        with open(self.json_path, 'w') as f:
            json.dump(self.json_data, f, indent=2)
    
    def log_csv_row(self, row: Dict[str, Any]):
        """Log row to CSV summary."""
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=row.keys())
            self.csv_writer.writeheader()
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def close(self):
        """Close all log files."""
        self.tb_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
