# Configuration loading and validation utilities
"""YAML configuration loading with validation."""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def save_config(config: DictConfig, save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(config), f, default_flow_style=False)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations with override taking precedence."""
    return OmegaConf.merge(base_config, override_config)
