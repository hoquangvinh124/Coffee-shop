"""Configuration Module"""

import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}
