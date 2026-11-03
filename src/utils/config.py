import yaml
import copy
from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """Returns the hardcoded default configuration dictionary."""
    return {
        "model": {
            "name": "pad_model",
            "backbone": "resnet18",
            "weights": None
        },
        "data": {
            "input_size": [224, 224],
            "batch_size": 32,
            "num_workers": 4
        },
        "training": {
            "epochs": 50,
            "learning_rate": 0.001
        }
    }

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merges two configuration dictionaries."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file and merges it with the defaults."""
    with open(config_path, 'r') as f:
        override = yaml.safe_load(f) or {}
    
    defaults = get_default_config()
    return merge_configs(defaults, override)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validates that required keys exist in the configuration."""
    required_keys = ["model", "data", "training"]
    for key in required_keys:
        if key not in config:
            return False
    return True
