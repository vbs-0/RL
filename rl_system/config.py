"""Configuration module for the RL system."""

import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration manager that loads settings from YAML files."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration with optional config file path."""
        self._config = {}
        if config_path:
            self.load_config(config_path)
        else:
            # Load default config
            default_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml')
            if os.path.exists(default_path):
                self.load_config(default_path)
    
    def load_config(self, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """Get a configuration value using dot notation (e.g., 'environment.width')."""
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set a configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return self._config.copy()


# Global configuration instance
config = Config()