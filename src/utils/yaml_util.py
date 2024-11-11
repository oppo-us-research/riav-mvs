"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import yaml
from typing import Dict, Optional

def update_dict_recursive(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively updates the base dictionary with values from the update dictionary.

    Args:
        base_dict: The original dictionary to be updated.
        update_dict: The dictionary with updates to be applied to the base dictionary.
    """
    for key, value in update_dict.items():
        if key not in base_dict:
            base_dict[key] = {}
        if isinstance(value, dict):
            update_dict_recursive(base_dict[key], value)
        else:
            base_dict[key] = value

def load_config(config_path: str, default_config_path: Optional[str] = None) -> Dict:
    """Loads a configuration file and optionally merges it with a default configuration.

    Args:
        config_path: Path to the primary configuration file.
        default_config_path: Path to the default configuration file (optional).

    Returns:
        A dictionary containing the loaded configuration, 
        with updates applied from the primary configuration.
    """
    # Load the primary configuration from the specified path
    with open(config_path, 'r') as config_file:
        primary_config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Check if the configuration should inherit from another configuration
    inherit_path = primary_config.get('inherit_from')

    # If inheritance is specified, load the inherited configuration first
    if inherit_path is not None:
        config = load_config(inherit_path, default_config_path)
    elif default_config_path is not None:
        # If no inheritance, load the default configuration if provided
        with open(default_config_path, 'r') as default_file:
            config = yaml.load(default_file, Loader=yaml.FullLoader)
    else:
        # If no default configuration, start with an empty dictionary
        config = {}

    # Update the configuration with values from the primary configuration
    update_dict_recursive(config, primary_config)

    return config

