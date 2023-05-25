import os
import sys

import yaml
from addict import Dict


def load_config(file_path: str) -> dict:
    if not os.path.exists(file_path):
        print("Config file not found!")
        sys.exit(1)
    with open(file_path, "r") as f:
        config = Dict(yaml.safe_load(f))
    return config


def save_config(config: dict, file_path: str) -> None:
    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
