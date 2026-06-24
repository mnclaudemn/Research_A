import yaml
import os

def load_config(path="configs/config.yaml"):
    """
    Load YAML configuration file and return it as a dictionary.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
