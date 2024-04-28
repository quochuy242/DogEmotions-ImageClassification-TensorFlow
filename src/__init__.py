import yaml
from pathlib import Path


def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
