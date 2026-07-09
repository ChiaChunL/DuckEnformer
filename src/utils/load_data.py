from pathlib import Path
from typing import Dict


from .logger_util import get_logger

logger = get_logger(__name__, log_type="Load data", output_dir="../../logs")

def load_yaml(path: Path) -> Dict:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_json(path: Path) -> dict:
    import json
    with open(path) as f:
        return json.load(f)