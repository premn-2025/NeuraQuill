from pathlib import Path

import yaml

from .config import GPTConfig


def load_config(path: str) -> GPTConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GPTConfig(**data)
