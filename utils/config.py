from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path | None = None) -> dict:
    """Load the simulation configuration from a YAML file."""

    config_path = Path(path) if path else Path(__file__).parent.parent / "config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
