from __future__ import annotations

from pathlib import Path

import yaml

# The original money world is the default for callers that do not name a
# world. Worlds that own a different config pass its path explicitly.
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "worlds" / "money" / "config.yaml"
)


def load_config(path: str | Path | None = None) -> dict:
    """Load a simulation configuration from a YAML file."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
