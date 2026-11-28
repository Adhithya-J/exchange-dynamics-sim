"""Shared replay serialization and validation primitives.

A replay is the boundary between a world and any renderer or analysis tool.
Each world owns its own schema and event vocabulary, but the mechanics of
turning Python values into deterministic JSON, writing compressed files, and
checking required numbers are shared here.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Return a JSON-serializable copy of ``value``."""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def replay_to_json(replay: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialize a replay deterministically."""

    separators = None if indent is not None else (",", ":")
    return json.dumps(
        replay,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=separators,
    )


def write_replay_json(
    replay: dict[str, Any],
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Write ``replay`` as UTF-8 JSON; ``.gz`` paths are compressed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = replay_to_json(replay, indent=indent)
    if output_path.suffix == ".gz":
        # mtime=0 and an empty filename keep the compressed bytes deterministic.
        with output_path.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                compressed.write(text.encode("utf-8"))
    else:
        output_path.write_text(text, encoding="utf-8")
    return output_path


def load_replay_json(path: str | Path) -> dict[str, Any]:
    """Read a replay written by :func:`write_replay_json`."""

    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with input_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def required_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def required_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def stock_ratio(total: float, initial_total: float) -> float:
    if initial_total <= 0:
        return 0.0
    return total / initial_total
