"""Versioned replay logs for the exchange dynamics simulation.

A replay is the boundary between the simulation and anything that renders or
analyses it. It contains only recorded facts: the configuration and seed,
stable agent metadata, one snapshot and metric row per iteration, and typed
events. It deliberately contains no hut positions, drawing code, or other
presentation details, so a viewer can change without touching the model.

Build one from a finished simulation::

    from main import ResourceSimulation
    from replay import build_replay, write_replay

    simulation = ResourceSimulation()
    simulation.run(iterations=1000)
    write_replay(build_replay(simulation), "replay.json.gz")

Or from the command line::

    python replay.py --output replay.json.gz --iterations 1000
    python main.py --replay replay.json.gz --iterations 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from main import ResourceSimulation

SCHEMA_VERSION = 1

# Stable identity colors. The replay records the default so every consumer
# colors an agent the same way; a viewer may still override them.
AGENT_COLORS = [
    "#2f6fed",
    "#d97706",
    "#0f9d76",
    "#d6336c",
    "#7c3aed",
    "#0e7490",
    "#ea580c",
    "#65a30d",
    "#9f1239",
    "#475569",
    "#a16207",
    "#0891b2",
]

# Phases follow the order in which the simulation applies them within a tick.
PHASE_ORDER = {
    "shock": 0,
    "exchange": 1,
    "market": 2,
    "cost": 3,
}
EVENT_TYPES = {"gift", "repayment", "trade", "death", "shock"}
DEATH_PHASES = {
    "gift": "exchange",
    "repayment": "exchange",
    "market": "market",
    "living_cost": "cost",
}


def export_replay(
    simulation: ResourceSimulation,
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Build a replay from ``simulation`` and write it to ``path``."""

    return write_replay(build_replay(simulation), path, indent=indent)


def build_replay(simulation: ResourceSimulation) -> dict[str, Any]:
    """Return a JSON-serializable replay for an initialized simulation."""

    if not simulation.agent_history:
        raise ValueError(
            "The simulation must be initialized or run before building a replay"
        )
    if not simulation.agents:
        raise ValueError("The simulation has no agents to record")

    agent_ids = [agent.id for agent in simulation.agents]
    agent_index = {agent_id: index for index, agent_id in enumerate(agent_ids)}
    history_by_tick = _history_by_tick(simulation.agent_history, agent_index)
    last_iteration = max(history_by_tick)
    metrics_by_iteration = {
        int(metrics["iteration"]): metrics
        for metrics in simulation.metrics_history
    }

    settings = simulation.config["SIMULATION"]
    agent_settings = simulation.config["AGENTS"]
    initial_total = float(
        sum(float(row["resources"]) for row in history_by_tick[0].values())
    )
    initial_resources = initial_total / len(agent_ids)
    agent_metadata = _agent_metadata(simulation, history_by_tick[0])

    ticks = []
    for iteration in range(last_iteration + 1):
        rows = history_by_tick.get(iteration)
        if rows is None:
            raise ValueError(f"Missing agent snapshot for iteration {iteration}")
        metrics = metrics_by_iteration.get(iteration)
        if metrics is None:
            metrics = _metrics_from_rows(rows)
        metrics = _jsonable(metrics)
        metrics.setdefault("iteration", iteration)
        metrics["stock_ratio"] = _stock_ratio(
            float(metrics.get("total", 0.0)), initial_total
        )
        ticks.append(
            {
                "iteration": iteration,
                "metrics": metrics,
                "agents": [
                    {
                        "id": agent_id,
                        "resources": float(rows[agent_id]["resources"]),
                        "goods": float(rows[agent_id].get("goods", 0.0)),
                        "alive": bool(rows[agent_id]["alive"]),
                        "risk_tolerance": float(
                            rows[agent_id].get("risk_tolerance", 0.5)
                        ),
                        "reputation": _optional_float(
                            rows[agent_id].get("reputation")
                        ),
                    }
                    for agent_id in agent_ids
                ],
            }
        )

    config = _jsonable(deepcopy(simulation.config))
    # Record what actually ran, not just what the input file asked for.
    recorded_settings = config.setdefault("SIMULATION", {})
    recorded_settings["ITERATIONS"] = last_iteration
    recorded_settings["AGENTS"] = len(agent_ids)
    events = _build_events(simulation, agent_index, last_iteration)
    market = config.get("MARKET") or {}
    trust = config.get("TRUST") or {}
    shocks = config.get("SHOCKS") or {}
    social = config.get("SOCIAL") or {}
    transfer_range = settings.get(
        "TRANSFER_AMOUNT_RANGE",
        [settings.get("TRANSFER_AMOUNT", 0.0)] * 2,
    )

    replay = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "generator": "exchange-dynamics-sim",
            "iterations": last_iteration,
            "seed": int(settings["RANDOM_SEED"]),
            "agent_count": len(agent_ids),
            "initial_resources": initial_resources,
            "initial_total": initial_total,
            "resource_floor": float(settings.get("RESOURCE_FLOOR", 0.0)),
            "living_cost": float(agent_settings.get("LIVING_COST", 0.0)),
            "transfer_range": _jsonable(transfer_range),
            "features": {
                "market": bool(market.get("ENABLED", False)),
                "trust": bool(trust.get("ENABLED", False)),
                "shocks": bool(shocks.get("ENABLED", False)),
                "reciprocity": bool(social.get("RECIPROCITY_ENABLED", False)),
            },
            "config": config,
        },
        "agents": agent_metadata,
        "ticks": ticks,
        "events": events,
    }
    validate_replay(replay)
    return replay


def _history_by_tick(
    history: Iterable[dict[str, Any]],
    agent_index: dict[str, int],
) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in history:
        agent_id = row["id"]
        if agent_id not in agent_index:
            continue
        grouped.setdefault(int(row["iteration"]), {})[agent_id] = row
    if not grouped:
        raise ValueError("The simulation history is empty")
    return grouped


def _agent_metadata(
    simulation: ResourceSimulation,
    first_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = []
    for index, agent in enumerate(simulation.agents):
        first = first_rows[agent.id]
        metadata.append(
            {
                "id": agent.id,
                "label": agent.id.replace("agent-", "A"),
                "color": AGENT_COLORS[index % len(AGENT_COLORS)],
                "generosity": float(first["generosity_score"]),
                "acceptance": float(first["acceptance_score"]),
                "initial_risk_tolerance": float(
                    first.get("risk_tolerance", 0.5)
                ),
                "production_rate": float(agent.production_rate),
                "consumption_need": float(agent.consumption_need),
                "initial_resources": float(first["resources"]),
            }
        )
    return metadata


def _metrics_from_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the summary metrics when no metrics row was recorded."""

    # Imported lazily so replay.py has no import-time dependency on the engine.
    from main import MetricsCalculator

    resources = [float(row["resources"]) for row in rows.values()]
    statistics = MetricsCalculator.calculate_statistics(
        _resources_frame(resources)
    )
    statistics["goods_total"] = float(
        sum(float(row.get("goods", 0.0)) for row in rows.values())
    )
    statistics["alive"] = int(
        sum(bool(row.get("alive", False)) for row in rows.values())
    )
    statistics["living_cost_paid"] = 0.0
    return statistics


def _resources_frame(resources: list[float]):
    import pandas as pd

    return pd.DataFrame({"resources": resources})


def _stock_ratio(total: float, initial_total: float) -> float:
    if initial_total <= 0:
        return 0.0
    return total / initial_total


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_events(
    simulation: ResourceSimulation,
    agent_index: dict[str, int],
    last_iteration: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for entry in getattr(simulation, "shock_log", []):
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "shock",
                "type": "shock",
                "actor": None,
                "target": None,
                "amount": None,
                "shock_type": str(entry["type"]),
                "magnitude": float(entry["magnitude"]),
                "duration": int(entry["duration"]),
            }
        )

    for entry in getattr(simulation, "transfer_log", []):
        if entry["giver"] not in agent_index:
            continue
        if entry["receiver"] not in agent_index:
            continue
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "exchange",
                "type": str(entry.get("type", "gift")),
                "actor": entry["giver"],
                "target": entry["receiver"],
                "amount": float(entry["amount"]),
            }
        )

    for entry in getattr(simulation, "trade_log", []):
        if entry["buyer"] not in agent_index or entry["seller"] not in agent_index:
            continue
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "market",
                "type": "trade",
                "actor": entry["buyer"],
                "target": entry["seller"],
                "amount": float(entry["amount"]),
                "quantity": float(entry["quantity"]),
                "price": float(entry["price"]),
            }
        )

    for entry in getattr(simulation, "death_log", []):
        if entry["agent"] not in agent_index:
            continue
        cause = str(entry["cause"])
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": DEATH_PHASES.get(cause, "cost"),
                "type": "death",
                "actor": entry["agent"],
                "target": None,
                "amount": None,
                "cause": cause,
            }
        )

    entries = [
        entry
        for entry in entries
        if 1 <= entry["tick"] <= last_iteration
    ]
    entries.sort(key=lambda entry: (entry["tick"], PHASE_ORDER[entry["phase"]]))

    sequence_by_tick: dict[int, int] = {}
    for entry in entries:
        tick = entry["tick"]
        sequence = sequence_by_tick.get(tick, 0)
        sequence_by_tick[tick] = sequence + 1
        entry["seq"] = sequence
        entry["id"] = f"{tick}:{sequence}"
    return entries


def validate_replay(replay: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``replay`` is not a valid v1 replay."""

    if not isinstance(replay, dict):
        raise ValueError("Replay must be a mapping")
    required = {"schema_version", "meta", "agents", "ticks", "events"}
    missing = required - replay.keys()
    if missing:
        raise ValueError(f"Replay is missing keys: {sorted(missing)}")
    if replay["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported replay schema version "
            f"{replay['schema_version']!r}; expected {SCHEMA_VERSION}"
        )

    meta = replay["meta"]
    if not isinstance(meta, dict):
        raise ValueError("Replay meta must be a mapping")
    for key in ("iterations", "seed", "agent_count", "config"):
        if key not in meta:
            raise ValueError(f"Replay meta is missing {key!r}")
    iterations = _required_int(meta["iterations"], "meta.iterations", minimum=0)
    _required_int(meta["seed"], "meta.seed")
    if not isinstance(meta["config"], dict):
        raise ValueError("Replay meta.config must be a mapping")

    agents = replay["agents"]
    if not isinstance(agents, list) or not agents:
        raise ValueError("Replay agents must be a non-empty list")
    agent_ids = []
    for agent in agents:
        if not isinstance(agent, dict) or "id" not in agent:
            raise ValueError("Every replay agent needs an id")
        agent_ids.append(str(agent["id"]))
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Replay agent ids must be unique")
    if len(agent_ids) != _required_int(
        meta["agent_count"], "meta.agent_count"
    ):
        raise ValueError("meta.agent_count does not match the agent list")

    ticks = replay["ticks"]
    if not isinstance(ticks, list):
        raise ValueError("Replay ticks must be a list")
    if len(ticks) != iterations + 1:
        raise ValueError(
            f"Replay has {len(ticks)} ticks for {iterations} iterations; "
            "expected one tick per iteration plus iteration zero"
        )
    expected_ids = set(agent_ids)
    for expected_iteration, tick in enumerate(ticks):
        if not isinstance(tick, dict):
            raise ValueError("Every replay tick must be a mapping")
        if tick.get("iteration") != expected_iteration:
            raise ValueError(
                "Replay ticks must be ordered from iteration 0 to "
                f"{iterations}"
            )
        if not isinstance(tick.get("metrics"), dict):
            raise ValueError("Every replay tick needs a metrics mapping")
        states = tick.get("agents")
        if not isinstance(states, list) or len(states) != len(agent_ids):
            raise ValueError("Every replay tick needs one state per agent")
        if {state.get("id") for state in states} != expected_ids:
            raise ValueError(
                f"Tick {expected_iteration} does not cover every agent"
            )
        for state in states:
            _required_float(
                state.get("resources"),
                f"tick {expected_iteration} resources",
            )
            _required_float(
                state.get("goods"), f"tick {expected_iteration} goods"
            )
            _required_float(
                state.get("risk_tolerance"),
                f"tick {expected_iteration} risk_tolerance",
            )
            if state.get("reputation") is not None:
                _required_float(
                    state["reputation"],
                    f"tick {expected_iteration} reputation",
                )
            if not isinstance(state.get("alive"), bool):
                raise ValueError(
                    f"Tick {expected_iteration} has a non-boolean alive flag"
                )

    events = replay["events"]
    if not isinstance(events, list):
        raise ValueError("Replay events must be a list")
    seen_ids: set[str] = set()
    seq_by_tick: dict[int, set[int]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Every replay event must be a mapping")
        tick = _required_int(event.get("tick"), "event.tick", minimum=1)
        if tick > iterations:
            raise ValueError(
                f"Event tick {tick} is beyond the final iteration {iterations}"
            )
        if event.get("type") not in EVENT_TYPES:
            raise ValueError(f"Unknown replay event type {event.get('type')!r}")
        if event.get("phase") not in PHASE_ORDER:
            raise ValueError(f"Unknown replay event phase {event.get('phase')!r}")
        sequence = _required_int(event.get("seq"), "event.seq", minimum=0)
        if sequence in seq_by_tick.setdefault(tick, set()):
            raise ValueError(f"Duplicate event sequence {sequence} at tick {tick}")
        seq_by_tick[tick].add(sequence)
        event_id = event.get("id")
        if event_id != f"{tick}:{sequence}":
            raise ValueError(f"Event id {event_id!r} does not match tick/seq")
        if event_id in seen_ids:
            raise ValueError(f"Duplicate replay event id {event_id!r}")
        seen_ids.add(event_id)
        for role in ("actor", "target"):
            value = event.get(role)
            if value is not None and value not in expected_ids:
                raise ValueError(f"Event {event_id} references unknown {role}")
        if event.get("amount") is not None:
            _required_float(event["amount"], f"event {event_id} amount")
        if event["type"] == "shock":
            if event.get("shock_type") not in {"scarcity", "information"}:
                raise ValueError(f"Event {event_id} has an invalid shock_type")
            _required_float(event["magnitude"], f"event {event_id} magnitude")
            _required_int(
                event.get("duration"), f"event {event_id} duration", minimum=1
            )
        if event["type"] == "death" and event.get("cause") not in DEATH_PHASES:
            raise ValueError(f"Event {event_id} has an invalid death cause")


def replay_to_json(replay: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialize ``replay`` deterministically."""

    separators = None if indent is not None else (",", ":")
    return json.dumps(
        replay,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=separators,
    )


def write_replay(
    replay: dict[str, Any],
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Validate and write ``replay``; ``.gz`` paths are gzip-compressed."""

    validate_replay(replay)
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


def load_replay(path: str | Path) -> dict[str, Any]:
    """Read and validate a replay written by :func:`write_replay`."""

    input_path = Path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            replay = json.load(handle)
    else:
        with input_path.open(encoding="utf-8") as handle:
            replay = json.load(handle)
    validate_replay(replay)
    return replay


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _required_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _required_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="Path to a custom YAML configuration"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the configured number of iterations",
    )
    parser.add_argument(
        "--output",
        default="replay.json.gz",
        help="Replay file to write (.gz is compressed)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print with this indent instead of compact JSON",
    )
    args = parser.parse_args(argv)

    from main import CONFIG, ResourceSimulation
    from utils import load_config

    config = load_config(args.config) if args.config else CONFIG
    simulation = ResourceSimulation(config)
    simulation.run(iterations=args.iterations)
    output_path = export_replay(simulation, args.output, indent=args.indent)
    print(f"Replay written to {output_path}")


if __name__ == "__main__":
    main()
