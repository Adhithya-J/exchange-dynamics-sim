"""Versioned replay format for the subsistence world.

A replay is the stable boundary between the model and the viewer. It records
only facts: the seed and configuration, agent skills, one snapshot per tick,
summary metrics, and typed gather/gift/trade/spoil/death events. It contains
no drawing coordinates or presentation state.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from utils import MetricsCalculator, load_config
from utils.replay import (
    jsonable,
    load_replay_json,
    required_float,
    required_int,
    stock_ratio,
    write_replay_json,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from worlds.subsistence.world import SubsistenceWorld

SCHEMA_VERSION = 2

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

PHASE_ORDER = {
    "gather": 0,
    "gift": 1,
    "trade": 2,
    "eat": 3,
    "spoil": 4,
}
EVENT_TYPES = {"gather", "gift", "trade", "spoil", "death"}
TRADE_FIELDS = ("berries", "wood", "price")


def export_replay(
    simulation: SubsistenceWorld,
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Build a replay from ``simulation`` and write it to ``path``."""

    return write_replay(build_replay(simulation), path, indent=indent)


def build_replay(simulation: SubsistenceWorld) -> dict[str, Any]:
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
    initial_metrics = _initial_metrics(simulation, history_by_tick[0])
    initial_berries = float(initial_metrics["total_berries"])
    initial_wood = float(initial_metrics["total_wood"])

    ticks = []
    for iteration in range(last_iteration + 1):
        rows = history_by_tick.get(iteration)
        if rows is None:
            raise ValueError(f"Missing agent snapshot for iteration {iteration}")
        metrics = (
            initial_metrics
            if iteration == 0
            else metrics_by_iteration.get(iteration)
        )
        if metrics is None:
            raise ValueError(f"Missing metrics row for iteration {iteration}")
        metrics = jsonable(metrics)
        metrics.setdefault("iteration", iteration)
        metrics["berries_ratio"] = stock_ratio(
            float(metrics.get("total_berries", 0.0)), initial_berries
        )
        metrics["wood_ratio"] = stock_ratio(
            float(metrics.get("total_wood", 0.0)), initial_wood
        )
        metrics["commons_berries_ratio"] = stock_ratio(
            float(metrics.get("commons_berries", 0.0)),
            simulation.initial_commons_berries,
        )
        metrics["commons_wood_ratio"] = stock_ratio(
            float(metrics.get("commons_wood", 0.0)),
            simulation.initial_commons_wood,
        )
        ticks.append(
            {
                "iteration": iteration,
                "metrics": metrics,
                "agents": [
                    {
                        "id": agent_id,
                        "health": int(rows[agent_id]["health"]),
                        "berries": int(rows[agent_id]["berries"]),
                        "wood": int(rows[agent_id]["wood"]),
                        "storage_capacity": int(
                            rows[agent_id]["storage_capacity"]
                        ),
                        "labor_berries": float(
                            rows[agent_id].get("labor_berries", 0.0)
                        ),
                        "labor_wood": float(
                            rows[agent_id].get("labor_wood", 0.0)
                        ),
                        "planned_berries": float(
                            rows[agent_id].get("planned_berries", 0.0)
                        ),
                        "planned_wood": float(
                            rows[agent_id].get("planned_wood", 0.0)
                        ),
                        "gathered_berries": int(
                            rows[agent_id].get("gathered_berries", 0)
                        ),
                        "gathered_wood": int(
                            rows[agent_id].get("gathered_wood", 0)
                        ),
                        "post_trade_berries": int(
                            rows[agent_id].get("post_trade_berries", 0)
                        ),
                        "post_trade_wood": int(
                            rows[agent_id].get("post_trade_wood", 0)
                        ),
                        "alive": bool(rows[agent_id]["alive"]),
                    }
                    for agent_id in agent_ids
                ],
            }
        )

    config = jsonable(deepcopy(simulation.config))
    recorded_settings = config.setdefault("SIMULATION", {})
    recorded_settings["ITERATIONS"] = last_iteration
    recorded_settings["AGENTS"] = len(agent_ids)
    events = _build_events(simulation, agent_index, last_iteration)

    replay = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "generator": "exchange-dynamics-sim",
            "world": "subsistence",
            "iterations": last_iteration,
            "seed": int(config["SIMULATION"]["RANDOM_SEED"]),
            "agent_count": len(agent_ids),
            "features": dict(simulation.features),
            "health": {
                "max": int(simulation.agent_settings["max_health"]),
                "food_per_iteration": int(
                    simulation.agent_settings["food_per_iteration"]
                ),
            },
            "time": {
                "minutes_per_iteration": float(
                    simulation.agent_settings["time_per_iteration"]
                ),
                "berry_minutes": float(
                    simulation.agent_settings["berry_time_cost"]
                ),
                "wood_minutes": float(
                    simulation.agent_settings["wood_time_cost"]
                ),
            },
            "storage": {
                "wood_per_container": int(
                    simulation.storage_settings["wood_per_container"]
                ),
                "berries_per_container": int(
                    simulation.storage_settings["berries_per_container"]
                ),
            },
            "resources": {
                "berries": {
                    "label": "Berries",
                    "perishable": True,
                    "unit": "berry",
                    "whole": True,
                },
                "wood": {
                    "label": "Wood",
                    "perishable": False,
                    "unit": "block",
                    "whole": True,
                },
            },
            "initial_commons": {
                "berries": int(simulation.initial_commons_berries),
                "wood": int(simulation.initial_commons_wood),
            },
            "initial_totals": {
                "berries": int(initial_berries),
                "wood": int(initial_wood),
            },
            "initial_price": int(simulation.initial_price),
            "config": config,
        },
        "agents": _agent_metadata(simulation, history_by_tick[0]),
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
    simulation: SubsistenceWorld,
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
                "berry_skill": float(agent.berry_skill),
                "wood_skill": float(agent.wood_skill),
                "labour": 1.0,
                "time_per_iteration": float(agent.time_per_iteration),
                "berry_time_cost": float(agent.berry_time_cost),
                "wood_time_cost": float(agent.wood_time_cost),
                "max_berries_per_day": float(agent.max_berries_per_day),
                "max_wood_per_day": float(agent.max_wood_per_day),
                "initial_health": int(first["health"]),
                "storage_wood_per_container": int(
                    agent.storage_wood_per_container
                ),
                "storage_berries_per_container": int(
                    agent.storage_berries_per_container
                ),
            }
        )
    return metadata


def _initial_metrics(
    simulation: SubsistenceWorld,
    first_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    berries = [int(row["berries"]) for row in first_rows.values()]
    wood = [int(row["wood"]) for row in first_rows.values()]
    health = [int(row["health"]) for row in first_rows.values()]
    skills_b = [float(row["berry_skill"]) for row in first_rows.values()]
    skills_w = [float(row["wood_skill"]) for row in first_rows.values()]
    total_berries = sum(berries)
    total_wood = sum(wood)
    return {
        "iteration": 0,
        "alive": len(first_rows),
        "mean_health": float(np.mean(health)) if health else 0.0,
        "total_berries": total_berries,
        "total_wood": total_wood,
        "commons_berries": int(simulation.initial_commons_berries),
        "commons_wood": int(simulation.initial_commons_wood),
        "total_berries_all": total_berries
        + int(simulation.initial_commons_berries),
        "total_wood_all": total_wood + int(simulation.initial_commons_wood),
        "gini_berries": MetricsCalculator.calculate_gini(berries) if berries else 0.0,
        "gini_wood": MetricsCalculator.calculate_gini(wood) if wood else 0.0,
        "barter_price": int(simulation.initial_price),
        "mean_berry_skill": float(np.mean(skills_b)) if skills_b else 0.0,
        "mean_wood_skill": float(np.mean(skills_w)) if skills_w else 0.0,
        "mean_labor_berries": 0.0,
        "specialisation": 0.0,
        "ppf_efficiency": 0.0,
        "berries_gathered": 0,
        "wood_gathered": 0,
        "berries_eaten": 0,
        "unmet_food": 0,
        "berries_spoiled": 0,
        "wood_decayed": 0,
        "commons_regenerated_berries": 0,
        "commons_regenerated_wood": 0,
        "commons_waste_berries": 0,
        "commons_waste_wood": 0,
        "storage_capacity": int(
            sum(int(row["storage_capacity"]) for row in first_rows.values())
        ),
        "storage_used": 0,
        "storage_utilization": 0.0,
        "trade_berries": 0,
        "trade_wood": 0,
        "gifts": 0,
        "gift_berries": 0,
        "extinction_iteration": None,
    }


def _build_events(
    simulation: SubsistenceWorld,
    agent_index: dict[str, int],
    last_iteration: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for entry in simulation.gather_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "gather",
                "type": "gather",
                "actor": entry["agent"],
                "target": None,
                "resource": str(entry["resource"]),
                "amount": int(entry["amount"]),
                "waste": int(entry.get("waste", 0)),
            }
        )

    for entry in simulation.gift_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "gift",
                "type": "gift",
                "actor": entry["giver"],
                "target": entry["receiver"],
                "resource": "berries",
                "amount": int(entry["amount"]),
            }
        )

    for entry in simulation.trade_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "trade",
                "type": "trade",
                "actor": entry["berry_seller"],
                "target": entry["wood_seller"],
                "berries": int(entry["berries"]),
                "wood": int(entry["wood"]),
                "price": int(entry["price"]),
                "amount": int(entry["berries"]),
            }
        )

    for entry in simulation.spoil_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "spoil",
                "type": "spoil",
                "actor": entry["agent"],
                "target": None,
                "resource": "berries",
                "amount": int(entry["amount"]),
            }
        )

    for entry in simulation.death_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "eat",
                "type": "death",
                "actor": entry["agent"],
                "target": None,
                "cause": str(entry["cause"]),
            }
        )

    entries = [
        entry
        for entry in entries
        if 1 <= entry["tick"] <= last_iteration
        and entry.get("actor") in agent_index
        and (entry.get("target") is None or entry["target"] in agent_index)
    ]
    entries.sort(
        key=lambda entry: (
            entry["tick"],
            PHASE_ORDER[entry["phase"]],
            str(entry.get("actor")),
            str(entry.get("target")),
        )
    )

    sequence_by_tick: dict[int, int] = {}
    for entry in entries:
        tick = entry["tick"]
        sequence = sequence_by_tick.get(tick, 0)
        sequence_by_tick[tick] = sequence + 1
        entry["seq"] = sequence
        entry["id"] = f"{tick}:{sequence}"
    return entries


def validate_replay(replay: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``replay`` is not a valid v2 replay."""

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
    if meta.get("world") != "subsistence":
        raise ValueError("Replay meta.world must be 'subsistence'")
    for key in ("iterations", "seed", "agent_count", "config", "features"):
        if key not in meta:
            raise ValueError(f"Replay meta is missing {key!r}")
    iterations = required_int(meta["iterations"], "meta.iterations", minimum=0)
    required_int(meta["seed"], "meta.seed")
    if not isinstance(meta["config"], dict):
        raise ValueError("Replay meta.config must be a mapping")
    if not isinstance(meta["features"], dict):
        raise ValueError("Replay meta.features must be a mapping")

    agents = replay["agents"]
    if not isinstance(agents, list) or not agents:
        raise ValueError("Replay agents must be a non-empty list")
    agent_ids = []
    for agent in agents:
        if not isinstance(agent, dict) or "id" not in agent:
            raise ValueError("Every replay agent needs an id")
        required_float(agent.get("berry_skill"), f"agent {agent['id']} berry_skill")
        required_float(agent.get("wood_skill"), f"agent {agent['id']} wood_skill")
        agent_ids.append(str(agent["id"]))
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("Replay agent ids must be unique")
    if len(agent_ids) != required_int(
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
            for field in (
                "health",
                "berries",
                "wood",
                "storage_capacity",
                "gathered_berries",
                "gathered_wood",
                "post_trade_berries",
                "post_trade_wood",
            ):
                required_int(
                    state.get(field),
                    f"tick {expected_iteration} {state.get('id')} {field}",
                    minimum=0,
                )
            for field in (
                "labor_berries",
                "labor_wood",
                "planned_berries",
                "planned_wood",
            ):
                required_float(
                    state.get(field),
                    f"tick {expected_iteration} {state.get('id')} {field}",
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
        tick = required_int(event.get("tick"), "event.tick", minimum=1)
        if tick > iterations:
            raise ValueError(
                f"Event tick {tick} is beyond the final iteration {iterations}"
            )
        if event.get("type") not in EVENT_TYPES:
            raise ValueError(f"Unknown replay event type {event.get('type')!r}")
        if event.get("phase") not in PHASE_ORDER:
            raise ValueError(f"Unknown replay event phase {event.get('phase')!r}")
        sequence = required_int(event.get("seq"), "event.seq", minimum=0)
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
            required_int(
                event["amount"], f"event {event_id} amount", minimum=0
            )
        if event.get("waste") is not None:
            required_int(
                event["waste"], f"event {event_id} waste", minimum=0
            )
        if event["type"] == "trade":
            for field in TRADE_FIELDS:
                required_int(
                    event.get(field), f"event {event_id} {field}", minimum=0
                )
            if event.get("price", 0) < 1:
                raise ValueError(f"Event {event_id} price must be positive")
        if event["type"] == "death" and event.get("cause") != "hunger":
            raise ValueError(f"Event {event_id} has an invalid death cause")


def write_replay(
    replay: dict[str, Any],
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Validate and write ``replay``; ``.gz`` paths are compressed."""

    validate_replay(replay)
    return write_replay_json(replay, path, indent=indent)


def load_replay(path: str | Path) -> dict[str, Any]:
    """Read and validate a subsistence replay."""

    replay = load_replay_json(path)
    validate_replay(replay)
    return replay


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
        default="replay-subsistence.json.gz",
        help="Replay file to write (.gz is compressed)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print with this indent instead of compact JSON",
    )
    args = parser.parse_args(argv)

    from worlds.subsistence.world import CONFIG, SubsistenceWorld

    config = load_config(args.config) if args.config else CONFIG
    simulation = SubsistenceWorld(config)
    simulation.run(iterations=args.iterations)
    output_path = export_replay(simulation, args.output, indent=args.indent)
    print(f"Replay written to {output_path}")


if __name__ == "__main__":
    main()
