"""Versioned replay format for the subsistence world.

A replay is the stable boundary between the model and the viewer. It records
only facts: the seed and configuration, agent skills, one snapshot per tick,
summary metrics, and typed gather/craft/gift/trade/credit/shock/spoil/death
events. It contains no drawing coordinates or presentation state.
"""

from __future__ import annotations

import argparse
import math
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

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({2, 3})

# Version 3 stores each agent's per-tick state as a positional row. The field
# order lives in ``meta.agent_state_fields`` so the format stays self-describing
# while cutting replay size by roughly three quarters: key names are written once
# instead of once per agent per tick.
AGENT_STATE_FIELDS = (
    "id",
    "health",
    "berries",
    "wood",
    "crates",
    "crate_durability",
    "craft_progress",
    "price_belief",
    "trust",
    "debt_wood",
    "debt_berries",
    "debt_crates",
    "credit_out_wood",
    "credit_out_berries",
    "credit_out_crates",
    "storage_capacity",
    "labor_berries",
    "labor_wood",
    "labor_craft",
    "wood_progress",
    "planned_berries",
    "planned_wood",
    "planned_crates",
    "crafted_crates",
    "gathered_berries",
    "gathered_wood",
    "effective_berries_per_day",
    "effective_wood_per_day",
    "max_berries_per_day",
    "max_wood_per_day",
    "post_trade_berries",
    "post_trade_wood",
    "alive",
)
AGENT_STATE_INDEX = {
    field: index for index, field in enumerate(AGENT_STATE_FIELDS)
}

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
    "craft": 0,
    "shock": 1,
    "gather": 2,
    "gift": 3,
    "trade": 4,
    "credit": 5,
    "eat": 6,
    "spoil": 7,
}
EVENT_TYPES = {
    "gather",
    "craft",
    "gift",
    "trade",
    "loan",
    "repay",
    "default",
    "write_off",
    "late",
    "shock",
    "spoil",
    "death",
}
TRADE_INT_FIELDS = ("berries", "wood")
TRADE_FLOAT_FIELDS = ("price",)


def export_replay(
    simulation: SubsistenceWorld,
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Build a replay from ``simulation`` and write it to ``path``."""

    return write_replay(
        build_replay(simulation), path, indent=indent, validate=False
    )


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
                    _agent_state_row(rows[agent_id]) for agent_id in agent_ids
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
            "agent_state_fields": list(AGENT_STATE_FIELDS),
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
                "base_craft_minutes": float(
                    simulation.storage_settings["base_craft_minutes"]
                ),
                "crate_wear_rate": float(
                    simulation.storage_settings["crate_wear_rate"]
                ),
            },
            "beliefs": {
                "fractional": True,
                "initial_price": float(simulation.initial_price),
            },
            "credit": jsonable(dict(getattr(simulation, "credit_settings", {}))),
            "trust": jsonable(dict(getattr(simulation, "trust_settings", {}))),
            "reciprocity": jsonable(
                dict(getattr(simulation, "reciprocity_settings", {}))
            ),
            "shocks": jsonable(dict(getattr(simulation, "shock_settings", {}))),
            "seasons": jsonable(dict(getattr(simulation, "season_settings", {}))),
            "commons": {
                "scale_with_agents": bool(
                    (simulation.config.get("COMMONS") or {}).get(
                        "SCALE_WITH_AGENTS", False
                    )
                ),
                "reference_agents": int(
                    (simulation.config.get("COMMONS") or {}).get(
                        "REFERENCE_AGENTS", 10
                    )
                ),
                "settings": jsonable(dict(simulation.commons_settings)),
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
            "initial_price": float(simulation.initial_price),
            "config": config,
        },
        "agents": _agent_metadata(simulation, history_by_tick[0]),
        "ticks": ticks,
        "events": events,
    }
    validate_replay(replay)
    return replay


def decode_tick_agents(tick: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one tick's agent states as mappings, whatever the schema."""

    states = tick["agents"]
    if states and isinstance(states[0], dict):
        return list(states)
    return [dict(zip(AGENT_STATE_FIELDS, row)) for row in states]


def _agent_state_row(row: dict[str, Any]) -> list[Any]:
    """Pack one agent snapshot into a positional row (schema 3)."""

    return [
        str(row["id"]),
        int(row["health"]),
        int(row["berries"]),
        int(row["wood"]),
        int(row.get("crates", 0)),
        round(float(row.get("crate_durability", 0.0)), 4),
        round(float(row.get("craft_progress", 0.0)), 4),
        round(float(row.get("price_belief", 5.0)), 4),
        round(float(row.get("trust", 0.0)), 4),
        int(row.get("debt_wood", 0)),
        int(row.get("debt_berries", 0)),
        int(row.get("debt_crates", 0)),
        int(row.get("credit_out_wood", 0)),
        int(row.get("credit_out_berries", 0)),
        int(row.get("credit_out_crates", 0)),
        int(row["storage_capacity"]),
        round(float(row.get("labor_berries", 0.0)), 4),
        round(float(row.get("labor_wood", 0.0)), 4),
        round(float(row.get("labor_craft", 0.0)), 4),
        round(float(row.get("wood_progress", 0.0)), 4),
        round(float(row.get("planned_berries", 0.0)), 4),
        round(float(row.get("planned_wood", 0.0)), 4),
        int(row.get("planned_crates", 0)),
        int(row.get("crafted_crates", 0)),
        int(row.get("gathered_berries", 0)),
        int(row.get("gathered_wood", 0)),
        round(float(row.get("effective_berries_per_day", 0.0)), 4),
        round(float(row.get("effective_wood_per_day", 0.0)), 4),
        round(float(row.get("max_berries_per_day", 0.0)), 4),
        round(float(row.get("max_wood_per_day", 0.0)), 4),
        int(row.get("post_trade_berries", 0)),
        int(row.get("post_trade_wood", 0)),
        bool(row["alive"]),
    ]


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
                "specialisation_z": float(agent.specialisation_z),
                "labour": 1.0,
                "time_per_iteration": float(agent.time_per_iteration),
                "berry_time_cost": float(agent.berry_time_cost),
                "wood_time_cost": float(agent.wood_time_cost),
                "max_berries_per_day": float(agent.max_berries_per_day),
                "max_wood_per_day": float(agent.max_wood_per_day),
                "initial_health": int(first["health"]),
                "price_belief": float(first.get("price_belief", 5.0)),
                "initial_crates": int(first.get("crates", 0)),
                "craft_minutes": float(simulation._craft_minutes(agent)),
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
    beliefs = [float(row.get("price_belief", 5.0)) for row in first_rows.values()]
    crates = [int(row.get("crates", 0)) for row in first_rows.values()]
    durabilities = [
        float(row.get("crate_durability", 0.0))
        for row in first_rows.values()
        if int(row.get("crates", 0)) > 0
    ]
    total_berries = sum(berries)
    total_wood = sum(wood)
    total_crates = sum(crates)
    storage_capacity = int(
        sum(int(row["storage_capacity"]) for row in first_rows.values())
    )
    storage_used = int(
        sum(
            min(int(row["berries"]), int(row["storage_capacity"]))
            for row in first_rows.values()
        )
    )
    return {
        "iteration": 0,
        "alive": len(first_rows),
        "mean_health": float(np.mean(health)) if health else 0.0,
        "total_berries": total_berries,
        "total_wood": total_wood,
        "total_crates": total_crates,
        "crate_durability": (
            float(np.mean(durabilities)) if durabilities else 0.0
        ),
        "crates_crafted": 0,
        "crates_worn": 0,
        "commons_berries": int(simulation.initial_commons_berries),
        "commons_wood": int(simulation.initial_commons_wood),
        "berries_commons_pct": (
            1.0 if simulation.initial_commons_berries > 0 else 0.0
        ),
        "wood_commons_pct": (
            1.0 if simulation.initial_commons_wood > 0 else 0.0
        ),
        "total_berries_all": total_berries
        + int(simulation.initial_commons_berries),
        "total_wood_all": total_wood + int(simulation.initial_commons_wood),
        "gini_berries": MetricsCalculator.calculate_gini(berries) if berries else 0.0,
        "gini_wood": MetricsCalculator.calculate_gini(wood) if wood else 0.0,
        "barter_price": float(simulation.initial_price),
        "last_crate_price": 0.0,
        "average_trade_price": float(simulation.initial_price),
        "gossip_price": float(simulation.initial_price),
        "mean_price_belief": (
            float(np.mean(beliefs)) if beliefs else 0.0
        ),
        "market_participants": 0,
        "market_traders": 0,
        "failed_trades": 0,
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
        "storage_capacity": storage_capacity,
        "storage_used": storage_used,
        "storage_utilization": (
            storage_used / storage_capacity if storage_capacity > 0 else 0.0
        ),
        "trade_berries": 0,
        "trade_wood": 0,
        "crate_trades": 0,
        "trade_crates": 0,
        "gifts": 0,
        "gift_berries": 0,
        "mean_trust": (
            float(np.mean([float(row.get("trust", 0.0)) for row in first_rows.values()]))
            if first_rows
            else 0.0
        ),
        "mean_defaults": 0.0,
        "loans_originated": 0,
        "loan_principal_wood": 0,
        "loan_principal_berries": 0,
        "loan_principal_crates": 0,
        "mean_interest_rate": 0.0,
        "loans_repaid": 0,
        "repayment_wood": 0,
        "repayment_berries": 0,
        "repayment_crates": 0,
        "loans_defaulted": 0,
        "default_loss_wood": 0,
        "default_loss_berries": 0,
        "default_loss_crates": 0,
        "loans_late": 0,
        "collateral_seized": 0,
        "loans_active": 0,
        "credit_outstanding_wood": 0,
        "credit_outstanding_berries": 0,
        "credit_outstanding_crates": 0,
        "gift_obligations": 0,
        "gift_obligation_berries": 0,
        "gift_repayments": 0,
        "gift_repayment_berries": 0,
        "shock_active": 0,
        "shocks": 0,
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

    for entry in simulation.craft_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "craft",
                "type": "craft",
                "actor": entry["agent"],
                "target": None,
                "resource": "crate",
                "amount": int(entry.get("crates", 1)),
                "wood": int(entry.get("wood", 0)),
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
        good = str(entry.get("good", "wood"))
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "trade",
                "type": "trade",
                "actor": entry["berry_seller"],
                "target": entry.get(
                    "wood_seller", entry.get("crate_seller")
                ),
                "good": good,
                "resource": "wood" if good == "wood" else "crate",
                "berries": int(entry["berries"]),
                "wood": int(entry.get("wood", 0)),
                "crates": int(entry.get("crates", 0)),
                "quantity": int(entry.get("quantity", 0)),
                "price": float(entry["price"]),
                "amount": int(entry["berries"]),
            }
        )

    for entry in simulation.credit_events:
        kind = entry.get("kind", "loan")
        event = entry.get("event")
        if event == "origination" and kind == "gift":
            continue
        base = {
            "tick": int(entry["iteration"]),
            "phase": "credit",
            "actor": entry["debtor"],
            "target": entry["creditor"],
            "resource": entry["good"],
            "amount": int(entry.get("amount", 0)),
            "principal": int(entry.get("principal", 0)),
            "interest_rate": float(entry.get("interest_rate", 0.0)),
            "due_iteration": int(entry.get("due_iteration") or 0),
            "collateral_crates": int(entry.get("collateral_crates", 0)),
            "collateral_durability": float(
                entry.get("collateral_durability", 0.0)
            ),
        }
        if event == "origination":
            base.update(
                {
                    "type": "loan",
                    "actor": entry["creditor"],
                    "target": entry["debtor"],
                    "amount": int(entry["principal"]),
                    "repayment": int(entry["amount_due"]),
                }
            )
        elif event == "repayment":
            base.update(
                {
                    "type": "repay",
                    "repayment": int(entry.get("amount", 0)),
                }
            )
        elif event == "default":
            base.update(
                {
                    "type": "default",
                    "recovered": int(round(entry.get("recovered", 0))),
                    "loss": int(entry.get("loss", 0)),
                    "collateral_seized": int(
                        entry.get("collateral_seized", 0)
                    ),
                    "collateral_excess": float(
                        entry.get("collateral_excess", 0.0)
                    ),
                    "reason": str(entry.get("reason", "")),
                }
            )
        elif event == "write_off":
            base.update(
                {
                    "type": "write_off",
                    "recovered": int(round(entry.get("recovered", 0))),
                    "loss": int(entry.get("loss", 0)),
                    "reason": str(entry.get("reason", "")),
                }
            )
        elif event == "late":
            base.update({"type": "late", "reason": "missed_payment"})
        else:
            continue
        if kind == "gift":
            base["kind"] = "gift"
        entries.append(base)

    for entry in simulation.shock_log:
        entries.append(
            {
                "tick": int(entry["iteration"]),
                "phase": "shock",
                "type": "shock",
                "actor": None,
                "target": None,
                "shock_type": str(entry.get("type", "drought")),
                "duration": int(entry.get("duration", 1)),
                "berry_factor": float(entry.get("berry_factor", 1.0)),
                "wood_factor": float(entry.get("wood_factor", 1.0)),
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
        and (entry.get("actor") is None or entry.get("actor") in agent_index)
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


INTEGER_AGENT_FIELDS = (
    "health",
    "berries",
    "wood",
    "crates",
    "debt_wood",
    "debt_berries",
    "debt_crates",
    "credit_out_wood",
    "credit_out_berries",
    "credit_out_crates",
    "storage_capacity",
    "planned_crates",
    "crafted_crates",
    "gathered_berries",
    "gathered_wood",
    "post_trade_berries",
    "post_trade_wood",
)
FLOAT_AGENT_FIELDS = (
    "crate_durability",
    "craft_progress",
    "price_belief",
    "trust",
    "labor_berries",
    "labor_wood",
    "labor_craft",
    "wood_progress",
    "planned_berries",
    "planned_wood",
    "effective_berries_per_day",
    "effective_wood_per_day",
    "max_berries_per_day",
    "max_wood_per_day",
)


def _validate_state_rows(
    states: list[Any],
    expected_ids: set[str],
    iteration: int,
) -> None:
    """Validate schema-3 positional state rows for one tick."""

    width = len(AGENT_STATE_FIELDS)
    index = AGENT_STATE_INDEX
    id_index = index["id"]
    alive_index = index["alive"]
    int_indexes = tuple(index[field] for field in INTEGER_AGENT_FIELDS)
    float_indexes = tuple(index[field] for field in FLOAT_AGENT_FIELDS)
    seen: set[str] = set()
    for state in states:
        if not isinstance(state, list) or len(state) != width:
            raise ValueError(
                f"Tick {iteration} needs {width} fields per agent state"
            )
        agent_id = state[id_index]
        if agent_id not in expected_ids:
            raise ValueError(
                f"Tick {iteration} references unknown agent {agent_id!r}"
            )
        seen.add(agent_id)
        for field_index in int_indexes:
            value = state[field_index]
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"tick {iteration} {agent_id} "
                    f"{AGENT_STATE_FIELDS[field_index]} must be a "
                    "non-negative integer"
                )
        for field_index in float_indexes:
            value = state[field_index]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(
                    f"tick {iteration} {agent_id} "
                    f"{AGENT_STATE_FIELDS[field_index]} must be finite"
                )
        if type(state[alive_index]) is not bool:
            raise ValueError(f"Tick {iteration} has a non-boolean alive flag")
    if seen != expected_ids:
        raise ValueError(f"Tick {iteration} does not cover every agent")


def _validate_state_dict(state: dict[str, Any], iteration: int) -> None:
    """Validate a legacy schema-2 agent state mapping."""

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
            f"tick {iteration} {state.get('id')} {field}",
            minimum=0,
        )
    for field in (
        "crates",
        "planned_crates",
        "debt_wood",
        "debt_berries",
        "debt_crates",
        "credit_out_wood",
        "credit_out_berries",
        "credit_out_crates",
    ):
        if field in state:
            required_int(
                state[field],
                f"tick {iteration} {state.get('id')} {field}",
                minimum=0,
            )
    for field in (
        "labor_berries",
        "labor_wood",
        "labor_craft",
        "planned_berries",
        "planned_wood",
        "craft_progress",
        "crate_durability",
        "price_belief",
        "trust",
    ):
        if field in state:
            required_float(
                state[field],
                f"tick {iteration} {state.get('id')} {field}",
            )
    if "wood_progress" in state:
        required_float(
            state["wood_progress"],
            f"tick {iteration} {state.get('id')} wood_progress",
        )
    if not isinstance(state.get("alive"), bool):
        raise ValueError(
            f"Tick {iteration} has a non-boolean alive flag"
        )


def validate_replay(replay: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``replay`` is a valid subsistence replay."""

    if not isinstance(replay, dict):
        raise ValueError("Replay must be a mapping")
    required = {"schema_version", "meta", "agents", "ticks", "events"}
    missing = required - replay.keys()
    if missing:
        raise ValueError(f"Replay is missing keys: {sorted(missing)}")
    version = replay["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported replay schema version "
            f"{version!r}; expected one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    compact = version >= 3

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
    if compact:
        fields = meta.get("agent_state_fields")
        if not isinstance(fields, list) or tuple(fields) != AGENT_STATE_FIELDS:
            raise ValueError(
                "meta.agent_state_fields must list the schema 3 agent fields"
            )

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
        if compact:
            _validate_state_rows(states, expected_ids, expected_iteration)
        else:
            if {state.get("id") for state in states} != expected_ids:
                raise ValueError(
                    f"Tick {expected_iteration} does not cover every agent"
                )
            for state in states:
                _validate_state_dict(state, expected_iteration)

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
        if event.get("crates") is not None:
            required_int(
                event["crates"], f"event {event_id} crates", minimum=0
            )
        if event["type"] == "trade":
            for field in TRADE_INT_FIELDS:
                required_int(
                    event.get(field), f"event {event_id} {field}", minimum=0
                )
            for field in TRADE_FLOAT_FIELDS:
                required_float(
                    event.get(field), f"event {event_id} {field}"
                )
            if event.get("price", 0) <= 0:
                raise ValueError(f"Event {event_id} price must be positive")
        if event["type"] in {
            "loan",
            "repay",
            "default",
            "write_off",
            "late",
        }:
            if event.get("resource") not in {"wood", "berries", "crates"}:
                raise ValueError(
                    f"Event {event_id} has an invalid credit resource"
                )
            for field in ("principal", "due_iteration", "collateral_crates"):
                if event.get(field) is not None:
                    required_int(
                        event[field],
                        f"event {event_id} {field}",
                        minimum=0,
                    )
            if event.get("interest_rate") is not None:
                required_float(
                    event["interest_rate"],
                    f"event {event_id} interest_rate",
                )
            if event["type"] == "loan":
                if not event.get("actor") or not event.get("target"):
                    raise ValueError(
                        f"Event {event_id} loan needs a creditor and debtor"
                    )
                required_int(
                    event.get("repayment"),
                    f"event {event_id} repayment",
                    minimum=0,
                )
            if event["type"] in {"default", "write_off"}:
                for field in ("recovered", "loss", "collateral_seized"):
                    if event.get(field) is not None:
                        required_int(
                            event[field],
                            f"event {event_id} {field}",
                            minimum=0,
                        )
                if event.get("collateral_excess") is not None:
                    required_float(
                        event["collateral_excess"],
                        f"event {event_id} collateral_excess",
                    )
            if event.get("collateral_durability") is not None:
                required_float(
                    event["collateral_durability"],
                    f"event {event_id} collateral_durability",
                )
        if event["type"] == "shock":
            required_int(
                event.get("duration"), f"event {event_id} duration", minimum=1
            )
            required_float(
                event.get("berry_factor"), f"event {event_id} berry_factor"
            )
            required_float(
                event.get("wood_factor"), f"event {event_id} wood_factor"
            )
        if event["type"] == "death" and event.get("cause") != "hunger":
            raise ValueError(f"Event {event_id} has an invalid death cause")


def write_replay(
    replay: dict[str, Any],
    path: str | Path,
    *,
    indent: int | None = None,
    validate: bool = True,
) -> Path:
    """Validate and write ``replay``; ``.gz`` paths are compressed."""

    if validate:
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
