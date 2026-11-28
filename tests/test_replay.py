import json
from copy import deepcopy
from pathlib import Path

import pytest

from main import CONFIG, ResourceSimulation
from replay import (
    SCHEMA_VERSION,
    build_replay,
    load_replay,
    replay_to_json,
    validate_replay,
    write_replay,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_replay.json"


def golden_config() -> dict:
    """A small run that exercises gifts, the living-cost sink, and deaths."""

    config = deepcopy(CONFIG)
    config["GIFTING"]["ENABLED"] = True
    config["SIMULATION"].update(
        {
            "AGENTS": 3,
            "ITERATIONS": 4,
            "INITIAL_RESOURCES": 10.0,
            "RESOURCE_FLOOR": 0.0,
            "RANDOM_SEED": 7,
            "TRANSFER_AMOUNT_RANGE": (1.0, 5.0),
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (0.5, 0.9),
            "ACCEPTANCE": (0.6, 0.95),
            "MIN_RESOURCES_AFTER_GIVING": 0.0,
            "MAX_RESOURCES_AFTER_RECEIVING": 10000.0,
            "LIVING_COST": 3.0,
        }
    )
    # Let the fund retain subsistence payments so the fixture still exercises
    # living-cost deaths and a non-zero fund balance.
    config["COMMONS"]["FUND_REDISTRIBUTION"] = 0.0
    return config


def layered_config() -> dict:
    """A run with every optional layer enabled, for conservation checks."""

    config = deepcopy(CONFIG)
    config["GIFTING"]["ENABLED"] = True
    config["SIMULATION"].update(
        {
            "AGENTS": 4,
            "ITERATIONS": 8,
            "INITIAL_RESOURCES": 60.0,
            "RESOURCE_FLOOR": 0.0,
            "RANDOM_SEED": 3,
            "TRANSFER_AMOUNT_RANGE": (1.0, 5.0),
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (0.5, 1.0),
            "ACCEPTANCE": (0.5, 1.0),
            "MIN_RESOURCES_AFTER_GIVING": 0.0,
            "MAX_RESOURCES_AFTER_RECEIVING": 10000.0,
            "LIVING_COST": 2.0,
        }
    )
    config["MARKET"].update(
        {
            "ENABLED": True,
            "PRICE": 2.0,
            "PRODUCTION_RANGE": (0.0, 2.0),
            "CONSUMPTION_RANGE": (0.5, 1.5),
        }
    )
    config["TRUST"]["ENABLED"] = True
    config["SOCIAL"]["RECIPROCITY_ENABLED"] = True
    config["SHOCKS"].update(
        {
            "ENABLED": True,
            "SCHEDULE": [
                {
                    "iteration": 3,
                    "type": "scarcity",
                    "magnitude": 0.5,
                    "duration": 2,
                }
            ],
        }
    )
    return config


def test_replay_matches_golden_fixture():
    simulation = ResourceSimulation(golden_config())
    simulation.run()

    replay = build_replay(simulation)
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert replay == expected


def test_replay_is_deterministic(tmp_path):
    first = ResourceSimulation(golden_config())
    second = ResourceSimulation(golden_config())
    first.run()
    second.run()

    first_replay = build_replay(first)
    second_replay = build_replay(second)
    assert replay_to_json(first_replay) == replay_to_json(second_replay)

    first_path = write_replay(first_replay, tmp_path / "first.json.gz")
    second_path = write_replay(second_replay, tmp_path / "second.json.gz")
    assert first_path.read_bytes() == second_path.read_bytes()


def test_replay_round_trips_through_gzip(tmp_path):
    simulation = ResourceSimulation(golden_config())
    simulation.run()
    replay = build_replay(simulation)

    output = write_replay(replay, tmp_path / "replay.json.gz")

    assert load_replay(output) == replay
    assert output.stat().st_size > 0


def test_replay_records_typed_events_and_deaths():
    simulation = ResourceSimulation(golden_config())
    simulation.run()

    replay = build_replay(simulation)

    assert replay["schema_version"] == SCHEMA_VERSION
    assert simulation.death_log
    event_types = {event["type"] for event in replay["events"]}
    assert {"gift", "death"} <= event_types

    agents = {agent["id"] for agent in replay["agents"]}
    seen_ids = set()
    for event in replay["events"]:
        assert event["id"] == f"{event['tick']}:{event['seq']}"
        assert event["id"] not in seen_ids
        assert event["phase"] in {"exchange", "market", "cost", "shock"}
        assert event["actor"] is None or event["actor"] in agents
        assert event["target"] is None or event["target"] in agents
        seen_ids.add(event["id"])

    deaths = [event for event in replay["events"] if event["type"] == "death"]
    assert len(deaths) == len(simulation.death_log)
    assert {death["cause"] for death in deaths} == {"living_cost"}


def test_replay_validator_rejects_unknown_agent():
    simulation = ResourceSimulation(golden_config())
    simulation.run()
    replay = build_replay(simulation)

    broken = deepcopy(replay)
    broken["events"][0]["actor"] = "agent-does-not-exist"

    with pytest.raises(ValueError, match="unknown actor"):
        validate_replay(broken)


def test_replay_validator_rejects_bad_schema_version():
    simulation = ResourceSimulation(golden_config())
    simulation.run()
    replay = build_replay(simulation)
    replay["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="Unsupported replay schema version"):
        validate_replay(replay)


def test_replay_conserves_money_from_ticks_alone():
    simulation = ResourceSimulation(layered_config())
    simulation.run()

    replay = build_replay(simulation)

    for previous, current in zip(replay["ticks"], replay["ticks"][1:]):
        assert current["metrics"]["total_money"] == pytest.approx(
            previous["metrics"]["total_money"], abs=1e-9
        )
        previous_goods = previous["metrics"].get("goods_total_all", 0.0)
        expected_goods = (
            previous_goods
            + current["metrics"]["goods_produced"]
            + current["metrics"]["commons_regenerated"]
            - current["metrics"]["goods_consumed"]
            - current["metrics"]["commons_waste"]
        )
        assert current["metrics"]["goods_total_all"] == pytest.approx(
            expected_goods, abs=1e-9
        )


def test_replay_can_be_built_after_initialization_only():
    simulation = ResourceSimulation(golden_config())
    simulation.initialize_agents()

    replay = build_replay(simulation)

    assert replay["meta"]["iterations"] == 0
    assert len(replay["ticks"]) == 1
    assert replay["events"] == []
    assert replay["ticks"][0]["metrics"]["total"] == 30.0


def test_replay_cli_writes_a_loadable_file(tmp_path):
    from replay import main as replay_main

    output = tmp_path / "cli-replay.json.gz"
    replay_main(["--iterations", "2", "--output", str(output)])

    replay = load_replay(output)

    assert replay["schema_version"] == SCHEMA_VERSION
    assert replay["meta"]["iterations"] == 2
    assert len(replay["ticks"]) == 3
