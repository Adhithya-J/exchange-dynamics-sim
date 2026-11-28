"""Replay format tests for the subsistence world."""

from copy import deepcopy

import pytest

from utils.replay import replay_to_json
from worlds.subsistence.replay import (
    SCHEMA_VERSION,
    build_replay,
    load_replay,
    validate_replay,
    write_replay,
)
from worlds.subsistence.world import CONFIG, SubsistenceWorld


def small_world() -> SubsistenceWorld:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 10, "RANDOM_SEED": 5})
    config["FEATURES"]["GIFTING"] = True
    config["GIFTING"].update(
        {"PROBABILITY": 1.0, "AMOUNT_RANGE": [1, 1], "MIN_RESERVE": 2}
    )
    world = SubsistenceWorld(config)
    world.run(iterations=100)
    return world


def test_build_replay_has_versioned_structure():
    world = small_world()
    replay = build_replay(world)

    assert replay["schema_version"] == SCHEMA_VERSION == 2
    assert replay["meta"]["world"] == "subsistence"
    assert replay["meta"]["iterations"] == 100
    assert replay["meta"]["agent_count"] == 10
    assert replay["meta"]["features"]["TRADE"] is True
    assert len(replay["ticks"]) == 101
    assert replay["ticks"][0]["iteration"] == 0
    assert replay["ticks"][-1]["iteration"] == 100
    assert len(replay["agents"]) == 10
    validate_replay(replay)


def test_replay_contains_typed_events():
    replay = build_replay(small_world())
    event_types = {event["type"] for event in replay["events"]}

    assert "gather" in event_types
    for event in replay["events"]:
        assert event["tick"] >= 1
        assert event["phase"] in {
            "craft",
            "gather",
            "gift",
            "trade",
            "eat",
            "spoil",
        }
        assert event["id"] == f"{event['tick']}:{event['seq']}"


def test_replay_records_pre_and_post_trade_points():
    replay = build_replay(small_world())
    state = replay["ticks"][1]["agents"][0]

    for field in (
        "labor_berries",
        "labor_wood",
        "wood_progress",
        "gathered_berries",
        "gathered_wood",
        "post_trade_berries",
        "post_trade_wood",
    ):
        assert field in state
        assert isinstance(state[field], (int, float))


def test_replay_records_the_specialisation_latent():
    replay = build_replay(small_world())

    for agent in replay["agents"]:
        assert "specialisation_z" in agent
        assert isinstance(agent["specialisation_z"], (int, float))


def test_replay_records_commons_density_and_effective_yields():
    replay = build_replay(small_world())
    tick = replay["ticks"][1]

    assert "berries_commons_pct" in tick["metrics"]
    assert "wood_commons_pct" in tick["metrics"]
    for state in tick["agents"]:
        assert "effective_berries_per_day" in state
        assert "effective_wood_per_day" in state
        assert state["effective_berries_per_day"] >= 0
        assert state["effective_wood_per_day"] >= 0


def test_replay_records_beliefs_crates_and_crafting():
    replay = build_replay(small_world())
    tick = replay["ticks"][1]
    state = tick["agents"][0]

    assert "price_belief" in state
    assert "crates" in state
    assert "crate_durability" in state
    assert "craft_progress" in state
    assert "labor_craft" in state
    assert "mean_price_belief" in tick["metrics"]
    assert "total_crates" in tick["metrics"]
    assert replay["meta"]["storage"]["base_craft_minutes"] > 0


def test_replay_round_trip_is_deterministic(tmp_path):
    replay = build_replay(small_world())
    path = write_replay(replay, tmp_path / "replay.json.gz")

    loaded = load_replay(path)

    assert loaded == replay
    assert replay_to_json(loaded) == replay_to_json(replay)


def test_same_seed_produces_the_same_replay():
    assert replay_to_json(build_replay(small_world())) == replay_to_json(
        build_replay(small_world())
    )


def test_validate_replay_rejects_unknown_schema():
    replay = build_replay(small_world())
    replay["schema_version"] = 1

    with pytest.raises(ValueError, match="Unsupported replay schema"):
        validate_replay(replay)


def test_validate_replay_rejects_wrong_world():
    replay = build_replay(small_world())
    replay["meta"]["world"] = "money"

    with pytest.raises(ValueError, match="world"):
        validate_replay(replay)


def test_validate_replay_rejects_missing_tick_rows():
    replay = build_replay(small_world())
    replay["ticks"] = replay["ticks"][:-1]

    with pytest.raises(ValueError, match="ticks"):
        validate_replay(replay)


def test_validate_replay_rejects_missing_agent_state():
    replay = build_replay(small_world())
    replay["ticks"][1]["agents"] = replay["ticks"][1]["agents"][:-1]

    with pytest.raises(ValueError, match="one state per agent"):
        validate_replay(replay)


def test_tick_zero_uses_the_initial_barter_price():
    replay = build_replay(small_world())
    initial = replay["meta"]["initial_price"]

    assert 1 <= initial <= 20
    assert replay["ticks"][0]["metrics"]["barter_price"] == initial
    assert replay["meta"]["initial_price"] == initial
