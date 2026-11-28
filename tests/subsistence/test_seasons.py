"""Season, commons-scaling, and compact-replay tests."""

from copy import deepcopy

import pytest

from worlds.subsistence.replay import (
    AGENT_STATE_FIELDS,
    build_replay,
    decode_tick_agents,
    validate_replay,
)
from worlds.subsistence.world import CONFIG, SubsistenceWorld


def make_config(**simulation) -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 4, "RANDOM_SEED": 7, **simulation})
    config["COMMONS"]["SCALE_WITH_AGENTS"] = False
    config["SEASONS"]["ENABLED"] = False
    return config


def season_world(**simulation) -> SubsistenceWorld:
    config = make_config(**simulation)
    config["SEASONS"] = {
        "ENABLED": True,
        "LENGTH": 2,
        "CYCLE": [
            {"NAME": "spring", "BERRY_REGEN": 1.0, "WOOD_REGEN": 1.0},
            {"NAME": "winter", "BERRY_REGEN": 0.0, "WOOD_REGEN": 1.0},
        ],
    }
    config["COMMONS"]["BERRIES"].update(
        {"INITIAL": 100, "CAPACITY": 1000, "REGENERATION": 20.0}
    )
    config["COMMONS"]["WOOD"].update(
        {"INITIAL": 10, "CAPACITY": 1000, "REGENERATION": 5.0}
    )
    return SubsistenceWorld(config)


def test_winter_stops_berries_but_not_wood():
    world = season_world(AGENTS=1)
    world.run(4)

    first, second, third, fourth = world.metrics_history[:4]
    assert first["season_name"] == "spring"
    assert first["commons_regenerated_berries"] == 20
    assert second["season_name"] == "spring"
    assert second["commons_regenerated_berries"] == 20
    assert third["season_name"] == "winter"
    assert third["commons_regenerated_berries"] == 0
    assert third["berry_regen_factor"] == 0.0
    assert third["commons_regenerated_wood"] == 5
    assert fourth["season_name"] == "winter"
    assert fourth["commons_regenerated_berries"] == 0


def test_season_cycle_wraps_and_numbers_from_one():
    world = season_world(AGENTS=1)

    assert world.season_for(1)["name"] == "spring"
    assert world.season_for(2)["name"] == "spring"
    assert world.season_for(3)["name"] == "winter"
    assert world.season_for(4)["name"] == "winter"
    assert world.season_for(5)["name"] == "spring"
    assert world.season_for(5)["index"] == 0


def test_disabled_seasons_keep_steady_regeneration():
    config = make_config(AGENTS=1)
    config["COMMONS"]["BERRIES"].update(
        {"INITIAL": 100, "CAPACITY": 1000, "REGENERATION": 20.0}
    )
    world = SubsistenceWorld(config)
    world.run(3)

    assert world.season_for(1)["name"] == "temperate"
    assert all(
        metrics["commons_regenerated_berries"] == 20
        for metrics in world.metrics_history
    )


def test_commons_scales_with_village_size():
    config = make_config(AGENTS=20)
    config["COMMONS"]["SCALE_WITH_AGENTS"] = True
    config["COMMONS"]["REFERENCE_AGENTS"] = 10
    world = SubsistenceWorld(config)
    world.initialize_agents()

    assert world.initial_commons_berries == 4000
    assert world.commons_settings["berries"]["capacity"] == 6000
    assert world.commons_settings["berries"]["regeneration"] == pytest.approx(240)
    assert world.initial_commons_wood == 160
    assert world.commons_settings["wood"]["regeneration"] == pytest.approx(2)


def test_commons_scaling_can_be_disabled():
    config = make_config(AGENTS=20)
    config["COMMONS"]["SCALE_WITH_AGENTS"] = False
    world = SubsistenceWorld(config)
    world.initialize_agents()

    assert world.initial_commons_berries == 2000
    assert world.initial_commons_wood == 80


def test_replay_meta_records_seasons_and_commons_scale():
    config = make_config(AGENTS=4)
    config["SEASONS"]["ENABLED"] = True
    config["COMMONS"]["SCALE_WITH_AGENTS"] = True
    world = SubsistenceWorld(config)
    world.run(3)

    replay = build_replay(world)

    assert replay["meta"]["seasons"]["enabled"] is True
    assert replay["meta"]["seasons"]["length"] == 45
    assert replay["meta"]["commons"]["scale_with_agents"] is True
    assert replay["meta"]["commons"]["settings"]["berries"]["initial"] == 800
    tick = replay["ticks"][1]
    assert tick["metrics"]["season_name"] == "spring"
    assert tick["metrics"]["berry_regen_factor"] == pytest.approx(1.0)


def test_compact_rows_decode_back_to_named_states():
    world = SubsistenceWorld(make_config(AGENTS=3))
    world.run(2)

    replay = build_replay(world)

    assert replay["schema_version"] == 3
    for tick in replay["ticks"]:
        for row in tick["agents"]:
            assert isinstance(row, list)
            assert len(row) == len(AGENT_STATE_FIELDS)

    states = decode_tick_agents(replay["ticks"][1])
    assert len(states) == 3
    for state in states:
        assert set(state) == set(AGENT_STATE_FIELDS)
        assert isinstance(state["alive"], bool)
        assert isinstance(state["berries"], int)
        assert isinstance(state["price_belief"], float)


def test_validator_rejects_a_short_agent_row():
    world = SubsistenceWorld(make_config(AGENTS=2))
    world.run(1)

    replay = build_replay(world)
    replay["ticks"][1]["agents"][0] = replay["ticks"][1]["agents"][0][:-1]

    with pytest.raises(ValueError, match="fields per agent state"):
        validate_replay(replay)


def test_version_two_replays_still_validate():
    world = SubsistenceWorld(make_config(AGENTS=2))
    world.run(1)

    replay = build_replay(world)
    replay["schema_version"] = 2
    del replay["meta"]["agent_state_fields"]
    for tick in replay["ticks"]:
        tick["agents"] = decode_tick_agents(tick)

    validate_replay(replay)
    assert decode_tick_agents(replay["ticks"][0])[0]["id"] == "agent-001"
