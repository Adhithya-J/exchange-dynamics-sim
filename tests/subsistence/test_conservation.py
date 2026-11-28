"""Conservation tests for berries and wood."""

from copy import deepcopy

import pytest

from worlds.subsistence.world import CONFIG, SubsistenceWorld


def active_config() -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 5, "RANDOM_SEED": 3})
    config["FEATURES"].update({"GIFTING": True, "TRADE": True})
    config["GIFTING"].update(
        {"PROBABILITY": 0.5, "AMOUNT_RANGE": [1, 1], "MIN_RESERVE": 2}
    )
    return config


def test_berry_and_wood_stocks_obey_the_ledger():
    world = SubsistenceWorld(active_config())
    world.initialize_agents()
    previous_berries = (
        sum(agent.berries for agent in world.agents) + world.commons_berries
    )
    previous_wood = sum(agent.wood for agent in world.agents) + world.commons_wood

    for _ in range(200):
        metrics = world.run_iteration()
        current_berries = (
            sum(agent.berries for agent in world.agents) + world.commons_berries
        )
        expected_berries = (
            previous_berries
            + metrics["commons_regenerated_berries"]
            - metrics["berries_eaten"]
            - metrics["berries_spoiled"]
            - metrics["commons_waste_berries"]
        )
        assert current_berries == pytest.approx(expected_berries, abs=1e-6)
        assert metrics["total_berries_all"] == pytest.approx(
            current_berries, abs=1e-6
        )

        current_wood = (
            sum(agent.wood for agent in world.agents) + world.commons_wood
        )
        expected_wood = (
            previous_wood
            + metrics["commons_regenerated_wood"]
            - metrics["wood_decayed"]
            - metrics["commons_waste_wood"]
        )
        assert current_wood == pytest.approx(expected_wood, abs=1e-6)
        assert metrics["total_wood_all"] == pytest.approx(current_wood, abs=1e-6)

        previous_berries = current_berries
        previous_wood = current_wood


def test_no_stock_becomes_negative():
    world = SubsistenceWorld(active_config())
    world.run(iterations=100)

    for agent in world.agents:
        assert agent.berries >= 0.0
        assert agent.wood >= 0.0
        assert 0.0 <= agent.health <= agent.max_health
    assert world.commons_berries >= 0.0
    assert world.commons_wood >= 0.0


def test_extraction_waste_is_the_only_commons_loss():
    config = active_config()
    config["COMMONS"]["BERRIES"]["EXTRACTION_EFFICIENCY"] = 0.5
    config["COMMONS"]["WOOD"]["EXTRACTION_EFFICIENCY"] = 0.5
    world = SubsistenceWorld(config)
    world.run(iterations=40)

    for metrics in world.metrics_history:
        assert metrics["commons_waste_berries"] >= 0.0
        assert metrics["commons_waste_wood"] >= 0.0
