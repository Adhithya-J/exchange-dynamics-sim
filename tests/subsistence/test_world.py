"""Core behaviour tests for the subsistence world."""

from copy import deepcopy

import pytest

from worlds.subsistence.world import CONFIG, Agent, SubsistenceWorld


def make_config(**overrides) -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 2, "RANDOM_SEED": 7})
    for section, values in overrides.items():
        config[section].update(values)
    return config


def quiet_config(**overrides) -> dict:
    """A world with no commons and no optional layers."""

    config = make_config(**overrides)
    config["FEATURES"].update(
        {"GIFTING": False, "TRADE": False, "STORAGE": False, "SPOILAGE": False}
    )
    config["COMMONS"]["BERRIES"].update(
        {"INITIAL": 0.0, "CAPACITY": 0.0, "REGENERATION": 0.0}
    )
    config["COMMONS"]["WOOD"].update(
        {"INITIAL": 0.0, "CAPACITY": 0.0, "REGENERATION": 0.0}
    )
    return config


def test_agents_start_with_nothing_but_full_health():
    world = SubsistenceWorld(make_config())
    world.initialize_agents()

    assert len(world.agents) == 2
    for agent in world.agents:
        assert agent.berries == 0.0
        assert agent.wood == 0.0
        assert agent.health == world.agent_settings["max_health"]
        assert agent.is_alive


def test_full_health_survives_exactly_max_health_missed_days():
    world = SubsistenceWorld(quiet_config(SIMULATION={"AGENTS": 1}))
    agent = world.initialize_agents()[0]
    max_health = world.agent_settings["max_health"]

    for iteration in range(1, max_health):
        world.run_iteration()
        assert agent.is_alive
        assert agent.health == pytest.approx(max_health - iteration)

    world.run_iteration()
    assert not agent.is_alive
    assert agent.health == 0.0
    assert world.extinction_iteration == max_health
    assert world.death_log[-1]["cause"] == "hunger"


def test_eating_restores_one_health():
    max_health = SubsistenceWorld(make_config()).agent_settings["max_health"]
    start_health = max_health - 1
    world = SubsistenceWorld(
        quiet_config(
            SIMULATION={"AGENTS": 1},
            AGENTS={
                "STARTING_HEALTH": start_health,
                "STARTING_BERRIES": 6,
            },
        )
    )
    agent = world.initialize_agents()[0]

    world.run_iteration()

    assert agent.health == pytest.approx(start_health + 1)


def test_no_storage_means_all_uneaten_berries_spoil():
    world = SubsistenceWorld(
        make_config(
            FEATURES={"STORAGE": False, "SPOILAGE": True, "TRADE": False}
        )
    )
    agent = world.initialize_agents()[0]
    agent.berries = 5
    agent.wood = 10

    world._decay_and_spoil()

    assert agent.storage_capacity == 0.0
    assert agent.berries == 0.0
    assert world.metrics_history == []


def test_storage_protects_berries_above_capacity():
    config = make_config(
        FEATURES={"STORAGE": True, "SPOILAGE": True, "TRADE": False}
    )
    config["STORAGE"]["WOOD_DECAY_CHANCE"] = 0.0
    world = SubsistenceWorld(config)
    agent = world.initialize_agents()[0]
    agent.berries = 12
    agent.wood = 8

    world._decay_and_spoil()

    assert agent.berries == 10
    assert world.spoil_log[-1]["amount"] == 2


def test_wood_decay_shrinks_storage_and_spoils_the_excess():
    config = make_config(
        FEATURES={"STORAGE": True, "SPOILAGE": True, "TRADE": False}
    )
    config["STORAGE"]["WOOD_DECAY_CHANCE"] = 1.0
    world = SubsistenceWorld(config)
    agent = world.initialize_agents()[0]
    agent.wood = 4
    agent.berries = 8

    world._decay_and_spoil()

    assert agent.wood == 0
    assert agent.berries == 0


def test_spoilage_off_keeps_berries_without_storage():
    world = SubsistenceWorld(
        make_config(
            FEATURES={"STORAGE": False, "SPOILAGE": False, "TRADE": False}
        )
    )
    agent = world.initialize_agents()[0]
    agent.berries = 5.0

    world._decay_and_spoil()

    assert agent.berries == pytest.approx(5.0)


def test_labour_allocation_follows_comparative_advantage():
    world = SubsistenceWorld(make_config())
    world.initialize_agents()
    berry_specialist = world.agents[0]
    wood_specialist = world.agents[1]
    berry_specialist.berry_skill = 4.0
    berry_specialist.wood_skill = 0.5
    berry_specialist.berries = 6
    wood_specialist.berry_skill = 0.8
    wood_specialist.wood_skill = 2.5
    wood_specialist.berries = 6
    # The wood specialist needs something to sell before it can specialise.
    wood_specialist.wood = 1
    world.trade_price = 4

    world._allocate_labour(berry_specialist)
    world._allocate_labour(wood_specialist)

    assert berry_specialist.planned_berries > 0
    assert berry_specialist.planned_wood == 0
    assert wood_specialist.planned_wood > 0
    assert wood_specialist.planned_berries == 0


def test_ppf_intercepts_come_from_time_and_skill():
    agent = Agent(
        id="a",
        berry_skill=1.5,
        wood_skill=1.0,
        time_per_iteration=100.0,
        berry_time_cost=10.0,
        wood_time_cost=80.0,
        health=10,
        max_health=10,
        storage_wood_per_container=8,
        storage_berries_per_container=10,
    )

    assert agent.max_berries_per_day == pytest.approx(15.0)
    assert agent.max_wood_per_day == pytest.approx(1.25)
    assert agent.opportunity_cost_of_wood == pytest.approx(12.0)


def test_skill_latent_is_negatively_correlated_and_clipped():
    world = SubsistenceWorld(
        make_config(SIMULATION={"AGENTS": 200, "RANDOM_SEED": 11})
    )
    agents = world.initialize_agents()
    berry = [agent.berry_skill for agent in agents]
    wood = [agent.wood_skill for agent in agents]

    mean_berry = sum(berry) / len(berry)
    mean_wood = sum(wood) / len(wood)
    covariance = sum(
        (b - mean_berry) * (w - mean_wood) for b, w in zip(berry, wood)
    )
    variance_berry = sum((b - mean_berry) ** 2 for b in berry)
    variance_wood = sum((w - mean_wood) ** 2 for w in wood)
    correlation = covariance / (variance_berry * variance_wood) ** 0.5

    assert correlation < -0.8
    for agent in agents:
        assert 0.30 <= agent.berry_skill <= 1.50
        assert 0.45 <= agent.wood_skill <= 1.45
        assert agent.berry_skill == pytest.approx(
            min(1.50, max(0.30, 0.85 + 0.35 * agent.specialisation_z))
        )
        assert agent.wood_skill == pytest.approx(
            min(1.45, max(0.45, 1.05 - 0.25 * agent.specialisation_z))
        )
    assert max(agent.max_berries_per_day for agent in agents) <= 15.0
    assert max(agent.max_wood_per_day for agent in agents) <= 100 * 1.45 / 120
    assert max(agent.max_wood_per_day for agent in agents) >= 1.0


def test_default_skill_and_health_parameters():
    settings = SubsistenceWorld(CONFIG).agent_settings

    assert settings["max_health"] == 4
    assert settings["starting_health"] == 4
    assert settings["food_per_iteration"] == 6
    assert settings["skill_z_range"] == pytest.approx(2.0)
    assert settings["berry_intercept"] == pytest.approx(0.85)
    assert settings["berry_slope"] == pytest.approx(0.35)
    assert settings["wood_intercept"] == pytest.approx(1.05)
    assert settings["wood_slope"] == pytest.approx(-0.25)
    assert settings["berry_skill_range"] == pytest.approx((0.30, 1.50))
    assert settings["wood_skill_range"] == pytest.approx((0.45, 1.45))


def test_partial_wood_work_carries_between_days():
    world = SubsistenceWorld(make_config())
    agent = world.initialize_agents()[0]
    world.commons_wood = 10

    world._extract(agent, "wood", 0.9)

    assert agent.wood == 0
    assert agent.wood_progress == pytest.approx(0.9)

    world._extract(agent, "wood", 0.2)

    assert agent.wood == 1
    assert agent.wood_progress == pytest.approx(0.1)
    assert world.commons_wood == 9


def test_trade_is_off_when_feature_is_disabled():
    world = SubsistenceWorld(make_config(FEATURES={"TRADE": False}))
    world.run(iterations=30)

    assert world.trade_log == []


def test_gifting_is_off_by_default():
    world = SubsistenceWorld(make_config())
    world.run(iterations=30)

    assert world.gift_log == []


def test_gifting_can_be_enabled():
    config = make_config(
        FEATURES={"GIFTING": True, "TRADE": False},
        GIFTING={"PROBABILITY": 1.0, "AMOUNT_RANGE": [1, 1], "MIN_RESERVE": 2},
    )
    config["AGENTS"]["STARTING_BERRIES"] = 5
    world = SubsistenceWorld(config)
    world.initialize_agents()

    world.run_iteration()

    assert world.gift_log
    assert world.metrics_history[-1]["gifts"] >= 1


def test_unsupported_feature_flags_are_rejected():
    with pytest.raises(ValueError, match="TRUST"):
        SubsistenceWorld(make_config(FEATURES={"TRUST": True}))


def test_wood_is_not_a_specialisation_without_storage():
    world = SubsistenceWorld(
        make_config(FEATURES={"STORAGE": False, "SPOILAGE": True, "TRADE": True})
    )
    world.initialize_agents()
    agent = world.agents[0]
    agent.berry_skill = 1.0
    agent.wood_skill = 10.0
    world.trade_price = 100

    world._allocate_labour(agent)

    assert agent.planned_wood == 0
    assert agent.planned_berries > 0


def test_wood_is_not_a_specialisation_without_spoilage():
    world = SubsistenceWorld(
        make_config(FEATURES={"STORAGE": True, "SPOILAGE": False, "TRADE": True})
    )
    world.initialize_agents()
    agent = world.agents[0]
    agent.berry_skill = 1.0
    agent.wood_skill = 10.0
    world.trade_price = 100

    world._allocate_labour(agent)

    assert agent.planned_wood == 0
    assert agent.planned_berries > 0
