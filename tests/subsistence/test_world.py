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
    agent_overrides = overrides.get("AGENTS", {})
    for key in ("STARTING_BERRIES", "STARTING_WOOD", "STARTING_CRATES"):
        if key not in agent_overrides:
            config["AGENTS"][key] = 0
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


def test_agents_start_with_a_starter_crate_and_food():
    world = SubsistenceWorld(make_config())
    world.initialize_agents()

    assert len(world.agents) == 2
    for agent in world.agents:
        assert agent.berries == 10
        assert agent.wood == 0
        assert agent.crate_count == 1
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
    world = SubsistenceWorld(config)
    agent = world.initialize_agents()[0]
    agent.crates = [1.0]
    agent.wood = 0
    agent.berries = 12

    world._decay_and_spoil()

    assert agent.storage_capacity == 10
    assert agent.berries == 10
    assert world.spoil_log[-1]["amount"] == 2


def test_crate_wear_shrinks_storage_and_spoils_the_excess():
    config = make_config(
        FEATURES={"STORAGE": True, "SPOILAGE": True, "TRADE": False}
    )
    config["STORAGE"]["CRATE_WEAR_RATE"] = 1.0
    world = SubsistenceWorld(config)
    agent = world.initialize_agents()[0]
    agent.crates = [1.0]
    agent.wood = 0
    agent.berries = 8

    world._decay_and_spoil()

    assert agent.crates == []
    assert agent.storage_capacity == 0
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
    world = SubsistenceWorld(
        make_config(
            AGENTS={"CHOICE_NOISE": 0.0, "EXPLORATION_RATE": 0.0}
        )
    )
    world.initialize_agents()
    berry_specialist = world.agents[0]
    wood_specialist = world.agents[1]
    berry_specialist.berry_skill = 4.0
    berry_specialist.wood_skill = 0.5
    berry_specialist.berries = 4
    wood_specialist.berry_skill = 0.8
    wood_specialist.wood_skill = 2.5
    wood_specialist.berries = 4
    # The wood specialist needs something to sell before it can specialise.
    wood_specialist.wood = 1
    world.trade_price = 4

    world._allocate_labour(berry_specialist)
    world._allocate_labour(wood_specialist)

    assert berry_specialist.planned_berries > 0
    assert wood_specialist.planned_wood > 0
    assert berry_specialist.planned_berries > wood_specialist.planned_berries
    assert wood_specialist.planned_wood > berry_specialist.planned_wood


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
    world = SubsistenceWorld(CONFIG)
    settings = world.agent_settings

    assert world.min_commons_density == pytest.approx(0.10)
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
    assert settings["reserve_days"] == 2
    assert settings["refill_fraction"] == pytest.approx(0.5)
    assert settings["max_crates_target"] == 2
    assert settings["wood_reserve_crates"] == 1
    assert settings["choice_noise"] > 0
    assert 0 <= settings["exploration_rate"] <= 1


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


def test_effective_yields_scale_with_commons_density():
    world = SubsistenceWorld(make_config(SIMULATION={"AGENTS": 1}))
    agent = world.initialize_agents()[0]
    full_berries = world.effective_max_berries_per_day(agent)
    full_wood = world.effective_max_wood_per_day(agent)

    world.commons_berries = world.initial_commons_berries // 2
    world.commons_wood = world.initial_commons_wood // 2
    world._current_berries_density = world._commons_density("berries")
    world._current_wood_density = world._commons_density("wood")

    assert world._current_berries_density == pytest.approx(0.5)
    assert world._current_wood_density == pytest.approx(0.5)
    assert world.effective_max_berries_per_day(agent) == pytest.approx(
        full_berries / 2
    )
    assert world.effective_max_wood_per_day(agent) == pytest.approx(
        full_wood / 2
    )

    world.commons_berries = 0
    world._current_berries_density = 0.0
    assert world.effective_max_berries_per_day(agent) == pytest.approx(
        full_berries * world.min_commons_density
    )


def test_berry_pickup_is_capped_and_leftover_goes_to_wood():
    world = SubsistenceWorld(
        make_config(
            SIMULATION={"AGENTS": 1},
            FEATURES={"TRADE": False},
        )
    )
    world.initialize_agents()
    agent = world.agents[0]
    agent.berry_skill = 4.0
    agent.wood_skill = 1.0
    agent.berries = 20
    agent.wood = 0
    world.trade_price = 5
    world._current_berries_density = 1.0
    world._current_wood_density = 1.0

    world._allocate_labour(agent)

    # The crate is empty, so the cap is just the six-berry ration. The agent
    # already has more than enough, so the leftover day goes to chopping.
    assert agent.planned_berries == 0
    assert agent.planned_wood > 0


def test_berry_cap_bounds_new_pickup_not_total_holding():
    world = SubsistenceWorld(make_config(SIMULATION={"AGENTS": 1}))
    world.initialize_agents()
    agent = world.agents[0]
    agent.crates = [1.0]
    agent.wood = 0
    agent.berries = 8

    # The reserve is two days of food, capped by the starter crate's ten
    # berries. With eight berries the larder is already above the refill
    # threshold, so the villager leaves it alone and chops instead.
    assert world._target_berries(agent) == 10
    assert world._berry_refill_threshold(agent) == 6
    assert world._rational_berry_target(agent) == 8

    # After the reserve runs down, it refills in one batch to the reserve plus
    # today's meal.
    agent.berries = 4
    assert world._rational_berry_target(agent) == 16

    agent.berries = 20
    assert world._rational_berry_target(agent) == 20


def test_wood_reserve_target_keeps_a_spare_crate_of_material():
    world = SubsistenceWorld(make_config(SIMULATION={"AGENTS": 1}))
    world.initialize_agents()
    agent = world.agents[0]

    # One crate owned, one missing, plus one spare crate of raw wood.
    agent.crates = [1.0]
    assert world._target_wood(agent) == 16

    agent.crates = [1.0, 1.0]
    assert world._target_wood(agent) == 8

    agent.crates = []
    assert world._target_wood(agent) == 16


def test_choice_sampling_can_swap_the_preferred_activity():
    world = SubsistenceWorld(
        make_config(
            SIMULATION={"AGENTS": 1},
            AGENTS={"CHOICE_NOISE": 0.0, "EXPLORATION_RATE": 1.0},
        )
    )
    world.initialize_agents()
    agent = world.agents[0]
    # A strong berry picker who would normally spend the day on berries.
    agent.berry_skill = 1.5
    agent.wood_skill = 0.45
    agent.price_belief = 5
    agent.berries = 0
    agent.crates = []
    world._current_berries_density = 1.0
    world._current_wood_density = 1.0

    world._allocate_labour(agent)

    # With exploration forced, the sampled choice flips to wood, so the agent
    # works the day for wood instead of picking every available berry.
    assert agent.planned_wood > 0


def test_zero_storage_agent_picks_ration_then_chops():
    world = SubsistenceWorld(make_config(SIMULATION={"AGENTS": 1}))
    world.initialize_agents()
    agent = world.agents[0]
    agent.crates = []
    agent.berries = 0
    agent.wood = 0
    agent.berry_skill = 0.74
    agent.wood_skill = 1.13
    agent.price_belief = 6
    world._current_berries_density = 1.0
    world._current_wood_density = 1.0

    world._allocate_labour(agent)

    assert agent.planned_berries <= 6.01
    assert agent.planned_wood > 0


def test_price_beliefs_start_as_bounded_integers():
    world = SubsistenceWorld(make_config())

    for agent in world.initialize_agents():
        assert isinstance(agent.price_belief, int)
        assert 1 <= agent.price_belief <= 20


def test_used_crates_wear_faster_than_idle_crates():
    world = SubsistenceWorld(make_config(SIMULATION={"AGENTS": 2}))
    busy, idle = world.initialize_agents()
    busy.crates = [1.0]
    busy.berries = 10
    idle.crates = [1.0]
    idle.berries = 0

    world._decay_and_spoil()

    assert busy.crate_durability < idle.crate_durability
    assert busy.crate_durability == pytest.approx(1.0 - 0.02)
    assert idle.crate_durability == pytest.approx(1.0 - 0.01)


def test_crate_crafting_consumes_wood_and_carries_progress():
    world = SubsistenceWorld(make_config(SIMULATION={"AGENTS": 1}))
    world.initialize_agents()
    agent = world.agents[0]
    agent.berry_skill = 0.1
    agent.wood_skill = 0.5
    agent.wood = 8
    agent.price_belief = 10
    world._current_berries_density = 1.0
    world._current_wood_density = 1.0

    world._allocate_labour(agent)

    # A 200-minute crate takes two full days for this villager.
    assert agent.planned_crates == 0
    assert agent.craft_progress == pytest.approx(100.0)

    world._reset_iteration_state()
    world._allocate_labour(agent)

    assert agent.planned_crates == 1
    assert agent.craft_progress == pytest.approx(0.0)
    assert agent.wood == 0
    assert agent.crate_count == 2


def test_crate_barter_moves_a_finished_crate():
    world = SubsistenceWorld(
        make_config(SIMULATION={"AGENTS": 2, "RANDOM_SEED": 3})
    )
    world.initialize_agents()
    seller, buyer = world.agents
    seller.berry_skill = 0.3
    seller.wood_skill = 1.4
    seller.crates = [1.0, 1.0, 1.0]
    seller.berries = 0
    seller.price_belief = 10
    buyer.berry_skill = 1.4
    buyer.wood_skill = 0.6
    buyer.crates = []
    buyer.berries = 200
    buyer.price_belief = 10
    world._current_berries_density = 1.0
    world._current_wood_density = 1.0

    world._process_trade()

    assert seller.crate_count == 2
    assert buyer.crate_count == 1
    assert seller.berries > 0
    assert buyer.berries < 200
    assert world.trade_log
    assert world.trade_log[0]["good"] == "crate"
    assert world.metrics_history == []


def test_belief_updates_happen_after_the_matching_phase():
    world = SubsistenceWorld(
        make_config(SIMULATION={"AGENTS": 2, "RANDOM_SEED": 4})
    )
    world.initialize_agents()
    seller, buyer = world.agents
    seller.berry_skill = 0.3
    seller.wood_skill = 1.4
    seller.wood = 10
    seller.berries = 0
    seller.price_belief = 4
    buyer.berry_skill = 1.4
    buyer.wood_skill = 0.6
    buyer.wood = 0
    buyer.berries = 100
    buyer.price_belief = 10
    world._current_berries_density = 1.0
    world._current_wood_density = 1.0
    start = {agent.id: agent.price_belief for agent in world.agents}
    observed = []
    original_match = world._match_market

    def wrapped(good, sellers, buyers):
        for entry in sellers + buyers:
            observed.append(entry["agent"].price_belief)
        return original_match(good, sellers, buyers)

    world._match_market = wrapped
    world._process_trade()

    assert observed
    assert all(value in (4, 10) for value in observed)
    assert seller.price_belief != start[seller.id]
    assert buyer.price_belief != start[buyer.id]


def test_gossip_and_failed_adjustments_only_touch_participants():
    world = SubsistenceWorld(
        make_config(SIMULATION={"AGENTS": 3, "RANDOM_SEED": 6})
    )
    living = world.initialize_agents()
    world.trade_settings["belief_band"] = 20
    trader, observer, absent = living
    start = {
        trader.id: 4,
        observer.id: 4,
        absent.id: 12,
    }
    for agent in living:
        agent.price_belief = start[agent.id]

    world._apply_belief_updates(
        living,
        start,
        [
            {
                "good": "wood",
                "seller": trader.id,
                "buyer": observer.id,
                "qty": 1,
                "price": 6,
            }
        ],
        {trader.id, observer.id, absent.id},
        [],
        [],
    )

    assert trader.price_belief == 5
    assert observer.price_belief == 5
    assert absent.price_belief == 11  # gossip moves one step toward 6
    assert absent.price_belief != start[absent.id]

    non_participant = start[absent.id]
    absent.price_belief = non_participant
    world._apply_belief_updates(
        living,
        {agent.id: agent.price_belief for agent in living},
        [],
        {trader.id, observer.id},
        [{"agent": trader}],
        [{"agent": observer}],
    )

    assert absent.price_belief == non_participant
    assert trader.price_belief == 4  # failed seller lowers
    assert observer.price_belief == 6  # failed buyer raises


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
    agent.berries = 0
    agent.price_belief = 100

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
    agent.berries = 0
    agent.price_belief = 100

    world._allocate_labour(agent)

    assert agent.planned_wood == 0
    assert agent.planned_berries > 0
