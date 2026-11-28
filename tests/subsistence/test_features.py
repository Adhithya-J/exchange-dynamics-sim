"""Composable feature-layer tests."""

from copy import deepcopy

import pytest

from worlds.subsistence.world import CONFIG, DEFAULT_FEATURES, SubsistenceWorld


def feature_config(**overrides) -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 3, "RANDOM_SEED": 7})
    features = overrides.pop("FEATURES", {})
    config["FEATURES"].update(features)
    for section, values in overrides.items():
        config[section].update(values)
    return config


def test_every_default_feature_has_a_registered_layer():
    world = SubsistenceWorld(feature_config())

    assert set(world.features) == set(DEFAULT_FEATURES)
    assert set(world.feature_registry.layers) == set(DEFAULT_FEATURES)


def test_dependencies_are_resolved_transitively():
    world = SubsistenceWorld(
        feature_config(FEATURES={"RECIPROCITY": True})
    )

    assert world.features["RECIPROCITY"]
    assert world.features["GIFTING"]
    assert world.features["TRUST"]
    assert not world.features["CREDIT"]


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError, match="Unknown FEATURES"):
        SubsistenceWorld(feature_config(FEATURES={"TELEPORT": True}))


def test_passive_core_layers_are_registered():
    world = SubsistenceWorld(feature_config())

    assert world.feature_registry.enabled("STORAGE")
    assert world.feature_registry.enabled("SPOILAGE")
    assert world.feature_registry.enabled("TRADE")


def test_gifting_without_reciprocity_creates_no_obligation():
    world = SubsistenceWorld(
        feature_config(
            FEATURES={"GIFTING": True},
            GIFTING={
                "PROBABILITY": 1.0,
                "AMOUNT_RANGE": [2, 2],
                "MIN_RESERVE": 2,
            },
        )
    )
    giver, receiver = world.initialize_agents()[:2]
    giver.berries = 20
    receiver.berries = 0
    receiver.health = 2

    world._current_iteration = 1
    world._process_gifts()

    assert world.gift_log
    assert world.obligations.gifts() == []


def test_reciprocity_turns_a_gift_into_a_soft_zero_interest_obligation():
    world = SubsistenceWorld(
        feature_config(
            SIMULATION={"AGENTS": 2},
            FEATURES={"GIFTING": True, "RECIPROCITY": True},
            GIFTING={
                "PROBABILITY": 1.0,
                "AMOUNT_RANGE": [2, 2],
                "MIN_RESERVE": 2,
            },
        )
    )
    giver, receiver = world.initialize_agents()[:2]
    giver.berries = 20
    receiver.berries = 0
    receiver.health = 2

    world._current_iteration = 1
    world._process_gifts()

    gifts = world.obligations.gifts()
    assert len(gifts) == 1
    obligation = gifts[0]
    assert obligation.creditor == giver.id
    assert obligation.debtor == receiver.id
    assert obligation.principal == 2
    assert obligation.amount_due == 2
    assert obligation.interest_rate == 0.0
    assert obligation.due_iteration is None

    receiver.berries = 20
    world.obligations.settle_soft_gifts(2, threshold=2)

    assert world.obligations.gifts() == []
    assert receiver.berries == 18
    assert giver.berries == 20


def test_shocks_reduce_commons_regeneration_and_are_logged():
    world = SubsistenceWorld(
        feature_config(
            FEATURES={"SHOCKS": True},
            SHOCKS={
                "PROBABILITY": 1.0,
                "DURATION_RANGE": [2, 2],
                "BERRY_REGEN_FACTOR": 0.5,
                "WOOD_REGEN_FACTOR": 0.5,
            },
        )
    )
    world.run(3)

    assert world.shock_log
    assert world.metrics_history[0]["shock_active"] == 1
    assert world.metrics_history[0]["shocks"] == 1
    assert world.metrics_history[0]["commons_regenerated_berries"] < 120


def test_trust_decay_pulls_reputation_back_to_default():
    world = SubsistenceWorld(
        feature_config(
            FEATURES={"TRUST": True},
            TRUST={"DEFAULT": 0.5, "DECAY": 0.5},
        )
    )
    agent = world.initialize_agents()[0]
    agent.trust = 1.0

    world.feature_registry.run("end_of_tick")

    assert agent.trust == pytest.approx(0.75)
