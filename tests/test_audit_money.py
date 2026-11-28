"""Regression tests for the money-world audit fixes."""

from copy import deepcopy

import pytest

from worlds.money.world import CONFIG, ResourceSimulation


def social_config() -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 2, "RANDOM_SEED": 1})
    config["GIFTING"]["ENABLED"] = True
    config["SOCIAL"].update(
        {"RECIPROCITY_ENABLED": True, "REPAY_DELAY": 1}
    )
    config["TRUST"].update({"ENABLED": True, "DECAY": 0.0})
    return config


def test_creditor_death_is_not_observed_as_borrower_default():
    simulation = ResourceSimulation(social_config())
    creditor, debtor = simulation.initialize_agents()
    creditor.resources = 0.0
    debtor.resources = 100.0
    debtor.repayment_probability = 1.0
    reputation_before = simulation.reputation[debtor.id]
    simulation._current_iteration = 1
    simulation._schedule_repayment(creditor, debtor, 10.0)

    simulation._current_iteration = 2
    simulation._process_repayments(2)

    assert simulation.written_off_debt == pytest.approx(10.0)
    assert simulation.reputation[debtor.id] == pytest.approx(reputation_before)
    assert simulation._repayment_opportunities[debtor.id] == 0


def test_repay_fraction_above_one_is_rejected():
    config = social_config()
    config["SOCIAL"]["REPAY_FRACTION"] = 1.5

    with pytest.raises(ValueError, match="REPAY_FRACTION"):
        ResourceSimulation(config)


def test_negative_living_cost_is_rejected():
    config = deepcopy(CONFIG)
    config["AGENTS"]["LIVING_COST"] = -1.0

    with pytest.raises(ValueError, match="LIVING_COST"):
        ResourceSimulation(config)


def test_negative_resource_floor_is_rejected():
    config = deepcopy(CONFIG)
    config["SIMULATION"]["RESOURCE_FLOOR"] = -1.0

    with pytest.raises(ValueError, match="RESOURCE_FLOOR"):
        ResourceSimulation(config)


def test_zero_agents_is_rejected():
    config = deepcopy(CONFIG)
    config["SIMULATION"]["AGENTS"] = 0

    with pytest.raises(ValueError, match="at least 1"):
        ResourceSimulation(config)


def test_existing_debt_reduces_acceptance_probability():
    simulation = ResourceSimulation(social_config())
    giver, receiver = simulation.initialize_agents()
    receiver.acceptance_score = 1.0
    simulation.affordability = type(
        "AlwaysAccept",
        (),
        {"effective_acceptance": lambda self, agent: 1.0},
    )()
    baseline = simulation._receive_probability(receiver, giver)
    simulation._repayment_queue.append(
        {
            "due": 99,
            "gift_iteration": 0,
            "debtor": receiver.id,
            "creditor": giver.id,
            "amount": 100.0,
            "remaining": 100.0,
            "defaults": 0,
            "attempts": 0,
        }
    )

    burdened = simulation._receive_probability(receiver, giver)

    assert burdened < baseline
