"""Credit-layer tests for both reciprocal and market modes."""

from copy import deepcopy

import pytest

from worlds.subsistence.world import CONFIG, SubsistenceWorld


def credit_config(mode: str = "reciprocal", **overrides) -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 2, "RANDOM_SEED": 3})
    config["FEATURES"]["CREDIT"] = True
    config["CREDIT"]["MODE"] = mode
    for section, values in overrides.items():
        config[section].update(values)
    return config


def prepare_pair(world: SubsistenceWorld):
    lender, borrower = world.initialize_agents()
    lender.wood = 10
    lender.crates = []
    lender.trust = 0.6
    borrower.wood = 0
    borrower.crates = [1.0]
    borrower.trust = 0.6
    borrower.health = 4
    borrower.berries = 6
    borrower.berry_skill = 0.3
    borrower.wood_skill = 1.45
    world._current_iteration = 1
    return lender, borrower


def test_credit_mode_must_be_known():
    config = credit_config("reciprocal")
    config["CREDIT"]["MODE"] = "barter"
    with pytest.raises(ValueError, match="CREDIT.MODE"):
        SubsistenceWorld(config)


def test_credit_is_off_by_default():
    world = SubsistenceWorld(CONFIG)
    world.run(15)

    assert world.loans == []
    assert world.credit_events == []
    assert world.metrics_history[-1]["loans_originated"] == 0


def test_reciprocal_loan_moves_wood_and_charges_no_interest():
    world = SubsistenceWorld(credit_config("reciprocal"))
    lender, borrower = prepare_pair(world)

    world._originate_credit()

    assert len(world.loans) == 1
    loan = world.loans[0]
    assert loan.principal == 8
    assert loan.amount_due == 8
    assert loan.interest_rate == 0.0
    assert loan.collateral_crates == 1
    assert loan.due_iteration == 1 + world.credit_settings["term_days"]
    assert lender.wood == 2
    assert borrower.wood == 8
    assert world.metrics_history == []
    assert world.credit_events[0]["event"] == "origination"


def test_reciprocal_credit_requires_trust_or_collateral():
    world = SubsistenceWorld(credit_config("reciprocal"))
    lender, borrower = prepare_pair(world)
    borrower.trust = 0.1
    borrower.crates = []

    world._originate_credit()
    assert world.loans == []

    borrower.crates = [1.0]
    world._originate_credit()

    assert len(world.loans) == 1
    assert world.loans[0].collateral_crates == 1


def test_repayment_moves_goods_back_and_raises_trust():
    world = SubsistenceWorld(credit_config("reciprocal"))
    lender, borrower = prepare_pair(world)
    world._originate_credit()
    loan = world.loans[0]
    loan.due_iteration = 2
    trust_before = borrower.trust

    world._current_iteration = 2
    world._settle_credit()

    assert loan.status == "repaid"
    assert loan.amount_due == 0
    assert lender.wood == 10
    assert borrower.wood == 0
    assert borrower.trust == pytest.approx(
        trust_before + world.trust_settings["gain"]
    )
    assert world.repayment_log[-1]["event"] == "repayment"


def test_default_seizes_pledged_crate_and_lowers_trust():
    world = SubsistenceWorld(
        credit_config("reciprocal", CREDIT={"GRACE_DAYS": 0})
    )
    lender, borrower = prepare_pair(world)
    world._originate_credit()
    loan = world.loans[0]
    loan.due_iteration = 2
    borrower.wood = 0
    trust_before = borrower.trust

    world._current_iteration = 2
    world._settle_credit()

    assert loan.status == "defaulted"
    assert loan.reason == "unable"
    assert loan.collateral_crates == 1
    assert lender.crate_count == 1
    assert borrower.crate_count == 0
    assert borrower.trust == pytest.approx(
        max(0.0, trust_before - world.trust_settings["loss"])
    )
    assert world.default_log[-1]["collateral_seized"] == 1


def test_death_defaults_the_loan_and_lender_takes_the_estate():
    world = SubsistenceWorld(credit_config("reciprocal"))
    lender, borrower = prepare_pair(world)
    world._originate_credit()
    loan = world.loans[0]
    loan.collateral_crates = 0
    borrower.crates = []
    borrower.wood = 8
    borrower.health = 0
    loan.due_iteration = 99

    world._current_iteration = 2
    world._settle_credit()

    assert loan.status == "defaulted"
    assert loan.reason == "death"
    assert lender.wood == 10
    assert borrower.wood == 0


def test_market_mode_prices_risk_into_a_positive_interest_rate():
    config = credit_config("market", CREDIT={"MAX_LOANS_PER_LENDER": 3})
    config["SIMULATION"]["AGENTS"] = 3
    world = SubsistenceWorld(config)
    lender, safe, risky = world.initialize_agents()
    lender.wood = 20
    lender.crates = [1.0]
    for borrower in (safe, risky):
        borrower.wood = 0
        borrower.crates = [1.0]
        borrower.berry_skill = 0.3
        borrower.wood_skill = 1.45
    safe.trust = 0.9
    safe.health = 4
    safe.berries = 6
    risky.trust = 0.35
    risky.health = 1
    risky.berries = 0

    world._current_iteration = 1
    world._originate_credit()

    assert len(world.loans) == 2
    by_debtor = {loan.debtor: loan for loan in world.loans}
    safe_loan = by_debtor[safe.id]
    risky_loan = by_debtor[risky.id]
    assert safe_loan.amount_due > safe_loan.principal
    assert risky_loan.amount_due > safe_loan.amount_due
    assert risky_loan.interest_rate > safe_loan.interest_rate > 0


def test_crate_capacity_can_be_lent_and_repaid():
    world = SubsistenceWorld(credit_config("reciprocal"))
    lender, borrower = prepare_pair(world)
    lender.crates = [1.0, 1.0]
    borrower.crates = []
    lender.wood = 0
    borrower.wood = 0

    world._originate_credit()

    assert len(world.loans) == 1
    loan = world.loans[0]
    assert loan.good == "crates"
    assert loan.principal == 1
    assert loan.amount_due == 1
    assert loan.collateral_crates == 1
    assert lender.crate_count == 1
    assert borrower.crate_count == 1

    loan.due_iteration = 2
    world._current_iteration = 2
    world._settle_credit()

    assert loan.status == "repaid"
    assert lender.crate_count == 2
    assert borrower.crate_count == 0


def test_credit_metrics_report_principal_interest_and_outstanding():
    world = SubsistenceWorld(credit_config("market"))
    lender, borrower = prepare_pair(world)
    world._originate_credit()

    metrics = world.credit_layer.metrics(1)

    assert metrics["loans_originated"] == 1
    assert metrics["loan_principal_wood"] == 8
    assert metrics["loan_principal_berries"] == 0
    assert metrics["mean_interest_rate"] > 0
    assert metrics["credit_outstanding_wood"] == world.loans[0].amount_due
    assert metrics["loans_active"] == 1
