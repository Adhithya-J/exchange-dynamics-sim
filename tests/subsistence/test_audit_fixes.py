"""Regression tests for the economic and accounting audit fixes."""

from copy import deepcopy

import pytest

from worlds.subsistence.replay import build_replay
from worlds.subsistence.world import CONFIG, SubsistenceWorld


def make_world(**overrides) -> SubsistenceWorld:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 2, "RANDOM_SEED": 3})
    features = overrides.pop("FEATURES", {})
    config["FEATURES"].update(features)
    for section, values in overrides.items():
        config[section].update(values)
    world = SubsistenceWorld(config)
    world.initialize_agents()
    world._current_iteration = 1
    return world


def test_market_prices_are_fractional_but_berry_payments_are_whole():
    world = make_world()
    seller, buyer = world.agents
    seller.wood = 3
    buyer.berries = 100

    trades, _, _ = world._match_market(
        "wood",
        [{"agent": seller, "qty": 3, "price": 2.4}],
        [{"agent": buyer, "qty": 3, "price": 2.8, "budget": 100}],
    )

    assert len(trades) == 1
    assert trades[0]["price"] == pytest.approx(2.6)
    assert isinstance(trades[0]["berries"], int)
    assert trades[0]["berries"] == 8
    assert trades[0]["qty"] == 3


def test_rounding_can_make_a_fractional_trade_affordable():
    world = make_world()
    seller, buyer = world.agents
    seller.wood = 1
    buyer.berries = 2

    trades, _, _ = world._match_market(
        "wood",
        [{"agent": seller, "qty": 1, "price": 2.4}],
        [{"agent": buyer, "qty": 1, "price": 2.4, "budget": 2}],
    )

    assert len(trades) == 1
    assert trades[0]["berries"] == 2
    assert trades[0]["price"] == pytest.approx(2.4)


def test_market_price_is_not_clamped():
    world = make_world()
    seller, buyer = world.agents
    seller.wood = 1
    buyer.berries = 1000

    trades, _, _ = world._match_market(
        "wood",
        [{"agent": seller, "qty": 1, "price": 80.0}],
        [{"agent": buyer, "qty": 1, "price": 80.0, "budget": 1000}],
    )

    assert trades and trades[0]["price"] == pytest.approx(80.0)


def test_market_does_not_allow_self_trades():
    world = make_world()
    agent = world.agents[0]
    agent.wood = 1
    agent.berries = 100

    trades, _, _ = world._match_market(
        "wood",
        [{"agent": agent, "qty": 1, "price": 5.0}],
        [{"agent": agent, "qty": 1, "price": 6.0, "budget": 100}],
    )

    assert trades == []
    assert agent.wood == 1
    assert agent.berries == 100


def test_formal_berry_debt_reserves_todays_food():
    world = make_world(FEATURES={"CREDIT": True}, CREDIT={"GRACE_DAYS": 2})
    lender, borrower = world.agents
    borrower.berries = 6
    borrower.health = 1
    loan = world.obligations.add(
        creditor=lender.id,
        debtor=borrower.id,
        good="berries",
        principal=6,
        amount_due=6,
        created_iteration=1,
        due_iteration=1,
        kind="loan",
    )

    world._settle_credit()

    assert loan.status == "active"
    assert loan.amount_due == 6
    assert loan.grace_used == 1
    assert borrower.berries == 6


def test_default_does_not_seize_todays_food():
    world = make_world(FEATURES={"CREDIT": True}, CREDIT={"GRACE_DAYS": 0})
    lender, borrower = world.agents
    borrower.berries = 6
    borrower.health = 1
    lender_berries = lender.berries
    loan = world.obligations.add(
        creditor=lender.id,
        debtor=borrower.id,
        good="berries",
        principal=6,
        amount_due=6,
        created_iteration=1,
        due_iteration=1,
        kind="loan",
    )

    world._settle_credit()

    assert loan.status == "defaulted"
    assert borrower.berries == 6
    assert lender.berries == lender_berries
    assert loan.loss == 6


def test_formal_loans_settle_before_soft_gifts():
    world = make_world(FEATURES={"CREDIT": True, "RECIPROCITY": True})
    giver, borrower = world.agents
    borrower.berries = 14
    borrower.health = 1
    gift = world.obligations.add(
        creditor=giver.id,
        debtor=borrower.id,
        good="berries",
        principal=6,
        amount_due=6,
        created_iteration=1,
        due_iteration=None,
        kind="gift",
    )
    loan = world.obligations.add(
        creditor=giver.id,
        debtor=borrower.id,
        good="berries",
        principal=8,
        amount_due=8,
        created_iteration=1,
        due_iteration=1,
        kind="loan",
    )

    world.feature_registry.run("start_of_tick")

    assert loan.status == "repaid"
    assert gift.status == "active"
    assert borrower.berries == 6
    world._eat_all()
    assert borrower.is_alive


def test_credit_demand_puts_food_first():
    world = make_world(FEATURES={"CREDIT": True})
    borrower = world.agents[1]
    borrower.berries = 0
    borrower.wood = 0
    borrower.crates = []

    demand = world.credit_layer._demand(borrower)

    assert demand and demand[0][0] == "berries"


def test_low_health_borrower_can_pledge_collateral_for_food():
    world = make_world(FEATURES={"CREDIT": True})
    lender, borrower = world.agents
    lender.berries = 20
    lender.wood = 0
    lender.crates = []
    lender.trust = 0.6
    borrower.berries = 0
    borrower.wood = 0
    borrower.crates = [1.0]
    borrower.trust = 0.1
    borrower.health = 1
    borrower.berry_skill = 0.3
    borrower.wood_skill = 1.45
    borrower.price_belief = 6.0

    world._originate_credit()

    assert len(world.loans) == 1
    loan = world.loans[0]
    assert loan.good == "berries"
    assert loan.collateral_crates == 1


def test_pledged_crates_cannot_be_sold_or_lent():
    world = make_world(
        FEATURES={"CREDIT": True},
        CREDIT={"LENDER_CRATE_RESERVE": 0},
        AGENTS={"RESERVE_DAYS": 1},
    )
    borrower, lender = world.agents
    borrower.crates = [1.0]
    borrower.berries = 0
    world.obligations.add(
        creditor=lender.id,
        debtor=borrower.id,
        good="berries",
        principal=1,
        amount_due=1,
        created_iteration=1,
        due_iteration=99,
        kind="loan",
        collateral_crates=1,
        collateral_durability=1.0,
    )

    sellers, _ = world._crate_market_plan([borrower], {borrower.id: 5.0})
    assert sellers == []
    assert "crates" not in world.credit_layer._supply(borrower)


def test_crate_sale_price_depends_on_durability():
    world = make_world()
    agent = world.agents[0]
    agent.berries = 0
    agent.health = agent.max_health

    new_price = world._crate_seller_min(agent, 5.0, durability=1.0)
    worn_price = world._crate_seller_min(agent, 5.0, durability=0.05)

    assert worn_price < new_price


def test_soft_gift_trust_gain_uses_configured_gain():
    world = make_world(FEATURES={"RECIPROCITY": True})
    giver, receiver = world.agents
    receiver.trust = 0.5
    receiver.berries = 9
    world.obligations.add(
        creditor=giver.id,
        debtor=receiver.id,
        good="berries",
        principal=4,
        amount_due=4,
        created_iteration=1,
        due_iteration=None,
        kind="gift",
    )

    world.obligations.settle_soft_gifts(1, threshold=2)

    expected = 0.5 + world.trust_settings["gain"] * 0.25
    assert receiver.trust == pytest.approx(expected)
    assert receiver.trust - 0.5 <= world.trust_settings["gain"]


def test_soft_gift_cap_does_not_create_free_gifts():
    world = make_world(
        FEATURES={"RECIPROCITY": True},
        GIFTING={
            "PROBABILITY": 1.0,
            "AMOUNT_RANGE": [2, 2],
            "MIN_RESERVE": 2,
        },
        RECIPROCITY={"MAX_OUTSTANDING": 2},
    )
    giver, receiver = world.agents
    giver.berries = 20
    receiver.berries = 0
    receiver.health = 2
    world._process_gifts()
    assert len(world.gift_log) == 1

    giver.berries = 20
    receiver.berries = 0
    receiver.health = 2
    world._current_iteration = 2
    world._process_gifts()

    assert len(world.gift_log) == 1
    assert len(world.obligations.gifts()) == 1


def test_soft_gift_default_penalises_trust_after_age_limit():
    world = make_world(FEATURES={"RECIPROCITY": True})
    giver, receiver = world.agents
    receiver.trust = 0.5
    obligation = world.obligations.add(
        creditor=giver.id,
        debtor=receiver.id,
        good="berries",
        principal=2,
        amount_due=2,
        created_iteration=1,
        due_iteration=None,
        kind="gift",
    )

    world.obligations.settle_soft_gifts(
        32, threshold=2, default_after_days=30
    )

    assert obligation.status == "defaulted"
    assert obligation.reason == "soft_default"
    assert receiver.trust < 0.5
    metrics = world.reciprocity_layer.metrics(32)
    assert metrics["gift_defaults"] == 1
    assert metrics["gift_default_berries"] == 2


def test_crate_credit_capacity_requires_real_crate_production():
    world = make_world(FEATURES={"CREDIT": True})
    agent = world.agents[0]
    agent.wood_skill = 0.45
    world._current_wood_density = 0.1
    world._current_berries_density = 0.1

    assert world.credit_layer._repayment_capacity(agent, "crates") == 0


def test_reinitializing_a_world_clears_active_shocks():
    world = make_world(
        FEATURES={"SHOCKS": True},
        SHOCKS={"PROBABILITY": 1.0, "DURATION_RANGE": [5, 5]},
    )
    world.run(2)
    assert world.shock_layer.active_until > 0

    world.initialize_agents()

    assert world.shock_layer.active_until == 0
    world.run(1)
    assert world.metrics_history[0]["shock_active"] == 1
    assert world.shock_log


def test_replay_metadata_uses_initial_crates_and_beliefs():
    world = make_world()
    initial_crates = [agent.crate_count for agent in world.agents]
    initial_beliefs = [agent.price_belief for agent in world.agents]
    world.run(20)

    replay = build_replay(world)

    assert [agent["initial_crates"] for agent in replay["agents"]] == initial_crates
    assert [agent["price_belief"] for agent in replay["agents"]] == pytest.approx(
        initial_beliefs
    )
    tick_zero = replay["ticks"][0]
    expected_storage = sum(
        min(state["berries"], state["storage_capacity"])
        for state in tick_zero["agents"]
    )
    assert tick_zero["metrics"]["storage_used"] == expected_storage


def test_zero_agents_is_rejected():
    config = deepcopy(CONFIG)
    config["SIMULATION"]["AGENTS"] = 0
    world = SubsistenceWorld(config)

    with pytest.raises(ValueError, match="at least 1"):
        world.initialize_agents()
