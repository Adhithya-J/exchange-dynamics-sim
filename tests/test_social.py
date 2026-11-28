"""Tests for the altruism and reciprocity layer."""

from copy import deepcopy
import math

import pytest

from main import CONFIG, ResourceSimulation


def make_config() -> dict:
    config = deepcopy(CONFIG)
    config["GIFTING"]["ENABLED"] = True
    return config


def social_config() -> dict:
    config = make_config()
    config["SIMULATION"].update(
        {
            "AGENTS": 2,
            "INITIAL_RESOURCES": 100.0,
            "TRANSFER_AMOUNT_RANGE": (10.0, 10.0),
            "RANDOM_SEED": 5,
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (0.0, 0.0),
            "ACCEPTANCE": (0.0, 0.0),
            "LIVING_COST": 0.0,
        }
    )
    config["SOCIAL"].update(
        {
            "RECIPROCITY_ENABLED": True,
            "REPAY_PROBABILITY": 1.0,
            "REPAY_DELAY": 2,
            "REPAY_FRACTION": 1.0,
        }
    )
    return config


def money_total(simulation: ResourceSimulation) -> float:
    return sum(agent.resources for agent in simulation.agents)


class TestReciprocityDisabled:
    def test_reciprocity_is_disabled_by_default(self):
        simulation = ResourceSimulation(make_config())
        simulation.initialize_agents()

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["repayments"] == 0
        assert metrics["repayment_volume"] == 0.0
        assert metrics["outstanding_debt"] == 0.0
        assert simulation._repayment_queue == []


class TestReciprocity:
    def test_gift_schedules_a_repayment(self):
        simulation = ResourceSimulation(social_config())
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1

        simulation._execute_transfer(giver, receiver, 10.0)

        assert simulation.total_gifts == 1
        assert len(simulation._repayment_queue) == 1
        entry = simulation._repayment_queue[0]
        assert entry["due"] == 3
        assert entry["debtor"] == receiver.id
        assert entry["creditor"] == giver.id
        assert entry["amount"] == 10.0

    def test_due_gift_is_repaid_and_conserves_money(self):
        simulation = ResourceSimulation(social_config())
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)
        simulation._current_iteration = 3

        result = simulation._process_repayments(3)

        assert result["repayments"] == 1
        assert result["repayment_volume"] == 10.0
        assert giver.resources == 100.0
        assert receiver.resources == 100.0
        assert math.isclose(money_total(simulation), 200.0)
        assert simulation.repayment_log[0]["delay"] == 2
        assert simulation.repayment_log[0]["amount"] == 10.0

    def test_repayment_probability_zero_leaves_the_gift_unpaid(self):
        config = social_config()
        config["SOCIAL"]["REPAY_PROBABILITY"] = 0.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)

        result = simulation._process_repayments(3)

        assert result["repayments"] == 0
        assert result["defaults"] == 1
        assert result["outstanding_debt"] == 10.5
        assert result["defaulted_debt"] == 10.5
        assert receiver.resources == 110.0
        assert giver.resources == 90.0
        assert len(simulation._repayment_queue) == 1

    def test_defaulted_debt_is_retried_and_can_be_settled(self):
        config = social_config()
        config["SOCIAL"]["REPAY_PROBABILITY"] = 0.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)

        simulation._process_repayments(3)
        receiver.repayment_probability = 1.0
        result = simulation._process_repayments(4)

        assert result["repayments"] == 1
        assert result["repayment_volume"] == 10.5
        assert result["outstanding_debt"] == 0.0
        assert simulation.total_defaults == 1

    def test_default_excludes_the_debtor_from_future_gifts(self):
        config = social_config()
        config["SOCIAL"]["REPAY_PROBABILITY"] = 0.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)
        simulation._process_repayments(3)

        assert simulation._find_receiver(giver) is None

        receiver.repayment_probability = 1.0
        simulation._process_repayments(4)

        assert simulation._find_receiver(giver).id == receiver.id

    def test_partial_repayment_uses_the_configured_fraction(self):
        config = social_config()
        config["SOCIAL"]["REPAY_FRACTION"] = 0.5
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)

        result = simulation._process_repayments(3)

        assert result["repayment_volume"] == 5.0
        assert result["outstanding_debt"] == 5.25
        assert giver.resources == 95.0
        assert receiver.resources == 105.0
        assert math.isclose(money_total(simulation), 200.0)

    def test_repayment_never_pushes_the_debtor_below_the_floor(self):
        simulation = ResourceSimulation(social_config())
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)
        receiver.resources = 1.0

        result = simulation._process_repayments(3)

        assert result["repayment_volume"] == 1.0
        assert result["outstanding_debt"] == 0.0
        assert result["written_off_debt"] == 9.0
        assert receiver.resources == 0.0
        assert giver.resources == 91.0
        assert math.isclose(money_total(simulation), 91.0)

    def test_reciprocity_rate_measures_value_not_events(self):
        config = social_config()
        config["SOCIAL"]["REPAY_FRACTION"] = 0.5
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        giver, receiver = simulation.agents
        simulation._current_iteration = 1
        simulation._execute_transfer(giver, receiver, 10.0)

        result = simulation._process_repayments(3)

        assert result["repayments"] == 1
        assert result["repayment_volume"] == 5.0
        assert result["reciprocity_rate"] == 0.5
        assert result["repayment_event_rate"] == 1.0

    def test_reciprocity_metrics_track_a_run(self):
        config = social_config()
        config["SIMULATION"]["TRANSFER_AMOUNT_RANGE"] = (1.0, 1.0)
        config["AGENTS"]["GENEROSITY"] = (1.0, 1.0)
        config["AGENTS"]["ACCEPTANCE"] = (1.0, 1.0)
        config["SOCIAL"]["REPAY_DELAY"] = 1
        simulation = ResourceSimulation(config)

        simulation.run(iterations=10)

        metrics = simulation.metrics_history[-1]
        assert metrics["total_repayments"] > 0
        assert 0.0 < metrics["reciprocity_rate"] <= 1.0
        assert metrics["outstanding_debt"] >= 0.0
        assert metrics["mean_repayment_delay"] >= 0.0
        assert math.isclose(money_total(simulation), 200.0)


class TestSocialSeparation:
    def test_market_trades_do_not_create_repayment_debts(self):
        config = social_config()
        config["MARKET"].update(
            {
                "ENABLED": True,
                "PRICE": 1.0,
                "PRODUCTION_RANGE": (1.0, 1.0),
                "CONSUMPTION_RANGE": (1.0, 1.0),
            }
        )
        simulation = ResourceSimulation(config)

        simulation.run(iterations=2)

        assert simulation.total_gifts == 0
        assert simulation._repayment_queue == []
        assert simulation.repayment_log == []


class TestSocialValidation:
    @pytest.mark.parametrize(
        ("setting", "value", "message"),
        [
            ("REPAY_PROBABILITY", 1.5, "REPAY_PROBABILITY"),
            ("REPAY_PROBABILITY", -0.1, "REPAY_PROBABILITY"),
            ("REPAY_DELAY", -1, "REPAY_DELAY"),
            ("REPAY_FRACTION", -1.0, "REPAY_FRACTION"),
        ],
    )
    def test_invalid_settings_are_rejected(self, setting, value, message):
        config = social_config()
        config["SOCIAL"][setting] = value
        simulation = ResourceSimulation(config)

        with pytest.raises(ValueError, match=message):
            simulation.initialize_agents()

    def test_non_numeric_setting_is_rejected(self):
        config = social_config()
        config["SOCIAL"]["REPAY_DELAY"] = "soon"
        simulation = ResourceSimulation(config)

        with pytest.raises(ValueError, match="must be numeric"):
            simulation.initialize_agents()
