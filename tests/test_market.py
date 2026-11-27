"""Behaviour and conservation tests for the optional goods market."""

from copy import deepcopy
import math

import pytest

from main import CONFIG, ResourceSimulation


def make_config() -> dict:
    return deepcopy(CONFIG)


def market_config(agents: int = 2, initial_resources: float = 100.0) -> dict:
    config = make_config()
    config["SIMULATION"].update(
        {
            "AGENTS": agents,
            "INITIAL_RESOURCES": initial_resources,
            "RESOURCE_FLOOR": 0.0,
            "RANDOM_SEED": 7,
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (0.0, 0.0),
            "ACCEPTANCE": (0.0, 0.0),
            "LIVING_COST": 0.0,
        }
    )
    config["MARKET"].update(
        {
            "ENABLED": True,
            "PRICE": 1.0,
            "PRODUCTION_RANGE": (0.0, 0.0),
            "CONSUMPTION_RANGE": (0.0, 0.0),
        }
    )
    return config


def money_total(simulation: ResourceSimulation) -> float:
    return sum(agent.resources for agent in simulation.agents)


class TestMarketDisabled:
    def test_market_is_disabled_by_default(self):
        simulation = ResourceSimulation(make_config())
        simulation.initialize_agents()

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_total"] == 0.0
        assert metrics["goods_produced"] == 0.0
        assert metrics["goods_consumed"] == 0.0
        assert metrics["unmet_demand"] == 0.0
        assert all(agent.goods == 0.0 for agent in simulation.agents)

    def test_missing_market_section_keeps_market_disabled(self):
        config = make_config()
        del config["MARKET"]

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        assert not simulation._market_settings()["enabled"]


class TestMarketCycle:
    def test_production_adds_goods_without_changing_money(self):
        config = market_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (2.0, 2.0)
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        before = money_total(simulation)

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_produced"] == 4.0
        assert metrics["goods_total"] == 4.0
        assert metrics["goods_consumed"] == 0.0
        assert math.isclose(money_total(simulation), before)

    def test_trade_moves_goods_and_conserves_money(self):
        config = market_config()
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        producer, buyer = simulation.agents
        producer.production_rate = 3.0
        producer.consumption_need = 1.0
        buyer.production_rate = 0.0
        buyer.consumption_need = 2.0

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_produced"] == 3.0
        assert metrics["goods_traded"] == 2.0
        assert metrics["trade_volume"] == 2.0
        assert metrics["goods_consumed"] == 3.0
        assert metrics["unmet_demand"] == 0.0
        assert metrics["goods_total"] == 0.0
        assert math.isclose(money_total(simulation), 200.0)
        assert producer.resources == 102.0
        assert buyer.resources == 98.0

    def test_unmet_demand_is_recorded_when_nothing_is_produced(self):
        config = market_config(agents=3)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        simulation = ResourceSimulation(config)

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_produced"] == 0.0
        assert metrics["goods_traded"] == 0.0
        assert metrics["unmet_demand"] == 3.0
        assert metrics["living_cost_due"] == 6.0
        assert metrics["living_cost_paid"] == 6.0
        assert math.isclose(money_total(simulation), 294.0)

    def test_unmet_demand_charges_the_subsistence_penalty(self):
        config = market_config(agents=1)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["MARKET"]["PRODUCTION_RANGE"] = (0.0, 0.0)
        config["MARKET"]["UNMET_DEMAND_PENALTY"] = 2.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        agent = simulation.agents[0]

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["unmet_demand"] == 1.0
        assert metrics["living_cost_due"] == 2.0
        assert metrics["living_cost_paid"] == 2.0
        assert agent.resources == 98.0

    def test_buyers_never_spend_below_the_resource_floor(self):
        config = market_config()
        config["SIMULATION"]["RESOURCE_FLOOR"] = 1.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        buyer, seller = simulation.agents
        buyer.resources = 2.0
        buyer.consumption_need = 10.0
        seller.production_rate = 10.0
        seller.consumption_need = 0.0

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert buyer.resources == 1.0
        assert metrics["goods_traded"] == 1.0
        assert math.isclose(money_total(simulation), 102.0)

    def test_dead_agents_do_not_produce_or_consume(self):
        config = market_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (1.0, 1.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        simulation.agents[0].resources = 0.0

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_produced"] == 1.0
        assert metrics["goods_consumed"] == 1.0
        assert metrics["goods_total"] == 0.0

    def test_goods_and_money_conserve_across_iterations(self):
        config = market_config(agents=6)
        config["MARKET"]["PRODUCTION_RANGE"] = (0.0, 2.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        simulation.run(iterations=10)

        previous_money = 600.0
        previous_goods = 0.0
        for metrics in simulation.metrics_history:
            assert math.isclose(
                metrics["total"], previous_money - metrics["living_cost_paid"]
            )
            expected = previous_goods + metrics["goods_produced"] - metrics["goods_consumed"]
            assert math.isclose(metrics["goods_total"], expected)
            previous_goods = metrics["goods_total"]
            previous_money = metrics["total"]

    def test_buyers_keep_back_the_fixed_living_cost(self):
        config = market_config()
        config["AGENTS"]["LIVING_COST"] = 3.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        buyer, seller = simulation.agents
        buyer.resources = 10.0
        buyer.consumption_need = 10.0
        seller.production_rate = 10.0
        seller.consumption_need = 0.0

        simulation._run_market_cycle()

        assert buyer.resources == 3.0
        assert seller.resources == 107.0

    def test_seller_income_respects_the_receiving_ceiling(self):
        config = market_config()
        config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"] = 10.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        buyer, seller = simulation.agents
        buyer.resources = 10.0
        buyer.consumption_need = 10.0
        seller.resources = 9.0
        seller.production_rate = 10.0
        seller.consumption_need = 0.0

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_traded"] == 1.0
        assert seller.resources == 10.0

    def test_buyers_refuse_goods_priced_above_their_value(self):
        config = market_config()
        config["MARKET"]["PRICE"] = 3.0
        config["MARKET"]["UNMET_DEMAND_PENALTY"] = 2.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        buyer, seller = simulation.agents
        buyer.consumption_need = 1.0
        seller.production_rate = 5.0

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["goods_traded"] == 0.0
        assert metrics["unmet_demand"] == 1.0

    def test_price_rises_when_demand_exceeds_supply(self):
        config = market_config(agents=2)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["MARKET"]["PRODUCTION_RANGE"] = (0.0, 0.0)
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        simulation.run_iteration()

        assert simulation.metrics_history[-1]["price"] == 1.0
        assert simulation.price > 1.0


class TestMarketValidation:
    def test_non_positive_price_is_rejected(self):
        config = market_config()
        config["MARKET"]["PRICE"] = 0.0

        with pytest.raises(ValueError, match="PRICE must be positive"):
            ResourceSimulation(config).initialize_agents()

    def test_reversed_production_range_is_rejected(self):
        config = market_config()
        config["MARKET"]["PRODUCTION_RANGE"] = [2.0, 1.0]

        with pytest.raises(ValueError, match="ascending order"):
            ResourceSimulation(config).initialize_agents()

    def test_malformed_consumption_range_is_rejected(self):
        config = market_config()
        config["MARKET"]["CONSUMPTION_RANGE"] = [1.0]

        with pytest.raises(ValueError, match="sequence of two values"):
            ResourceSimulation(config).initialize_agents()

    def test_negative_production_is_rejected(self):
        config = market_config()
        config["MARKET"]["PRODUCTION_RANGE"] = [-1.0, 1.0]

        with pytest.raises(ValueError, match="non-negative"):
            ResourceSimulation(config).initialize_agents()
