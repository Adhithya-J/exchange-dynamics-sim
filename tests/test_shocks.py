"""Tests for scheduled environmental shocks and adaptation."""

from copy import deepcopy
import math

import pytest

from main import CONFIG, ResourceSimulation


def make_config() -> dict:
    return deepcopy(CONFIG)


def shock_config() -> dict:
    config = make_config()
    config["SIMULATION"].update(
        {
            "AGENTS": 2,
            "INITIAL_RESOURCES": 100.0,
            "TRANSFER_AMOUNT_RANGE": (1.0, 1.0),
            "RANDOM_SEED": 9,
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (0.0, 0.0),
            "ACCEPTANCE": (0.0, 0.0),
            "LIVING_COST": 0.0,
            "RISK_TOLERANCE": (0.5, 0.5),
        }
    )
    config["MARKET"].update(
        {
            "ENABLED": True,
            "PRICE": 1.0,
            "PRODUCTION_RANGE": (2.0, 2.0),
            "CONSUMPTION_RANGE": (0.0, 0.0),
        }
    )
    config["SHOCKS"].update(
        {
            "ENABLED": True,
            "SCHEDULE": [],
            "ADAPTATION_RATE": 0.05,
        }
    )
    # These tests isolate production and unmet demand; no free extraction.
    config["COMMONS"].update(
        {
            "INITIAL_GOODS": 0.0,
            "CAPACITY": 0.0,
            "REGENERATION": 0.0,
            "EXTRACTION_LIMIT": 0.0,
        }
    )
    return config


class TestShocksDisabled:
    def test_shocks_are_disabled_by_default(self):
        simulation = ResourceSimulation(make_config())
        simulation.initialize_agents()

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert metrics["shock_active"] == 0
        assert metrics["shock_count"] == 0
        assert metrics["production_modifier"] == 1.0
        assert metrics["reputation_noise"] == 0.0
        assert simulation._active_shocks == []


class TestScarcityShock:
    def test_scarcity_reduces_production_for_its_duration(self):
        config = shock_config()
        config["SHOCKS"]["SCHEDULE"] = [
            {
                "iteration": 2,
                "type": "scarcity",
                "magnitude": 0.5,
                "duration": 2,
            }
        ]
        simulation = ResourceSimulation(config)

        simulation.run(iterations=4)

        produced = [m["goods_produced"] for m in simulation.metrics_history]
        assert produced == [4.0, 2.0, 2.0, 4.0]
        assert [m["shock_active"] for m in simulation.metrics_history] == [
            0,
            1,
            1,
            0,
        ]
        assert [m["production_modifier"] for m in simulation.metrics_history] == [
            1.0,
            0.5,
            0.5,
            1.0,
        ]
        assert [m["iterations_since_shock"] for m in simulation.metrics_history] == [
            0,
            0,
            1,
            2,
        ]
        assert simulation.metrics_history[-1]["shock_count"] == 1
        assert len(simulation.shock_log) == 1

    def test_goods_conserve_during_scarcity(self):
        config = shock_config()
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["SHOCKS"]["SCHEDULE"] = [
            {
                "iteration": 2,
                "type": "scarcity",
                "magnitude": 0.5,
                "duration": 3,
            }
        ]
        simulation = ResourceSimulation(config)

        simulation.run(iterations=5)

        previous_goods = 0.0
        for metrics in simulation.metrics_history:
            expected = (
                previous_goods
                + metrics["goods_produced"]
                - metrics["goods_consumed"]
            )
            assert math.isclose(metrics["goods_total"], expected)
            previous_goods = metrics["goods_total"]


class TestInformationShock:
    def test_information_shock_adds_reputation_noise(self):
        config = shock_config()
        config["SHOCKS"]["SCHEDULE"] = [
            {
                "iteration": 1,
                "type": "information",
                "magnitude": 0.25,
                "duration": 2,
            }
        ]
        simulation = ResourceSimulation(config)

        simulation.run(iterations=3)

        assert [m["reputation_noise"] for m in simulation.metrics_history] == [
            0.25,
            0.25,
            0.0,
        ]


class TestAdaptation:
    def test_unmet_demand_raises_risk_tolerance(self):
        config = shock_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (0.0, 0.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["SHOCKS"]["ADAPTATION_RATE"] = 0.1
        simulation = ResourceSimulation(config)

        simulation.run(iterations=2)

        assert all(
            agent.risk_tolerance == pytest.approx(0.7)
            for agent in simulation.agents
        )
        assert simulation.metrics_history[-1]["mean_risk_tolerance"] == pytest.approx(
            0.7
        )

    def test_fulfilled_demand_relaxes_risk_tolerance(self):
        config = shock_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (1.0, 1.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["SHOCKS"]["ADAPTATION_RATE"] = 0.1
        simulation = ResourceSimulation(config)

        simulation.run(iterations=2)

        assert all(
            agent.risk_tolerance == pytest.approx(0.4)
            for agent in simulation.agents
        )

    def test_risk_tolerance_is_capped_at_one(self):
        config = shock_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (0.0, 0.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["SHOCKS"]["ADAPTATION_RATE"] = 0.5
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        for agent in simulation.agents:
            agent.risk_tolerance = 0.9

        simulation.run(iterations=2)

        assert all(agent.risk_tolerance == 1.0 for agent in simulation.agents)

    def test_risk_tolerance_does_not_become_negative(self):
        config = shock_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (1.0, 1.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["SHOCKS"]["ADAPTATION_RATE"] = 0.5
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        for agent in simulation.agents:
            agent.risk_tolerance = 0.1

        simulation.run(iterations=1)

        assert all(agent.risk_tolerance == 0.0 for agent in simulation.agents)


class TestResilience:
    def test_recovery_time_after_scarcity(self):
        config = shock_config()
        config["MARKET"]["PRODUCTION_RANGE"] = (1.0, 1.0)
        config["MARKET"]["CONSUMPTION_RANGE"] = (1.0, 1.0)
        config["SHOCKS"]["SCHEDULE"] = [
            {
                "iteration": 2,
                "type": "scarcity",
                "magnitude": 0.75,
                "duration": 2,
            }
        ]
        simulation = ResourceSimulation(config)

        simulation.run(iterations=6)

        assert simulation.recovery_time() == 2
        assert simulation.metrics_history[-1]["unmet_demand"] == 0.0

    def test_recovery_time_without_shocks_is_none(self):
        simulation = ResourceSimulation(shock_config())

        simulation.run(iterations=2)

        assert simulation.recovery_time() is None


class TestShockValidation:
    @pytest.mark.parametrize(
        ("schedule", "message"),
        [
            ("not-a-list", "must be a sequence"),
            ([1], "must be a mapping"),
            ([{"type": "scarcity"}], "requires iteration and type"),
            (
                [{"iteration": 1, "type": "unknown"}],
                "scarcity or information",
            ),
            (
                [{"iteration": 0, "type": "scarcity"}],
                "at least 1",
            ),
            (
                [{"iteration": 1, "type": "scarcity", "magnitude": -1}],
                "must not be negative",
            ),
            (
                [{"iteration": 1, "type": "scarcity", "duration": 0}],
                "at least 1",
            ),
            (
                [{"iteration": 1, "type": "scarcity", "duration": "long"}],
                "must be numeric",
            ),
        ],
    )
    def test_invalid_schedules_are_rejected(self, schedule, message):
        config = shock_config()
        config["SHOCKS"]["SCHEDULE"] = schedule
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        with pytest.raises(ValueError, match=message):
            simulation.run_iteration()

    def test_negative_adaptation_rate_is_rejected(self):
        config = shock_config()
        config["SHOCKS"]["ADAPTATION_RATE"] = -0.1
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        with pytest.raises(ValueError, match="must not be negative"):
            simulation.run_iteration()

    def test_non_numeric_adaptation_rate_is_rejected(self):
        config = shock_config()
        config["SHOCKS"]["ADAPTATION_RATE"] = "fast"
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        with pytest.raises(ValueError, match="must be a number"):
            simulation.run_iteration()
