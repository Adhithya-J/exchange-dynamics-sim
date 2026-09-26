from copy import deepcopy
import math

import pandas as pd
import pytest

from main import (
    CONFIG,
    Agent,
    AffordabilityCalculator,
    MetricsCalculator,
    ResourceSimulation,
)


def make_config() -> dict:
    return deepcopy(CONFIG)


class TestMetricsCalculator:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([5, 5, 5, 5, 5], 0.0),
            ([1, 2, 3, 4, 5], 0.2666666667),
            ([1, 1, 1, 1, 10], 0.5142857143),
            ([0, 0, 1], 0.6666666667),
            ([42], 0.0),
        ],
    )
    def test_gini_calculation(self, values, expected):
        assert math.isclose(
            MetricsCalculator.calculate_gini(values), expected, rel_tol=1e-6
        )

    def test_gini_rejects_empty_and_negative_values(self):
        with pytest.raises(ValueError, match="must not be empty"):
            MetricsCalculator.calculate_gini([])
        with pytest.raises(ValueError, match="negative"):
            MetricsCalculator.calculate_gini([1, -1])

    def test_statistics(self):
        frame = pd.DataFrame({"resources": [1.0, 2.0, 3.0]})

        result = MetricsCalculator.calculate_statistics(frame)

        assert result["mean"] == 2.0
        assert result["median"] == 2.0
        assert result["min"] == 1.0
        assert result["max"] == 3.0
        assert result["total"] == 6.0


class TestAffordabilityCalculator:
    def setup_method(self):
        self.calculator = AffordabilityCalculator(make_config())

    @pytest.mark.parametrize(
        ("resources", "expected"),
        [(20, 0.0), (50, 0.0), (175, 0.5), (300, 1.0), (350, 1.0)],
    )
    def test_affordability_boundaries(self, resources, expected):
        assert self.calculator.affordability(resources) == expected

    def test_effective_scores_are_capped(self):
        poor = Agent("poor", 1.0, 1.0, 50)
        wealthy = Agent("wealthy", 1.0, 1.0, 500)

        assert self.calculator.effective_generosity(poor) == 0.0
        assert self.calculator.effective_generosity(wealthy) == 1.0
        assert self.calculator.effective_acceptance(poor) == 1.0
        assert self.calculator.effective_acceptance(wealthy) == 1.0

    def test_invalid_resource_range_is_rejected(self):
        config = make_config()
        config["AFFORDABILITY"]["RESOURCE_MAX"] = 50

        with pytest.raises(ValueError, match="greater"):
            AffordabilityCalculator(config).affordability(100)


class TestAgent:
    def test_initialization_and_serialization(self):
        agent = Agent("test-id", 0.8, 0.7, 1000, memory_size=2)
        agent.memory.append("other-id")

        snapshot = agent.to_dict()

        assert agent.id == "test-id"
        assert agent.generosity_score == 0.8
        assert agent.acceptance_score == 0.7
        assert agent.resources == 1000
        assert snapshot["memory"] == ["other-id"]
        assert agent.is_alive

    def test_giving_and_receiving_constraints(self):
        agent = Agent("test-id", 0.8, 0.7, 10)

        assert agent.can_give(giving_floor=1, transfer_size=1)
        assert not agent.can_give(giving_floor=10, transfer_size=1)
        assert agent.can_receive(receiving_ceil=11, transfer_size=1)
        assert not agent.can_receive(receiving_ceil=10, transfer_size=1)


class TestResourceSimulation:
    def test_initialization_is_deterministic(self):
        config = make_config()
        config["ENV_INIT"]["N_AGENTS"] = 3

        first = ResourceSimulation(config)
        second = ResourceSimulation(config)

        first.initialize_agents()
        second.initialize_agents()

        assert [agent.to_dict() for agent in first.agents] == [
            agent.to_dict() for agent in second.agents
        ]

    def test_iteration_preserves_resources_and_records_memory(self):
        config = make_config()
        config["ENV_INIT"].update({"N_AGENTS": 4, "MAX_RESOURCES": 10})
        config["AGENTS_INIT"].update(
            {
                "GENEROSITY_RANGE": (1.0, 1.0),
                "ACCEPTANCE_RANGE": (1.0, 1.0),
                "GIVING_FLOOR": 0.0,
                "COST_OF_LIVING": 0.0,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        transfers = simulation.run_iteration()

        assert transfers
        assert sum(agent.resources for agent in simulation.agents) == 40
        assert any(agent.memory for agent in simulation.agents)
        assert len(simulation.metrics_history) == 1

    def test_receiving_ceiling_is_not_exceeded(self):
        config = make_config()
        config["ENV_INIT"].update({"N_AGENTS": 3, "MAX_RESOURCES": 1})
        config["AGENTS_INIT"].update(
            {
                "GENEROSITY_RANGE": (1.0, 1.0),
                "ACCEPTANCE_RANGE": (1.0, 1.0),
                "GIVING_FLOOR": 0.0,
                "RECEIVING_CEIL": 2.0,
                "COST_OF_LIVING": 0.0,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.run(iterations=1)

        assert max(agent.resources for agent in simulation.agents) <= 2.0

    def test_living_cost_reduces_resources_after_transfer(self):
        config = make_config()
        config["ENV_INIT"].update({"N_AGENTS": 2, "MAX_RESOURCES": 10})
        config["AGENTS_INIT"].update(
            {
                "GENEROSITY_RANGE": (1.0, 1.0),
                "ACCEPTANCE_RANGE": (1.0, 1.0),
                "GIVING_FLOOR": 0.0,
                "COST_OF_LIVING": 0.1,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.run(iterations=1)

        assert math.isclose(simulation.metrics_history[-1]["total"], 18.0)

    def test_negative_iterations_are_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            ResourceSimulation().run(iterations=-1)
