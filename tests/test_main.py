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
        config["AFFORDABILITY"]["MAX_RESOURCES"] = 50

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
        config["SIMULATION"]["AGENTS"] = 3

        first = ResourceSimulation(config)
        second = ResourceSimulation(config)

        first.initialize_agents()
        second.initialize_agents()

        assert [agent.to_dict() for agent in first.agents] == [
            agent.to_dict() for agent in second.agents
        ]

    def test_iteration_preserves_resources_and_records_memory(self):
        config = make_config()
        config["SIMULATION"].update({"AGENTS": 4, "INITIAL_RESOURCES": 10})
        config["AGENTS"].update(
            {
                "GENEROSITY": (1.0, 1.0),
                "ACCEPTANCE": (1.0, 1.0),
                "MIN_RESOURCES_AFTER_GIVING": 0.0,
                "LIVING_COST": 0.0,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        transfers = simulation.run_iteration()

        assert transfers
        assert sum(agent.resources for agent in simulation.agents) == 40
        assert any(agent.memory for agent in simulation.agents)
        assert len(simulation.metrics_history) == 1
        assert simulation.metrics_history[-1]["total_transfers"] == len(transfers)

    def test_receiving_ceiling_is_not_exceeded(self):
        config = make_config()
        config["SIMULATION"].update({"AGENTS": 3, "INITIAL_RESOURCES": 1})
        config["AGENTS"].update(
            {
                "GENEROSITY": (1.0, 1.0),
                "ACCEPTANCE": (1.0, 1.0),
                "MIN_RESOURCES_AFTER_GIVING": 0.0,
                "MAX_RESOURCES_AFTER_RECEIVING": 2.0,
                "LIVING_COST": 0.0,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.run(iterations=1)

        assert max(agent.resources for agent in simulation.agents) <= 2.0

    def test_living_cost_reduces_resources_after_transfer(self):
        config = make_config()
        config["SIMULATION"].update({"AGENTS": 2, "INITIAL_RESOURCES": 10})
        config["AGENTS"].update(
            {
                "GENEROSITY": (1.0, 1.0),
                "ACCEPTANCE": (1.0, 1.0),
                "MIN_RESOURCES_AFTER_GIVING": 0.0,
                "LIVING_COST": 0.1,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.run(iterations=1)

        assert math.isclose(simulation.metrics_history[-1]["total"], 19.8)

    def test_agents_at_resource_floor_are_dead_and_inactive(self):
        config = make_config()
        config["SIMULATION"].update(
            {
                "AGENTS": 2,
                "INITIAL_RESOURCES": 1,
                "RESOURCE_FLOOR": 0.0,
                "TRANSFER_AMOUNT_RANGE": (1.0, 1.0),
            }
        )
        config["AGENTS"].update(
            {
                "MIN_RESOURCES_AFTER_GIVING": 0.0,
                "MAX_RESOURCES_AFTER_RECEIVING": 10.0,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        simulation._perform_transfers([("agent-001", "agent-002", 1.0)])

        assert not simulation.agents[0].is_alive
        assert simulation.agents[0].resources == 0.0
        assert not simulation.agents[0].can_receive(10.0, 1.0)
        assert simulation.agents[1].is_alive

    def test_transfer_amounts_are_variable_and_conserve_resources(self):
        config = make_config()
        config["SIMULATION"].update(
            {
                "AGENTS": 4,
                "INITIAL_RESOURCES": 10,
                "TRANSFER_AMOUNT_RANGE": (1.0, 5.0),
            }
        )
        config["AGENTS"].update(
            {
                "GENEROSITY": (1.0, 1.0),
                "ACCEPTANCE": (1.0, 1.0),
                "MIN_RESOURCES_AFTER_GIVING": 0.0,
                "LIVING_COST": 0.0,
            }
        )

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        transfers = simulation.run_iteration()

        assert transfers
        assert all(1.0 <= amount <= 5.0 for _, _, amount in transfers)
        assert math.isclose(sum(agent.resources for agent in simulation.agents), 40.0)

    def test_living_cost_does_not_top_up_agents_to_floor(self):
        config = make_config()
        config["SIMULATION"].update(
            {
                "AGENTS": 1,
                "INITIAL_RESOURCES": 1.0,
                "RESOURCE_FLOOR": 1.0,
                "TRANSFER_AMOUNT_RANGE": (1.0, 1.0),
            }
        )
        config["AGENTS"]["LIVING_COST"] = 0.1

        simulation = ResourceSimulation(config)
        simulation.run(iterations=1)

        assert simulation.agents[0].resources == 1.0
        assert not simulation.agents[0].is_alive

    def test_negative_iterations_are_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            ResourceSimulation().run(iterations=-1)

    def test_memory_fraction_must_be_between_zero_and_one(self):
        config = make_config()
        config["MEMORY"]["PREFERENCE_FRACTION"] = 1.1
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        with pytest.raises(ValueError, match="between 0 and 1"):
            simulation._find_receiver(simulation.agents[0])

    def test_memory_recency_and_frequency_weighting(self):
        config = make_config()
        config["MEMORY"]["PREFERENCE_FRACTION"] = 1.0
        simulation = ResourceSimulation(config)

        sender = Agent("sender", 1.0, 1.0, 100)
        c1 = Agent("c1", 1.0, 1.0, 100)
        c2 = Agent("c2", 1.0, 1.0, 100)
        simulation.agents = [sender, c1, c2]

        sender.memory.append("c1")  # idx 0 -> score 1
        sender.memory.append("c2")  # idx 1 -> score 2
        sender.memory.append("c2")  # idx 2 -> score 3 (c2 total = 5, c1 total = 1)

        chosen = [simulation._find_receiver(sender) for _ in range(600)]
        c2_count = sum(1 for agent in chosen if agent.id == "c2")
        c1_count = sum(1 for agent in chosen if agent.id == "c1")

        assert c2_count > c1_count * 3
