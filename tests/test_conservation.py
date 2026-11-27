"""Invariant tests for the rules documented in MODEL.md."""

from copy import deepcopy
import math
import random

from main import CONFIG, Agent, ResourceSimulation


def make_config() -> dict:
    return deepcopy(CONFIG)


def money_total(simulation: ResourceSimulation) -> float:
    return sum(agent.resources for agent in simulation.agents)


class TestConservation:
    def test_social_transfers_conserve_money(self):
        config = make_config()
        config["SIMULATION"].update(
            {
                "AGENTS": 4,
                "INITIAL_RESOURCES": 100.0,
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
        before = money_total(simulation)

        transfers = simulation.run_iteration()

        assert transfers
        assert math.isclose(money_total(simulation), before)

    def test_living_cost_is_the_only_money_sink(self):
        config = make_config()
        config["SIMULATION"].update({"AGENTS": 3, "INITIAL_RESOURCES": 100.0})
        config["AGENTS"].update({"GENEROSITY": (0.0, 0.0), "LIVING_COST": 2.5})

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        simulation.run_iteration()

        assert math.isclose(money_total(simulation), 300.0 - 3 * 2.5)

    def test_dead_agents_are_not_charged_living_cost(self):
        config = make_config()
        config["AGENTS"]["LIVING_COST"] = 1.0
        simulation = ResourceSimulation(config)
        simulation.agents = [
            Agent("alive", 0.5, 0.5, 10.0),
            Agent("dead", 0.5, 0.5, 0.0),
        ]

        simulation._apply_living_cost()

        assert simulation.agents[0].resources == 9.0
        assert simulation.agents[1].resources == 0.0

    def test_money_is_conserved_across_many_transfer_iterations(self):
        config = make_config()
        config["SIMULATION"].update(
            {
                "AGENTS": 5,
                "INITIAL_RESOURCES": 50.0,
                "TRANSFER_AMOUNT_RANGE": (1.0, 10.0),
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

        simulation.run(iterations=25)

        assert math.isclose(money_total(simulation), 250.0)

    def test_random_configurations_preserve_invariants(self):
        rng = random.Random(2026)
        for _ in range(8):
            config = make_config()
            config["SIMULATION"].update(
                {
                    "AGENTS": rng.randint(2, 6),
                    "ITERATIONS": 10,
                    "INITIAL_RESOURCES": rng.uniform(20.0, 200.0),
                    "RESOURCE_FLOOR": 0.0,
                    "RANDOM_SEED": rng.randint(0, 10000),
                    "TRANSFER_AMOUNT_RANGE": (1.0, 5.0),
                }
            )
            config["AGENTS"].update(
                {
                    "MIN_RESOURCES_AFTER_GIVING": 0.0,
                    "LIVING_COST": rng.choice([0.0, 0.5, 1.0]),
                }
            )
            config["MARKET"]["ENABLED"] = rng.choice([True, False])
            config["TRUST"]["ENABLED"] = rng.choice([True, False])
            config["SOCIAL"]["RECIPROCITY_ENABLED"] = rng.choice(
                [True, False]
            )
            config["SHOCKS"]["ENABLED"] = rng.choice([True, False])
            config["SHOCKS"]["SCHEDULE"] = [
                {
                    "iteration": 5,
                    "type": "scarcity",
                    "magnitude": 0.5,
                    "duration": 2,
                }
            ]

            simulation = ResourceSimulation(config)
            simulation.initialize_agents()
            previous_money = money_total(simulation)
            previous_goods = 0.0
            for _ in range(10):
                simulation.run_iteration()
                metrics = simulation.metrics_history[-1]
                current_money = money_total(simulation)
                assert current_money <= previous_money + 1e-9
                assert math.isclose(
                    metrics["goods_total"],
                    previous_goods
                    + metrics["goods_produced"]
                    - metrics["goods_consumed"],
                )
                assert all(
                    agent.resources >= 0.0 for agent in simulation.agents
                )
                assert all(agent.goods >= -1e-9 for agent in simulation.agents)
                if config["TRUST"]["ENABLED"]:
                    assert all(
                        0.0 <= value <= 1.0
                        for value in simulation.reputation.values()
                    )
                    assert all(
                        0.0 <= value <= 1.0
                        for agent in simulation.agents
                        for value in agent.beliefs.values()
                    )
                assert metrics["outstanding_debt"] >= 0.0
                previous_money = current_money
                previous_goods = metrics["goods_total"]

    def test_living_cost_never_pushes_money_below_the_floor(self):
        config = make_config()
        config["SIMULATION"].update(
            {"AGENTS": 2, "INITIAL_RESOURCES": 1.0, "RESOURCE_FLOOR": 0.0}
        )
        config["AGENTS"].update(
            {
                "GENEROSITY": (0.0, 0.0),
                "LIVING_COST": 5.0,
            }
        )
        simulation = ResourceSimulation(config)

        simulation.run(iterations=3)

        assert all(agent.resources >= 0.0 for agent in simulation.agents)
        assert all(not agent.is_alive for agent in simulation.agents)
