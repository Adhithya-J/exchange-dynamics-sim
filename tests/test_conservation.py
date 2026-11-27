"""Invariant tests for the rules documented in MODEL.md."""

from copy import deepcopy
import math

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
