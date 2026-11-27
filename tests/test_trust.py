"""Tests for private trust beliefs and noisy public reputation."""

from copy import deepcopy
import math

import pytest

from main import CONFIG, ResourceSimulation


def make_config() -> dict:
    return deepcopy(CONFIG)


def trust_config(agents: int = 2, initial_resources: float = 100.0) -> dict:
    config = make_config()
    config["SIMULATION"].update(
        {
            "AGENTS": agents,
            "INITIAL_RESOURCES": initial_resources,
            "TRANSFER_AMOUNT_RANGE": (1.0, 1.0),
            "RANDOM_SEED": 3,
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (1.0, 1.0),
            "ACCEPTANCE": (1.0, 1.0),
            "MIN_RESOURCES_AFTER_GIVING": 0.0,
            "MAX_RESOURCES_AFTER_RECEIVING": 10000.0,
            "LIVING_COST": 0.0,
        }
    )
    config["TRUST"].update(
        {
            "ENABLED": True,
            "INITIAL_BELIEF": 0.5,
            "LEARNING_RATE": 0.2,
            "DECAY": 0.0,
            "NOISE": 0.0,
            "DELAY": 0,
            "BIAS": 0.0,
            "PUBLIC_WEIGHT": 0.5,
        }
    )
    config["SOCIAL"].update(
        {
            "RECIPROCITY_ENABLED": True,
            "REPAY_PROBABILITY": 1.0,
            "REPAY_DELAY": 0,
            "REPAY_FRACTION": 1.0,
        }
    )
    return config


class TestTrustDisabled:
    def test_trust_is_disabled_by_default(self):
        simulation = ResourceSimulation(make_config())
        simulation.initialize_agents()

        assert simulation.reputation == {}
        assert all(agent.beliefs == {} for agent in simulation.agents)

        simulation.run(iterations=2)

        metrics = simulation.metrics_history[-1]
        assert metrics["mean_trust"] == 0.0
        assert metrics["mean_reputation"] == 0.0
        assert metrics["trust_observations"] == 0

    def test_missing_trust_section_keeps_trust_disabled(self):
        config = make_config()
        del config["TRUST"]

        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        assert not simulation._trust_settings()["enabled"]
        assert simulation.reputation == {}


class TestTrustUpdates:
    def test_initial_reputation_uses_the_prior(self):
        simulation = ResourceSimulation(trust_config())
        simulation.initialize_agents()

        assert simulation.reputation == {
            "agent-001": 0.5,
            "agent-002": 0.5,
        }
        assert all(agent.beliefs == {} for agent in simulation.agents)

    def test_repayment_raises_private_belief_and_public_reputation(self):
        simulation = ResourceSimulation(trust_config())
        simulation.initialize_agents()

        simulation.run(iterations=2)

        metrics = simulation.metrics_history[-1]
        assert metrics["trust_observations"] >= 1
        assert all(value > 0.5 for value in simulation.reputation.values())
        observed_beliefs = [
            value
            for agent in simulation.agents
            for value in agent.beliefs.values()
        ]
        assert observed_beliefs
        assert all(0.5 < value <= 1.0 for value in observed_beliefs)

    def test_public_weight_one_makes_private_belief_match_reputation(self):
        config = trust_config()
        config["TRUST"]["PUBLIC_WEIGHT"] = 1.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        simulation.run(iterations=2)

        for agent in simulation.agents:
            for other_id, belief in agent.beliefs.items():
                assert math.isclose(belief, simulation.reputation[other_id])

    def test_negative_bias_lowers_reputation(self):
        config = trust_config()
        config["TRUST"]["BIAS"] = -1.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        simulation.run(iterations=2)

        assert all(value < 0.5 for value in simulation.reputation.values())

    def test_delay_holds_observations_until_due(self):
        config = trust_config()
        config["TRUST"]["DELAY"] = 2
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()

        simulation.run(iterations=1)
        assert all(value == 0.5 for value in simulation.reputation.values())

        simulation.run(iterations=3)
        assert all(value > 0.5 for value in simulation.reputation.values())

    def test_decay_returns_values_to_the_prior(self):
        config = trust_config()
        config["TRUST"]["DECAY"] = 1.0
        simulation = ResourceSimulation(config)
        simulation.initialize_agents()
        simulation.reputation["agent-001"] = 0.9
        simulation.agents[0].beliefs["agent-002"] = 0.9

        simulation._decay_trust()

        assert simulation.reputation["agent-001"] == 0.5
        assert simulation.agents[0].beliefs["agent-002"] == 0.5

    def test_reputation_error_shrinks_as_agents_cooperate(self):
        simulation = ResourceSimulation(trust_config())
        simulation.initialize_agents()

        simulation.run(iterations=30)

        metrics = simulation.metrics_history[-1]
        assert metrics["reputation_error"] < 0.05
        assert metrics["reputation_brier"] < 0.05

    def test_belief_falls_back_to_public_reputation(self):
        simulation = ResourceSimulation(trust_config())
        simulation.initialize_agents()
        simulation.reputation["agent-002"] = 0.8

        belief = simulation.belief_for(simulation.agents[0], "agent-002")

        assert belief == 0.8


class TestTrustValidation:
    @pytest.mark.parametrize(
        ("setting", "value", "message"),
        [
            ("INITIAL_BELIEF", 1.5, "INITIAL_BELIEF"),
            ("LEARNING_RATE", -0.1, "LEARNING_RATE"),
            ("DECAY", 1.1, "DECAY"),
            ("NOISE", -1.0, "NOISE"),
            ("DELAY", -1, "DELAY"),
            ("PUBLIC_WEIGHT", 2.0, "PUBLIC_WEIGHT"),
        ],
    )
    def test_invalid_settings_are_rejected(self, setting, value, message):
        config = trust_config()
        config["TRUST"][setting] = value

        with pytest.raises(ValueError, match=message):
            ResourceSimulation(config).initialize_agents()

    def test_non_numeric_setting_is_rejected(self):
        config = trust_config()
        config["TRUST"]["LEARNING_RATE"] = "fast"

        with pytest.raises(ValueError, match="must be numeric"):
            ResourceSimulation(config).initialize_agents()
