"""Tests for trust-based giving, receiving, and receiver selection."""

from copy import deepcopy
import math

import pytest

from main import CONFIG, Agent, ResourceSimulation


def make_config() -> dict:
    return deepcopy(CONFIG)


def decision_config(agents: int = 3) -> dict:
    config = make_config()
    config["SIMULATION"].update(
        {
            "AGENTS": agents,
            "INITIAL_RESOURCES": 1000.0,
            "TRANSFER_AMOUNT_RANGE": (1.0, 1.0),
            "RANDOM_SEED": 11,
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (1.0, 1.0),
            "ACCEPTANCE": (1.0, 1.0),
            "MIN_RESOURCES_AFTER_GIVING": 0.0,
            "MAX_RESOURCES_AFTER_RECEIVING": 10000.0,
            "LIVING_COST": 0.0,
            "RISK_TOLERANCE": (0.5, 0.5),
        }
    )
    config["MEMORY"]["PREFERENCE_FRACTION"] = 0.0
    config["TRUST"].update(
        {
            "ENABLED": True,
            "INITIAL_BELIEF": 0.5,
            "LEARNING_RATE": 0.0,
            "DECAY": 0.0,
            "NOISE": 0.0,
            "DELAY": 0,
            "BIAS": 0.0,
            "PUBLIC_WEIGHT": 0.0,
            "DECISION_WEIGHT": 1.0,
            "RETURN_MULTIPLIER": 2.0,
        }
    )
    return config


class TestBeliefDecisions:
    def test_higher_belief_raises_give_probability(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        sender, trusted, distrusted = simulation.agents
        sender.beliefs[trusted.id] = 0.9
        sender.beliefs[distrusted.id] = 0.1

        trusted_probability = simulation._give_probability(sender, trusted)
        distrusted_probability = simulation._give_probability(sender, distrusted)

        assert trusted_probability > distrusted_probability
        assert trusted_probability == pytest.approx(
            1.0 / (1.0 + math.exp(-0.8 / 0.5))
        )

    def test_higher_belief_raises_accept_probability(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        receiver, trusted, distrusted = simulation.agents
        receiver.beliefs[trusted.id] = 0.9
        receiver.beliefs[distrusted.id] = 0.1

        assert simulation._receive_probability(
            receiver, trusted
        ) > simulation._receive_probability(receiver, distrusted)

    def test_risk_tolerance_flattens_positive_expected_payoff(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        other = Agent("other", 1.0, 1.0, 1000.0)
        simulation.reputation[other.id] = 0.5
        cautious = Agent("cautious", 1.0, 1.0, 1000.0, risk_tolerance=0.05)
        bold = Agent("bold", 1.0, 1.0, 1000.0, risk_tolerance=1.0)
        cautious.beliefs[other.id] = 0.9
        bold.beliefs[other.id] = 0.9

        assert simulation._trust_decision_probability(
            cautious, other
        ) > simulation._trust_decision_probability(bold, other)

    def test_receiver_selection_favours_trusted_agents(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        sender, trusted, distrusted = simulation.agents
        sender.beliefs[trusted.id] = 0.95
        sender.beliefs[distrusted.id] = 0.05

        chosen = [simulation._find_receiver(sender) for _ in range(600)]
        trusted_count = sum(1 for agent in chosen if agent.id == trusted.id)
        distrusted_count = sum(1 for agent in chosen if agent.id == distrusted.id)

        assert trusted_count > distrusted_count * 3

    def test_decision_log_records_inputs(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()

        simulation.run_iteration()

        assert simulation.decision_log
        entry = simulation.decision_log[0]
        for key in (
            "iteration",
            "giver",
            "receiver",
            "belief",
            "risk_tolerance",
            "give_probability",
            "accept_probability",
            "offered_amount",
            "gave",
            "accepted",
        ):
            assert key in entry
        assert 0.0 <= entry["belief"] <= 1.0
        assert 0.0 <= entry["give_probability"] <= 1.0
        assert 0.0 <= entry["accept_probability"] <= 1.0
        assert entry["risk_tolerance"] == 0.5

    def test_metrics_report_mean_decision_probabilities(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()

        simulation.run_iteration()

        metrics = simulation.metrics_history[-1]
        assert 0.0 <= metrics["mean_give_probability"] <= 1.0
        assert 0.0 <= metrics["mean_accept_probability"] <= 1.0

    def test_risk_tolerance_is_drawn_from_config(self):
        config = decision_config()
        config["AGENTS"]["RISK_TOLERANCE"] = (0.2, 0.2)
        simulation = ResourceSimulation(config)

        simulation.initialize_agents()

        assert all(agent.risk_tolerance == 0.2 for agent in simulation.agents)

    def test_zero_decision_weight_preserves_legacy_transfers(self):
        config = decision_config()
        config["TRUST"]["DECISION_WEIGHT"] = 0.0
        with_trust = deepcopy(config)
        with_trust["TRUST"]["ENABLED"] = True
        without_trust = deepcopy(config)
        without_trust["TRUST"]["ENABLED"] = False

        outcomes = []
        for run_config in (with_trust, without_trust):
            simulation = ResourceSimulation(run_config)
            simulation.run(iterations=20)
            metrics = simulation.metrics_history[-1]
            outcomes.append(
                (metrics["total_transfers"], metrics["total"])
            )

        assert outcomes[0] == outcomes[1]


class TestDecisionValidation:
    def test_decision_weight_must_be_between_zero_and_one(self):
        config = decision_config()
        config["TRUST"]["DECISION_WEIGHT"] = 1.5

        with pytest.raises(ValueError, match="DECISION_WEIGHT"):
            ResourceSimulation(config).initialize_agents()

    def test_return_multiplier_must_be_positive(self):
        config = decision_config()
        config["TRUST"]["RETURN_MULTIPLIER"] = 0.0

        with pytest.raises(ValueError, match="RETURN_MULTIPLIER"):
            ResourceSimulation(config).initialize_agents()

    def test_negative_risk_tolerance_is_rejected(self):
        config = decision_config()
        config["AGENTS"]["RISK_TOLERANCE"] = [-0.1, 0.5]

        with pytest.raises(ValueError, match="non-negative"):
            ResourceSimulation(config).initialize_agents()

    def test_reversed_risk_tolerance_range_is_rejected(self):
        config = decision_config()
        config["AGENTS"]["RISK_TOLERANCE"] = [1.0, 0.5]

        with pytest.raises(ValueError, match="ascending order"):
            ResourceSimulation(config).initialize_agents()
