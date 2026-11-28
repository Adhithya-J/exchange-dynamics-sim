"""Tests for trust-based giving, receiving, and receiver selection."""

from copy import deepcopy

import pytest

from main import CONFIG, Agent, ResourceSimulation


def make_config() -> dict:
    config = deepcopy(CONFIG)
    config["GIFTING"]["ENABLED"] = True
    return config


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


class TestBeliefDecisions:
    def test_higher_belief_raises_selection_trust(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        sender, trusted, distrusted = simulation.agents
        sender.beliefs[trusted.id] = 0.9
        sender.beliefs[distrusted.id] = 0.1

        assert simulation._trust_decision_probability(
            sender, trusted
        ) > simulation._trust_decision_probability(sender, distrusted)
        # Giving itself uses the legacy affordability score; trust is counted
        # once, in receiver selection, not again in the give probability.
        assert simulation._give_probability(sender, trusted) == pytest.approx(
            simulation.affordability.effective_generosity(sender)
        )

    def test_acceptance_does_not_depend_on_trust(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        receiver, trusted, distrusted = simulation.agents
        receiver.beliefs[trusted.id] = 0.9
        receiver.beliefs[distrusted.id] = 0.1

        assert simulation._receive_probability(
            receiver, trusted
        ) == simulation._receive_probability(receiver, distrusted)

    def test_risk_tolerance_increases_response_to_belief(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        other = Agent("other", 1.0, 1.0, 1000.0)
        simulation.reputation[other.id] = 0.5
        cautious = Agent("cautious", 1.0, 1.0, 1000.0, risk_tolerance=0.05)
        bold = Agent("bold", 1.0, 1.0, 1000.0, risk_tolerance=1.0)
        cautious.beliefs[other.id] = 0.9
        bold.beliefs[other.id] = 0.9

        # Risk tolerance scales the response to the belief rather than acting
        # as decision noise, so a risk-tolerant agent differentiates more.
        assert simulation._trust_decision_probability(
            bold, other
        ) > simulation._trust_decision_probability(cautious, other)

    def test_zero_risk_tolerance_ignores_trust_beliefs(self):
        simulation = ResourceSimulation(decision_config())
        simulation.initialize_agents()
        other = Agent("other", 1.0, 1.0, 1000.0)
        simulation.reputation[other.id] = 0.5
        risk_averse = Agent("averse", 1.0, 1.0, 1000.0, risk_tolerance=0.0)
        risk_averse.beliefs[other.id] = 0.99

        assert simulation._trust_decision_probability(
            risk_averse, other
        ) == pytest.approx(
            simulation._trust_decision_probability(
                Agent("same", 1.0, 1.0, 1000.0, risk_tolerance=0.0), other
            )
        )

    def test_expected_return_is_derived_from_repay_fraction(self):
        full = ResourceSimulation(decision_config())
        full.initialize_agents()
        other = Agent("other", 1.0, 1.0, 1000.0)
        sender = full.agents[0]
        sender.beliefs[other.id] = 0.9
        full.reputation[other.id] = 0.5

        partial_config = decision_config()
        partial_config["SOCIAL"]["REPAY_FRACTION"] = 0.5
        partial = ResourceSimulation(partial_config)
        partial.initialize_agents()
        partial_sender = partial.agents[0]
        partial_sender.beliefs[other.id] = 0.9
        partial.reputation[other.id] = 0.5

        assert full._trust_decision_probability(
            sender, other
        ) > partial._trust_decision_probability(partial_sender, other)

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

    def test_decision_temperature_must_be_positive(self):
        config = decision_config()
        config["TRUST"]["DECISION_TEMPERATURE"] = 0.0

        with pytest.raises(ValueError, match="DECISION_TEMPERATURE"):
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
