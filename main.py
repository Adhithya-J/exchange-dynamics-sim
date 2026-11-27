"""A small agent-based simulation of resource sharing.

The simulation keeps agents as ordinary Python objects. Pandas is used only
for presenting the current state and calculating summary statistics.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from random import Random
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils import load_config

CONFIG: dict[str, dict[str, Any]] = load_config()

# Keep the old lowercase name as a small compatibility convenience for users
# who imported the original module-level configuration.
config = CONFIG

# Market settings used when a custom configuration omits the section. The
# market is opt-in so existing transfer-only configurations keep working.
DEFAULT_MARKET_SETTINGS: dict[str, Any] = {
    "ENABLED": False,
    "PRICE": 1.0,
    "PRODUCTION_RANGE": (0.0, 0.0),
    "CONSUMPTION_RANGE": (0.0, 0.0),
}

# Trust settings used when a custom configuration omits the section. Trust is
# also opt-in so the legacy transfer model is unchanged by default.
DEFAULT_TRUST_SETTINGS: dict[str, Any] = {
    "ENABLED": False,
    "INITIAL_BELIEF": 0.5,
    "LEARNING_RATE": 0.2,
    "DECAY": 0.02,
    "NOISE": 0.0,
    "DELAY": 0,
    "BIAS": 0.0,
    "PUBLIC_WEIGHT": 0.3,
    "DECISION_WEIGHT": 0.0,
    "RETURN_MULTIPLIER": 2.0,
}

# Social settings used when a custom configuration omits the section.
DEFAULT_SOCIAL_SETTINGS: dict[str, Any] = {
    "RECIPROCITY_ENABLED": False,
    "REPAY_PROBABILITY": 0.5,
    "REPAY_DELAY": 3,
    "REPAY_FRACTION": 1.0,
}

# Shock settings used when a custom configuration omits the section.
DEFAULT_SHOCK_SETTINGS: dict[str, Any] = {
    "ENABLED": False,
    "SCHEDULE": (),
    "ADAPTATION_RATE": 0.05,
}


class Agent:
    """An individual with resources, social tendencies, and recipient memory."""

    def __init__(
        self,
        id: str,
        generosity_score: float,
        acceptance_score: float,
        initial_resources: float,
        memory_size: int = 10,
        resource_floor: float = 0.0,
        goods: float = 0.0,
        production_rate: float = 0.0,
        consumption_need: float = 0.0,
        risk_tolerance: float = 0.5,
    ) -> None:
        self.id = id
        self.generosity_score = float(generosity_score)
        self.acceptance_score = float(acceptance_score)
        self.resources = float(initial_resources)
        self.resource_floor = float(resource_floor)
        self.goods = float(goods)
        self.production_rate = float(production_rate)
        self.consumption_need = float(consumption_need)
        self.risk_tolerance = float(risk_tolerance)
        self.memory: deque[str] = deque(maxlen=memory_size)
        # Private belief that another agent will reciprocate, keyed by id.
        self.beliefs: dict[str, float] = {}

    @property
    def is_alive(self) -> bool:
        """Whether the agent has resources strictly above the death floor."""

        return self.resources > self.resource_floor

    def can_give(self, giving_floor: float, transfer_size: float) -> bool:
        """Return whether a transfer can leave the agent above the giving floor."""

        return (
            self.is_alive
            and self.resources >= transfer_size
            and self.resources - transfer_size >= giving_floor
        )

    def can_receive(self, receiving_ceil: float, transfer_size: float) -> bool:
        """Return whether a transfer would keep the agent below its ceiling."""

        return self.is_alive and self.resources + transfer_size <= receiving_ceil

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the agent."""

        return {
            "id": self.id,
            "generosity_score": self.generosity_score,
            "acceptance_score": self.acceptance_score,
            "resources": self.resources,
            "goods": self.goods,
            "production_rate": self.production_rate,
            "consumption_need": self.consumption_need,
            "risk_tolerance": self.risk_tolerance,
            "alive": self.is_alive,
            "memory": list(self.memory),
            "beliefs": dict(self.beliefs),
        }


class AffordabilityCalculator:
    """Convert resources into a normalized affordability score."""

    def __init__(self, config: dict[str, dict[str, Any]]) -> None:
        self.config = config

    def affordability(self, resources: float) -> float:
        low = self.config["AFFORDABILITY"]["MIN_RESOURCES"]
        high = self.config["AFFORDABILITY"]["MAX_RESOURCES"]
        min_score = self.config["AFFORDABILITY"]["MIN_SCORE"]
        max_score = self.config["AFFORDABILITY"]["MAX_SCORE"]
        if high <= low:
            raise ValueError("MAX_RESOURCES must be greater than MIN_RESOURCES")
        if resources <= low:
            return min_score
        if resources >= high:
            return max_score
        return min_score + (max_score - min_score) * (resources - low) / (high - low)

    def effective_generosity(self, agent: Agent) -> float:
        score = agent.generosity_score * self.affordability(agent.resources)
        lower_limit = self.config["AGENTS"]["GENEROSITY"][0]
        upper_limit = self.config["AGENTS"]["GENEROSITY"][1]
        return float(np.clip(score, lower_limit, upper_limit))

    def effective_acceptance(self, agent: Agent) -> float:
        score = agent.acceptance_score * (2.0 - self.affordability(agent.resources))
        lower_limit = self.config["AGENTS"]["ACCEPTANCE"][0]
        upper_limit = self.config["AGENTS"]["ACCEPTANCE"][1]
        return float(np.clip(score, lower_limit, upper_limit))


class MetricsCalculator:
    """Calculate inequality and resource statistics."""

    @staticmethod
    def calculate_gini(values: Iterable[float]) -> float:
        values_array = np.asarray(list(values), dtype=float)
        if values_array.size == 0:
            raise ValueError("Input array must not be empty")
        if np.any(values_array < 0):
            raise ValueError("Gini is undefined for negative values")

        total = values_array.sum()
        if total == 0:
            return 0.0

        differences = np.abs(values_array[:, None] - values_array[None, :])
        return float(differences.sum() / (2 * values_array.size * total))

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> dict[str, float]:
        if "resources" not in df:
            raise ValueError("DataFrame must contain a resources column")
        resources = df["resources"].to_numpy(dtype=float)
        if resources.size == 0:
            raise ValueError("Cannot calculate statistics for an empty DataFrame")
        return {
            "gini": MetricsCalculator.calculate_gini(resources),
            "mean": float(np.mean(resources)),
            "median": float(np.median(resources)),
            "std": float(np.std(resources)),
            "min": float(np.min(resources)),
            "max": float(np.max(resources)),
            "total": float(np.sum(resources)),
        }


class ResourceSimulation:
    """Run resource transfers between a collection of agents."""

    def __init__(self, config: dict[str, dict[str, Any]] | None = None) -> None:
        self.config = deepcopy(config or CONFIG)
        self.random = Random(self.config["SIMULATION"]["RANDOM_SEED"])
        self.affordability = AffordabilityCalculator(self.config)
        self.agents: list[Agent] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.agent_history: list[dict[str, Any]] = []
        self.total_transfers = 0
        self.total_transfer_volume = 0.0
        self._current_iteration = 0
        self.reputation: dict[str, float] = {}
        self._reputation_queue: list[tuple[int, str, float]] = []
        self._giving_opportunities: dict[str, int] = {}
        self._giving_successes: dict[str, int] = {}
        self._iteration_observations: list[tuple[float, float]] = []
        self.decision_log: list[dict[str, Any]] = []
        self._repayment_queue: list[dict[str, Any]] = []
        self.repayment_log: list[dict[str, Any]] = []
        # Replay-friendly logs. They record exchanges that already happened and
        # are never used to decide the next simulation step.
        self.transfer_log: list[dict[str, Any]] = []
        self.trade_log: list[dict[str, Any]] = []
        self.death_log: list[dict[str, Any]] = []
        self.total_gifts = 0
        self.total_repayments = 0
        self.total_repayment_volume = 0.0
        self.total_repayment_delay = 0
        self._active_shocks: list[dict[str, Any]] = []
        self.shock_log: list[dict[str, Any]] = []
        self.shock_count = 0
        self._last_shock_iteration = 0
        self._last_unmet_by_agent: dict[str, float] = {}

    @staticmethod
    def _validated_range(value: Any, name: str) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a sequence of two values")
        lower, upper = float(value[0]), float(value[1])
        if lower < 0 or upper < lower:
            raise ValueError(
                f"{name} must contain non-negative values in ascending order"
            )
        return lower, upper

    def _market_settings(self) -> dict[str, Any]:
        """Return validated market settings, defaulting to a disabled market."""

        configured = self.config.get("MARKET") or {}
        merged = {**DEFAULT_MARKET_SETTINGS, **configured}
        enabled = bool(merged["ENABLED"])
        try:
            price = float(merged["PRICE"])
        except (TypeError, ValueError) as exc:
            raise ValueError("MARKET.PRICE must be a number") from exc
        production_range = self._validated_range(
            merged["PRODUCTION_RANGE"], "MARKET.PRODUCTION_RANGE"
        )
        consumption_range = self._validated_range(
            merged["CONSUMPTION_RANGE"], "MARKET.CONSUMPTION_RANGE"
        )
        if enabled and price <= 0:
            raise ValueError("MARKET.PRICE must be positive")
        return {
            "enabled": enabled,
            "price": price,
            "production_range": production_range,
            "consumption_range": consumption_range,
        }

    def _trust_settings(self) -> dict[str, Any]:
        """Return validated trust settings, defaulting to a disabled layer."""

        configured = self.config.get("TRUST") or {}
        merged = {**DEFAULT_TRUST_SETTINGS, **configured}
        enabled = bool(merged["ENABLED"])
        try:
            initial_belief = float(merged["INITIAL_BELIEF"])
            learning_rate = float(merged["LEARNING_RATE"])
            decay = float(merged["DECAY"])
            noise = float(merged["NOISE"])
            delay = int(merged["DELAY"])
            bias = float(merged["BIAS"])
            public_weight = float(merged["PUBLIC_WEIGHT"])
            decision_weight = float(merged["DECISION_WEIGHT"])
            return_multiplier = float(merged["RETURN_MULTIPLIER"])
        except (TypeError, ValueError) as exc:
            raise ValueError("TRUST settings must be numeric") from exc
        if not 0 <= initial_belief <= 1:
            raise ValueError("TRUST.INITIAL_BELIEF must be between 0 and 1")
        if not 0 <= learning_rate <= 1:
            raise ValueError("TRUST.LEARNING_RATE must be between 0 and 1")
        if not 0 <= decay <= 1:
            raise ValueError("TRUST.DECAY must be between 0 and 1")
        if noise < 0:
            raise ValueError("TRUST.NOISE must not be negative")
        if delay < 0:
            raise ValueError("TRUST.DELAY must not be negative")
        if not 0 <= public_weight <= 1:
            raise ValueError("TRUST.PUBLIC_WEIGHT must be between 0 and 1")
        if not 0 <= decision_weight <= 1:
            raise ValueError("TRUST.DECISION_WEIGHT must be between 0 and 1")
        if return_multiplier <= 0:
            raise ValueError("TRUST.RETURN_MULTIPLIER must be positive")
        return {
            "enabled": enabled,
            "initial_belief": initial_belief,
            "learning_rate": learning_rate,
            "decay": decay,
            "noise": noise,
            "delay": delay,
            "bias": bias,
            "public_weight": public_weight,
            "decision_weight": decision_weight,
            "return_multiplier": return_multiplier,
        }

    def _social_settings(self) -> dict[str, Any]:
        """Return validated altruism and reciprocity settings."""

        configured = self.config.get("SOCIAL") or {}
        merged = {**DEFAULT_SOCIAL_SETTINGS, **configured}
        enabled = bool(merged["RECIPROCITY_ENABLED"])
        try:
            probability = float(merged["REPAY_PROBABILITY"])
            delay = int(merged["REPAY_DELAY"])
            fraction = float(merged["REPAY_FRACTION"])
        except (TypeError, ValueError) as exc:
            raise ValueError("SOCIAL settings must be numeric") from exc
        if not 0 <= probability <= 1:
            raise ValueError("SOCIAL.REPAY_PROBABILITY must be between 0 and 1")
        if delay < 0:
            raise ValueError("SOCIAL.REPAY_DELAY must not be negative")
        if fraction < 0:
            raise ValueError("SOCIAL.REPAY_FRACTION must not be negative")
        return {
            "enabled": enabled,
            "repay_probability": probability,
            "repay_delay": delay,
            "repay_fraction": fraction,
        }

    def _schedule_repayment(
        self, giver: Agent, receiver: Agent, amount: float
    ) -> None:
        """Record a gift and schedule a possible reciprocal repayment."""

        self.total_gifts += 1
        settings = self._social_settings()
        if not settings["enabled"]:
            return
        self._repayment_queue.append(
            {
                "due": self._current_iteration + settings["repay_delay"],
                "gift_iteration": self._current_iteration,
                "debtor": receiver.id,
                "creditor": giver.id,
                "amount": amount,
            }
        )

    def _process_repayments(self, iteration: int) -> dict[str, Any]:
        """Pay due gifts back when the debtor can afford it."""

        settings = self._social_settings()
        empty = {
            "repayments": 0,
            "repayment_volume": 0.0,
            "total_repayments": self.total_repayments,
            "reciprocity_rate": (
                self.total_repayments / self.total_gifts
                if self.total_gifts
                else 0.0
            ),
            "outstanding_debt": float(
                sum(entry["amount"] for entry in self._repayment_queue)
            ),
            "mean_repayment_delay": (
                self.total_repayment_delay / self.total_repayments
                if self.total_repayments
                else 0.0
            ),
        }
        if not settings["enabled"] or not self._repayment_queue:
            return empty

        due = [
            entry for entry in self._repayment_queue if entry["due"] <= iteration
        ]
        self._repayment_queue = [
            entry
            for entry in self._repayment_queue
            if entry["due"] > iteration
        ]
        count = 0
        volume = 0.0
        floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        for entry in due:
            if self.random.random() >= settings["repay_probability"]:
                continue
            debtor = self._agent_by_id(entry["debtor"])
            creditor = self._agent_by_id(entry["creditor"])
            if not debtor.is_alive or not creditor.is_alive:
                continue
            spendable = max(0.0, debtor.resources - floor)
            amount = min(
                entry["amount"] * settings["repay_fraction"], spendable
            )
            if amount <= 0:
                continue
            debtor.resources -= amount
            creditor.resources += amount
            if not debtor.is_alive:
                self._log_death(debtor, "repayment")
            delay = iteration - entry["gift_iteration"]
            self.repayment_log.append(
                {
                    "gift_iteration": entry["gift_iteration"],
                    "repayment_iteration": iteration,
                    "delay": delay,
                    "debtor": entry["debtor"],
                    "creditor": entry["creditor"],
                    "scheduled_amount": entry["amount"],
                    "amount": amount,
                }
            )
            self.transfer_log.append(
                {
                    "iteration": iteration,
                    "type": "repayment",
                    "giver": entry["debtor"],
                    "receiver": entry["creditor"],
                    "amount": amount,
                }
            )
            count += 1
            volume += amount
            self.total_repayments += 1
            self.total_repayment_volume += amount
            self.total_repayment_delay += delay

        return {
            "repayments": count,
            "repayment_volume": volume,
            "total_repayments": self.total_repayments,
            "reciprocity_rate": (
                self.total_repayments / self.total_gifts
                if self.total_gifts
                else 0.0
            ),
            "outstanding_debt": float(
                sum(entry["amount"] for entry in self._repayment_queue)
            ),
            "mean_repayment_delay": (
                self.total_repayment_delay / self.total_repayments
                if self.total_repayments
                else 0.0
            ),
        }

    def _shock_settings(self) -> dict[str, Any]:
        """Return validated shock settings and the normalized schedule."""

        configured = self.config.get("SHOCKS") or {}
        merged = {**DEFAULT_SHOCK_SETTINGS, **configured}
        enabled = bool(merged["ENABLED"])
        try:
            adaptation_rate = float(merged["ADAPTATION_RATE"])
        except (TypeError, ValueError) as exc:
            raise ValueError("SHOCKS.ADAPTATION_RATE must be a number") from exc
        if adaptation_rate < 0:
            raise ValueError("SHOCKS.ADAPTATION_RATE must not be negative")

        schedule: list[dict[str, Any]] = []
        if enabled:
            raw_schedule = merged["SCHEDULE"]
            if not isinstance(raw_schedule, (list, tuple)):
                raise ValueError("SHOCKS.SCHEDULE must be a sequence")
            for index, event in enumerate(raw_schedule):
                if not isinstance(event, dict):
                    raise ValueError(
                        f"SHOCKS.SCHEDULE[{index}] must be a mapping"
                    )
                if "iteration" not in event or "type" not in event:
                    raise ValueError(
                        f"SHOCKS.SCHEDULE[{index}] requires iteration and type"
                    )
                try:
                    iteration = int(event["iteration"])
                    magnitude = float(event.get("magnitude", 0.5))
                    duration = int(event.get("duration", 1))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"SHOCKS.SCHEDULE[{index}] values must be numeric"
                    ) from exc
                shock_type = str(event["type"]).lower()
                if shock_type not in {"scarcity", "information"}:
                    raise ValueError(
                        f"SHOCKS.SCHEDULE[{index}].type must be "
                        "scarcity or information"
                    )
                if iteration < 1:
                    raise ValueError(
                        "SHOCKS.SCHEDULE iteration must be at least 1"
                    )
                if magnitude < 0:
                    raise ValueError(
                        "SHOCKS.SCHEDULE magnitude must not be negative"
                    )
                if duration < 1:
                    raise ValueError(
                        "SHOCKS.SCHEDULE duration must be at least 1"
                    )
                schedule.append(
                    {
                        "iteration": iteration,
                        "type": shock_type,
                        "magnitude": magnitude,
                        "duration": duration,
                    }
                )
        return {
            "enabled": enabled,
            "adaptation_rate": adaptation_rate,
            "schedule": schedule,
        }

    def _update_shocks(self, iteration: int) -> None:
        """Expire old shocks and activate events scheduled for this iteration."""

        settings = self._shock_settings()
        if not settings["enabled"]:
            self._active_shocks = []
            return
        for shock in self._active_shocks:
            shock["remaining"] -= 1
        self._active_shocks = [
            shock for shock in self._active_shocks if shock["remaining"] > 0
        ]
        for event in settings["schedule"]:
            if event["iteration"] != iteration:
                continue
            self._active_shocks.append(
                {
                    "type": event["type"],
                    "magnitude": event["magnitude"],
                    "remaining": event["duration"],
                }
            )
            self._last_shock_iteration = iteration
            self.shock_count += 1
            self.shock_log.append(
                {
                    "iteration": iteration,
                    "type": event["type"],
                    "magnitude": event["magnitude"],
                    "duration": event["duration"],
                }
            )

    def _production_modifier(self) -> float:
        """Return the combined scarcity multiplier for goods production."""

        modifier = 1.0
        for shock in self._active_shocks:
            if shock["type"] == "scarcity":
                modifier *= max(0.0, 1.0 - shock["magnitude"])
        return modifier

    def _information_noise(self) -> float:
        """Return extra reputation noise from active information shocks."""

        return float(
            sum(
                shock["magnitude"]
                for shock in self._active_shocks
                if shock["type"] == "information"
            )
        )

    def _adapt_to_scarcity(self) -> None:
        """Raise risk tolerance when demand is unmet, otherwise relax it."""

        settings = self._shock_settings()
        if not settings["enabled"]:
            return
        rate = settings["adaptation_rate"]
        for agent in self.agents:
            if not agent.is_alive:
                continue
            if self._last_unmet_by_agent.get(agent.id, 0.0) > 0:
                agent.risk_tolerance = min(1.0, agent.risk_tolerance + rate)
            else:
                agent.risk_tolerance = max(
                    0.0, agent.risk_tolerance - rate * 0.5
                )

    def _shock_metrics(self) -> dict[str, Any]:
        return {
            "shock_active": int(bool(self._active_shocks)),
            "shock_count": self.shock_count,
            "production_modifier": self._production_modifier(),
            "reputation_noise": self._information_noise(),
            "mean_risk_tolerance": (
                float(
                    np.mean([agent.risk_tolerance for agent in self.agents])
                )
                if self.agents
                else 0.0
            ),
            "iterations_since_shock": (
                self._current_iteration - self._last_shock_iteration
                if self._last_shock_iteration
                else 0
            ),
        }

    def recovery_time(self) -> int | None:
        """Return iterations from the last shock until unmet demand recovers."""

        if not self.shock_log:
            return None
        shock_iteration = self.shock_log[-1]["iteration"]
        before = [
            metrics
            for metrics in self.metrics_history
            if metrics["iteration"] < shock_iteration
        ]
        baseline = before[-1].get("unmet_demand", 0.0) if before else 0.0
        for metrics in self.metrics_history:
            if metrics["iteration"] < shock_iteration:
                continue
            if metrics.get("unmet_demand", 0.0) <= baseline:
                return metrics["iteration"] - shock_iteration
        return None

    def _initialize_trust(self) -> None:
        settings = self._trust_settings()
        self._reputation_queue = []
        self._giving_opportunities = {agent.id: 0 for agent in self.agents}
        self._giving_successes = {agent.id: 0 for agent in self.agents}
        self._iteration_observations = []
        for agent in self.agents:
            agent.beliefs = {}
        if not settings["enabled"]:
            self.reputation = {}
            return
        self.reputation = {
            agent.id: settings["initial_belief"] for agent in self.agents
        }

    def _decay_trust(self) -> None:
        settings = self._trust_settings()
        if not settings["enabled"]:
            return
        prior = settings["initial_belief"]
        keep = 1.0 - settings["decay"]
        for agent_id, value in self.reputation.items():
            self.reputation[agent_id] = float(
                np.clip(prior + keep * (value - prior), 0.0, 1.0)
            )
        for agent in self.agents:
            for other_id, value in agent.beliefs.items():
                agent.beliefs[other_id] = float(
                    np.clip(prior + keep * (value - prior), 0.0, 1.0)
                )

    def _apply_reputation_observation(self, agent_id: str, observed: float) -> None:
        settings = self._trust_settings()
        current = self.reputation.get(agent_id, settings["initial_belief"])
        updated = current + settings["learning_rate"] * (observed - current)
        self.reputation[agent_id] = float(np.clip(updated, 0.0, 1.0))

    def _process_reputation_queue(self) -> None:
        settings = self._trust_settings()
        if not settings["enabled"] or not self._reputation_queue:
            return
        due = [
            event
            for event in self._reputation_queue
            if event[0] <= self._current_iteration
        ]
        self._reputation_queue = [
            event
            for event in self._reputation_queue
            if event[0] > self._current_iteration
        ]
        for _, agent_id, observed in due:
            self._apply_reputation_observation(agent_id, observed)

    def belief_for(self, observer: Agent, other_id: str) -> float:
        """Return a private belief, falling back to public reputation."""

        settings = self._trust_settings()
        if other_id in observer.beliefs:
            return observer.beliefs[other_id]
        return self.reputation.get(other_id, settings["initial_belief"])

    def _observe_reciprocation(self, giver: Agent, receiver: Agent) -> None:
        """Update private and public trust after a successful transfer."""

        settings = self._trust_settings()
        if not settings["enabled"]:
            return
        prior = settings["initial_belief"]
        self._giving_successes[giver.id] = (
            self._giving_successes.get(giver.id, 0) + 1
        )

        observed = 1.0 + settings["bias"]
        noise = settings["noise"] + self._information_noise()
        if noise > 0:
            observed += self.random.gauss(0.0, noise)
        observed = float(np.clip(observed, 0.0, 1.0))

        predicted = self.reputation.get(giver.id, prior)
        self._iteration_observations.append((predicted, 1.0))

        if settings["delay"] == 0:
            self._apply_reputation_observation(giver.id, observed)
        else:
            self._reputation_queue.append(
                (
                    self._current_iteration + settings["delay"],
                    giver.id,
                    observed,
                )
            )

        private = receiver.beliefs.get(giver.id, prior)
        private = private + settings["learning_rate"] * (1.0 - private)
        private = (
            (1.0 - settings["public_weight"]) * private
            + settings["public_weight"] * self.reputation.get(giver.id, prior)
        )
        receiver.beliefs[giver.id] = float(np.clip(private, 0.0, 1.0))

    def _trust_metrics(self) -> dict[str, Any]:
        settings = self._trust_settings()
        if not settings["enabled"]:
            return {
                "mean_trust": 0.0,
                "mean_reputation": 0.0,
                "reputation_error": 0.0,
                "reputation_brier": 0.0,
                "trust_observations": 0,
                "mean_give_probability": 0.0,
                "mean_accept_probability": 0.0,
            }

        prior = settings["initial_belief"]
        beliefs = [
            value for agent in self.agents for value in agent.beliefs.values()
        ]
        reputation_values = list(self.reputation.values())
        errors = []
        for agent in self.agents:
            opportunities = self._giving_opportunities.get(agent.id, 0)
            if opportunities == 0:
                continue
            giving_rate = self._giving_successes.get(agent.id, 0) / opportunities
            errors.append(
                abs(self.reputation.get(agent.id, prior) - giving_rate)
            )
        if self._iteration_observations:
            brier = float(
                np.mean(
                    [
                        (predicted - outcome) ** 2
                        for predicted, outcome in self._iteration_observations
                    ]
                )
            )
        else:
            brier = 0.0
        give_probabilities = [
            decision["give_probability"] for decision in self.decision_log
        ]
        accept_probabilities = [
            decision["accept_probability"] for decision in self.decision_log
        ]
        return {
            "mean_trust": float(np.mean(beliefs)) if beliefs else prior,
            "mean_reputation": (
                float(np.mean(reputation_values)) if reputation_values else prior
            ),
            "reputation_error": float(np.mean(errors)) if errors else 0.0,
            "reputation_brier": brier,
            "trust_observations": len(self._iteration_observations),
            "mean_give_probability": (
                float(np.mean(give_probabilities))
                if give_probabilities
                else 0.0
            ),
            "mean_accept_probability": (
                float(np.mean(accept_probabilities))
                if accept_probabilities
                else 0.0
            ),
        }

    def initialize_agents(self) -> list[Agent]:
        """Create the initial population using the configured random seed."""

        generosity_range = self.config["AGENTS"]["GENEROSITY"]
        acceptance_range = self.config["AGENTS"]["ACCEPTANCE"]
        memory_size = self.config["MEMORY"]["SIZE"]
        initial_resources = self.config["SIMULATION"]["INITIAL_RESOURCES"]
        resource_floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        count = self.config["SIMULATION"]["AGENTS"]
        market = self._market_settings()
        trust = self._trust_settings()
        risk_lower, risk_upper = self._validated_range(
            self.config["AGENTS"].get("RISK_TOLERANCE", [0.0, 1.0]),
            "AGENTS.RISK_TOLERANCE",
        )
        use_trust_decisions = trust["enabled"] and trust["decision_weight"] > 0

        self.agents = [
            Agent(
                id=f"agent-{index + 1:03d}",
                generosity_score=self.random.uniform(*generosity_range),
                acceptance_score=self.random.uniform(*acceptance_range),
                initial_resources=initial_resources,
                memory_size=memory_size,
                resource_floor=resource_floor,
                production_rate=(
                    self.random.uniform(*market["production_range"])
                    if market["enabled"]
                    else 0.0
                ),
                consumption_need=(
                    self.random.uniform(*market["consumption_range"])
                    if market["enabled"]
                    else 0.0
                ),
                risk_tolerance=(
                    self.random.uniform(risk_lower, risk_upper)
                    if use_trust_decisions
                    else 0.5
                ),
            )
            for index in range(count)
        ]
        self.metrics_history = []
        self.agent_history = []
        self.total_transfers = 0
        self.total_transfer_volume = 0.0
        self._current_iteration = 0
        self._repayment_queue = []
        self.repayment_log = []
        self.transfer_log = []
        self.trade_log = []
        self.total_gifts = 0
        self.total_repayments = 0
        self.total_repayment_volume = 0.0
        self.total_repayment_delay = 0
        self._active_shocks = []
        self.shock_log = []
        self.death_log = []
        self.shock_count = 0
        self._last_shock_iteration = 0
        self._last_unmet_by_agent = {}
        self._initialize_trust()
        self._record_agent_snapshot(iteration=0)
        return self.agents

    def _record_agent_snapshot(self, iteration: int) -> None:
        """Record the state needed to visualize every agent over time."""

        for agent in self.agents:
            self.agent_history.append(
                {
                    "iteration": iteration,
                    "id": agent.id,
                    "resources": agent.resources,
                    "goods": agent.goods,
                    "generosity_score": agent.generosity_score,
                    "acceptance_score": agent.acceptance_score,
                    "risk_tolerance": agent.risk_tolerance,
                    "alive": agent.is_alive,
                    "reputation": self.reputation.get(agent.id),
                }
            )

    def _agent_by_id(self, agent_id: str) -> Agent:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Unknown agent: {agent_id}")

    def _transfer_amount_bounds(self) -> tuple[float, float]:
        """Return the configured minimum and maximum transfer amounts."""

        simulation_settings = self.config["SIMULATION"]
        if "TRANSFER_AMOUNT_RANGE" in simulation_settings:
            bounds = simulation_settings["TRANSFER_AMOUNT_RANGE"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    "TRANSFER_AMOUNT_RANGE must be a sequence of two values"
                )
            lower, upper = bounds
        else:
            # Keep compatibility with configurations that still use the old
            # fixed transfer amount.
            lower = upper = simulation_settings["TRANSFER_AMOUNT"]
        lower = float(lower)
        upper = float(upper)
        if lower <= 0 or upper < lower:
            raise ValueError(
                "TRANSFER_AMOUNT_RANGE must contain positive values in ascending order"
            )
        return lower, upper

    def _memory_weights(
        self,
        sender: Agent,
        candidates: list[Agent],
        memory_fraction: float,
    ) -> list[float] | None:
        """Return memory-based weights, or None when memory is not used."""

        remembered_candidates = [
            candidate for candidate in candidates if candidate.id in sender.memory
        ]
        if not remembered_candidates or memory_fraction == 0:
            return None

        candidate_memory_scores: dict[str, float] = {}
        for idx, agent_id in enumerate(sender.memory):
            candidate_memory_scores[agent_id] = (
                candidate_memory_scores.get(agent_id, 0.0) + (idx + 1)
            )

        total_memory_score = sum(
            candidate_memory_scores[candidate.id]
            for candidate in remembered_candidates
        )

        uniform_share = (1 - memory_fraction) / len(candidates)
        weights = []
        for candidate in candidates:
            weight = uniform_share
            if candidate.id in candidate_memory_scores:
                memory_share = candidate_memory_scores[candidate.id] / total_memory_score
                weight += memory_fraction * memory_share
            weights.append(weight)
        return weights

    def _find_receiver(
        self, sender: Agent, minimum_transfer: float | None = None
    ) -> Agent | None:
        if minimum_transfer is None:
            minimum_transfer, _ = self._transfer_amount_bounds()
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        candidates = [
            agent
            for agent in self.agents
            if agent.id != sender.id
            and agent.can_receive(receiving_ceil, minimum_transfer)
        ]
        if not candidates:
            return None

        memory_fraction = self.config["MEMORY"]["PREFERENCE_FRACTION"]
        if not 0 <= memory_fraction <= 1:
            raise ValueError("PREFERENCE_FRACTION must be between 0 and 1")

        weights = self._memory_weights(sender, candidates, memory_fraction)
        trust = self._trust_settings()
        decision_weight = trust["decision_weight"] if trust["enabled"] else 0.0
        if decision_weight > 0:
            if weights is None:
                weights = [1.0 / len(candidates)] * len(candidates)
            trust_weights = []
            for candidate, base_weight in zip(candidates, weights):
                probability = self._trust_decision_probability(sender, candidate)
                factor = (1.0 - decision_weight) + decision_weight * probability
                trust_weights.append(base_weight * (0.01 + factor))
            return self.random.choices(
                candidates, weights=trust_weights, k=1
            )[0]

        if weights is None:
            return self.random.choice(candidates)
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _trust_decision_probability(self, agent: Agent, other: Agent) -> float:
        """Return the probability of cooperating based on expected payoff."""

        settings = self._trust_settings()
        belief = self.belief_for(agent, other.id)
        expected_net = belief * settings["return_multiplier"] - 1.0
        temperature = max(agent.risk_tolerance, 0.05)
        return float(1.0 / (1.0 + np.exp(-expected_net / temperature)))

    def _give_probability(self, agent: Agent, receiver: Agent | None = None) -> float:
        legacy = self.affordability.effective_generosity(agent)
        if receiver is None:
            return legacy
        settings = self._trust_settings()
        weight = settings["decision_weight"] if settings["enabled"] else 0.0
        if weight <= 0:
            return legacy
        trust = self._trust_decision_probability(agent, receiver)
        return float(np.clip((1.0 - weight) * legacy + weight * trust, 0.0, 1.0))

    def _receive_probability(self, agent: Agent, giver: Agent | None = None) -> float:
        legacy = self.affordability.effective_acceptance(agent)
        if giver is None:
            return legacy
        settings = self._trust_settings()
        weight = settings["decision_weight"] if settings["enabled"] else 0.0
        if weight <= 0:
            return legacy
        trust = self._trust_decision_probability(agent, giver)
        return float(np.clip((1.0 - weight) * legacy + weight * trust, 0.0, 1.0))

    def _can_give(
        self,
        agent: Agent,
        transfer_size: float,
        receiver: Agent | None = None,
    ) -> bool:
        giving_floor = self.config["AGENTS"]["MIN_RESOURCES_AFTER_GIVING"]
        return (
            agent.can_give(giving_floor, transfer_size)
            and self.random.random() < self._give_probability(agent, receiver)
        )

    def _can_receive(
        self, agent: Agent, transfer_size: float, giver: Agent | None = None
    ) -> bool:
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        return (
            agent.can_receive(receiving_ceil, transfer_size)
            and self.random.random() < self._receive_probability(agent, giver)
        )

    def _log_death(self, agent: Agent, cause: str) -> None:
        """Record a death observed immediately after a cost was paid."""

        self.death_log.append(
            {
                "iteration": self._current_iteration,
                "agent": agent.id,
                "cause": cause,
            }
        )

    def _execute_transfer(
        self, giver: Agent, receiver: Agent, transfer_size: float
    ) -> tuple[str, str, float]:
        giver.resources -= transfer_size
        receiver.resources += transfer_size
        if not giver.is_alive:
            self._log_death(giver, "gift")
        receiver.memory.append(giver.id)
        self._observe_reciprocation(giver, receiver)
        self._schedule_repayment(giver, receiver, transfer_size)
        self.transfer_log.append(
            {
                "iteration": self._current_iteration,
                "type": "gift",
                "giver": giver.id,
                "receiver": receiver.id,
                "amount": transfer_size,
            }
        )
        return (giver.id, receiver.id, transfer_size)

    def _process_transfers(self) -> list[tuple[str, str, float]]:
        """Process transfer actions sequentially in randomized order with immediate updates."""
        minimum_transfer, maximum_transfer = self._transfer_amount_bounds()
        giving_floor = self.config["AGENTS"]["MIN_RESOURCES_AFTER_GIVING"]
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        completed = []

        senders = list(self.agents)
        self.random.shuffle(senders)
        settings = self._trust_settings()
        trust_enabled = settings["enabled"]
        use_trust_decisions = trust_enabled and settings["decision_weight"] > 0

        for sender in senders:
            if not sender.is_alive:
                continue
            if trust_enabled:
                self._giving_opportunities[sender.id] = (
                    self._giving_opportunities.get(sender.id, 0) + 1
                )
            if use_trust_decisions:
                if not sender.can_give(giving_floor, minimum_transfer):
                    continue
                receiver = self._find_receiver(sender, minimum_transfer)
                if receiver is None:
                    continue
                feasible_maximum = min(
                    maximum_transfer,
                    sender.resources - giving_floor,
                    receiving_ceil - receiver.resources,
                )
                if feasible_maximum < minimum_transfer:
                    continue
                transfer_size = self.random.uniform(
                    minimum_transfer, feasible_maximum
                )
                give_probability = self._give_probability(sender, receiver)
                accept_probability = self._receive_probability(receiver, sender)
                gave = self.random.random() < give_probability
                accepted = False
                if gave:
                    accepted = self.random.random() < accept_probability
                self.decision_log.append(
                    {
                        "iteration": self._current_iteration,
                        "giver": sender.id,
                        "receiver": receiver.id,
                        "belief": self.belief_for(sender, receiver.id),
                        "risk_tolerance": sender.risk_tolerance,
                        "give_probability": give_probability,
                        "accept_probability": accept_probability,
                        "offered_amount": transfer_size,
                        "gave": gave,
                        "accepted": accepted,
                    }
                )
                if gave and accepted:
                    completed.append(
                        self._execute_transfer(sender, receiver, transfer_size)
                    )
                continue

            if not self._can_give(sender, minimum_transfer):
                continue
            receiver = self._find_receiver(sender, minimum_transfer)
            if receiver is None:
                continue

            feasible_maximum = min(
                maximum_transfer,
                sender.resources - giving_floor,
                receiving_ceil - receiver.resources,
            )
            if feasible_maximum < minimum_transfer:
                continue
            transfer_size = self.random.uniform(minimum_transfer, feasible_maximum)
            if self._can_receive(receiver, transfer_size):
                completed.append(
                    self._execute_transfer(sender, receiver, transfer_size)
                )
        return completed

    def _perform_transfers(
        self, actions: Iterable[tuple[str, str, float]]
    ) -> list[tuple[str, str, float]]:
        giving_floor = self.config["AGENTS"]["MIN_RESOURCES_AFTER_GIVING"]
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        completed = []

        for giver_id, receiver_id, transfer_size in actions:
            giver = self._agent_by_id(giver_id)
            receiver = self._agent_by_id(receiver_id)
            if not giver.can_give(giving_floor, transfer_size):
                continue
            if not receiver.can_receive(receiving_ceil, transfer_size):
                continue
            completed.append(self._execute_transfer(giver, receiver, transfer_size))
        return completed

    def _apply_living_cost(self) -> float:
        """Charge living agents and return the money removed by the sink."""

        cost = self.config["AGENTS"]["LIVING_COST"]
        minimum = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        paid = 0.0
        for agent in self.agents:
            if not agent.is_alive:
                continue
            before = agent.resources
            agent.resources -= cost
            if agent.resources <= minimum:
                # Reaching the floor is death, not a resource top-up.
                agent.resources = minimum
            paid += before - agent.resources
            if not agent.is_alive:
                self._log_death(agent, "living_cost")
        return paid

    def _produce_goods(self) -> float:
        """Add this iteration's production and return the goods created."""

        modifier = self._production_modifier()
        produced = 0.0
        for agent in self.agents:
            if not agent.is_alive or agent.production_rate <= 0:
                continue
            amount = agent.production_rate * modifier
            agent.goods += amount
            produced += amount
        return produced

    def _run_market_cycle(self) -> dict[str, float]:
        """Produce, trade, and consume goods, returning per-iteration totals."""

        settings = self._market_settings()
        self._last_unmet_by_agent = {}
        empty_cycle = {
            "price": settings["price"],
            "goods_produced": 0.0,
            "goods_consumed": 0.0,
            "goods_traded": 0.0,
            "unmet_demand": 0.0,
            "trade_volume": 0.0,
        }
        if not settings["enabled"]:
            return empty_cycle

        price = settings["price"]
        floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        produced = self._produce_goods()
        goods_traded = 0.0
        trade_volume = 0.0

        participants = [agent for agent in self.agents if agent.is_alive]
        self.random.shuffle(participants)
        for buyer in participants:
            shortfall = buyer.consumption_need - buyer.goods
            if shortfall <= 0:
                continue
            spendable = max(0.0, buyer.resources - floor)
            requested = min(shortfall, spendable / price)
            if requested <= 0:
                continue
            sellers = [
                seller
                for seller in participants
                if seller.id != buyer.id
                and seller.goods > seller.consumption_need
            ]
            self.random.shuffle(sellers)
            for seller in sellers:
                if requested <= 0:
                    break
                surplus = seller.goods - seller.consumption_need
                quantity = min(requested, surplus)
                if quantity <= 0:
                    continue
                cost = quantity * price
                seller.goods -= quantity
                buyer.goods += quantity
                buyer.resources -= cost
                seller.resources += cost
                if not buyer.is_alive:
                    self._log_death(buyer, "market")
                goods_traded += quantity
                trade_volume += cost
                requested -= quantity
                self.trade_log.append(
                    {
                        "iteration": self._current_iteration,
                        "buyer": buyer.id,
                        "seller": seller.id,
                        "quantity": quantity,
                        "price": price,
                        "amount": cost,
                    }
                )

        consumed = 0.0
        unmet = 0.0
        for agent in self.agents:
            if not agent.is_alive:
                continue
            consumed_here = min(agent.consumption_need, agent.goods)
            agent.goods -= consumed_here
            unmet_here = agent.consumption_need - consumed_here
            consumed += consumed_here
            unmet += unmet_here
            self._last_unmet_by_agent[agent.id] = unmet_here

        return {
            "price": price,
            "goods_produced": produced,
            "goods_consumed": consumed,
            "goods_traded": goods_traded,
            "unmet_demand": unmet,
            "trade_volume": trade_volume,
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the current agent state as a dataframe for analysis."""

        columns = [
            "id",
            "generosity_score",
            "acceptance_score",
            "resources",
            "goods",
            "production_rate",
            "consumption_need",
            "risk_tolerance",
            "alive",
            "memory",
        ]
        if not self.agents:
            return pd.DataFrame(columns=columns).set_index("id")
        return pd.DataFrame([agent.to_dict() for agent in self.agents]).set_index("id")

    def agent_history_frame(self) -> pd.DataFrame:
        """Return one row per agent and iteration for visualization."""

        columns = [
            "iteration",
            "id",
            "resources",
            "goods",
            "generosity_score",
            "acceptance_score",
            "risk_tolerance",
            "alive",
            "reputation",
        ]
        return pd.DataFrame(self.agent_history, columns=columns)

    def run_iteration(self) -> list[tuple[str, str, float]]:
        """Run one transfer-and-living-cost step and return completed transfers."""

        if not self.agents:
            self.initialize_agents()
        iteration = len(self.metrics_history) + 1
        self._current_iteration = iteration
        self._update_shocks(iteration)
        self._iteration_observations = []
        self.decision_log = []
        self._decay_trust()
        self._process_reputation_queue()
        repayments = self._process_repayments(iteration)
        completed = self._process_transfers()
        market = self._run_market_cycle()
        living_cost_paid = self._apply_living_cost()
        self._adapt_to_scarcity()
        self.total_transfers += len(completed)
        transfer_volume = sum(amount for _, _, amount in completed)
        self.total_transfer_volume += transfer_volume
        statistics = MetricsCalculator.calculate_statistics(self.to_frame())
        statistics.update(
            {
                "iteration": iteration,
                "transfers": len(completed),
                "total_transfers": self.total_transfers,
                "transfer_volume": transfer_volume,
                "total_transfer_volume": self.total_transfer_volume,
                "goods_total": float(sum(agent.goods for agent in self.agents)),
                "alive": int(sum(agent.is_alive for agent in self.agents)),
                "altruism_volume": transfer_volume,
                "living_cost_paid": living_cost_paid,
                **repayments,
                **market,
                **self._trust_metrics(),
                **self._shock_metrics(),
            }
        )
        self.metrics_history.append(statistics)
        self._record_agent_snapshot(iteration)
        return completed

    def run(self, iterations: int | None = None) -> pd.DataFrame:
        """Run the configured number of iterations and return the final state."""

        if not self.agents:
            self.initialize_agents()
        count = self.config["SIMULATION"]["ITERATIONS"] if iterations is None else iterations
        if count < 0:
            raise ValueError("iterations must not be negative")
        for _ in range(count):
            self.run_iteration()
        return self.to_frame()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the configured number of iterations",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Write a versioned replay log to this path (.gz is compressed)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else None
    simulation = ResourceSimulation(cfg)
    final_state = simulation.run(iterations=args.iterations)
    if args.replay:
        from replay import export_replay

        print(f"Replay written to {export_replay(simulation, args.replay)}")
    print(final_state.head())
    print("\nFinal statistics:")
    print(simulation.metrics_history[-1])


if __name__ == "__main__":
    main()
