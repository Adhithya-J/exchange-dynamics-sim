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

from utils import MetricsCalculator, load_config

CONFIG: dict[str, dict[str, Any]] = load_config()

# Keep the old lowercase name as a small compatibility convenience for users
# who imported the original module-level configuration.
config = CONFIG

# Market settings used when a custom configuration omits the section. The
# market is opt-in so existing transfer-only configurations keep working.
DEFAULT_MARKET_SETTINGS: dict[str, Any] = {
    "ENABLED": False,
    "PRICE": 1.0,
    "PRICE_ADJUSTMENT": 0.0,
    "MIN_PRICE": 0.1,
    "MAX_PRICE": 10.0,
    "PRODUCTION_RANGE": (0.0, 0.0),
    "CONSUMPTION_RANGE": (0.0, 0.0),
    "UNMET_DEMAND_PENALTY": 0.0,
    "SELLER_RESERVATION_PRICE": 0.0,
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
    "DECISION_TEMPERATURE": 0.1,
}

# Gifting is opt-in. Missing sections keep the new no-gift default.
DEFAULT_GIFTING_SETTINGS: dict[str, Any] = {
    "ENABLED": False,
}

# Social settings used when a custom configuration omits the section.
DEFAULT_SOCIAL_SETTINGS: dict[str, Any] = {
    "RECIPROCITY_ENABLED": False,
    "REPAY_PROBABILITY_RANGE": (0.2, 0.8),
    "REPAY_DELAY": 3,
    "REPAY_FRACTION": 1.0,
    "REPAY_RETRY_DELAY": 1,
    "DEFAULT_PENALTY_RATE": 0.05,
    "EXCLUDE_DEFAULTERS": True,
}

# Shock settings used when a custom configuration omits the section.
DEFAULT_SHOCK_SETTINGS: dict[str, Any] = {
    "ENABLED": False,
    "SCHEDULE": (),
    "ADAPTATION_RATE": 0.05,
}

# Commons settings used when a custom configuration omits the section. A missing
# section has no natural stock, but still recycles money so it is not destroyed.
DEFAULT_COMMONS_SETTINGS: dict[str, Any] = {
    "INITIAL_GOODS": 0.0,
    "CAPACITY": 0.0,
    "REGENERATION": 0.0,
    "EXTRACTION_EFFICIENCY": 0.5,
    "EXTRACTION_LIMIT": 0.0,
    "EXTRACTION_COST": 0.0,
    "FUND_REDISTRIBUTION": 0.0,
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
        repayment_probability: float = 0.5,
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
        # Hidden trait: the probability this agent repays a due gift. It is the
        # behaviour that reputation is supposed to forecast.
        self.repayment_probability = float(repayment_probability)
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
            "repayment_probability": self.repayment_probability,
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


class ResourceSimulation:
    """Run resource transfers between a collection of agents."""

    def __init__(self, config: dict[str, dict[str, Any]] | None = None) -> None:
        self.config = deepcopy(config or CONFIG)
        self._validate_config()
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
        self._repayment_opportunities: dict[str, int] = {}
        self._repayment_successes: dict[str, float] = {}
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
        self.total_gift_volume = 0.0
        self.total_repayments = 0
        self.total_repayment_volume = 0.0
        self.total_repayment_delay = 0
        self.total_defaults = 0
        self.written_off_debt = 0.0
        self._active_shocks: list[dict[str, Any]] = []
        self.shock_log: list[dict[str, Any]] = []
        self.shock_count = 0
        self._last_shock_iteration = 0
        self._last_unmet_by_agent: dict[str, float] = {}
        self._last_living_cost_due = 0.0
        self.extinction_iteration: int | None = None
        self.price = self._market_settings()["price"]
        commons = self._commons_settings()
        self.initial_commons_goods = commons["initial_goods"]
        self.commons_goods = commons["initial_goods"]
        self.commons_fund = 0.0
        self._extracted_by_agent: dict[str, float] = {}
        self._last_commons_metrics: dict[str, float] = {
            "commons_regenerated": 0.0,
            "commons_extracted": 0.0,
            "commons_depleted": 0.0,
            "commons_waste": 0.0,
        }

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

    def _validate_config(self) -> None:
        """Reject settings that would break conservation or survival rules."""

        simulation = self.config["SIMULATION"]
        agents = self.config["AGENTS"]
        memory = self.config.get("MEMORY") or {}
        affordability = self.config.get("AFFORDABILITY") or {}
        try:
            count = int(simulation["AGENTS"])
            iterations = int(simulation["ITERATIONS"])
            initial = float(simulation["INITIAL_RESOURCES"])
            floor = float(simulation["RESOURCE_FLOOR"])
            living_cost = float(agents["LIVING_COST"])
            giving_floor = float(agents["MIN_RESOURCES_AFTER_GIVING"])
            receiving_ceiling = float(
                agents["MAX_RESOURCES_AFTER_RECEIVING"]
            )
            memory_size = int(memory.get("SIZE", 10))
            affordability_low = float(
                affordability.get("MIN_RESOURCES", 50.0)
            )
            affordability_high = float(
                affordability.get("MAX_RESOURCES", 300.0)
            )
            score_low = float(affordability.get("MIN_SCORE", 0.0))
            score_high = float(affordability.get("MAX_SCORE", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SIMULATION and AGENTS settings must be numeric"
            ) from exc
        if count < 1:
            raise ValueError("SIMULATION.AGENTS must be at least 1")
        if iterations < 0:
            raise ValueError("SIMULATION.ITERATIONS must not be negative")
        if floor < 0:
            raise ValueError("SIMULATION.RESOURCE_FLOOR must not be negative")
        if initial < floor:
            raise ValueError(
                "SIMULATION.INITIAL_RESOURCES must not be below RESOURCE_FLOOR"
            )
        if living_cost < 0:
            raise ValueError("AGENTS.LIVING_COST must not be negative")
        if giving_floor < floor:
            raise ValueError(
                "AGENTS.MIN_RESOURCES_AFTER_GIVING must be at least "
                "RESOURCE_FLOOR"
            )
        if receiving_ceiling < giving_floor:
            raise ValueError(
                "AGENTS.MAX_RESOURCES_AFTER_RECEIVING must be at least "
                "MIN_RESOURCES_AFTER_GIVING"
            )
        if memory_size < 0:
            raise ValueError("MEMORY.SIZE must not be negative")
        if affordability_high <= affordability_low:
            raise ValueError(
                "AFFORDABILITY.MAX_RESOURCES must exceed MIN_RESOURCES"
            )
        if not 0 <= score_low <= score_high <= 1:
            raise ValueError(
                "AFFORDABILITY scores must be in [0, 1] and ascending"
            )
        for name in ("GENEROSITY", "ACCEPTANCE", "RISK_TOLERANCE"):
            self._validated_range(agents.get(name, [0.0, 1.0]), name)
            lower, upper = self._validated_range(
                agents.get(name, [0.0, 1.0]), name
            )
            if upper > 1:
                raise ValueError(f"AGENTS.{name} values must be at most 1")
        self._transfer_amount_bounds()
        # Force section validators to run at construction time instead of
        # surfacing bad values only when the first iteration happens.
        self._market_settings()
        self._commons_settings()
        self._trust_settings()
        self._social_settings()
        self._gifting_settings()

    def _market_settings(self) -> dict[str, Any]:
        """Return validated market settings, defaulting to a disabled market."""

        configured = self.config.get("MARKET") or {}
        merged = {**DEFAULT_MARKET_SETTINGS, **configured}
        enabled = bool(merged["ENABLED"])
        try:
            price = float(merged["PRICE"])
            price_adjustment = float(merged["PRICE_ADJUSTMENT"])
            min_price = float(merged["MIN_PRICE"])
            max_price = float(merged["MAX_PRICE"])
            unmet_penalty = float(merged["UNMET_DEMAND_PENALTY"])
            seller_reservation = float(merged["SELLER_RESERVATION_PRICE"])
        except (TypeError, ValueError) as exc:
            raise ValueError("MARKET settings must be numeric") from exc
        production_range = self._validated_range(
            merged["PRODUCTION_RANGE"], "MARKET.PRODUCTION_RANGE"
        )
        consumption_range = self._validated_range(
            merged["CONSUMPTION_RANGE"], "MARKET.CONSUMPTION_RANGE"
        )
        if enabled and price <= 0:
            raise ValueError("MARKET.PRICE must be positive")
        if not 0 <= price_adjustment <= 1:
            raise ValueError("MARKET.PRICE_ADJUSTMENT must be between 0 and 1")
        if min_price <= 0 or max_price < min_price:
            raise ValueError(
                "MARKET.MIN_PRICE must be positive and not exceed MAX_PRICE"
            )
        if unmet_penalty < 0:
            raise ValueError("MARKET.UNMET_DEMAND_PENALTY must not be negative")
        if seller_reservation < 0:
            raise ValueError(
                "MARKET.SELLER_RESERVATION_PRICE must not be negative"
            )
        return {
            "enabled": enabled,
            "price": price,
            "price_adjustment": price_adjustment,
            "min_price": min_price,
            "max_price": max_price,
            "production_range": production_range,
            "consumption_range": consumption_range,
            "unmet_demand_penalty": unmet_penalty,
            "seller_reservation_price": seller_reservation,
        }

    def _commons_settings(self) -> dict[str, Any]:
        """Return validated natural-commons and fund settings."""

        configured = self.config.get("COMMONS") or {}
        merged = {**DEFAULT_COMMONS_SETTINGS, **configured}
        try:
            initial_goods = float(merged["INITIAL_GOODS"])
            capacity = float(merged["CAPACITY"])
            regeneration = float(merged["REGENERATION"])
            efficiency = float(merged["EXTRACTION_EFFICIENCY"])
            extraction_limit = float(merged["EXTRACTION_LIMIT"])
            extraction_cost = float(merged["EXTRACTION_COST"])
            fund_redistribution = float(merged["FUND_REDISTRIBUTION"])
        except (TypeError, ValueError) as exc:
            raise ValueError("COMMONS settings must be numeric") from exc
        if initial_goods < 0 or capacity < 0:
            raise ValueError("COMMONS stock values must not be negative")
        if capacity < initial_goods:
            raise ValueError("COMMONS.CAPACITY must be at least INITIAL_GOODS")
        if regeneration < 0:
            raise ValueError("COMMONS.REGENERATION must not be negative")
        if not 0 < efficiency <= 1:
            raise ValueError(
                "COMMONS.EXTRACTION_EFFICIENCY must be in (0, 1]"
            )
        if extraction_limit < 0:
            raise ValueError("COMMONS.EXTRACTION_LIMIT must not be negative")
        if extraction_cost < 0:
            raise ValueError("COMMONS.EXTRACTION_COST must not be negative")
        if not 0 <= fund_redistribution <= 1:
            raise ValueError(
                "COMMONS.FUND_REDISTRIBUTION must be between 0 and 1"
            )
        return {
            "initial_goods": initial_goods,
            "capacity": capacity,
            "regeneration": regeneration,
            "extraction_efficiency": efficiency,
            "extraction_limit": extraction_limit,
            "extraction_cost": extraction_cost,
            "fund_redistribution": fund_redistribution,
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
            decision_temperature = float(merged["DECISION_TEMPERATURE"])
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
        if decision_temperature <= 0:
            raise ValueError("TRUST.DECISION_TEMPERATURE must be positive")
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
            "decision_temperature": decision_temperature,
        }

    def _gifting_settings(self) -> dict[str, Any]:
        """Return whether direct transfers are enabled."""

        configured = self.config.get("GIFTING") or {}
        merged = {**DEFAULT_GIFTING_SETTINGS, **configured}
        return {"enabled": bool(merged["ENABLED"])}

    def _social_settings(self) -> dict[str, Any]:
        """Return validated altruism and reciprocity settings."""

        configured = self.config.get("SOCIAL") or {}
        merged = {**DEFAULT_SOCIAL_SETTINGS, **configured}
        enabled = bool(merged["RECIPROCITY_ENABLED"])
        try:
            delay = int(merged["REPAY_DELAY"])
            fraction = float(merged["REPAY_FRACTION"])
            retry_delay = int(merged["REPAY_RETRY_DELAY"])
            default_penalty = float(merged["DEFAULT_PENALTY_RATE"])
        except (TypeError, ValueError) as exc:
            raise ValueError("SOCIAL settings must be numeric") from exc
        if "REPAY_PROBABILITY" in configured:
            # Backwards-compatible single-probability form.
            try:
                probability = float(configured["REPAY_PROBABILITY"])
            except (TypeError, ValueError) as exc:
                raise ValueError("SOCIAL settings must be numeric") from exc
            if not 0 <= probability <= 1:
                raise ValueError(
                    "SOCIAL.REPAY_PROBABILITY must be between 0 and 1"
                )
            probability_range = (probability, probability)
        else:
            probability_range = self._validated_range(
                merged["REPAY_PROBABILITY_RANGE"],
                "SOCIAL.REPAY_PROBABILITY_RANGE",
            )
            if probability_range[1] > 1:
                raise ValueError(
                    "SOCIAL.REPAY_PROBABILITY_RANGE values must be between 0 and 1"
                )
        if delay < 0:
            raise ValueError("SOCIAL.REPAY_DELAY must not be negative")
        if not 0 <= fraction <= 1:
            raise ValueError("SOCIAL.REPAY_FRACTION must be between 0 and 1")
        if retry_delay < 1:
            raise ValueError("SOCIAL.REPAY_RETRY_DELAY must be at least 1")
        if default_penalty < 0:
            raise ValueError("SOCIAL.DEFAULT_PENALTY_RATE must not be negative")
        return {
            "enabled": enabled,
            "repay_probability_range": probability_range,
            "repay_delay": delay,
            "repay_fraction": fraction,
            "repay_retry_delay": retry_delay,
            "default_penalty_rate": default_penalty,
            "exclude_defaulters": bool(merged["EXCLUDE_DEFAULTERS"]),
        }

    def _schedule_repayment(
        self, giver: Agent, receiver: Agent, amount: float
    ) -> None:
        """Record a gift and schedule a possible reciprocal repayment."""

        self.total_gifts += 1
        self.total_gift_volume += amount
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
                "remaining": amount,
                "defaults": 0,
                "attempts": 0,
            }
        )

    def _process_repayments(self, iteration: int) -> dict[str, Any]:
        """Pay due gifts back and keep unpaid debt on the books.

        A missed or partial payment is a default: it is observed by trust,
        charged a late fee, and retried later. If either party dies the
        unrecoverable remainder is written off. Debt therefore does not silently
        disappear, and ``outstanding_debt`` stays meaningful.
        """

        settings = self._social_settings()

        def summary(defaults: int = 0) -> dict[str, Any]:
            outstanding = float(
                sum(entry["remaining"] for entry in self._repayment_queue)
            )
            defaulted = float(
                sum(
                    entry["remaining"]
                    for entry in self._repayment_queue
                    if entry["defaults"] > 0
                )
            )
            return {
                "repayments": 0,
                "repayment_volume": 0.0,
                "total_repayments": self.total_repayments,
                "reciprocity_rate": (
                    min(
                        1.0,
                        self.total_repayment_volume / self.total_gift_volume,
                    )
                    if self.total_gift_volume > 0
                    else 0.0
                ),
                "repayment_event_rate": (
                    min(1.0, self.total_repayments / self.total_gifts)
                    if self.total_gifts
                    else 0.0
                ),
                "outstanding_debt": outstanding,
                "defaulted_debt": defaulted,
                "defaults": defaults,
                "total_defaults": self.total_defaults,
                "written_off_debt": self.written_off_debt,
                "mean_repayment_delay": (
                    self.total_repayment_delay / self.total_repayments
                    if self.total_repayments
                    else 0.0
                ),
            }

        if not settings["enabled"] or not self._repayment_queue:
            return summary()

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
        defaults = 0
        floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        epsilon = 1e-9
        for entry in due:
            debtor = self._agent_by_id(entry["debtor"])
            creditor = self._agent_by_id(entry["creditor"])
            entry["attempts"] += 1
            due_amount = entry["remaining"] * settings["repay_fraction"]
            amount = 0.0
            if (
                due_amount > 0
                and debtor.is_alive
                and creditor.is_alive
                and self.random.random() < debtor.repayment_probability
            ):
                spendable = max(0.0, debtor.resources - floor)
                amount = min(due_amount, spendable)
            outcome = amount / due_amount if due_amount > 0 else 0.0
            # Only a live creditor can observe repayment behaviour. A
            # creditor's death is a write-off, not a borrower default.
            if debtor.is_alive and creditor.is_alive:
                self._observe_repayment(debtor, creditor, outcome)
            if amount > 0:
                debtor.resources -= amount
                creditor.resources += amount
                if not debtor.is_alive:
                    self._log_death(debtor, "repayment")
                delay = iteration - entry["gift_iteration"]
                entry["remaining"] -= amount
                self.repayment_log.append(
                    {
                        "gift_iteration": entry["gift_iteration"],
                        "repayment_iteration": iteration,
                        "delay": delay,
                        "debtor": entry["debtor"],
                        "creditor": entry["creditor"],
                        "scheduled_amount": entry["amount"],
                        "amount": amount,
                        "remaining": max(0.0, entry["remaining"]),
                        "full": entry["remaining"] <= epsilon,
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

            if entry["remaining"] <= epsilon:
                continue
            if not debtor.is_alive or not creditor.is_alive:
                # No one can settle the remainder; write it off instead of
                # retrying a dead account forever.
                self.written_off_debt += entry["remaining"]
                continue
            defaults += 1
            self.total_defaults += 1
            entry["defaults"] += 1
            entry["remaining"] *= 1.0 + settings["default_penalty_rate"]
            entry["due"] = iteration + settings["repay_retry_delay"]
            self._repayment_queue.append(entry)

        result = summary(defaults=defaults)
        result["repayments"] = count
        result["repayment_volume"] = volume
        return result

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
        if not self._market_settings()["enabled"]:
            return
        if not any(
            shock["type"] == "scarcity" for shock in self._active_shocks
        ):
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
        living = [agent for agent in self.agents if agent.is_alive]
        return {
            "shock_active": int(bool(self._active_shocks)),
            "shock_count": self.shock_count,
            "production_modifier": self._production_modifier(),
            "reputation_noise": self._information_noise(),
            "mean_risk_tolerance": (
                float(np.mean([agent.risk_tolerance for agent in living]))
                if living
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
        last_shock = self.shock_log[-1]
        if last_shock.get("type") != "scarcity":
            # Information shocks do not move unmet demand, so a demand-based
            # recovery time would be meaningless.
            return None
        shock_iteration = last_shock["iteration"]
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
        self._repayment_opportunities = {agent.id: 0 for agent in self.agents}
        self._repayment_successes = {agent.id: 0.0 for agent in self.agents}
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

    def _observe_repayment(
        self, debtor: Agent, creditor: Agent, outcome: float
    ) -> None:
        """Update trust from whether a due gift was actually repaid."""

        settings = self._trust_settings()
        if not settings["enabled"]:
            return
        outcome = float(np.clip(outcome, 0.0, 1.0))
        prior = settings["initial_belief"]
        self._repayment_opportunities[debtor.id] = (
            self._repayment_opportunities.get(debtor.id, 0) + 1
        )
        self._repayment_successes[debtor.id] = (
            self._repayment_successes.get(debtor.id, 0.0) + outcome
        )

        observed = outcome + settings["bias"]
        noise = settings["noise"] + self._information_noise()
        if noise > 0:
            observed += self.random.gauss(0.0, noise)
        observed = float(np.clip(observed, 0.0, 1.0))

        predicted = self.reputation.get(debtor.id, prior)
        self._iteration_observations.append((predicted, outcome))

        if settings["delay"] == 0:
            self._apply_reputation_observation(debtor.id, observed)
        else:
            self._reputation_queue.append(
                (
                    self._current_iteration + settings["delay"],
                    debtor.id,
                    observed,
                )
            )

        private = creditor.beliefs.get(debtor.id, prior)
        private = private + settings["learning_rate"] * (observed - private)
        private = (
            (1.0 - settings["public_weight"]) * private
            + settings["public_weight"] * self.reputation.get(debtor.id, prior)
        )
        creditor.beliefs[debtor.id] = float(np.clip(private, 0.0, 1.0))

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
            opportunities = self._repayment_opportunities.get(agent.id, 0)
            if opportunities == 0:
                continue
            repayment_rate = (
                self._repayment_successes.get(agent.id, 0.0) / opportunities
            )
            errors.append(
                abs(self.reputation.get(agent.id, prior) - repayment_rate)
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
        social = self._social_settings()
        risk_lower, risk_upper = self._validated_range(
            self.config["AGENTS"].get("RISK_TOLERANCE", [0.0, 1.0]),
            "AGENTS.RISK_TOLERANCE",
        )
        use_trust_decisions = trust["enabled"] and trust["decision_weight"] > 0
        repay_lower, repay_upper = social["repay_probability_range"]

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
                repayment_probability=(
                    self.random.uniform(repay_lower, repay_upper)
                    if social["enabled"]
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
        self.total_gift_volume = 0.0
        self.total_repayments = 0
        self.total_repayment_volume = 0.0
        self.total_repayment_delay = 0
        self.total_defaults = 0
        self.written_off_debt = 0.0
        self._active_shocks = []
        self.shock_log = []
        self.death_log = []
        self.shock_count = 0
        self._last_shock_iteration = 0
        self._last_unmet_by_agent = {}
        self._last_living_cost_due = 0.0
        self.extinction_iteration: int | None = None
        self.price = market["price"]
        commons = self._commons_settings()
        self.initial_commons_goods = commons["initial_goods"]
        self.commons_goods = commons["initial_goods"]
        self.commons_fund = 0.0
        self._extracted_by_agent = {}
        self._last_commons_metrics = {
            "commons_regenerated": 0.0,
            "commons_extracted": 0.0,
            "commons_depleted": 0.0,
            "commons_waste": 0.0,
        }
        self.decision_log = []
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
        social = self._social_settings()
        candidates = [
            agent
            for agent in self.agents
            if agent.id != sender.id
            and agent.can_receive(receiving_ceil, minimum_transfer)
        ]
        if social["enabled"] and social["exclude_defaulters"]:
            defaulting = {
                entry["debtor"]
                for entry in self._repayment_queue
                if entry["creditor"] == sender.id
                and entry["defaults"] > 0
                and entry["remaining"] > 1e-9
            }
            candidates = [
                candidate for candidate in candidates if candidate.id not in defaulting
            ]
        if not candidates:
            return None

        memory_settings = self.config.get("MEMORY") or {}
        memory_fraction = memory_settings["PREFERENCE_FRACTION"]
        if not 0 <= memory_fraction <= 1:
            raise ValueError("PREFERENCE_FRACTION must be between 0 and 1")
        if social["enabled"] and not bool(
            memory_settings.get("USE_WITH_RECIPROCITY", False)
        ):
            # A gift already creates a debt. Keeping the memory preference on
            # would reward the same gift twice.
            memory_fraction = 0.0

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
                trust_weights.append(base_weight * factor)
            if sum(trust_weights) <= 0:
                trust_weights = weights
            return self.random.choices(
                candidates, weights=trust_weights, k=1
            )[0]

        if weights is None:
            return self.random.choice(candidates)
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _trust_decision_probability(self, agent: Agent, other: Agent) -> float:
        """Return how strongly the agent acts on its belief about ``other``.

        Risk tolerance scales how far the agent moves away from the prior in
        response to a belief. A risk-tolerant agent acts on a strong belief; a
        risk-averse one falls back toward the prior. The logistic temperature is
        a separate decision-noise parameter, so higher risk tolerance never
        makes every choice more random.
        """

        settings = self._trust_settings()
        social = self._social_settings()
        prior = settings["initial_belief"]
        belief = self.belief_for(agent, other.id)
        adjusted_belief = prior + agent.risk_tolerance * (belief - prior)
        expected_return = (
            adjusted_belief * social["repay_fraction"]
            if social["enabled"]
            else 0.0
        )
        expected_net = expected_return - 1.0
        temperature = settings["decision_temperature"]
        return float(1.0 / (1.0 + np.exp(-expected_net / temperature)))

    def _give_probability(self, agent: Agent, receiver: Agent | None = None) -> float:
        """Return the legacy giving probability.

        Trust affects receiver selection, not the give probability again. This
        keeps the belief from being counted in selection and in the decision.
        """

        return self.affordability.effective_generosity(agent)

    def _receive_probability(self, agent: Agent, giver: Agent | None = None) -> float:
        """Return the acceptance probability.

        Accepting a gift creates a repayment obligation when reciprocity is
        enabled, and default is costly. Existing debt therefore reduces
        willingness to accept a new one instead of being ignored.
        """

        probability = self.affordability.effective_acceptance(agent)
        social = self._social_settings()
        if social["enabled"]:
            debt = sum(
                entry["remaining"]
                for entry in self._repayment_queue
                if entry["debtor"] == agent.id
            )
            if debt > 0:
                burden = debt / max(debt + max(agent.resources, 0.0), 1e-9)
                probability *= max(0.0, 1.0 - 0.5 * burden)
        return float(probability)

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
        use_trust_decisions = settings["enabled"] and settings["decision_weight"] > 0

        for sender in senders:
            if not sender.is_alive:
                continue
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
        """Charge subsistence costs and return the money removed by the sink.

        Every living agent owes the fixed living cost plus an emergency cost for
        each unit of unmet demand. The emergency cost is why goods have value:
        consuming a good avoids it. The charge is capped by what the agent can
        pay while staying at or above the resource floor, so ``paid`` is the
        exact reduction in total money.
        """

        cost = max(0.0, float(self.config["AGENTS"]["LIVING_COST"]))
        market = self._market_settings()
        penalty_rate = (
            market["unmet_demand_penalty"] if market["enabled"] else 0.0
        )
        minimum = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        paid = 0.0
        due_total = 0.0
        for agent in self.agents:
            if not agent.is_alive:
                continue
            unmet = self._last_unmet_by_agent.get(agent.id, 0.0)
            due = cost + penalty_rate * unmet
            due_total += due
            before = agent.resources
            spendable = max(0.0, before - minimum)
            paid_here = min(due, spendable)
            agent.resources = before - paid_here
            if agent.resources <= minimum:
                # Reaching the floor is death, not a resource top-up. If the
                # agent somehow began below the floor, never create money by
                # raising it to the floor.
                agent.resources = min(before, minimum)
            paid += before - agent.resources
            if not agent.is_alive:
                self._log_death(agent, "living_cost")
        # Subsistence payments are a transfer to the commons, not destruction.
        self.commons_fund += paid
        self._last_living_cost_due = due_total
        return paid

    def _distribute_commons_dividend(self) -> float:
        """Return fund money to living agents and record the amount paid out."""

        settings = self._commons_settings()
        if self.commons_fund <= 0 or settings["fund_redistribution"] <= 0:
            return 0.0
        living = [agent for agent in self.agents if agent.is_alive]
        if not living:
            return 0.0
        distributable = self.commons_fund * settings["fund_redistribution"]
        share = distributable / len(living)
        for agent in living:
            agent.resources += share
        self.commons_fund -= distributable
        return distributable

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

    def _extract_from_commons(
        self, agent: Agent, requested: float
    ) -> tuple[float, float, float]:
        """Extract usable goods from the commons, returning usable/raw/waste."""

        settings = self._commons_settings()
        already = self._extracted_by_agent.get(agent.id, 0.0)
        remaining_limit = max(
            0.0, settings["extraction_limit"] - already
        )
        if requested <= 0 or remaining_limit <= 0 or self.commons_goods <= 0:
            return 0.0, 0.0, 0.0
        efficiency = settings["extraction_efficiency"]
        usable = min(
            requested, remaining_limit, self.commons_goods * efficiency
        )
        if usable <= 0:
            return 0.0, 0.0, 0.0
        raw = usable / efficiency
        self.commons_goods = max(0.0, self.commons_goods - raw)
        agent.goods += usable
        self._extracted_by_agent[agent.id] = already + usable
        return usable, raw, raw - usable

    def _buy_from_market(
        self,
        buyer: Agent,
        requested: float,
        price: float,
        participants: list[Agent],
    ) -> tuple[float, float]:
        """Buy up to ``requested`` goods and return (quantity, money spent)."""

        floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        living_cost = max(0.0, self.config["AGENTS"]["LIVING_COST"])
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        market = self._market_settings()
        if price < market["seller_reservation_price"]:
            return 0.0, 0.0
        spendable = max(0.0, buyer.resources - floor - living_cost)
        requested = min(requested, spendable / price)
        if requested <= 0:
            return 0.0, 0.0
        traded = 0.0
        volume = 0.0
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
            seller_room = max(0.0, receiving_ceil - seller.resources)
            quantity = min(
                requested,
                surplus,
                spendable / price,
                seller_room / price,
            )
            if quantity <= 0:
                continue
            cost = quantity * price
            seller.goods -= quantity
            buyer.goods += quantity
            buyer.resources -= cost
            seller.resources += cost
            spendable -= cost
            if not buyer.is_alive:
                self._log_death(buyer, "market")
            traded += quantity
            volume += cost
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
        return traded, volume

    def _run_market_cycle(self) -> dict[str, float]:
        """Produce, trade, extract, and consume goods for one iteration."""

        settings = self._market_settings()
        commons = self._commons_settings()
        self._last_unmet_by_agent = {}
        self._extracted_by_agent = {}
        self._last_commons_metrics = {
            "commons_regenerated": 0.0,
            "commons_extracted": 0.0,
            "commons_depleted": 0.0,
            "commons_waste": 0.0,
        }
        empty_cycle = {
            "price": self.price,
            "goods_produced": 0.0,
            "goods_consumed": 0.0,
            "goods_traded": 0.0,
            "unmet_demand": 0.0,
            "trade_volume": 0.0,
            "commons_goods": self.commons_goods,
            "commons_regenerated": 0.0,
            "commons_extracted": 0.0,
            "commons_depleted": 0.0,
            "commons_waste": 0.0,
        }
        if not settings["enabled"]:
            return empty_cycle

        price = self.price
        penalty = settings["unmet_demand_penalty"]
        floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        living_cost = max(0.0, self.config["AGENTS"]["LIVING_COST"])
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]

        regenerated = min(
            max(0.0, commons["capacity"] - self.commons_goods),
            commons["regeneration"],
        )
        self.commons_goods += regenerated
        produced = self._produce_goods()
        goods_traded = 0.0
        trade_volume = 0.0
        extracted = 0.0
        depleted = 0.0
        waste = 0.0

        participants = [agent for agent in self.agents if agent.is_alive]
        total_demand = float(sum(agent.consumption_need for agent in participants))
        self.random.shuffle(participants)
        for buyer in participants:
            shortfall = buyer.consumption_need - buyer.goods
            if shortfall <= 0:
                continue
            can_buy = price <= penalty
            # The market is preferred while its price is no higher than the
            # effort-equivalent cost of extracting. Above that, extract first.
            extract_first = price > commons["extraction_cost"]
            if extract_first or not can_buy:
                usable, raw, lost = self._extract_from_commons(buyer, shortfall)
                extracted += usable
                depleted += raw
                waste += lost
                shortfall -= usable
                if shortfall > 0 and can_buy:
                    bought, volume = self._buy_from_market(
                        buyer, shortfall, price, participants
                    )
                    goods_traded += bought
                    trade_volume += volume
            else:
                bought, volume = self._buy_from_market(
                    buyer, shortfall, price, participants
                )
                goods_traded += bought
                trade_volume += volume
                shortfall -= bought
                if shortfall > 0:
                    usable, raw, lost = self._extract_from_commons(
                        buyer, shortfall
                    )
                    extracted += usable
                    depleted += raw
                    waste += lost

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

        # Simple price discovery: excess demand raises next iteration's price,
        # excess inventory lowers it. Bounds keep the walk stable.
        leftover = float(sum(agent.goods for agent in participants))
        if settings["price_adjustment"] > 0 and total_demand > 0:
            imbalance = (unmet - leftover) / total_demand
            next_price = price * (1.0 + settings["price_adjustment"] * imbalance)
            self.price = float(
                np.clip(
                    next_price, settings["min_price"], settings["max_price"]
                )
            )

        self._last_commons_metrics = {
            "commons_regenerated": regenerated,
            "commons_extracted": extracted,
            "commons_depleted": depleted,
            "commons_waste": waste,
        }
        return {
            "price": price,
            "goods_produced": produced,
            "goods_consumed": consumed,
            "goods_traded": goods_traded,
            "unmet_demand": unmet,
            "trade_volume": trade_volume,
            "commons_goods": self.commons_goods,
            **self._last_commons_metrics,
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
        if self._gifting_settings()["enabled"]:
            completed = self._process_transfers()
        else:
            completed = []
        market = self._run_market_cycle()
        living_cost_paid = self._apply_living_cost()
        dividend_paid = self._distribute_commons_dividend()
        self._adapt_to_scarcity()
        self.total_transfers += len(completed)
        transfer_volume = sum(amount for _, _, amount in completed)
        self.total_transfer_volume += transfer_volume
        statistics = MetricsCalculator.calculate_statistics(self.to_frame())
        alive = int(sum(agent.is_alive for agent in self.agents))
        if alive == 0 and self.extinction_iteration is None:
            self.extinction_iteration = iteration
        social = self._social_settings()
        altruism_volume = (
            max(0.0, transfer_volume - repayments["repayment_volume"])
            if social["enabled"]
            else transfer_volume
        )
        statistics.update(
            {
                "iteration": iteration,
                "transfers": len(completed),
                "total_transfers": self.total_transfers,
                "transfer_volume": transfer_volume,
                "total_transfer_volume": self.total_transfer_volume,
                "goods_total": float(sum(agent.goods for agent in self.agents)),
                "goods_total_all": (
                    float(sum(agent.goods for agent in self.agents))
                    + self.commons_goods
                ),
                "alive": alive,
                "altruism_volume": altruism_volume,
                "gift_volume": transfer_volume,
                "net_gift_volume": (
                    self.total_gift_volume - self.total_repayment_volume
                ),
                "living_cost_paid": living_cost_paid,
                "living_cost_due": self._last_living_cost_due,
                "dividend_paid": dividend_paid,
                "commons_fund": self.commons_fund,
                "total_money": statistics["total"] + self.commons_fund,
                "extinction_iteration": self.extinction_iteration,
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
        from worlds.money.replay import export_replay

        print(f"Replay written to {export_replay(simulation, args.replay)}")
    print(final_state.head())
    if simulation.metrics_history:
        print("\nFinal statistics:")
        print(simulation.metrics_history[-1])
    else:
        print("\nNo iterations run; no final statistics.")


if __name__ == "__main__":
    main()
