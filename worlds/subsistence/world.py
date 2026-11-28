"""A goods-only subsistence world with whole resources and finite days.

Agents start with a starter kit of ten berries and one finished crate. Each
iteration is a finite working day: time is spent picking whole berries or
chopping whole wood from a shared, depletable commons. Berries are food and
spoil outside storage; wood is slow to produce, slow to regrow, and used to
build storage. Health is a whole-meal buffer. There is no money anywhere in the
model. Prices are fractional berries-per-wood ratios, while goods themselves
are always whole.

The optional layers are deliberately small:

- ``FEATURES.TRADE`` enables whole-unit barter at a berries-per-wood ratio.
- ``FEATURES.STORAGE`` lets finished crates provide storage capacity.
- ``FEATURES.SPOILAGE`` makes unprotected berries spoil.
- ``FEATURES.GIFTING`` enables need-based whole-berry gifts.
- ``FEATURES.TRUST`` adds public credit reputation.
- ``FEATURES.RECIPROCITY`` turns gifts into soft zero-interest obligations.
- ``FEATURES.CREDIT`` enables reciprocal or risk-priced loans.
- ``FEATURES.SHOCKS`` adds temporary commons droughts.

Feature code lives in ``worlds/subsistence/features``; the core world keeps the
resource, labour, and market engine.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils import MetricsCalculator, load_config
from worlds.subsistence.features import (
    CreditLayer,
    FeatureRegistry,
    GiftingLayer,
    GoodsLedger,
    Obligation,
    ObligationBook,
    PassiveLayer,
    ReciprocityLayer,
    ShockLayer,
    TradeLayer,
    TrustLayer,
)

CONFIG: dict[str, dict[str, Any]] = load_config(
    Path(__file__).resolve().parent / "config.yaml"
)
config = CONFIG

EPSILON = 1e-9

DEFAULT_FEATURES: dict[str, bool] = {
    "GIFTING": False,
    "TRADE": True,
    "CREDIT": False,
    "STORAGE": True,
    "SPOILAGE": True,
    "TRUST": False,
    "RECIPROCITY": False,
    "SHOCKS": False,
}

DEFAULT_AGENT_SETTINGS: dict[str, Any] = {
    "SKILL_Z_RANGE": 2.0,
    "BERRY_INTERCEPT": 0.85,
    "BERRY_SLOPE": 0.35,
    "WOOD_INTERCEPT": 1.05,
    "WOOD_SLOPE": -0.25,
    "BERRY_SKILL_RANGE": (0.30, 1.50),
    "WOOD_SKILL_RANGE": (0.45, 1.45),
    "TIME_PER_ITERATION": 100.0,
    "BERRY_TIME_COST": 10.0,
    "WOOD_TIME_COST": 120.0,
    "MAX_HEALTH": 4,
    "STARTING_HEALTH": 4,
    "FOOD_PER_ITERATION": 6,
    "HEAL_PER_MEAL": 1,
    "STARVATION_RATE": 1,
    "STARTING_BERRIES": 10,
    "STARTING_WOOD": 0,
    "STARTING_CRATES": 1,
    "RESERVE_DAYS": 2,
    "REFILL_FRACTION": 0.5,
    "RESERVE_JITTER": 0.15,
    "TRADE_SURPLUS": 1,
    "WOOD_RESERVE_CRATES": 1,
    "MAX_WOOD_TARGET": 20,
    "MAX_CRATES_TARGET": 2,
    "CHOICE_NOISE": 0.25,
    "EXPLORATION_RATE": 0.08,
}

DEFAULT_RESOURCE_SETTINGS: dict[str, float] = {
    "INITIAL": 0,
    "CAPACITY": 0,
    "REGENERATION": 0.0,
    "EXTRACTION_EFFICIENCY": 1.0,
}

DEFAULT_COMMONS_MIN_DENSITY = 0.10

DEFAULT_SEASONS: dict[str, Any] = {
    "ENABLED": False,
    "LENGTH": 90,
    "CYCLE": [
        {"NAME": "spring", "BERRY_REGEN": 1.0, "WOOD_REGEN": 1.0},
        {"NAME": "summer", "BERRY_REGEN": 1.0, "WOOD_REGEN": 1.0},
        {"NAME": "autumn", "BERRY_REGEN": 1.0, "WOOD_REGEN": 1.0},
        {"NAME": "winter", "BERRY_REGEN": 1.0, "WOOD_REGEN": 1.0},
    ],
}

DEFAULT_STORAGE_SETTINGS: dict[str, float] = {
    "WOOD_PER_CONTAINER": 8,
    "BERRIES_PER_CONTAINER": 10,
    "BASE_CRAFT_MINUTES": 100.0,
    "CRATE_WEAR_RATE": 0.01,
    "CRATE_USE_WEAR_FACTOR": 1.0,
    "WOOD_DECAY_CHANCE": 0.0,
    "STORED_SPOIL_RATE": 0.0,
}

DEFAULT_TRADE_SETTINGS: dict[str, float] = {
    "INITIAL_PRICE": 5.0,
    "BELIEF_NOISE": 1.0,
    "GOSSIP_STEP": 1.0,
}

# Backwards-compatible name for the generic obligation record.
Loan = Obligation


class Agent:
    """A villager with a finite day, whole goods, health, and two skills."""

    def __init__(
        self,
        id: str,
        berry_skill: float,
        wood_skill: float,
        time_per_iteration: float,
        berry_time_cost: float,
        wood_time_cost: float,
        health: int,
        max_health: int,
        storage_wood_per_container: int,
        storage_berries_per_container: int,
        berries: int = 0,
        wood: int = 0,
        specialisation_z: float = 0.0,
        crates: list[float] | None = None,
        price_belief: float = 5.0,
    ) -> None:
        self.id = id
        self.berry_skill = float(berry_skill)
        self.wood_skill = float(wood_skill)
        self.specialisation_z = float(specialisation_z)
        self.time_per_iteration = float(time_per_iteration)
        self.berry_time_cost = float(berry_time_cost)
        self.wood_time_cost = float(wood_time_cost)
        self.health = int(health)
        self.max_health = int(max_health)
        self.storage_wood_per_container = int(storage_wood_per_container)
        self.storage_berries_per_container = int(storage_berries_per_container)
        self.berries = int(berries)
        self.wood = int(wood)
        self.crates: list[float] = [float(value) for value in (crates or [])]
        self.price_belief = float(price_belief)
        # Per-villager taste for a days-of-food reserve. The world draws this
        # once at initialization so stocks differ across the population.
        self.reserve_days_preference = 0.0
        # Credit reputation: a public score updated by repayment and default.
        self.trust = 0.5
        self.defaults = 0
        self.repayments = 0
        self.loans_taken = 0
        self.loans_given = 0
        # Fractional wood work completed toward the next whole block. It
        # persists across iterations so slow chopping is not lost each day.
        self.wood_progress = 0.0
        # Minutes invested in the crate currently being built.
        self.craft_progress = 0.0

        # Per-iteration observations, reset before each gathering step.
        self.labor_berries = 0.0
        self.labor_wood = 0.0
        self.labor_craft = 0.0
        self.planned_berries = 0.0
        self.planned_wood = 0.0
        self.planned_crates = 0
        self.crafted_crates = 0
        self.gathered_berries = 0
        self.gathered_wood = 0
        self.meals_eaten = 0
        self.unmet_food = 0
        self.post_trade_berries = 0
        self.post_trade_wood = 0

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def specialisation(self) -> float:
        """Alias for the shared skill latent, kept for readability."""

        return self.specialisation_z

    @property
    def crate_count(self) -> int:
        return len(self.crates)

    @property
    def crate_durability(self) -> float:
        if not self.crates:
            return 0.0
        return float(sum(self.crates) / len(self.crates))

    @property
    def storage_capacity(self) -> int:
        if self.storage_berries_per_container <= 0:
            return 0
        return self.crate_count * self.storage_berries_per_container

    @property
    def max_berries_per_day(self) -> float:
        if self.berry_time_cost <= 0:
            return 0.0
        return self.time_per_iteration * self.berry_skill / self.berry_time_cost

    @property
    def max_wood_per_day(self) -> float:
        if self.wood_time_cost <= 0:
            return 0.0
        return self.time_per_iteration * self.wood_skill / self.wood_time_cost

    @property
    def opportunity_cost_of_wood(self) -> float:
        """Berries forgone per whole unit of wood at full-time work."""

        if self.max_wood_per_day <= 0:
            return float("inf")
        return self.max_berries_per_day / self.max_wood_per_day

    @property
    def potential_berries(self) -> float:
        return self.max_berries_per_day

    @property
    def potential_wood(self) -> float:
        return self.max_wood_per_day

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "health": self.health,
            "max_health": self.max_health,
            "berries": self.berries,
            "wood": self.wood,
            "crates": self.crate_count,
            "crate_durability": self.crate_durability,
            "craft_progress": self.craft_progress,
            "price_belief": self.price_belief,
            "trust": self.trust,
            "defaults": self.defaults,
            "repayments": self.repayments,
            "loans_taken": self.loans_taken,
            "loans_given": self.loans_given,
            "storage_capacity": self.storage_capacity,
            "berry_skill": self.berry_skill,
            "wood_skill": self.wood_skill,
            "specialisation_z": self.specialisation_z,
            "time_per_iteration": self.time_per_iteration,
            "berry_time_cost": self.berry_time_cost,
            "wood_time_cost": self.wood_time_cost,
            "max_berries_per_day": self.max_berries_per_day,
            "max_wood_per_day": self.max_wood_per_day,
            "labor_berries": self.labor_berries,
            "labor_wood": self.labor_wood,
            "labor_craft": self.labor_craft,
            "wood_progress": self.wood_progress,
            "planned_berries": self.planned_berries,
            "planned_wood": self.planned_wood,
            "planned_crates": self.planned_crates,
            "crafted_crates": self.crafted_crates,
            "gathered_berries": self.gathered_berries,
            "gathered_wood": self.gathered_wood,
            "meals_eaten": self.meals_eaten,
            "unmet_food": self.unmet_food,
            "post_trade_berries": self.post_trade_berries,
            "post_trade_wood": self.post_trade_wood,
            "alive": self.is_alive,
        }


class SubsistenceWorld:
    """Run a whole-unit berries-and-wood economy with no money."""

    def __init__(self, config: dict[str, dict[str, Any]] | None = None) -> None:
        self.config = deepcopy(config or CONFIG)
        self.random = Random(self.config["SIMULATION"]["RANDOM_SEED"])
        self.agents: list[Agent] = []
        self._agent_index: dict[str, Agent] = {}
        self.metrics_history: list[dict[str, Any]] = []
        self.agent_history: list[dict[str, Any]] = []

        self.agent_settings = self._agent_settings()
        self.commons_settings = {
            name: self._resource_settings(name) for name in ("berries", "wood")
        }
        self.min_commons_density = self._commons_min_density()
        self.storage_settings = self._storage_settings()
        self.trade_settings = self._trade_settings()
        self.season_settings = self._season_settings()

        self.ledger = GoodsLedger(self)
        self.obligations = ObligationBook(self)
        self.feature_registry = FeatureRegistry(self)
        self.feature_registry.register(PassiveLayer(self, "STORAGE"))
        self.feature_registry.register(PassiveLayer(self, "SPOILAGE"))
        self.feature_registry.register(TradeLayer(self))
        self.feature_registry.register(GiftingLayer(self))
        self.feature_registry.register(TrustLayer(self))
        # Formal credit settles before soft gift obligations. The registry
        # order is also the start_of_tick order, so Credit must precede
        # Reciprocity.
        self.feature_registry.register(CreditLayer(self))
        self.feature_registry.register(ReciprocityLayer(self))
        self.feature_registry.register(ShockLayer(self))
        self.features = self.feature_registry.resolve(self._requested_features())

        self.commons_berries = 0
        self.commons_wood = 0
        self.initial_commons_berries = 0
        self.initial_commons_wood = 0
        self._current_berries_density = 0.0
        self._current_wood_density = 0.0
        self._berry_regen_factor = 1.0
        self._wood_regen_factor = 1.0
        self.initial_price = float(self.trade_settings["initial_price"])
        self.last_market_price = self.initial_price
        self.last_crate_price = 0.0
        self.average_trade_price = self.initial_price
        self._market_participants = 0
        self._market_traders = 0
        self._failed_trades = 0
        self._gossip_price = self.initial_price
        self._current_iteration = 0
        self._season: dict[str, Any] = {
            "index": -1,
            "name": "temperate",
            "berry_regen": 1.0,
            "wood_regen": 1.0,
        }
        self.extinction_iteration: int | None = None

        self.gather_log: list[dict[str, Any]] = []
        self.trade_log: list[dict[str, Any]] = []
        self.craft_log: list[dict[str, Any]] = []
        self.gift_log: list[dict[str, Any]] = []
        self.spoil_log: list[dict[str, Any]] = []
        self.death_log: list[dict[str, Any]] = []
        self.loan_log: list[dict[str, Any]] = []
        self.repayment_log: list[dict[str, Any]] = []
        self.default_log: list[dict[str, Any]] = []
        self.write_off_log: list[dict[str, Any]] = []
        self.shock_log: list[dict[str, Any]] = []
        self._dead_logged: set[str] = set()

        self._last_gathered_berries = 0
        self._last_gathered_wood = 0
        self._last_waste_berries = 0
        self._last_waste_wood = 0
        self._last_blocked_wood = 0
        self._last_regenerated_berries = 0
        self._last_regenerated_wood = 0
        self._last_berries_spoiled = 0
        self._last_wood_decayed = 0
        self._last_crates_crafted = 0
        self._last_crates_worn = 0
        self._last_wood_crafted = 0
        self._last_trade_berries = 0
        self._last_trade_wood = 0
        self._last_crate_trades = 0
        self._last_trade_crates = 0
        self._last_gifts = 0
        self._last_gift_berries = 0

    @property
    def loans(self) -> list[Obligation]:
        return [
            obligation
            for obligation in self.obligations.obligations
            if obligation.kind == "loan"
        ]

    @property
    def credit_events(self) -> list[dict[str, Any]]:
        return self.obligations.events

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
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

    @staticmethod
    def _validated_int_range(value: Any, name: str) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a sequence of two values")
        lower, upper = int(value[0]), int(value[1])
        if lower < 0 or upper < lower:
            raise ValueError(
                f"{name} must contain non-negative whole values in ascending order"
            )
        return lower, upper

    def _requested_features(self) -> dict[str, bool]:
        configured = self.config.get("FEATURES") or {}
        unknown = set(configured) - set(DEFAULT_FEATURES)
        if unknown:
            raise ValueError(
                f"Unknown FEATURES flags: {sorted(unknown)}; valid flags are "
                f"{sorted(DEFAULT_FEATURES)}"
            )
        merged = {**DEFAULT_FEATURES, **configured}
        return {key: bool(value) for key, value in merged.items()}

    def _agent_settings(self) -> dict[str, Any]:
        configured = self.config.get("AGENTS") or {}
        merged = {**DEFAULT_AGENT_SETTINGS, **configured}
        try:
            time_per_iteration = float(merged["TIME_PER_ITERATION"])
            berry_time_cost = float(merged["BERRY_TIME_COST"])
            wood_time_cost = float(merged["WOOD_TIME_COST"])
            skill_z_range = float(merged["SKILL_Z_RANGE"])
            berry_intercept = float(merged["BERRY_INTERCEPT"])
            berry_slope = float(merged["BERRY_SLOPE"])
            wood_intercept = float(merged["WOOD_INTERCEPT"])
            wood_slope = float(merged["WOOD_SLOPE"])
            max_health = int(merged["MAX_HEALTH"])
            starting_health = int(merged["STARTING_HEALTH"])
            food = int(merged["FOOD_PER_ITERATION"])
            heal = int(merged["HEAL_PER_MEAL"])
            starvation = int(merged["STARVATION_RATE"])
            starting_berries = int(merged["STARTING_BERRIES"])
            starting_wood = int(merged["STARTING_WOOD"])
            starting_crates = int(merged["STARTING_CRATES"])
            reserve_days = int(merged["RESERVE_DAYS"])
            refill_fraction = float(merged["REFILL_FRACTION"])
            reserve_jitter = float(merged["RESERVE_JITTER"])
            trade_surplus = int(merged["TRADE_SURPLUS"])
            wood_reserve_crates = int(merged["WOOD_RESERVE_CRATES"])
            max_wood_target = int(merged["MAX_WOOD_TARGET"])
            max_crates_target = int(merged["MAX_CRATES_TARGET"])
            choice_noise = float(merged["CHOICE_NOISE"])
            exploration_rate = float(merged["EXPLORATION_RATE"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AGENTS health, food, and starting stock settings must be whole "
                "numbers, and time and skill settings must be numeric"
            ) from exc
        berry_range = self._validated_range(
            merged["BERRY_SKILL_RANGE"], "AGENTS.BERRY_SKILL_RANGE"
        )
        wood_range = self._validated_range(
            merged["WOOD_SKILL_RANGE"], "AGENTS.WOOD_SKILL_RANGE"
        )
        if time_per_iteration <= 0:
            raise ValueError("AGENTS.TIME_PER_ITERATION must be positive")
        if berry_time_cost <= 0 or wood_time_cost <= 0:
            raise ValueError("AGENTS time costs must be positive")
        if skill_z_range <= 0:
            raise ValueError("AGENTS.SKILL_Z_RANGE must be positive")
        if berry_slope < 0 or wood_slope > 0:
            raise ValueError(
                "AGENTS skill slopes must describe complementary skills: "
                "BERRY_SLOPE >= 0 and WOOD_SLOPE <= 0"
            )
        if berry_slope == 0 and wood_slope == 0:
            raise ValueError(
                "AGENTS.BERRY_SLOPE and AGENTS.WOOD_SLOPE cannot both be zero"
            )
        if max_health <= 0:
            raise ValueError("AGENTS.MAX_HEALTH must be positive")
        if not 0 < starting_health <= max_health:
            raise ValueError(
                "AGENTS.STARTING_HEALTH must be positive and at most MAX_HEALTH"
            )
        if food <= 0:
            raise ValueError("AGENTS.FOOD_PER_ITERATION must be a positive whole number")
        if heal < 0:
            raise ValueError("AGENTS.HEAL_PER_MEAL must not be negative")
        if starvation < 0:
            raise ValueError("AGENTS.STARVATION_RATE must not be negative")
        if starting_berries < 0 or starting_wood < 0 or starting_crates < 0:
            raise ValueError("AGENTS starting goods must not be negative")
        if reserve_days < 0:
            raise ValueError("AGENTS.RESERVE_DAYS must not be negative")
        if not 0 <= refill_fraction <= 1:
            raise ValueError("AGENTS.REFILL_FRACTION must be between 0 and 1")
        if not 0 <= reserve_jitter <= 1:
            raise ValueError("AGENTS.RESERVE_JITTER must be between 0 and 1")
        if trade_surplus < 0:
            raise ValueError("AGENTS.TRADE_SURPLUS must not be negative")
        if wood_reserve_crates < 0:
            raise ValueError("AGENTS.WOOD_RESERVE_CRATES must not be negative")
        if max_wood_target < 0:
            raise ValueError("AGENTS.MAX_WOOD_TARGET must not be negative")
        if max_crates_target < 0:
            raise ValueError("AGENTS.MAX_CRATES_TARGET must not be negative")
        if choice_noise < 0:
            raise ValueError("AGENTS.CHOICE_NOISE must not be negative")
        if not 0 <= exploration_rate <= 1:
            raise ValueError("AGENTS.EXPLORATION_RATE must be between 0 and 1")
        return {
            "skill_z_range": skill_z_range,
            "berry_intercept": berry_intercept,
            "berry_slope": berry_slope,
            "wood_intercept": wood_intercept,
            "wood_slope": wood_slope,
            "berry_skill_range": berry_range,
            "wood_skill_range": wood_range,
            "time_per_iteration": time_per_iteration,
            "berry_time_cost": berry_time_cost,
            "wood_time_cost": wood_time_cost,
            "max_health": max_health,
            "starting_health": starting_health,
            "food_per_iteration": food,
            "heal_per_meal": heal,
            "starvation_rate": starvation,
            "starting_berries": starting_berries,
            "starting_wood": starting_wood,
            "starting_crates": starting_crates,
            "reserve_days": reserve_days,
            "refill_fraction": refill_fraction,
            "reserve_jitter": reserve_jitter,
            "trade_surplus": trade_surplus,
            "wood_reserve_crates": wood_reserve_crates,
            "max_wood_target": max_wood_target,
            "max_crates_target": max_crates_target,
            "choice_noise": choice_noise,
            "exploration_rate": exploration_rate,
        }

    def _commons_scale(self) -> float:
        """Multiplier that keeps the ecology comparable at any village size."""

        commons = self.config.get("COMMONS") or {}
        if not commons.get("SCALE_WITH_AGENTS", False):
            return 1.0
        reference = int(commons.get("REFERENCE_AGENTS", 10) or 10)
        if reference <= 0:
            raise ValueError("COMMONS.REFERENCE_AGENTS must be positive")
        agents = int(self.config["SIMULATION"]["AGENTS"])
        return max(0.0, agents / reference)

    def _resource_settings(self, name: str) -> dict[str, float]:
        configured = (self.config.get("COMMONS") or {}).get(name.upper()) or {}
        merged = {**DEFAULT_RESOURCE_SETTINGS, **configured}
        try:
            initial = int(merged["INITIAL"])
            capacity = int(merged["CAPACITY"])
            regeneration = float(merged["REGENERATION"])
            efficiency = float(merged["EXTRACTION_EFFICIENCY"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"COMMONS.{name.upper()} settings must be numeric; stock values "
                "must be whole numbers"
            ) from exc
        if initial < 0 or capacity < 0:
            raise ValueError(
                f"COMMONS.{name.upper()} stock values must not be negative"
            )
        if capacity < initial:
            raise ValueError(
                f"COMMONS.{name.upper()}.CAPACITY must be at least INITIAL"
            )
        if regeneration < 0:
            raise ValueError(
                f"COMMONS.{name.upper()}.REGENERATION must not be negative"
            )
        if not 0 < efficiency <= 1:
            raise ValueError(
                f"COMMONS.{name.upper()}.EXTRACTION_EFFICIENCY must be in (0, 1]"
            )
        scale = self._commons_scale()
        return {
            "initial": int(round(initial * scale)),
            "capacity": int(round(capacity * scale)),
            "regeneration": regeneration * scale,
            "extraction_efficiency": efficiency,
        }

    def _commons_min_density(self) -> float:
        configured = self.config.get("COMMONS") or {}
        try:
            min_density = float(
                configured.get("MIN_DENSITY", DEFAULT_COMMONS_MIN_DENSITY)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("COMMONS.MIN_DENSITY must be numeric") from exc
        if not 0 < min_density <= 1:
            raise ValueError("COMMONS.MIN_DENSITY must be in (0, 1]")
        return min_density

    def _season_settings(self) -> dict[str, Any]:
        configured = self.config.get("SEASONS") or {}
        merged = {**DEFAULT_SEASONS, **configured}
        try:
            length = int(merged["LENGTH"])
        except (TypeError, ValueError) as exc:
            raise ValueError("SEASONS.LENGTH must be a whole number") from exc
        if length <= 0:
            raise ValueError("SEASONS.LENGTH must be positive")
        cycle: list[dict[str, Any]] = []
        for index, season in enumerate(merged.get("CYCLE") or []):
            if not isinstance(season, dict):
                raise ValueError("SEASONS.CYCLE entries must be mappings")
            try:
                berry = float(season.get("BERRY_REGEN", 1.0))
                wood = float(season.get("WOOD_REGEN", 1.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "SEASONS.CYCLE regeneration factors must be numeric"
                ) from exc
            if berry < 0 or wood < 0:
                raise ValueError(
                    "SEASONS.CYCLE regeneration factors must not be negative"
                )
            cycle.append(
                {
                    "name": str(season.get("NAME", f"season-{index + 1}")),
                    "berry_regen": berry,
                    "wood_regen": wood,
                }
            )
        if not cycle:
            raise ValueError("SEASONS.CYCLE must contain at least one season")
        return {"enabled": bool(merged["ENABLED"]), "length": length, "cycle": cycle}

    def season_for(self, iteration: int) -> dict[str, Any]:
        """The season covering ``iteration`` (one-based)."""

        settings = self.season_settings
        if not settings["enabled"]:
            return {
                "index": -1,
                "name": "temperate",
                "berry_regen": 1.0,
                "wood_regen": 1.0,
            }
        cycle = settings["cycle"]
        index = ((max(1, int(iteration)) - 1) // settings["length"]) % len(cycle)
        season = cycle[index]
        return {
            "index": index,
            "name": season["name"],
            "berry_regen": season["berry_regen"],
            "wood_regen": season["wood_regen"],
        }

    def _storage_settings(self) -> dict[str, float]:
        configured = self.config.get("STORAGE") or {}
        merged = {**DEFAULT_STORAGE_SETTINGS, **configured}
        try:
            wood_per_container = int(merged["WOOD_PER_CONTAINER"])
            berries_per_container = int(merged["BERRIES_PER_CONTAINER"])
            base_craft_minutes = float(merged["BASE_CRAFT_MINUTES"])
            crate_wear_rate = float(merged["CRATE_WEAR_RATE"])
            crate_use_wear = float(merged["CRATE_USE_WEAR_FACTOR"])
            wood_decay = float(merged["WOOD_DECAY_CHANCE"])
            stored_spoil = float(merged["STORED_SPOIL_RATE"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "STORAGE settings must be numeric; container sizes must be whole "
                "numbers"
            ) from exc
        if wood_per_container <= 0:
            raise ValueError("STORAGE.WOOD_PER_CONTAINER must be positive")
        if berries_per_container < 0:
            raise ValueError("STORAGE.BERRIES_PER_CONTAINER must not be negative")
        if base_craft_minutes <= 0:
            raise ValueError("STORAGE.BASE_CRAFT_MINUTES must be positive")
        if not 0 <= crate_wear_rate <= 1:
            raise ValueError("STORAGE.CRATE_WEAR_RATE must be between 0 and 1")
        if crate_use_wear < 0:
            raise ValueError(
                "STORAGE.CRATE_USE_WEAR_FACTOR must not be negative"
            )
        if not 0 <= wood_decay <= 1:
            raise ValueError("STORAGE.WOOD_DECAY_CHANCE must be between 0 and 1")
        if not 0 <= stored_spoil <= 1:
            raise ValueError("STORAGE.STORED_SPOIL_RATE must be between 0 and 1")
        return {
            "wood_per_container": wood_per_container,
            "berries_per_container": berries_per_container,
            "base_craft_minutes": base_craft_minutes,
            "crate_wear_rate": crate_wear_rate,
            "crate_use_wear_factor": crate_use_wear,
            "wood_decay_chance": wood_decay,
            "stored_spoil_rate": stored_spoil,
        }

    def _trade_settings(self) -> dict[str, float]:
        configured = self.config.get("TRADE") or {}
        merged = {**DEFAULT_TRADE_SETTINGS, **configured}
        try:
            initial_price = float(merged["INITIAL_PRICE"])
            belief_noise = float(merged["BELIEF_NOISE"])
            gossip_step = float(merged["GOSSIP_STEP"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "TRADE prices and belief settings must be numeric"
            ) from exc
        if initial_price <= 0:
            raise ValueError("TRADE.INITIAL_PRICE must be positive")
        if belief_noise < 0:
            raise ValueError("TRADE.BELIEF_NOISE must not be negative")
        if gossip_step < 0:
            raise ValueError("TRADE.GOSSIP_STEP must not be negative")
        return {
            "initial_price": initial_price,
            "belief_noise": belief_noise,
            "gossip_step": gossip_step,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize_agents(self) -> list[Agent]:
        """Create the population and reset the commons and logs."""

        simulation = self.config["SIMULATION"]
        count = int(simulation["AGENTS"])
        if count < 1:
            raise ValueError("SIMULATION.AGENTS must be at least 1")
        # Re-initialization is a fresh run: clear layer-local state such as an
        # active drought, not just the world logs.
        self.feature_registry.reset()
        self._berry_regen_factor = 1.0
        self._wood_regen_factor = 1.0
        settings = self.agent_settings
        storage = self.storage_settings
        berry_low, berry_high = settings["berry_skill_range"]
        wood_low, wood_high = settings["wood_skill_range"]
        self.agents = []
        for index in range(count):
            # One shared specialisation latent per villager. A high z is a
            # berry specialist and a low z is a wood specialist, so the two
            # skills are negatively correlated by construction.
            z = self.random.uniform(
                -settings["skill_z_range"], settings["skill_z_range"]
            )
            berry_skill = min(
                berry_high,
                max(
                    berry_low,
                    settings["berry_intercept"] + settings["berry_slope"] * z,
                ),
            )
            wood_skill = min(
                wood_high,
                max(
                    wood_low,
                    settings["wood_intercept"] + settings["wood_slope"] * z,
                ),
            )
            self.agents.append(
                Agent(
                    id=f"agent-{index + 1:03d}",
                    berry_skill=berry_skill,
                    wood_skill=wood_skill,
                    time_per_iteration=settings["time_per_iteration"],
                    berry_time_cost=settings["berry_time_cost"],
                    wood_time_cost=settings["wood_time_cost"],
                    health=settings["starting_health"],
                    max_health=settings["max_health"],
                    storage_wood_per_container=(
                        storage["wood_per_container"]
                        if self.features["STORAGE"]
                        else 0
                    ),
                    storage_berries_per_container=(
                        storage["berries_per_container"]
                        if self.features["STORAGE"]
                        else 0
                    ),
                    berries=settings["starting_berries"],
                    wood=settings["starting_wood"],
                    crates=(
                        [1.0] * settings["starting_crates"]
                        if self.features["STORAGE"]
                        else []
                    ),
                    specialisation_z=z,
                )
            )
        self._agent_index = {agent.id: agent for agent in self.agents}
        baseline = self._initial_price_belief()
        beliefs = self.trade_settings
        jitter = settings["reserve_jitter"]
        for agent in self.agents:
            noise = (
                self.random.uniform(
                    -beliefs["belief_noise"], beliefs["belief_noise"]
                )
                if beliefs["belief_noise"] > 0
                else 0.0
            )
            # Beliefs are fractional and unbounded; only positivity is
            # enforced so a market price remains meaningful.
            agent.price_belief = max(float(EPSILON), baseline + noise)
            # A small stable taste for how many days of food this villager
            # likes to keep, within the configured reserve window.
            agent.reserve_days_preference = max(
                0.0,
                settings["reserve_days"]
                * self.random.uniform(1.0 - jitter, 1.0 + jitter),
            )
            trust_layer = getattr(self, "trust_layer", None)
            if trust_layer is not None:
                trust_layer.reset_agent(agent)
            else:
                agent.trust = 0.0
                agent.defaults = 0
                agent.repayments = 0
                agent.loans_taken = 0
                agent.loans_given = 0
        self.initial_price = baseline
        self.last_market_price = baseline
        self.last_crate_price = 0.0
        self.average_trade_price = baseline
        self.metrics_history = []
        self.agent_history = []
        self._current_iteration = 0
        self.extinction_iteration = None
        self.gather_log = []
        self.trade_log = []
        self.craft_log = []
        self.gift_log = []
        self.spoil_log = []
        self.death_log = []
        self.loan_log = []
        self.repayment_log = []
        self.default_log = []
        self.write_off_log = []
        self.shock_log = []
        self.obligations.reset()
        self._dead_logged = set()
        berries = self.commons_settings["berries"]
        wood = self.commons_settings["wood"]
        self.initial_commons_berries = berries["initial"]
        self.initial_commons_wood = wood["initial"]
        self.commons_berries = berries["initial"]
        self.commons_wood = wood["initial"]
        self._current_berries_density = self._commons_density("berries")
        self._current_wood_density = self._commons_density("wood")
        self._reset_iteration_state()
        self._record_agent_snapshot(iteration=0)
        return self.agents

    def _initial_price_belief(self) -> float:
        """Village baseline berries-per-wood opportunity cost.

        The baseline is fractional. There is deliberately no upper or lower
        market clamp; the initial value is only a starting point for learning.
        """

        settings = self.trade_settings
        costs = [
            agent.opportunity_cost_of_wood
            for agent in self.agents
            if math.isfinite(agent.opportunity_cost_of_wood)
            and agent.opportunity_cost_of_wood > 0
        ]
        baseline = (
            float(np.median(costs))
            if costs
            else float(settings["initial_price"])
        )
        return max(float(EPSILON), baseline)

    def _commons_density(self, resource: str) -> float:
        """Current stock as a fraction of the initial stock, capped at 1."""

        if resource == "berries":
            initial = self.initial_commons_berries
            stock = self.commons_berries
        else:
            initial = self.initial_commons_wood
            stock = self.commons_wood
        if initial <= 0:
            return 0.0
        return min(1.0, max(0.0, float(stock) / float(initial)))

    @property
    def berries_commons_pct(self) -> float:
        return self._commons_density("berries")

    @property
    def wood_commons_pct(self) -> float:
        return self._commons_density("wood")

    def _effective_density(self, density: float) -> float:
        return max(float(density), self.min_commons_density)

    def effective_berry_minutes(
        self, agent: Agent, density: float | None = None
    ) -> float:
        if agent.berry_skill <= 0 or agent.berry_time_cost <= 0:
            return float("inf")
        current = self.berries_commons_pct if density is None else density
        return agent.berry_time_cost / (
            agent.berry_skill * self._effective_density(current)
        )

    def effective_wood_minutes(
        self, agent: Agent, density: float | None = None
    ) -> float:
        if agent.wood_skill <= 0 or agent.wood_time_cost <= 0:
            return float("inf")
        current = self.wood_commons_pct if density is None else density
        return agent.wood_time_cost / (
            agent.wood_skill * self._effective_density(current)
        )

    def effective_max_berries_per_day(
        self, agent: Agent, density: float | None = None
    ) -> float:
        minutes = self.effective_berry_minutes(agent, density)
        if not math.isfinite(minutes) or minutes <= 0:
            return 0.0
        return agent.time_per_iteration / minutes

    def effective_max_wood_per_day(
        self, agent: Agent, density: float | None = None
    ) -> float:
        minutes = self.effective_wood_minutes(agent, density)
        if not math.isfinite(minutes) or minutes <= 0:
            return 0.0
        return agent.time_per_iteration / minutes

    def _reset_iteration_state(self) -> None:
        for agent in self.agents:
            agent.labor_berries = 0.0
            agent.labor_wood = 0.0
            agent.labor_craft = 0.0
            agent.planned_berries = 0.0
            agent.planned_wood = 0.0
            agent.planned_crates = 0
            agent.crafted_crates = 0
            agent.gathered_berries = 0
            agent.gathered_wood = 0
            agent.meals_eaten = 0
            agent.unmet_food = 0
            agent.post_trade_berries = 0
            agent.post_trade_wood = 0
        self._last_gathered_berries = 0
        self._last_gathered_wood = 0
        self._last_waste_berries = 0
        self._last_waste_wood = 0
        self._last_blocked_wood = 0
        self._last_regenerated_berries = 0
        self._last_regenerated_wood = 0
        self._last_berries_spoiled = 0
        self._last_wood_decayed = 0
        self._last_crates_crafted = 0
        self._last_crates_worn = 0
        self._last_wood_crafted = 0
        self._last_trade_berries = 0
        self._last_trade_wood = 0
        self._last_crate_trades = 0
        self._last_trade_crates = 0
        self._last_gifts = 0
        self._last_gift_berries = 0
        self._market_participants = 0
        self._market_traders = 0
        self._failed_trades = 0
        self._gossip_price = self.last_market_price

    def run_iteration(self) -> dict[str, Any]:
        """Run one finite day: gather, exchange, eat, then spoil."""

        if not self.agents:
            self.initialize_agents()
        iteration = len(self.metrics_history) + 1
        self._current_iteration = iteration
        self._reset_iteration_state()

        self.feature_registry.run("before_regenerate")
        self._regenerate_commons()
        self._current_berries_density = self._commons_density("berries")
        self._current_wood_density = self._commons_density("wood")
        self.feature_registry.run("start_of_tick")
        self._gather_all()
        self.feature_registry.run("after_gather")
        self.feature_registry.run("market")
        self.feature_registry.run("before_consume")
        for agent in self.agents:
            if agent.is_alive:
                agent.post_trade_berries = agent.berries
                agent.post_trade_wood = agent.wood
        self._eat_all()
        self._decay_and_spoil()
        self.feature_registry.run("end_of_tick")

        metrics = self._metrics(iteration)
        self.metrics_history.append(metrics)
        self._record_agent_snapshot(iteration)
        return metrics

    def run(self, iterations: int | None = None) -> pd.DataFrame:
        if not self.agents:
            self.initialize_agents()
        count = (
            self.config["SIMULATION"]["ITERATIONS"]
            if iterations is None
            else iterations
        )
        if count < 0:
            raise ValueError("iterations must not be negative")
        for _ in range(count):
            self.run_iteration()
        return self.to_frame()

    # ------------------------------------------------------------------
    # Commons and gathering
    # ------------------------------------------------------------------
    def _stochastic_round(self, value: float) -> int:
        """Turn a fractional expected yield into whole units."""

        if value <= 0:
            return 0
        lower = int(math.floor(value))
        fraction = value - lower
        if fraction > 0 and self.random.random() < fraction:
            return lower + 1
        return lower

    def _regenerate_commons(self) -> None:
        season = self.season_for(self._current_iteration)
        self._season = season
        season_factors = {
            "berries": float(season["berry_regen"]),
            "wood": float(season["wood_regen"]),
        }
        for name in ("berries", "wood"):
            settings = self.commons_settings[name]
            attribute = f"commons_{name}"
            current = getattr(self, attribute)
            factor = (
                self._berry_regen_factor
                if name == "berries"
                else self._wood_regen_factor
            )
            amount = self._stochastic_round(
                settings["regeneration"] * float(factor) * season_factors[name]
            )
            amount = min(amount, max(0, settings["capacity"] - current))
            setattr(self, attribute, current + amount)
            if name == "berries":
                self._last_regenerated_berries = amount
            else:
                self._last_regenerated_wood = amount

    def _reserve_days(self, agent: Agent) -> float:
        """This villager's preferred food-reserve length in days."""

        preferred = agent.reserve_days_preference
        if preferred <= 0:
            return float(self.agent_settings["reserve_days"])
        return float(preferred)

    def _target_crates(self, agent: Agent) -> int:
        if not self.features["STORAGE"] or not self.features["SPOILAGE"]:
            return 0
        settings = self.storage_settings
        berries_per_crate = settings["berries_per_container"]
        if berries_per_crate <= 0:
            return 0
        reserve = min(
            self._reserve_days(agent) * self.agent_settings["food_per_iteration"],
            self.agent_settings["max_crates_target"] * berries_per_crate,
        )
        return int(math.ceil(reserve / berries_per_crate))

    def _target_wood(self, agent: Agent) -> int:
        if not self.features["STORAGE"] or not self.features["SPOILAGE"]:
            return 0
        settings = self.storage_settings
        wood_per_crate = settings["wood_per_container"]
        if wood_per_crate <= 0:
            return 0
        missing = max(0, self._target_crates(agent) - agent.crate_count)
        # Once a villager owns a crate, keep one crate's worth of raw wood as
        # a replacement reserve. Raw wood is durable and does not wear out.
        spare = (
            self.agent_settings["wood_reserve_crates"]
            if agent.crate_count > 0
            else 0
        )
        material = (missing + spare) * wood_per_crate
        return max(
            0, min(self.agent_settings["max_wood_target"], material)
        )

    def _craft_minutes(self, agent: Agent) -> float:
        if agent.wood_skill <= 0:
            return float("inf")
        return (
            self.storage_settings["base_craft_minutes"] / agent.wood_skill
        )

    def _complete_crafts(self, agent: Agent) -> None:
        craft_minutes = self._craft_minutes(agent)
        wood_per_crate = self.storage_settings["wood_per_container"]
        if not math.isfinite(craft_minutes) or craft_minutes <= 0:
            return
        while (
            agent.craft_progress + EPSILON >= craft_minutes
            and agent.wood >= wood_per_crate
        ):
            agent.craft_progress = max(
                0.0, agent.craft_progress - craft_minutes
            )
            self.ledger.consume(agent, "wood", wood_per_crate)
            agent.crates.append(1.0)
            agent.planned_crates += 1
            agent.crafted_crates += 1
            self._last_crates_crafted += 1
            self._last_wood_crafted += wood_per_crate
            self.craft_log.append(
                {
                    "iteration": self._current_iteration,
                    "agent": agent.id,
                    "crates": 1,
                    "wood": wood_per_crate,
                }
            )

    def _target_berries(self, agent: Agent) -> int:
        """Berries to keep after eating: a few days, capped by storage.

        This is deliberately a *post-meal* reserve. The old rule added a
        whole daily ration on top of full crate capacity, so the target was
        unreachable after spoilage and villagers topped up every single day.
        """

        food = self.agent_settings["food_per_iteration"]
        reserve = self._reserve_days(agent) * food
        if not self.features["SPOILAGE"]:
            return int(max(food, reserve))
        capacity = agent.storage_capacity if self.features["STORAGE"] else 0
        if capacity <= 0:
            # Without a crate only today's ration can survive.
            return food
        return int(min(capacity, max(food, reserve)))

    def _berry_refill_threshold(self, agent: Agent) -> int:
        """Refill storage only once the reserve has run down."""

        food = self.agent_settings["food_per_iteration"]
        if not self.features["SPOILAGE"]:
            return 0
        target = self._target_berries(agent)
        fraction = self.agent_settings["refill_fraction"]
        return int(max(food, round(target * fraction)))

    def _rational_berry_cap(self, agent: Agent) -> float:
        """Most new berries worth picking today, before planned barter.

        The cap is one daily ration plus whatever empty crate space remains.
        With spoilage disabled, berries keep indefinitely, so there is no
        rational cap.
        """

        if not self.features["SPOILAGE"]:
            return float("inf")
        ration = self.agent_settings["food_per_iteration"]
        capacity = agent.storage_capacity if self.features["STORAGE"] else 0
        remaining_crate = max(0, capacity - agent.berries)
        return float(ration + remaining_crate)

    def _rational_berry_target(self, agent: Agent) -> float:
        """Total berry holding to aim for after non-barter gathering.

        When the larder is above the refill threshold the villager leaves it
        alone and spends the whole day on wood or carpentry. Otherwise it
        refills toward ``reserve + today's ration`` in one batch.
        """

        food = self.agent_settings["food_per_iteration"]
        if not self.features["SPOILAGE"]:
            return float(self._target_berries(agent))
        capacity = agent.storage_capacity if self.features["STORAGE"] else 0
        if capacity <= 0:
            return float(max(agent.berries, food))
        target = self._target_berries(agent)
        if agent.berries >= self._berry_refill_threshold(agent):
            return float(agent.berries)
        harvest_target = food + target
        new_allowed = self._rational_berry_cap(agent)
        return float(
            min(harvest_target, agent.berries + new_allowed)
        )

    def _allocate_labour(self, agent: Agent) -> None:
        """Spend the finite day between berries and wood.

        With trade, specialise where one minute of work is worth more at the
        current exchange ratio. Without trade, aim for self-sufficiency:
        today's meal, then wood for storage, then a berry reserve.
        """

        settings = self.agent_settings
        food = settings["food_per_iteration"]
        remaining = agent.time_per_iteration
        berry_density = self._effective_density(self._current_berries_density)
        wood_density = self._effective_density(self._current_wood_density)
        berry_minutes = self.effective_berry_minutes(
            agent, self._current_berries_density
        )
        wood_minutes = self.effective_wood_minutes(
            agent, self._current_wood_density
        )
        craft_minutes = self._craft_minutes(agent)
        wood_per_crate = self.storage_settings["wood_per_container"]
        price = max(float(EPSILON), float(agent.price_belief))
        berry_value_per_minute = (
            agent.berry_skill * berry_density / agent.berry_time_cost
            if agent.berry_time_cost > 0 and math.isfinite(berry_minutes)
            else 0.0
        )
        wood_value_per_minute = (
            agent.wood_skill
            * wood_density
            * price
            / agent.wood_time_cost
            if agent.wood_time_cost > 0 and math.isfinite(wood_minutes)
            else 0.0
        )
        wood_has_value = self.features["STORAGE"] and self.features["SPOILAGE"]
        if not wood_has_value:
            # Wood is only fuel for storage here; do not spend berry time on it.
            wood_value_per_minute = 0.0

        def sample_value(value: float) -> float:
            """A log-normal taste shock, so daily choices are not identical."""

            if value <= 0 or settings["choice_noise"] <= 0:
                return value
            return value * math.exp(
                self.random.gauss(0.0, settings["choice_noise"])
            )

        berry_choice_value = sample_value(berry_value_per_minute)
        wood_choice_value = sample_value(wood_value_per_minute)
        # Epsilon-greedy exploration: occasionally try the other activity even
        # when the comparison favours one side.
        explored = self.random.random() < settings["exploration_rate"]
        if (
            explored
            and berry_value_per_minute > 0
            and wood_value_per_minute > 0
        ):
            berry_choice_value, wood_choice_value = (
                wood_choice_value,
                berry_choice_value,
            )
        prefer_wood = wood_choice_value > berry_choice_value

        target_crates = self._target_crates(agent)
        own_crate_need = max(0, target_crates - agent.crate_count)
        finish_craft_minutes = (
            max(0.0, craft_minutes - agent.craft_progress)
            if math.isfinite(craft_minutes)
            else float("inf")
        )

        def spend_berries(amount: float) -> None:
            nonlocal remaining
            if amount <= 0 or remaining <= 0 or not math.isfinite(berry_minutes):
                return
            used = min(remaining, amount * berry_minutes)
            if used <= 0:
                return
            agent.planned_berries += used / berry_minutes
            agent.labor_berries += used / agent.time_per_iteration
            remaining -= used

        def spend_wood(amount: float) -> None:
            nonlocal remaining
            if amount <= 0 or remaining <= 0 or not math.isfinite(wood_minutes):
                return
            used = min(remaining, amount * wood_minutes)
            if used <= 0:
                return
            agent.planned_wood += used / wood_minutes
            agent.labor_wood += used / agent.time_per_iteration
            remaining -= used

        def spend_craft(minutes: float) -> None:
            nonlocal remaining
            if (
                minutes <= 0
                or remaining <= 0
                or not math.isfinite(craft_minutes)
                or agent.wood < wood_per_crate
            ):
                return
            used = min(remaining, minutes)
            if used <= 0:
                return
            agent.craft_progress += used
            agent.labor_craft += used / agent.time_per_iteration
            remaining -= used
            self._complete_crafts(agent)

        def spend_leftover_wood() -> None:
            if remaining <= 0 or wood_value_per_minute <= 0:
                return
            surplus = (
                settings["trade_surplus"] if self.features["TRADE"] else 0
            )
            wood_cap = self._target_wood(agent) + surplus
            spend_wood(max(0.0, wood_cap - agent.wood - agent.planned_wood))

        # Finish any crate whose craft minutes were already accumulated.
        self._complete_crafts(agent)

        # With trade, pick the activity with the higher density-adjusted
        # market value per minute. Wood specialists prioritize crafting when
        # they have material, then chop; berry specialists cover food and the
        # crate, then use leftover time for carpentry or wood.
        if self.features["TRADE"]:
            rational_target = self._rational_berry_target(agent)
            target_wood = self._target_wood(agent)
            can_craft = (
                agent.wood >= wood_per_crate
                and math.isfinite(craft_minutes)
            )

            if prefer_wood:
                # Eat first, then specialise. A wood worker picks up to its
                # full meal when the larder is short, and otherwise spends the
                # day chopping or crafting.
                potential = self.effective_max_berries_per_day(
                    agent, self._current_berries_density
                )
                if agent.berries < food:
                    spend_berries(
                        max(0.0, min(food, potential) - agent.berries)
                    )
                should_craft = can_craft and (
                    own_crate_need > 0
                    or agent.craft_progress > 0
                    or agent.berries >= food
                    or agent.crate_count > target_crates
                )
                # Now and then defer a non-urgent craft for a day of chopping.
                if (
                    should_craft
                    and own_crate_need <= 0
                    and agent.craft_progress <= 0
                    and agent.berries >= food
                    and self.random.random() < settings["exploration_rate"]
                ):
                    should_craft = False
                if should_craft:
                    minutes = min(
                        remaining,
                        finish_craft_minutes
                        if finish_craft_minutes > 0
                        else craft_minutes,
                    )
                    spend_craft(minutes)
                if remaining > 0:
                    wood_cap = target_wood + settings["trade_surplus"]
                    spend_wood(
                        max(
                            0.0,
                            wood_cap - agent.wood - agent.planned_wood,
                        )
                    )
                if remaining > 0:
                    spend_berries(max(0, food - agent.berries))
                    spend_berries(
                        max(
                            0.0,
                            rational_target
                            - agent.berries
                            - agent.planned_berries,
                        )
                    )
            else:
                spend_berries(max(0, food - agent.berries))
                spend_berries(
                    max(
                        0.0,
                        rational_target
                        - agent.berries
                        - agent.planned_berries,
                    )
                )
                # Extra berry picking for speculative barter is deliberately
                # omitted: without a matching trade it would only rot. Any
                # existing surplus above target is still offered in the market.
                if (
                    remaining > 0
                    and can_craft
                    and own_crate_need > 0
                    and agent.berries + agent.planned_berries >= food
                ):
                    spend_craft(
                        min(
                            remaining,
                            finish_craft_minutes
                            if finish_craft_minutes > 0
                            else craft_minutes,
                        )
                    )
            spend_leftover_wood()
            return

        # Without trade, aim for self-sufficiency.
        spend_berries(max(0, food - agent.berries))

        if own_crate_need > 0 and agent.wood >= wood_per_crate:
            spend_craft(
                min(
                    remaining,
                    finish_craft_minutes
                    if finish_craft_minutes > 0
                    else craft_minutes,
                )
            )

        if self.features["STORAGE"] and self.features["SPOILAGE"]:
            spend_wood(
                max(
                    0.0,
                    self._target_wood(agent)
                    - agent.wood
                    - agent.planned_wood,
                )
            )

        if self.features["STORAGE"] and self.features["SPOILAGE"]:
            target = self._rational_berry_target(agent)
            spend_berries(
                max(0.0, target - agent.berries - agent.planned_berries)
            )

        if remaining > 0 and not self.features["SPOILAGE"]:
            if math.isfinite(berry_minutes):
                spend_berries(remaining / berry_minutes)

        spend_leftover_wood()

    def _extract(self, agent: Agent, resource: str, requested: float) -> int:
        settings = self.commons_settings[resource]
        attribute = f"commons_{resource}"
        stock = int(getattr(self, attribute))
        efficiency = settings["extraction_efficiency"]
        carried = 0.0
        if resource == "wood":
            # Chopping is slow, so carry the fractional part of a block into
            # the next day. Whole blocks are still the only wood ever stored.
            # The carried fraction stays below one block, so a fractional
            # chopper can never bank a windfall and convert two blocks at once.
            carried = agent.wood_progress
            total = carried + max(0.0, requested)
            requested_whole = int(math.floor(total + EPSILON))
            remainder = max(0.0, min(0.999999, total - requested_whole))
        else:
            requested_whole = self._stochastic_round(requested)
            remainder = 0.0
        if requested_whole <= 0:
            if resource == "wood":
                agent.wood_progress = remainder
            return 0
        if stock <= 0:
            # The commons is empty. No new work converts, but the fraction of
            # a block already chopped is not silently discarded.
            if resource == "wood":
                agent.wood_progress = carried
                self._last_blocked_wood += requested_whole
            return 0
        usable_cap = int(stock * efficiency)
        usable = min(requested_whole, usable_cap)
        if resource == "wood":
            if usable <= 0:
                agent.wood_progress = carried
                self._last_blocked_wood += requested_whole
                return 0
            # Work that the thinning forest could not supply is wasted, like
            # standing at an empty copse; the finished fraction carries on.
            agent.wood_progress = remainder
            self._last_blocked_wood += max(0, requested_whole - usable)
        if usable <= 0:
            return 0
        raw = (
            usable
            if efficiency >= 1
            else min(stock, math.ceil(usable / efficiency))
        )
        raw = max(usable, raw)
        setattr(self, attribute, stock - raw)
        setattr(agent, resource, getattr(agent, resource) + usable)
        waste = raw - usable
        if resource == "berries":
            agent.gathered_berries += usable
            self._last_gathered_berries += usable
            self._last_waste_berries += waste
        else:
            agent.gathered_wood += usable
            self._last_gathered_wood += usable
            self._last_waste_wood += waste
        self.gather_log.append(
            {
                "iteration": self._current_iteration,
                "agent": agent.id,
                "resource": resource,
                "amount": usable,
                "raw": raw,
                "waste": waste,
            }
        )
        return usable

    def _gather_all(self) -> None:
        living = [agent for agent in self.agents if agent.is_alive]
        self.random.shuffle(living)
        for agent in living:
            self._allocate_labour(agent)
            self._extract(agent, "berries", agent.planned_berries)
            self._extract(agent, "wood", agent.planned_wood)

    # ------------------------------------------------------------------
    # Gifts and barter
    # ------------------------------------------------------------------
    def _process_gifts(self) -> None:
        self.feature_registry.run("after_gather")

    def _settle_credit(self) -> None:
        self.feature_registry.layer("CREDIT").start_of_tick()

    def _originate_credit(self) -> None:
        self.feature_registry.layer("CREDIT").before_consume()

    def _survival_urgency(self, agent: Agent) -> int:
        food = self.agent_settings["food_per_iteration"]
        if agent.health <= 1 or agent.berries <= 0:
            return 2
        if agent.berries < food:
            return 1
        return 0

    def _effective_wood_opportunity(self, agent: Agent) -> float:
        max_wood = self.effective_max_wood_per_day(
            agent, self._current_wood_density
        )
        max_berries = self.effective_max_berries_per_day(
            agent, self._current_berries_density
        )
        if max_wood <= 0:
            return float("inf")
        return max_berries / max_wood

    def _effective_berry_rate(self, agent: Agent) -> float:
        if agent.time_per_iteration <= 0:
            return 0.0
        return (
            self.effective_max_berries_per_day(
                agent, self._current_berries_density
            )
            / agent.time_per_iteration
        )

    def _crate_cost_and_belief(
        self, agent: Agent, belief: float, durability: float = 1.0
    ) -> tuple[float, float]:
        wood_per_crate = self.storage_settings["wood_per_container"]
        wood_opportunity = self._effective_wood_opportunity(agent)
        craft_minutes = self._craft_minutes(agent)
        berry_rate = self._effective_berry_rate(agent)
        time_value = (
            craft_minutes * berry_rate if math.isfinite(craft_minutes) else 0.0
        )
        scale = max(0.0, min(1.0, float(durability)))
        if scale <= 0:
            return 0.0, 0.0
        cost = scale * (wood_per_crate * wood_opportunity + time_value)
        believed = scale * (wood_per_crate * float(belief) + time_value)
        return cost, believed

    def _offered_crate_durability(self, agent: Agent, quantity: int) -> float:
        """Average durability of the crates a sale would actually move."""

        if quantity <= 0 or not agent.crates:
            return 0.0
        offered = sorted(agent.crates)[: int(quantity)]
        return float(sum(offered) / len(offered))

    def crate_berry_value(
        self,
        agent: Agent | None = None,
        belief: float | None = None,
        durability: float = 1.0,
    ) -> float:
        """Common berry-equivalent value of one crate.

        The last observed crate price is used when available; otherwise the
        value is anchored to the villager's own material and craft opportunity
        cost. Durability scales the value because a worn crate is less useful.
        """

        scale = max(0.0, min(1.0, float(durability)))
        if scale <= 0:
            return 0.0
        if agent is None:
            agent = next(
                (candidate for candidate in self.agents if candidate.is_alive),
                self.agents[0] if self.agents else None,
            )
        if agent is None:
            value = self.storage_settings["wood_per_container"] * max(
                float(self.trade_settings["initial_price"]), float(EPSILON)
            )
            return max(float(EPSILON), value * scale)
        if self.last_crate_price > 0:
            value = float(self.last_crate_price)
        else:
            price_belief = (
                float(agent.price_belief)
                if belief is None
                else float(belief)
            )
            opportunity, believed = self._crate_cost_and_belief(
                agent, price_belief, 1.0
            )
            value = 0.8 * opportunity + 0.2 * believed
            if not math.isfinite(value):
                value = self.storage_settings["wood_per_container"] * max(
                    price_belief, float(EPSILON)
                )
        return max(float(EPSILON), value * scale)

    def crate_wood_value(
        self,
        agent: Agent | None = None,
        durability: float = 1.0,
    ) -> float:
        """Crate value expressed in whole wood-equivalent units."""

        berry_value = self.crate_berry_value(
            agent, durability=durability
        )
        if self.last_market_price > 0:
            return max(float(EPSILON), berry_value / self.last_market_price)
        return max(
            float(EPSILON),
            self.storage_settings["wood_per_container"]
            * max(0.0, min(1.0, float(durability))),
        )

    def _wood_seller_min(self, agent: Agent, belief: float) -> float:
        food = self.agent_settings["food_per_iteration"]
        wood_rate = self.effective_max_wood_per_day(
            agent, self._current_wood_density
        )
        subsistence = food / max(0.5, wood_rate)
        opportunity = self._effective_wood_opportunity(agent)
        raw = 0.2 * opportunity + 0.8 * float(belief)
        if agent.berries < food:
            # A seller who needs to eat today has a low outside option. Do not
            # make hunger raise the ask: a hungry seller undercuts, and a
            # near-starving one accepts almost any positive price.
            if agent.health <= 2:
                raw = float(EPSILON)
            elif math.isfinite(raw):
                raw = raw * 0.5
            else:
                raw = float(EPSILON)
        elif math.isfinite(raw):
            raw = max(raw, subsistence)
        return max(float(EPSILON), raw)

    def _wood_buyer_max(self, agent: Agent, belief: float) -> float:
        food = self.agent_settings["food_per_iteration"]
        budget = max(0, agent.berries - food)
        opportunity = self._effective_wood_opportunity(agent)
        raw = 0.2 * opportunity + 0.8 * float(belief)
        urgency = 1.0 if agent.berries > agent.storage_capacity else 0.0
        raw += urgency
        if budget > 0:
            # Savings can smooth a thin market: a buyer may spend down to
            # today's meal, not only its daily production surplus.
            raw = min(raw, float(budget))
        return max(float(EPSILON), raw)

    def _crate_seller_min(
        self, agent: Agent, belief: float, durability: float = 1.0
    ) -> float:
        opportunity, believed = self._crate_cost_and_belief(
            agent, belief, durability
        )
        raw = 0.8 * opportunity + 0.2 * believed
        food = self.agent_settings["food_per_iteration"]
        if agent.berries < food:
            # A hungry crate seller discounts heavily; near starvation it
            # accepts almost any positive price.
            if agent.health <= 2 or not math.isfinite(raw):
                raw = float(EPSILON)
            else:
                raw = raw * 0.5
        return max(float(EPSILON), raw)

    def _crate_buyer_max(
        self, agent: Agent, belief: float, durability: float = 1.0
    ) -> float:
        opportunity, believed = self._crate_cost_and_belief(
            agent, belief, durability
        )
        raw = 0.8 * opportunity + 0.2 * believed
        urgency = 1.0 if agent.berries > agent.storage_capacity else 0.0
        raw += 2.0 * urgency
        food = self.agent_settings["food_per_iteration"]
        budget = max(0, agent.berries - food)
        if budget > 0:
            raw = min(raw, float(budget))
        return max(float(EPSILON), raw)

    def _crate_market_plan(
        self, agents: Iterable[Agent], beliefs: dict[str, float]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        food = self.agent_settings["food_per_iteration"]
        sellers: list[dict[str, Any]] = []
        buyers: list[dict[str, Any]] = []
        for agent in agents:
            target_crates = self._target_crates(agent)
            berry_target = self._target_berries(agent)
            berry_deficit = max(0, berry_target - agent.berries)
            # Spend only berries above the reserve, so a bargain does not eat
            # into the few days of food the villager is keeping.
            berry_budget = max(0, agent.berries - berry_target)
            hungry = agent.berries < food
            keep = target_crates
            if hungry:
                keep = max(0, target_crates - 1)
            pledged = self.obligations.pledged_crates(agent.id)
            unpledged = max(0, agent.crate_count - pledged)
            surplus = max(0, unpledged - keep)
            need = max(0, target_crates - agent.crate_count)
            listed_as_seller = False
            if surplus > 0 and berry_deficit > 0:
                durability = self._offered_crate_durability(agent, surplus)
                sellers.append(
                    {
                        "agent": agent,
                        "qty": surplus,
                        "price": self._crate_seller_min(
                            agent, beliefs[agent.id], durability
                        ),
                    }
                )
                listed_as_seller = True
            if need > 0 and berry_budget > 0 and not listed_as_seller:
                buyers.append(
                    {
                        "agent": agent,
                        "qty": need,
                        "price": self._crate_buyer_max(
                            agent, beliefs[agent.id]
                        ),
                        "budget": berry_budget,
                    }
                )
        return sellers, buyers

    def _wood_market_plan(
        self, agents: Iterable[Agent], beliefs: dict[str, float]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        food = self.agent_settings["food_per_iteration"]
        wood_per_crate = self.storage_settings["wood_per_container"]
        sellers: list[dict[str, Any]] = []
        buyers: list[dict[str, Any]] = []
        for agent in agents:
            berry_target = self._target_berries(agent)
            berry_deficit = max(0, berry_target - agent.berries)
            # Keep today's meal back; the rest can buy wood or a crate.
            berry_budget = max(0, agent.berries - food)
            hungry = agent.berries < food
            # Keep at most one crate's material for own use. A hungry agent
            # sells down to zero. A wood specialist sells at most one block a
            # day while it has food, but may sell two when it is starving.
            if agent.wood_skill > agent.berry_skill:
                reserve = 0
            else:
                reserve = (
                    0
                    if hungry
                    else min(agent.wood, wood_per_crate)
                )
            surplus = max(0, agent.wood - reserve)
            if agent.wood_skill > agent.berry_skill:
                surplus = min(surplus, 2 if hungry else 1)
            target_wood = self._target_wood(agent)
            need = max(
                0, min(target_wood, wood_per_crate) - agent.wood
            )
            listed_as_seller = False
            if surplus > 0 and berry_deficit > 0:
                sellers.append(
                    {
                        "agent": agent,
                        "qty": surplus,
                        "price": self._wood_seller_min(
                            agent, beliefs[agent.id]
                        ),
                    }
                )
                listed_as_seller = True
            # Do not let one agent be both a buyer and a seller of the same
            # good in the same tick. A villager that is already offering wood
            # is specialising on the supply side.
            if need > 0 and berry_budget > 0 and not listed_as_seller:
                buyers.append(
                    {
                        "agent": agent,
                        "qty": need,
                        "price": self._wood_buyer_max(
                            agent, beliefs[agent.id]
                        ),
                        "budget": berry_budget,
                    }
                )
        return sellers, buyers

    def _match_market(
        self,
        good: str,
        sellers: list[dict[str, Any]],
        buyers: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        # Price priority: the cheapest ask meets the highest bid first, with a
        # random tie-break. This lets a desperate seller reach the buyers who
        # can actually pay, instead of clearing at a random low bid.
        sellers = sorted(
            sellers,
            key=lambda entry: (entry["price"], self.random.random()),
        )
        buyers = sorted(
            buyers,
            key=lambda entry: (-entry["price"], self.random.random()),
        )
        seller_done = {id(entry): 0 for entry in sellers}
        buyer_done = {id(entry): 0 for entry in buyers}
        trades: list[dict[str, Any]] = []

        for seller in sellers:
            seller_agent = seller["agent"]
            for buyer in buyers:
                buyer_agent = buyer["agent"]
                if seller_agent.id == buyer_agent.id:
                    continue
                if seller["qty"] <= 0 or buyer["qty"] <= 0:
                    continue
                if buyer["budget"] <= 0:
                    continue
                if seller["price"] > buyer["price"]:
                    continue
                # Prices are fractional. Goods are whole, so the berry leg is
                # rounded to whole berries and the residual is an accepted
                # indivisibility cost.
                price = round(
                    (float(seller["price"]) + float(buyer["price"])) / 2.0,
                    10,
                )
                if not math.isfinite(price) or price <= 0:
                    continue
                affordable = int(math.floor(buyer["budget"] / price))
                if affordable <= 0 and buyer["budget"] >= 1:
                    # Rounding can make one whole berry enough for a unit even
                    # when the fractional price times one exceeds the budget.
                    affordable = 1
                qty = min(seller["qty"], buyer["qty"], affordable)
                qty, berries = self._whole_berry_trade(
                    qty, price, int(buyer["budget"])
                )
                if qty <= 0:
                    continue
                if good == "wood":
                    moved = self.ledger.move(
                        seller_agent, buyer_agent, "wood", qty, "trade"
                    )
                    if moved != qty:
                        qty = moved
                        qty, berries = self._whole_berry_trade(
                            qty, price, int(buyer["budget"])
                        )
                    if qty <= 0:
                        continue
                    self._last_trade_wood += qty
                    self.trade_log.append(
                        {
                            "iteration": self._current_iteration,
                            "good": "wood",
                            "seller": seller_agent.id,
                            "buyer": buyer_agent.id,
                            "berry_seller": buyer_agent.id,
                            "wood_seller": seller_agent.id,
                            "quantity": qty,
                            "wood": qty,
                            "berries": berries,
                            "price": price,
                        }
                    )
                else:
                    moved = self.ledger.move(
                        seller_agent, buyer_agent, "crates", qty, "trade"
                    )
                    if moved != qty:
                        qty = moved
                        qty, berries = self._whole_berry_trade(
                            qty, price, int(buyer["budget"])
                        )
                    if qty <= 0:
                        continue
                    self._last_crate_trades += qty
                    self._last_trade_crates += qty
                    self.trade_log.append(
                        {
                            "iteration": self._current_iteration,
                            "good": "crate",
                            "seller": seller_agent.id,
                            "buyer": buyer_agent.id,
                            "berry_seller": buyer_agent.id,
                            "crate_seller": seller_agent.id,
                            "quantity": qty,
                            "crates": qty,
                            "berries": berries,
                            "price": price,
                        }
                    )
                paid = self.ledger.move(
                    buyer_agent, seller_agent, "berries", berries, "trade"
                )
                if paid != berries:
                    # The plans already reserve the berries, so this is a
                    # defensive guard rather than an expected path.
                    berries = paid
                self._last_trade_berries += berries
                buyer["budget"] -= berries
                seller["qty"] -= qty
                buyer["qty"] -= qty
                seller_done[id(seller)] += qty
                buyer_done[id(buyer)] += qty
                trades.append(
                    {
                        "good": good,
                        "seller": seller_agent.id,
                        "buyer": buyer_agent.id,
                        "qty": qty,
                        "price": price,
                        "berries": berries,
                    }
                )
                if seller["qty"] <= 0:
                    break
                if buyer["qty"] <= 0:
                    continue

        failed_sellers = [
            seller for seller in sellers if seller_done[id(seller)] == 0
        ]
        failed_buyers = [
            buyer for buyer in buyers if buyer_done[id(buyer)] == 0
        ]
        return trades, failed_sellers, failed_buyers

    def _whole_berry_trade(
        self, qty: int, price: float, budget: int
    ) -> tuple[int, int]:
        """Return a whole-quantity trade whose whole-berry payment fits budget.

        The price is a fractional ratio. The berry leg is rounded to the
        nearest whole berry (at least one) and the quantity is reduced by a
        unit if rounding would breach the buyer's budget. This is where the
        model deliberately absorbs the indivisibility of goods.
        """

        remaining = int(qty)
        while remaining > 0:
            berries = int(math.floor(remaining * price + 0.5))
            berries = max(1, berries)
            if berries <= int(budget):
                return remaining, berries
            remaining -= 1
        return 0, 0

    def _process_trade(self) -> None:
        living = [agent for agent in self.agents if agent.is_alive]
        if len(living) < 2:
            return

        # Strict information lag: every reservation and trade this phase uses
        # the villager's belief at the start of the tick. Updates are applied
        # only after all markets have cleared.
        start_beliefs = {agent.id: agent.price_belief for agent in living}

        crate_sellers, crate_buyers = self._crate_market_plan(
            living, start_beliefs
        )
        crate_trades, crate_failed_sellers, crate_failed_buyers = (
            self._match_market("crate", crate_sellers, crate_buyers)
        )

        # Recompute the raw-wood market with post-crate holdings, but still
        # with start-of-tick beliefs.
        wood_sellers, wood_buyers = self._wood_market_plan(
            living, start_beliefs
        )
        wood_trades, wood_failed_sellers, wood_failed_buyers = (
            self._match_market("wood", wood_sellers, wood_buyers)
        )

        participants = {
            entry["agent"].id
            for entry in (
                crate_sellers
                + crate_buyers
                + wood_sellers
                + wood_buyers
            )
        }
        trades = crate_trades + wood_trades
        failed_sellers = crate_failed_sellers + wood_failed_sellers
        failed_buyers = crate_failed_buyers + wood_failed_buyers
        self._market_participants = len(participants)
        self._market_traders = len(
            {trade["seller"] for trade in trades}
            | {trade["buyer"] for trade in trades}
        )
        self._failed_trades = len(failed_sellers) + len(failed_buyers)

        self._apply_belief_updates(
            living,
            start_beliefs,
            trades,
            participants,
            wood_failed_sellers,
            wood_failed_buyers,
        )

    def _apply_belief_updates(
        self,
        living: Iterable[Agent],
        start_beliefs: dict[str, float],
        trades: list[dict[str, Any]],
        participants: set[str],
        failed_sellers: list[dict[str, Any]],
        failed_buyers: list[dict[str, Any]],
    ) -> None:
        settings = self.trade_settings
        trade_prices: dict[str, list[float]] = {}
        wood_prices: list[float] = []
        crate_prices: list[float] = []

        for trade in trades:
            if trade["good"] == "wood":
                price = float(trade["price"])
                wood_prices.append(price)
                for agent_id in (trade["seller"], trade["buyer"]):
                    trade_prices.setdefault(agent_id, []).append(price)
            else:
                # A crate price bundles wood, craft time, and storage. Do not
                # use it as a wood price; keep the two markets separate.
                crate_prices.append(float(trade["price"]))

        gossip_price = (
            float(np.mean(wood_prices)) if wood_prices else None
        )
        if gossip_price is not None:
            self.last_market_price = gossip_price
        if crate_prices:
            self.last_crate_price = float(np.mean(crate_prices))
        self.average_trade_price = self.last_market_price
        self._gossip_price = (
            gossip_price if gossip_price is not None else self.last_market_price
        )

        updates = dict(start_beliefs)
        for agent_id, prices in trade_prices.items():
            target = float(np.mean(prices))
            current = start_beliefs[agent_id]
            if target > current:
                step = 1.0
            elif target < current:
                step = -1.0
            else:
                step = 0.0
            updates[agent_id] = current + step
        if gossip_price is not None:
            for agent_id in participants:
                if agent_id in trade_prices:
                    continue
                current = start_beliefs[agent_id]
                if gossip_price > current:
                    step = settings["gossip_step"]
                elif gossip_price < current:
                    step = -settings["gossip_step"]
                else:
                    step = 0.0
                updates[agent_id] = current + step
        food = int(self.agent_settings["food_per_iteration"])
        for entry in failed_sellers:
            agent = entry["agent"]
            agent_id = agent.id
            current = updates.get(
                agent_id, start_beliefs[agent_id]
            )
            if current <= float(EPSILON):
                continue
            opportunity = self._effective_wood_opportunity(agent)
            if not math.isfinite(opportunity):
                # The seller cannot produce wood at all; its ask is already
                # non-finite and a belief change cannot make the order viable.
                continue
            if agent.berries >= food:
                wood_rate = self.effective_max_wood_per_day(
                    agent, self._current_wood_density
                )
                subsistence = food / max(0.5, wood_rate)
                raw = 0.2 * opportunity + 0.8 * current
                if raw <= subsistence:
                    # The ask is already at its subsistence floor; a lower
                    # belief would not change the order.
                    continue
            updates[agent_id] = current - 1.0
        for entry in failed_buyers:
            agent = entry["agent"]
            agent_id = agent.id
            current = updates.get(
                agent_id, start_beliefs[agent_id]
            )
            budget = max(0, agent.berries - food)
            if budget <= current:
                # The bid is already budget-capped; raising the belief cannot
                # change the price the buyer can actually offer.
                continue
            updates[agent_id] = current + 1.0

        for agent in living:
            if agent.id not in participants:
                continue
            belief = updates.get(agent.id, start_beliefs[agent.id])
            # Positivity is the only guard. The level is not clamped: the
            # market is allowed to discover any positive exchange ratio.
            agent.price_belief = max(float(EPSILON), float(belief))

    # ------------------------------------------------------------------
    # Consumption and spoilage
    # ------------------------------------------------------------------
    def _eat_all(self) -> None:
        settings = self.agent_settings
        food = settings["food_per_iteration"]
        for agent in self.agents:
            if not agent.is_alive:
                continue
            meal = min(agent.berries, food)
            self.ledger.consume(agent, "berries", meal)
            agent.meals_eaten = meal
            agent.unmet_food = max(0, food - meal)
            if meal >= food:
                agent.health = min(
                    settings["max_health"],
                    agent.health + settings["heal_per_meal"],
                )
            else:
                # A short or missed day costs one day of health, so a full
                # buffer always lasts MAX_HEALTH days without food.
                agent.health -= settings["starvation_rate"]
            if agent.health <= 0:
                agent.health = 0
                self._mark_dead(agent, "hunger")

    def _decay_and_spoil(self) -> None:
        settings = self.storage_settings
        for agent in self.agents:
            if agent.wood > 0 and settings["wood_decay_chance"] > 0:
                decayed = 0
                for _ in range(agent.wood):
                    if self.random.random() < settings["wood_decay_chance"]:
                        decayed += 1
                self.ledger.consume(agent, "wood", decayed)
                self._last_wood_decayed += decayed
            wear = settings["crate_wear_rate"]
            if agent.crates and wear > 0:
                capacity = (
                    agent.storage_capacity
                    if self.features["STORAGE"]
                    else 0
                )
                used = min(agent.berries, capacity)
                utilization = used / capacity if capacity > 0 else 0.0
                wear *= 1.0 + (
                    settings["crate_use_wear_factor"] * utilization
                )
                worn_crates = [
                    durability - wear for durability in agent.crates
                ]
                worn_count = sum(
                    1 for durability in worn_crates if durability <= 0
                )
                agent.crates = [
                    min(1.0, durability)
                    for durability in worn_crates
                    if durability > 0
                ]
                self._last_crates_worn += worn_count
            capacity = agent.storage_capacity if self.features["STORAGE"] else 0
            if not self.features["SPOILAGE"]:
                continue
            spoiled = max(0, agent.berries - capacity)
            self.ledger.consume(agent, "berries", spoiled)
            if settings["stored_spoil_rate"] > 0:
                stored = min(agent.berries, capacity)
                stored_spoiled = self._stochastic_round(
                    stored * settings["stored_spoil_rate"]
                )
                stored_spoiled = min(stored_spoiled, stored)
                self.ledger.consume(agent, "berries", stored_spoiled)
                spoiled += stored_spoiled
            self._last_berries_spoiled += spoiled
            if spoiled > 0:
                self.spoil_log.append(
                    {
                        "iteration": self._current_iteration,
                        "agent": agent.id,
                        "amount": spoiled,
                    }
                )
        for agent in self.agents:
            agent.berries = max(0, int(agent.berries))
            agent.wood = max(0, int(agent.wood))
            agent.health = min(
                agent.max_health, max(0, int(agent.health))
            )

    def _mark_dead(self, agent: Agent, cause: str) -> None:
        if agent.id in self._dead_logged:
            return
        self._dead_logged.add(agent.id)
        self.death_log.append(
            {
                "iteration": self._current_iteration,
                "agent": agent.id,
                "cause": cause,
            }
        )

    # ------------------------------------------------------------------
    # Metrics and snapshots
    # ------------------------------------------------------------------
    def _metrics(self, iteration: int) -> dict[str, Any]:
        living = [agent for agent in self.agents if agent.is_alive]
        alive = len(living)
        price = self.last_market_price
        total_berries = int(sum(agent.berries for agent in self.agents))
        total_wood = int(sum(agent.wood for agent in self.agents))
        total_crates = int(sum(agent.crate_count for agent in self.agents))
        crate_durabilities = [
            agent.crate_durability
            for agent in living
            if agent.crate_count > 0
        ]
        crate_durability = (
            float(np.mean(crate_durabilities))
            if crate_durabilities
            else 0.0
        )
        storage_capacity = int(
            sum(agent.storage_capacity for agent in living)
        )
        storage_used = int(
            sum(
                min(agent.berries, agent.storage_capacity)
                for agent in living
            )
        )
        potential_value = float(
            sum(
                agent.planned_berries
                + price * agent.planned_wood
                for agent in living
            )
        )
        actual_value = float(
            sum(
                agent.gathered_berries + price * agent.gathered_wood
                for agent in living
            )
        )
        specialisation_values = []
        for agent in living:
            shares = np.array(
                [
                    agent.labor_berries,
                    agent.labor_wood,
                    agent.labor_craft,
                ],
                dtype=float,
            )
            total = float(shares.sum())
            if total <= 0:
                specialisation_values.append(0.0)
                continue
            shares /= total
            # Normalised Herfindahl index over berry, wood, and craft work:
            # 0 means evenly split across all three activities, 1 means the
            # whole day was one activity.
            concentration = float(np.square(shares).sum())
            specialisation_values.append(
                max(0.0, min(1.0, (concentration - 1.0 / 3.0) / (2.0 / 3.0)))
            )
        specialisation = (
            float(np.mean(specialisation_values))
            if specialisation_values
            else 0.0
        )
        if alive == 0 and self.extinction_iteration is None:
            self.extinction_iteration = iteration
        metrics = {
            "iteration": iteration,
            "alive": alive,
            "mean_health": (
                float(np.mean([agent.health for agent in living]))
                if living
                else 0.0
            ),
            "total_berries": total_berries,
            "total_wood": total_wood,
            "total_crates": total_crates,
            "crate_durability": crate_durability,
            "crates_crafted": self._last_crates_crafted,
            "crates_worn": self._last_crates_worn,
            "wood_crafted": self._last_wood_crafted,
            "commons_berries": self.commons_berries,
            "commons_wood": self.commons_wood,
            "total_berries_all": total_berries + self.commons_berries,
            "total_wood_all": total_wood + self.commons_wood,
            "gini_berries": MetricsCalculator.calculate_gini(
                [agent.berries for agent in self.agents]
            ),
            "gini_wood": MetricsCalculator.calculate_gini(
                [agent.wood for agent in self.agents]
            ),
            "barter_price": price,
            "last_crate_price": self.last_crate_price,
            "average_trade_price": self.average_trade_price,
            "gossip_price": self._gossip_price,
            "mean_price_belief": (
                float(np.mean([agent.price_belief for agent in living]))
                if living
                else 0.0
            ),
            "market_participants": self._market_participants,
            "market_traders": self._market_traders,
            "failed_trades": self._failed_trades,
            "mean_berry_skill": (
                float(np.mean([agent.berry_skill for agent in living]))
                if living
                else 0.0
            ),
            "mean_wood_skill": (
                float(np.mean([agent.wood_skill for agent in living]))
                if living
                else 0.0
            ),
            "mean_labor_berries": (
                float(np.mean([agent.labor_berries for agent in living]))
                if living
                else 0.0
            ),
            "mean_labor_craft": (
                float(np.mean([agent.labor_craft for agent in living]))
                if living
                else 0.0
            ),
            "specialisation": specialisation,
            "ppf_efficiency": (
                min(1.0, actual_value / potential_value)
                if potential_value > 0
                else 0.0
            ),
            "berries_gathered": self._last_gathered_berries,
            "wood_gathered": self._last_gathered_wood,
            "wood_blocked": self._last_blocked_wood,
            "berries_eaten": int(
                sum(agent.meals_eaten for agent in self.agents)
            ),
            "unmet_food": int(sum(agent.unmet_food for agent in self.agents)),
            "berries_spoiled": self._last_berries_spoiled,
            "wood_decayed": self._last_wood_decayed,
            "commons_regenerated_berries": self._last_regenerated_berries,
            "commons_regenerated_wood": self._last_regenerated_wood,
            "commons_waste_berries": self._last_waste_berries,
            "commons_waste_wood": self._last_waste_wood,
            "storage_capacity": storage_capacity,
            "storage_used": storage_used,
            "storage_utilization": (
                storage_used / storage_capacity if storage_capacity > 0 else 0.0
            ),
            "berries_commons_pct": self._current_berries_density,
            "wood_commons_pct": self._current_wood_density,
            "season": self._season["index"],
            "season_name": self._season["name"],
            "berry_regen_factor": self._season["berry_regen"],
            "wood_regen_factor": self._season["wood_regen"],
            "trade_berries": self._last_trade_berries,
            "trade_wood": self._last_trade_wood,
            "crate_trades": self._last_crate_trades,
            "trade_crates": self._last_trade_crates,
            "gifts": self._last_gifts,
            "gift_berries": self._last_gift_berries,
            "extinction_iteration": self.extinction_iteration,
        }
        metrics.update(self.feature_registry.metrics(iteration))
        return metrics

    def _record_agent_snapshot(self, iteration: int) -> None:
        for agent in self.agents:
            row = agent.to_dict()
            row["effective_berries_per_day"] = (
                self.effective_max_berries_per_day(
                    agent, self._current_berries_density
                )
            )
            row["effective_wood_per_day"] = self.effective_max_wood_per_day(
                agent, self._current_wood_density
            )
            row["debt_wood"] = self.obligations.outstanding(agent.id, "wood")
            row["debt_berries"] = self.obligations.outstanding(
                agent.id, "berries"
            )
            row["debt_crates"] = self.obligations.outstanding(
                agent.id, "crates"
            )
            row["credit_out_wood"] = self.obligations.receivable(
                agent.id, "wood"
            )
            row["credit_out_berries"] = self.obligations.receivable(
                agent.id, "berries"
            )
            row["credit_out_crates"] = self.obligations.receivable(
                agent.id, "crates"
            )
            self.agent_history.append({"iteration": iteration, **row})

    def to_frame(self) -> pd.DataFrame:
        columns = [
            "id",
            "health",
            "max_health",
            "berries",
            "wood",
            "storage_capacity",
            "berry_skill",
            "wood_skill",
            "specialisation_z",
            "price_belief",
            "trust",
            "debt_wood",
            "debt_berries",
            "debt_crates",
            "credit_out_wood",
            "credit_out_berries",
            "credit_out_crates",
            "max_berries_per_day",
            "max_wood_per_day",
            "labor_berries",
            "labor_wood",
            "wood_progress",
            "planned_berries",
            "planned_wood",
            "gathered_berries",
            "gathered_wood",
            "meals_eaten",
            "unmet_food",
            "post_trade_berries",
            "post_trade_wood",
            "alive",
        ]
        if not self.agents:
            return pd.DataFrame(columns=columns).set_index("id")
        rows = []
        for agent in self.agents:
            row = agent.to_dict()
            row["debt_wood"] = self.obligations.outstanding(agent.id, "wood")
            row["debt_berries"] = self.obligations.outstanding(
                agent.id, "berries"
            )
            row["debt_crates"] = self.obligations.outstanding(
                agent.id, "crates"
            )
            row["credit_out_wood"] = self.obligations.receivable(
                agent.id, "wood"
            )
            row["credit_out_berries"] = self.obligations.receivable(
                agent.id, "berries"
            )
            row["credit_out_crates"] = self.obligations.receivable(
                agent.id, "crates"
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns).set_index("id")

    def agent_history_frame(self) -> pd.DataFrame:
        columns = [
            "iteration",
            "id",
            "health",
            "max_health",
            "berries",
            "wood",
            "storage_capacity",
            "berry_skill",
            "wood_skill",
            "specialisation_z",
            "price_belief",
            "trust",
            "debt_wood",
            "debt_berries",
            "debt_crates",
            "credit_out_wood",
            "credit_out_berries",
            "credit_out_crates",
            "max_berries_per_day",
            "max_wood_per_day",
            "effective_berries_per_day",
            "effective_wood_per_day",
            "labor_berries",
            "labor_wood",
            "wood_progress",
            "planned_berries",
            "planned_wood",
            "gathered_berries",
            "gathered_wood",
            "meals_eaten",
            "unmet_food",
            "post_trade_berries",
            "post_trade_wood",
            "alive",
        ]
        return pd.DataFrame(self.agent_history, columns=columns)
