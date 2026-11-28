"""A goods-only subsistence world with whole resources and finite days.

Agents start with no goods. Each iteration is a finite working day: time is
spent picking whole berries or chopping whole wood from a shared, depletable
commons. Berries are food and spoil outside storage; wood is slow to produce,
slow to regrow, and used to build storage. Health is a whole-meal buffer.
There is no money anywhere in the model.

The optional layers are deliberately small:

- ``FEATURES.TRADE`` enables whole-unit barter at a berries-per-wood ratio.
- ``FEATURES.STORAGE`` lets wood provide storage capacity.
- ``FEATURES.SPOILAGE`` makes unprotected berries spoil.
- ``FEATURES.GIFTING`` enables need-based whole-berry gifts.

Trust, reciprocity, and shocks are reserved for shared utilities and are not
active in this world yet.
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
from utils.social import GiftSettings, process_berry_gifts, validate_gift_settings

CONFIG: dict[str, dict[str, Any]] = load_config(
    Path(__file__).resolve().parent / "config.yaml"
)
config = CONFIG

EPSILON = 1e-9

DEFAULT_FEATURES: dict[str, bool] = {
    "GIFTING": False,
    "TRADE": True,
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
    "STARTING_BERRIES": 0,
    "STARTING_WOOD": 0,
    "RESERVE_DAYS": 4,
    "TRADE_SURPLUS": 1,
    "MAX_WOOD_TARGET": 20,
}

DEFAULT_RESOURCE_SETTINGS: dict[str, float] = {
    "INITIAL": 0,
    "CAPACITY": 0,
    "REGENERATION": 0.0,
    "EXTRACTION_EFFICIENCY": 1.0,
}

DEFAULT_STORAGE_SETTINGS: dict[str, float] = {
    "WOOD_PER_CONTAINER": 8,
    "BERRIES_PER_CONTAINER": 10,
    "WOOD_DECAY_CHANCE": 0.03,
    "STORED_SPOIL_RATE": 0.0,
}

DEFAULT_TRADE_SETTINGS: dict[str, float] = {
    "INITIAL_PRICE": 5,
    "PRICE_ADJUSTMENT": 0.1,
    "MIN_PRICE": 1,
    "MAX_PRICE": 6,
}

DEFAULT_GIFTING_SETTINGS: dict[str, Any] = {
    "PROBABILITY": 0.25,
    "AMOUNT_RANGE": (1, 3),
    "MIN_RESERVE": 2,
}

UNSUPPORTED_FEATURES = ("TRUST", "RECIPROCITY", "SHOCKS")


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
        # Fractional wood work completed toward the next whole block. It
        # persists across iterations so slow chopping is not lost each day.
        self.wood_progress = 0.0

        # Per-iteration observations, reset before each gathering step.
        self.labor_berries = 0.0
        self.labor_wood = 0.0
        self.planned_berries = 0.0
        self.planned_wood = 0.0
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
    def storage_capacity(self) -> int:
        if self.storage_wood_per_container <= 0:
            return 0
        return (
            self.wood * self.storage_berries_per_container
        ) // self.storage_wood_per_container

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
            "wood_progress": self.wood_progress,
            "planned_berries": self.planned_berries,
            "planned_wood": self.planned_wood,
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

        self.features = self._feature_settings()
        self.agent_settings = self._agent_settings()
        self.commons_settings = {
            name: self._resource_settings(name) for name in ("berries", "wood")
        }
        self.storage_settings = self._storage_settings()
        self.trade_settings = self._trade_settings()
        self.gifting_settings = self._gifting_settings()

        self.commons_berries = 0
        self.commons_wood = 0
        self.initial_commons_berries = 0
        self.initial_commons_wood = 0
        self.price = float(self.trade_settings["initial_price"])
        self.trade_price = int(self.trade_settings["initial_price"])
        self.initial_price = self.trade_price
        self._last_trade_price = self.trade_price
        self._current_iteration = 0
        self.extinction_iteration: int | None = None

        self.gather_log: list[dict[str, Any]] = []
        self.trade_log: list[dict[str, Any]] = []
        self.gift_log: list[dict[str, Any]] = []
        self.spoil_log: list[dict[str, Any]] = []
        self.death_log: list[dict[str, Any]] = []
        self._dead_logged: set[str] = set()

        self._last_gathered_berries = 0
        self._last_gathered_wood = 0
        self._last_waste_berries = 0
        self._last_waste_wood = 0
        self._last_regenerated_berries = 0
        self._last_regenerated_wood = 0
        self._last_berries_spoiled = 0
        self._last_wood_decayed = 0
        self._last_trade_berries = 0
        self._last_trade_wood = 0
        self._last_gifts = 0
        self._last_gift_berries = 0

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

    def _feature_settings(self) -> dict[str, bool]:
        configured = self.config.get("FEATURES") or {}
        merged = {**DEFAULT_FEATURES, **configured}
        features = {key: bool(value) for key, value in merged.items()}
        for key in UNSUPPORTED_FEATURES:
            if features.get(key):
                raise ValueError(
                    f"FEATURES.{key} is not active in the subsistence world yet; "
                    "it is reserved for the shared social utilities"
                )
        return features

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
            reserve_days = int(merged["RESERVE_DAYS"])
            trade_surplus = int(merged["TRADE_SURPLUS"])
            max_wood_target = int(merged["MAX_WOOD_TARGET"])
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
        if starting_berries < 0 or starting_wood < 0:
            raise ValueError("AGENTS starting goods must not be negative")
        if reserve_days < 0:
            raise ValueError("AGENTS.RESERVE_DAYS must not be negative")
        if trade_surplus < 0:
            raise ValueError("AGENTS.TRADE_SURPLUS must not be negative")
        if max_wood_target < 0:
            raise ValueError("AGENTS.MAX_WOOD_TARGET must not be negative")
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
            "reserve_days": reserve_days,
            "trade_surplus": trade_surplus,
            "max_wood_target": max_wood_target,
        }

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
        return {
            "initial": initial,
            "capacity": capacity,
            "regeneration": regeneration,
            "extraction_efficiency": efficiency,
        }

    def _storage_settings(self) -> dict[str, float]:
        configured = self.config.get("STORAGE") or {}
        merged = {**DEFAULT_STORAGE_SETTINGS, **configured}
        try:
            wood_per_container = int(merged["WOOD_PER_CONTAINER"])
            berries_per_container = int(merged["BERRIES_PER_CONTAINER"])
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
        if not 0 <= wood_decay <= 1:
            raise ValueError("STORAGE.WOOD_DECAY_CHANCE must be between 0 and 1")
        if not 0 <= stored_spoil <= 1:
            raise ValueError("STORAGE.STORED_SPOIL_RATE must be between 0 and 1")
        return {
            "wood_per_container": wood_per_container,
            "berries_per_container": berries_per_container,
            "wood_decay_chance": wood_decay,
            "stored_spoil_rate": stored_spoil,
        }

    def _trade_settings(self) -> dict[str, float]:
        configured = self.config.get("TRADE") or {}
        merged = {**DEFAULT_TRADE_SETTINGS, **configured}
        try:
            initial_price = int(merged["INITIAL_PRICE"])
            adjustment = float(merged["PRICE_ADJUSTMENT"])
            min_price = int(merged["MIN_PRICE"])
            max_price = int(merged["MAX_PRICE"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "TRADE prices must be whole numbers and PRICE_ADJUSTMENT numeric"
            ) from exc
        if initial_price <= 0:
            raise ValueError("TRADE.INITIAL_PRICE must be positive")
        if not 0 <= adjustment <= 1:
            raise ValueError("TRADE.PRICE_ADJUSTMENT must be between 0 and 1")
        if min_price <= 0 or max_price < min_price:
            raise ValueError(
                "TRADE.MIN_PRICE must be positive and not exceed MAX_PRICE"
            )
        if not min_price <= initial_price <= max_price:
            raise ValueError(
                "TRADE.INITIAL_PRICE must be within MIN_PRICE and MAX_PRICE"
            )
        return {
            "initial_price": initial_price,
            "price_adjustment": adjustment,
            "min_price": min_price,
            "max_price": max_price,
        }

    def _gifting_settings(self) -> GiftSettings:
        configured = self.config.get("GIFTING") or {}
        merged = {**DEFAULT_GIFTING_SETTINGS, **configured}
        try:
            probability = float(merged["PROBABILITY"])
            amount_range = self._validated_int_range(
                merged["AMOUNT_RANGE"], "GIFTING.AMOUNT_RANGE"
            )
            min_reserve = int(merged["MIN_RESERVE"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "GIFTING settings must be numeric; amounts and reserves must be "
                "whole berries"
            ) from exc
        settings = GiftSettings(
            probability=probability,
            amount_range=amount_range,
            min_reserve=min_reserve,
        )
        validate_gift_settings(settings)
        return settings

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize_agents(self) -> list[Agent]:
        """Create the population and reset the commons and logs."""

        simulation = self.config["SIMULATION"]
        count = int(simulation["AGENTS"])
        if count < 0:
            raise ValueError("SIMULATION.AGENTS must not be negative")
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
                    specialisation_z=z,
                )
            )
        self._agent_index = {agent.id: agent for agent in self.agents}
        self.metrics_history = []
        self.agent_history = []
        self._current_iteration = 0
        self.extinction_iteration = None
        self.gather_log = []
        self.trade_log = []
        self.gift_log = []
        self.spoil_log = []
        self.death_log = []
        self._dead_logged = set()
        berries = self.commons_settings["berries"]
        wood = self.commons_settings["wood"]
        self.initial_commons_berries = berries["initial"]
        self.initial_commons_wood = wood["initial"]
        self.commons_berries = berries["initial"]
        self.commons_wood = wood["initial"]
        self.price = float(self.trade_settings["initial_price"])
        self.trade_price = int(self.trade_settings["initial_price"])
        self.initial_price = self.trade_price
        self._last_trade_price = self.trade_price
        self._reset_iteration_state()
        self._record_agent_snapshot(iteration=0)
        return self.agents

    def _reset_iteration_state(self) -> None:
        for agent in self.agents:
            agent.labor_berries = 0.0
            agent.labor_wood = 0.0
            agent.planned_berries = 0.0
            agent.planned_wood = 0.0
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
        self._last_regenerated_berries = 0
        self._last_regenerated_wood = 0
        self._last_berries_spoiled = 0
        self._last_wood_decayed = 0
        self._last_trade_berries = 0
        self._last_trade_wood = 0
        self._last_trade_price = self.trade_price
        self._last_gifts = 0
        self._last_gift_berries = 0

    def run_iteration(self) -> dict[str, Any]:
        """Run one finite day: gather, exchange, eat, then spoil."""

        if not self.agents:
            self.initialize_agents()
        iteration = len(self.metrics_history) + 1
        self._current_iteration = iteration
        self._reset_iteration_state()

        self._regenerate_commons()
        self._gather_all()
        if self.features["GIFTING"]:
            self._process_gifts()
        if self.features["TRADE"]:
            self._process_trade()
        for agent in self.agents:
            if agent.is_alive:
                agent.post_trade_berries = agent.berries
                agent.post_trade_wood = agent.wood
        self._eat_all()
        self._decay_and_spoil()

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
        for name in ("berries", "wood"):
            settings = self.commons_settings[name]
            attribute = f"commons_{name}"
            current = getattr(self, attribute)
            amount = self._stochastic_round(settings["regeneration"])
            amount = min(amount, max(0, settings["capacity"] - current))
            setattr(self, attribute, current + amount)
            if name == "berries":
                self._last_regenerated_berries = amount
            else:
                self._last_regenerated_wood = amount

    def _target_wood(self, agent: Agent) -> int:
        if not self.features["STORAGE"] or not self.features["SPOILAGE"]:
            return 0
        settings = self.storage_settings
        berries_per_container = settings["berries_per_container"]
        wood_per_container = settings["wood_per_container"]
        if berries_per_container <= 0 or wood_per_container <= 0:
            return 0
        max_storage = (
            self.agent_settings["max_wood_target"] * berries_per_container
        ) // wood_per_container
        reserve = min(
            self.agent_settings["reserve_days"]
            * self.agent_settings["food_per_iteration"],
            max_storage,
        )
        target = math.ceil(reserve * wood_per_container / berries_per_container)
        return max(0, min(self.agent_settings["max_wood_target"], target))

    def _target_berries(self, agent: Agent) -> int:
        food = self.agent_settings["food_per_iteration"]
        reserve = self.agent_settings["reserve_days"] * food
        if not self.features["SPOILAGE"]:
            return max(food, reserve)
        capacity = agent.storage_capacity if self.features["STORAGE"] else 0
        return max(food, min(capacity, reserve))

    def _allocate_labour(self, agent: Agent) -> None:
        """Spend the finite day between berries and wood.

        With trade, specialise where one minute of work is worth more at the
        current exchange ratio. Without trade, aim for self-sufficiency:
        today's meal, then wood for storage, then a berry reserve.
        """

        settings = self.agent_settings
        food = settings["food_per_iteration"]
        remaining = agent.time_per_iteration
        berry_minutes = (
            agent.berry_time_cost / agent.berry_skill
            if agent.berry_skill > 0
            else float("inf")
        )
        wood_minutes = (
            agent.wood_time_cost / agent.wood_skill
            if agent.wood_skill > 0
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

        # With trade, pick the activity with the higher market value per
        # minute. A wood specialist chops first, finishing the next whole
        # block, then gathers food with any remaining time. That lets a
        # low-berry villager produce a tradable block instead of spending the
        # whole day failing to gather a full ration. A berry specialist covers
        # food and its reserve, then produces a small surplus to buy wood.
        if self.features["TRADE"]:
            target_berries = self._target_berries(agent)
            target_wood = self._target_wood(agent)
            berry_value_per_minute = (
                agent.berry_skill / agent.berry_time_cost
                if agent.berry_time_cost > 0
                else 0.0
            )
            wood_value_per_minute = (
                agent.wood_skill * self.trade_price / agent.wood_time_cost
                if agent.wood_time_cost > 0
                else 0.0
            )
            if not (self.features["STORAGE"] and self.features["SPOILAGE"]):
                # Wood has no preservation value in these control settings.
                wood_value_per_minute = 0.0

            if wood_value_per_minute > berry_value_per_minute:
                wood_cap = target_wood + self.agent_settings["trade_surplus"]
                if agent.wood < wood_cap and math.isfinite(wood_minutes):
                    needed = max(0.0, 1.0 - agent.wood_progress)
                    minutes_for_block = min(remaining, needed * wood_minutes)
                    if minutes_for_block > 0:
                        spend_wood(minutes_for_block / wood_minutes)
                if remaining > 0:
                    spend_berries(max(0, food - agent.berries))
                    spend_berries(
                        max(
                            0.0,
                            target_berries
                            - agent.berries
                            - agent.planned_berries,
                        )
                    )
            else:
                spend_berries(max(0, food - agent.berries))
                spend_berries(
                    max(
                        0.0,
                        target_berries - agent.berries - agent.planned_berries,
                    )
                )
                wood_deficit = max(0, target_wood - agent.wood)
                purchase_target = 1 if wood_deficit > 0 else 0
                berries_for_wood = min(
                    purchase_target * self.trade_price, 8
                )
                berry_cap = (
                    target_berries
                    + berries_for_wood
                    + self.agent_settings["trade_surplus"]
                )
                spend_berries(
                    max(
                        0.0,
                        berry_cap - agent.berries - agent.planned_berries,
                    )
                )
            return

        # Without trade, aim for self-sufficiency.
        spend_berries(max(0, food - agent.berries))

        if self.features["STORAGE"]:
            spend_wood(max(0, self._target_wood(agent) - agent.wood))

        if self.features["STORAGE"] and self.features["SPOILAGE"]:
            target = self._target_berries(agent)
            spend_berries(
                max(0.0, target - agent.berries - agent.planned_berries)
            )

        if remaining > 0 and not self.features["SPOILAGE"]:
            if math.isfinite(berry_minutes):
                spend_berries(remaining / berry_minutes)

    def _extract(self, agent: Agent, resource: str, requested: float) -> int:
        settings = self.commons_settings[resource]
        attribute = f"commons_{resource}"
        stock = int(getattr(self, attribute))
        efficiency = settings["extraction_efficiency"]
        if resource == "wood":
            # Chopping is slow, so carry the fractional part of a block into
            # the next day. Whole blocks are still the only wood ever stored.
            total = agent.wood_progress + max(0.0, requested)
            requested_whole = int(math.floor(total + EPSILON))
            agent.wood_progress = max(0.0, total - requested_whole)
        else:
            requested_whole = self._stochastic_round(requested)
        if requested_whole <= 0 or stock <= 0:
            return 0
        usable_cap = int(stock * efficiency)
        usable = min(requested_whole, usable_cap)
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
        gifts = process_berry_gifts(
            self.agents,
            settings=self.gifting_settings,
            random=self.random,
        )
        for gift in gifts:
            self.gift_log.append(
                {
                    "iteration": self._current_iteration,
                    "giver": gift["giver"],
                    "receiver": gift["receiver"],
                    "amount": gift["amount"],
                }
            )
            self._last_gifts += 1
            self._last_gift_berries += gift["amount"]

    def _trade_offers(
        self, agents: Iterable[Agent]
    ) -> tuple[dict[str, int], dict[str, int]]:
        price = self.trade_price
        if price <= 0:
            return {}, {}
        food = self.agent_settings["food_per_iteration"]
        berry_offers: dict[str, int] = {}
        wood_offers: dict[str, int] = {}
        for agent in agents:
            berry_target = self._target_berries(agent)
            wood_target = self._target_wood(agent)
            if agent.berries < food:
                # A hungry agent will sell storage wood before it starves.
                wood_target = 0
            berry_surplus = max(0, agent.berries - berry_target)
            wood_surplus = max(0, agent.wood - wood_target)
            berry_deficit = max(0, berry_target - agent.berries)
            wood_deficit = max(0, wood_target - agent.wood)
            if berry_surplus > 0 and wood_deficit > 0:
                wood_wanted = min(wood_deficit, berry_surplus // price)
                if wood_wanted > 0:
                    berry_offers[agent.id] = wood_wanted * price
            elif wood_surplus > 0 and berry_deficit > 0:
                wood_wanted = min(
                    wood_surplus, math.ceil(berry_deficit / price)
                )
                if wood_wanted > 0:
                    wood_offers[agent.id] = wood_wanted
            elif agent.wood > 0 and berry_deficit > 0:
                # A wood-rich agent below its own wood target will still sell
                # storage wood to cover a berry deficit, especially food.
                wood_wanted = min(
                    agent.wood, math.ceil(berry_deficit / price)
                )
                if wood_wanted > 0:
                    wood_offers[agent.id] = wood_wanted
        return berry_offers, wood_offers

    def _process_trade(self) -> None:
        living = [agent for agent in self.agents if agent.is_alive]
        if len(living) < 2:
            return
        price = self.trade_price
        berry_offers, wood_offers = self._trade_offers(living)
        total_berry_supply = int(sum(berry_offers.values()))
        total_wood_supply = int(sum(wood_offers.values()))
        berry_demand = total_wood_supply * price

        berry_traded = min(total_berry_supply, berry_demand)
        if berry_traded <= 0:
            self._update_barter_price(price, total_berry_supply, berry_demand)
            return

        berry_left = {agent_id: amount for agent_id, amount in berry_offers.items()}
        wood_left = {agent_id: amount for agent_id, amount in wood_offers.items()}
        berry_sellers = [self._agent_index[agent_id] for agent_id in berry_left]
        wood_sellers = [self._agent_index[agent_id] for agent_id in wood_left]
        self.random.shuffle(berry_sellers)
        self.random.shuffle(wood_sellers)
        berry_iterator = iter(
            [agent for agent in berry_sellers if berry_left[agent.id] > 0]
        )
        wood_iterator = iter(
            [agent for agent in wood_sellers if wood_left[agent.id] > 0]
        )
        berry_seller = next(berry_iterator, None)
        wood_seller = next(wood_iterator, None)

        while berry_seller is not None and wood_seller is not None:
            if berry_left[berry_seller.id] < price:
                berry_seller = next(berry_iterator, None)
                continue
            if wood_left[wood_seller.id] <= 0:
                wood_seller = next(wood_iterator, None)
                continue
            wood_trade = min(
                berry_left[berry_seller.id] // price,
                wood_left[wood_seller.id],
            )
            if wood_trade <= 0:
                break
            berries = wood_trade * price
            berry_seller.berries -= berries
            wood_seller.berries += berries
            wood_seller.wood -= wood_trade
            berry_seller.wood += wood_trade
            berry_left[berry_seller.id] -= berries
            wood_left[wood_seller.id] -= wood_trade
            self._last_trade_berries += berries
            self._last_trade_wood += wood_trade
            self.trade_log.append(
                {
                    "iteration": self._current_iteration,
                    "berry_seller": berry_seller.id,
                    "wood_seller": wood_seller.id,
                    "berries": berries,
                    "wood": wood_trade,
                    "price": price,
                }
            )
            if berry_left[berry_seller.id] < price:
                berry_seller = next(berry_iterator, None)
            if wood_left[wood_seller.id] <= 0:
                wood_seller = next(wood_iterator, None)

        self._update_barter_price(price, total_berry_supply, berry_demand)

    def _update_barter_price(
        self,
        price: int,
        berry_supply: int,
        berry_demand: int,
    ) -> None:
        """Set next iteration's whole-berry price from this iteration's offers.

        ``price`` is berries per wood. Scarce berries raise the value of
        berries relative to wood, so one wood buys fewer berries: the ratio
        falls when berry demand exceeds supply.
        """

        if berry_supply <= 0 and berry_demand <= 0:
            return
        denominator = max(berry_supply, berry_demand)
        imbalance = (berry_demand - berry_supply) / denominator
        settings = self.trade_settings
        self.price = float(
            np.clip(
                self.price * (1.0 - settings["price_adjustment"] * imbalance),
                settings["min_price"],
                settings["max_price"],
            )
        )
        self.trade_price = int(
            np.clip(
                round(self.price),
                settings["min_price"],
                settings["max_price"],
            )
        )

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
            agent.berries -= meal
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
                agent.wood -= decayed
                self._last_wood_decayed += decayed
            capacity = agent.storage_capacity if self.features["STORAGE"] else 0
            if not self.features["SPOILAGE"]:
                continue
            spoiled = max(0, agent.berries - capacity)
            agent.berries -= spoiled
            if settings["stored_spoil_rate"] > 0:
                stored = min(agent.berries, capacity)
                stored_spoiled = self._stochastic_round(
                    stored * settings["stored_spoil_rate"]
                )
                stored_spoiled = min(stored_spoiled, stored)
                agent.berries -= stored_spoiled
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
        price = self._last_trade_price
        total_berries = int(sum(agent.berries for agent in self.agents))
        total_wood = int(sum(agent.wood for agent in self.agents))
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
        specialisation = float(
            np.mean(
                [
                    abs(agent.labor_berries - 0.5) * 2.0
                    for agent in living
                ]
            )
            if living
            else 0.0
        )
        if alive == 0 and self.extinction_iteration is None:
            self.extinction_iteration = iteration
        return {
            "iteration": iteration,
            "alive": alive,
            "mean_health": (
                float(np.mean([agent.health for agent in living]))
                if living
                else 0.0
            ),
            "total_berries": total_berries,
            "total_wood": total_wood,
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
            "specialisation": specialisation,
            "ppf_efficiency": (
                min(1.0, actual_value / potential_value)
                if potential_value > 0
                else 0.0
            ),
            "berries_gathered": self._last_gathered_berries,
            "wood_gathered": self._last_gathered_wood,
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
            "trade_berries": self._last_trade_berries,
            "trade_wood": self._last_trade_wood,
            "gifts": self._last_gifts,
            "gift_berries": self._last_gift_berries,
            "extinction_iteration": self.extinction_iteration,
        }

    def _record_agent_snapshot(self, iteration: int) -> None:
        for agent in self.agents:
            self.agent_history.append(
                {
                    "iteration": iteration,
                    **agent.to_dict(),
                }
            )

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
        return pd.DataFrame([agent.to_dict() for agent in self.agents]).set_index("id")

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
        return pd.DataFrame(self.agent_history, columns=columns)
