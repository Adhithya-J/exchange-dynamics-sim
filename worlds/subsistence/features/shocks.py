"""Temporary commons-regeneration shocks."""

from __future__ import annotations

from typing import Any

from .base import FeatureLayer

DEFAULT_SHOCK_SETTINGS: dict[str, Any] = {
    "PROBABILITY": 0.04,
    "DURATION_RANGE": (3, 8),
    "BERRY_REGEN_FACTOR": 0.40,
    "WOOD_REGEN_FACTOR": 0.60,
}


class ShockLayer(FeatureLayer):
    name = "SHOCKS"

    def setup(self) -> None:
        configured = self.world.config.get("SHOCKS") or {}
        merged = {**DEFAULT_SHOCK_SETTINGS, **configured}
        try:
            probability = float(merged["PROBABILITY"])
            duration_range = self.world._validated_int_range(
                merged["DURATION_RANGE"], "SHOCKS.DURATION_RANGE"
            )
            berry_factor = float(merged["BERRY_REGEN_FACTOR"])
            wood_factor = float(merged["WOOD_REGEN_FACTOR"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SHOCKS settings must be numeric; durations must be whole days"
            ) from exc
        if not 0 <= probability <= 1:
            raise ValueError("SHOCKS.PROBABILITY must be between 0 and 1")
        if duration_range[0] < 1:
            raise ValueError("SHOCKS.DURATION_RANGE must start at 1 or more")
        if not 0 <= berry_factor <= 1 or not 0 <= wood_factor <= 1:
            raise ValueError("SHOCKS regeneration factors must be in [0, 1]")
        self.settings = {
            "probability": probability,
            "duration_range": duration_range,
            "berry_factor": berry_factor,
            "wood_factor": wood_factor,
        }
        self.world.shock_settings = self.settings
        self.world.shock_layer = self
        self.active_until = 0
        self._last_started = 0

    def before_regenerate(self) -> None:
        world = self.world
        iteration = world._current_iteration
        if iteration > self.active_until and world.random.random() < float(
            self.settings["probability"]
        ):
            duration = world.random.randint(*self.settings["duration_range"])
            self.active_until = iteration + duration - 1
            self._last_started = iteration
            world.shock_log.append(
                {
                    "iteration": iteration,
                    "type": "drought",
                    "duration": duration,
                    "berry_factor": float(self.settings["berry_factor"]),
                    "wood_factor": float(self.settings["wood_factor"]),
                }
            )
        active = iteration <= self.active_until
        world._berry_regen_factor = (
            float(self.settings["berry_factor"]) if active else 1.0
        )
        world._wood_regen_factor = (
            float(self.settings["wood_factor"]) if active else 1.0
        )

    def metrics(self, iteration: int) -> dict[str, Any]:
        started = sum(
            1 for entry in self.world.shock_log if entry["iteration"] == iteration
        )
        return {
            "shock_active": 1 if iteration <= self.active_until else 0,
            "shocks": started,
        }

    def default_metrics(self) -> dict[str, Any]:
        return {"shock_active": 0, "shocks": 0}
