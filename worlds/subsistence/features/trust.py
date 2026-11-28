"""Public credit reputation."""

from __future__ import annotations

from typing import Any

from .base import FeatureLayer

DEFAULT_TRUST_SETTINGS: dict[str, float] = {
    "DEFAULT": 0.50,
    "GAIN": 0.08,
    "LOSS": 0.25,
    "DECAY": 0.0,
}


class TrustLayer(FeatureLayer):
    name = "TRUST"

    def setup(self) -> None:
        configured = self.world.config.get("TRUST") or {}
        merged = {**DEFAULT_TRUST_SETTINGS, **configured}
        try:
            default = float(merged["DEFAULT"])
            gain = float(merged["GAIN"])
            loss = float(merged["LOSS"])
            decay = float(merged["DECAY"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "TRUST settings must be numeric"
            ) from exc
        if not 0 <= default <= 1:
            raise ValueError("TRUST.DEFAULT must be between 0 and 1")
        if not 0 <= gain <= 1:
            raise ValueError("TRUST.GAIN must be between 0 and 1")
        if not 0 <= loss <= 1:
            raise ValueError("TRUST.LOSS must be between 0 and 1")
        if not 0 <= decay <= 1:
            raise ValueError("TRUST.DECAY must be between 0 and 1")
        self.settings = {
            "default": default,
            "gain": gain,
            "loss": loss,
            "decay": decay,
        }
        self.world.trust_settings = self.settings
        self.world.trust_layer = self

    def reset_agent(self, agent: Any) -> None:
        agent.trust = float(self.settings["default"])
        agent.defaults = 0
        agent.repayments = 0
        agent.loans_taken = 0
        agent.loans_given = 0

    def award(self, agent: Any, gain: float | None = None) -> None:
        step = self.settings["gain"] if gain is None else float(gain)
        agent.trust = min(1.0, float(agent.trust) + max(0.0, step))
        agent.repayments += 1

    def penalize(self, agent: Any) -> None:
        agent.trust = max(0.0, float(agent.trust) - self.settings["loss"])
        agent.defaults += 1

    def end_of_tick(self) -> None:
        decay = float(self.settings["decay"])
        if decay <= 0:
            return
        default = float(self.settings["default"])
        for agent in self.world.agents:
            if not agent.is_alive:
                continue
            agent.trust += (default - agent.trust) * decay

    def metrics(self, iteration: int) -> dict[str, Any]:
        living = [agent for agent in self.world.agents if agent.is_alive]
        return {
            "mean_trust": (
                float(sum(agent.trust for agent in living) / len(living))
                if living
                else 0.0
            ),
            "mean_defaults": (
                float(sum(agent.defaults for agent in living) / len(living))
                if living
                else 0.0
            ),
        }

    def default_metrics(self) -> dict[str, Any]:
        return {"mean_trust": 0.0, "mean_defaults": 0.0}
