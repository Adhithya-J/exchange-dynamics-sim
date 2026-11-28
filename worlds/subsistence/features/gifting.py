"""Need-based whole-berry gifts."""

from __future__ import annotations

from typing import Any

from utils.social import GiftSettings, process_berry_gifts, validate_gift_settings

from .base import FeatureLayer

DEFAULT_GIFTING_SETTINGS: dict[str, Any] = {
    "PROBABILITY": 0.25,
    "AMOUNT_RANGE": (1, 3),
    "MIN_RESERVE": 2,
}


class GiftingLayer(FeatureLayer):
    name = "GIFTING"

    def __init__(self, world: Any) -> None:
        super().__init__(world)
        self.last_gifts: list[dict[str, Any]] = []

    def setup(self) -> None:
        configured = self.world.config.get("GIFTING") or {}
        merged = {**DEFAULT_GIFTING_SETTINGS, **configured}
        try:
            probability = float(merged["PROBABILITY"])
            amount_range = self.world._validated_int_range(
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
        self.settings = {
            "probability": settings.probability,
            "amount_range": settings.amount_range,
            "min_reserve": settings.min_reserve,
        }
        self.world.gifting_settings = settings
        self.world.gifting_layer = self

    def after_gather(self) -> None:
        self.last_gifts = process_berry_gifts(
            self.world.agents,
            settings=self.world.gifting_settings,
            random=self.world.random,
            transfer=lambda giver, receiver, amount: self.world.ledger.move(
                giver, receiver, "berries", amount, "gift"
            ),
        )
        for gift in self.last_gifts:
            self.world.gift_log.append(
                {
                    "iteration": self.world._current_iteration,
                    "giver": gift["giver"],
                    "receiver": gift["receiver"],
                    "amount": gift["amount"],
                }
            )
            self.world._last_gifts += 1
            self.world._last_gift_berries += gift["amount"]

    def run(self) -> None:
        """Compatibility alias used by the world's public helper."""

        self.after_gather()
