"""Soft zero-interest gift obligations."""

from __future__ import annotations

from typing import Any

from .base import FeatureLayer

DEFAULT_RECIPROCITY_SETTINGS: dict[str, int] = {
    "REPAY_THRESHOLD": 2,
    "MIN_GIFT": 1,
    "MAX_OUTSTANDING": 6,
    # A soft obligation that is never serviced is written off and penalised
    # after this many days. Set 0 to keep soft claims open forever.
    "DEFAULT_AFTER_DAYS": 30,
}


class ReciprocityLayer(FeatureLayer):
    name = "RECIPROCITY"
    dependencies = ("GIFTING", "TRUST")

    def setup(self) -> None:
        configured = self.world.config.get("RECIPROCITY") or {}
        merged = {**DEFAULT_RECIPROCITY_SETTINGS, **configured}
        try:
            repay_threshold = int(merged["REPAY_THRESHOLD"])
            min_gift = int(merged["MIN_GIFT"])
            max_outstanding = int(merged["MAX_OUTSTANDING"])
            default_after_days = int(merged["DEFAULT_AFTER_DAYS"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RECIPROCITY settings must be whole numbers"
            ) from exc
        if (
            repay_threshold < 0
            or min_gift < 0
            or max_outstanding < 0
            or default_after_days < 0
        ):
            raise ValueError("RECIPROCITY settings must not be negative")
        self.settings = {
            "repay_threshold": repay_threshold,
            "min_gift": min_gift,
            "max_outstanding": max_outstanding,
            "default_after_days": default_after_days,
        }
        self.world.reciprocity_settings = self.settings
        self.world.reciprocity_layer = self

    def start_of_tick(self) -> None:
        self.world.obligations.settle_soft_gifts(
            self.world._current_iteration,
            threshold=int(self.settings["repay_threshold"]),
            default_after_days=int(self.settings["default_after_days"]),
        )

    def after_gather(self) -> None:
        gifting = getattr(self.world, "gifting_layer", None)
        if gifting is None:
            return
        for gift in gifting.last_gifts:
            amount = int(gift["amount"])
            if amount < int(self.settings["min_gift"]):
                continue
            debtor = gift["receiver"]
            outstanding = self.world.obligations.outstanding(
                debtor, "berries", kind="gift"
            )
            if outstanding + amount > int(self.settings["max_outstanding"]):
                continue
            self.world.obligations.add(
                creditor=gift["giver"],
                debtor=debtor,
                good="berries",
                principal=amount,
                amount_due=amount,
                created_iteration=self.world._current_iteration,
                due_iteration=None,
                kind="gift",
                interest_rate=0.0,
            )

    def metrics(self, iteration: int) -> dict[str, Any]:
        gifts = self.world.obligations.gifts()
        outstanding = sum(obligation.amount_due for obligation in gifts)
        likely = [
            event
            for event in self.world.obligations.events
            if event["iteration"] == iteration
            and event["kind"] == "gift"
        ]
        return {
            "gift_obligations": len(gifts),
            "gift_obligation_berries": int(outstanding),
            "gift_repayments": sum(
                1 for event in likely if event["event"] == "repayment"
            ),
            "gift_repayment_berries": sum(
                event["amount"]
                for event in likely
                if event["event"] == "repayment"
            ),
            "gift_defaults": sum(
                1 for event in likely if event["event"] == "default"
            ),
            "gift_default_berries": sum(
                event["loss"]
                for event in likely
                if event["event"] == "default"
            ),
        }

    def default_metrics(self) -> dict[str, Any]:
        return {
            "gift_obligations": 0,
            "gift_obligation_berries": 0,
            "gift_repayments": 0,
            "gift_repayment_berries": 0,
            "gift_defaults": 0,
            "gift_default_berries": 0,
        }
