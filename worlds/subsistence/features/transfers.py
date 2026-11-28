"""Shared whole-good transfers and deferred-payment obligations.

Gifts, reciprocal gifts, loans, repayments, collateral seizure, and estate
settlement all need to move the same three stocks. Keeping those moves in one
ledger makes the feature layers small and keeps conservation auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

GOODS = ("berries", "wood", "crates")


@dataclass
class Obligation:
    """A claim on a future whole-good transfer."""

    id: int
    creditor: str
    debtor: str
    good: str
    principal: int
    amount_due: int
    created_iteration: int
    due_iteration: int | None
    kind: str = "loan"
    interest_rate: float = 0.0
    collateral_crates: int = 0
    collateral_durability: float = 0.0
    collateral_excess: float = 0.0
    grace_used: int = 0
    status: str = "active"
    recovered: float = 0.0
    loss: float = 0.0
    reason: str = ""

    @property
    def active(self) -> bool:
        return self.status == "active"


class GoodsLedger:
    """Move whole goods between agents without creating or destroying them."""

    def __init__(self, world: Any) -> None:
        self.world = world

    @staticmethod
    def amount(agent: Any, good: str) -> int:
        if good == "crates":
            return len(agent.crates)
        return int(getattr(agent, good))

    def move(
        self,
        source: Any,
        target: Any,
        good: str,
        quantity: int,
        reason: str,
        *,
        crate_preference: str = "low",
    ) -> int:
        if source is None or target is None or quantity <= 0:
            return 0
        if good not in GOODS:
            raise ValueError(f"Unknown good {good!r}")
        if good == "crates":
            moved = min(int(quantity), len(source.crates))
            if moved <= 0:
                return 0
            if crate_preference not in {"low", "high"}:
                raise ValueError("crate_preference must be 'low' or 'high'")
            indices = sorted(
                range(len(source.crates)),
                key=lambda index: source.crates[index],
                reverse=crate_preference == "high",
            )[:moved]
            for index in sorted(indices, reverse=True):
                target.crates.append(source.crates.pop(index))
            return moved
        available = int(getattr(source, good))
        moved = min(int(quantity), available)
        if moved <= 0:
            return 0
        setattr(source, good, available - moved)
        setattr(target, good, int(getattr(target, good)) + moved)
        return moved

    def consume(self, agent: Any, good: str, quantity: int) -> int:
        """Remove whole goods without a receiver (food, crafting, spoilage)."""

        if agent is None or quantity <= 0 or good not in ("berries", "wood"):
            return 0
        available = int(getattr(agent, good))
        removed = min(int(quantity), available)
        setattr(agent, good, available - removed)
        return removed


class ObligationBook:
    """Track loans and soft reciprocal-gift claims in one place."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.ledger: GoodsLedger = world.ledger
        self.obligations: list[Obligation] = []
        self.events: list[dict[str, Any]] = []
        self._next_id = 1

    def reset(self) -> None:
        self.obligations = []
        self.events = []
        self._next_id = 1

    def add(
        self,
        *,
        creditor: str,
        debtor: str,
        good: str,
        principal: int,
        amount_due: int,
        created_iteration: int,
        due_iteration: int | None,
        kind: str = "loan",
        interest_rate: float = 0.0,
        collateral_crates: int = 0,
        collateral_durability: float = 0.0,
    ) -> Obligation:
        obligation = Obligation(
            id=self._next_id,
            creditor=creditor,
            debtor=debtor,
            good=good,
            principal=int(principal),
            amount_due=int(amount_due),
            created_iteration=int(created_iteration),
            due_iteration=due_iteration,
            kind=kind,
            interest_rate=float(interest_rate),
            collateral_crates=int(collateral_crates),
            collateral_durability=float(collateral_durability),
        )
        self._next_id += 1
        self.obligations.append(obligation)
        self._record(
            "origination",
            obligation,
            iteration=int(created_iteration),
            amount=obligation.principal,
        )
        return obligation

    def active(self, kind: str | None = None) -> list[Obligation]:
        return [
            obligation
            for obligation in self.obligations
            if obligation.active and (kind is None or obligation.kind == kind)
        ]

    def loans(self) -> list[Obligation]:
        return self.active("loan")

    def gifts(self) -> list[Obligation]:
        return self.active("gift")

    def for_debtor(
        self, debtor: str, kind: str | None = None
    ) -> list[Obligation]:
        return [
            obligation
            for obligation in self.active(kind)
            if obligation.debtor == debtor
        ]

    def for_creditor(
        self, creditor: str, kind: str | None = None
    ) -> list[Obligation]:
        return [
            obligation
            for obligation in self.active(kind)
            if obligation.creditor == creditor
        ]

    def pledged_crates(self, debtor: str) -> int:
        return sum(
            obligation.collateral_crates
            for obligation in self.active()
            if obligation.debtor == debtor
        )

    def outstanding(
        self, debtor: str, good: str, kind: str | None = None
    ) -> int:
        return sum(
            obligation.amount_due
            for obligation in self.for_debtor(debtor, kind)
            if obligation.good == good
        )

    def receivable(
        self, creditor: str, good: str, kind: str | None = None
    ) -> int:
        return sum(
            obligation.amount_due
            for obligation in self.for_creditor(creditor, kind)
            if obligation.good == good
        )

    def settle_due(self, iteration: int) -> None:
        for obligation in list(self.obligations):
            if not obligation.active or obligation.due_iteration is None:
                continue
            debtor = self.world._agent_index.get(obligation.debtor)
            creditor = self.world._agent_index.get(obligation.creditor)
            if debtor is None or creditor is None:
                continue
            debtor_alive = debtor.is_alive
            if obligation.due_iteration > iteration and debtor_alive:
                continue
            if not creditor.is_alive:
                # A dead creditor cannot collect. This is a write-off, not a
                # borrower default, and it must not penalise the borrower.
                self._write_off(obligation, iteration, "creditor_death")
            elif not debtor_alive:
                self._default(obligation, iteration, "death")
            else:
                available = self._available_for_repayment(
                    debtor, obligation.good
                )
                if available >= obligation.amount_due:
                    self._repay(obligation, iteration)
                elif available > 0:
                    self._repay(obligation, iteration, amount=available)
                    if obligation.active and obligation.amount_due > 0:
                        self._missed_payment(obligation, iteration)
                else:
                    self._missed_payment(obligation, iteration)

    def settle_soft_gifts(
        self,
        iteration: int,
        threshold: int,
        default_after_days: int = 0,
    ) -> None:
        """Repay gift obligations when the debtor has a comfortable surplus."""

        food = int(self.world.agent_settings["food_per_iteration"])
        for obligation in list(self.obligations):
            if (
                not obligation.active
                or obligation.kind != "gift"
                or obligation.good != "berries"
            ):
                continue
            debtor = self.world._agent_index.get(obligation.debtor)
            creditor = self.world._agent_index.get(obligation.creditor)
            if debtor is None or creditor is None:
                continue
            if not debtor.is_alive or not creditor.is_alive:
                self._write_off(obligation, iteration, "death")
                continue
            age = iteration - int(obligation.created_iteration)
            if default_after_days > 0 and age > default_after_days:
                # A soft claim that is never serviced is not free. Write it
                # off, lower the debtor's public trust, and let it be replaced
                # only through a fresh, trust-priced relationship.
                obligation.loss = int(obligation.amount_due)
                obligation.amount_due = 0
                obligation.status = "defaulted"
                obligation.reason = "soft_default"
                self._penalize_trust(debtor)
                self._record(
                    "default",
                    obligation,
                    iteration=iteration,
                    amount=0,
                    loss=obligation.loss,
                    reason="soft_default",
                )
                continue
            available = int(debtor.berries) - food - int(threshold)
            if available <= 0:
                continue
            amount = min(available, obligation.amount_due)
            moved = self.ledger.move(
                debtor, creditor, "berries", amount, "gift_repayment"
            )
            if moved <= 0:
                continue
            obligation.amount_due -= moved
            obligation.recovered += moved
            trust = getattr(self.world, "trust_settings", {})
            gain = float(trust.get("gain", 0.0)) * min(
                1.0, moved / max(1, obligation.principal)
            )
            self._award_trust(debtor, gain=gain)
            self._record(
                "repayment",
                obligation,
                iteration=iteration,
                amount=moved,
                partial=obligation.amount_due > 0,
            )
            if obligation.amount_due <= 0:
                obligation.status = "repaid"
                obligation.amount_due = 0

    def _available_for_repayment(self, debtor: Any, good: str) -> int:
        """Goods that can be repaid without taking today's meal.

        Berry debt is subordinate to survival: a debtor keeps
        ``FOOD_PER_ITERATION`` berries before servicing a formal loan. Wood and
        crates have no equivalent daily subsistence claim.
        """

        available = self.ledger.amount(debtor, good)
        if good == "berries":
            food = int(self.world.agent_settings["food_per_iteration"])
            available = max(0, available - food)
        return available

    def _can_pay(self, debtor: Any, good: str, amount: int) -> bool:
        return self._available_for_repayment(debtor, good) >= int(amount)

    def _repay(
        self,
        obligation: Obligation,
        iteration: int,
        amount: int | None = None,
    ) -> None:
        debtor = self.world._agent_index[obligation.debtor]
        creditor = self.world._agent_index[obligation.creditor]
        requested = (
            int(obligation.amount_due) if amount is None else int(amount)
        )
        requested = min(requested, int(obligation.amount_due))
        if requested <= 0:
            self._missed_payment(obligation, iteration)
            return
        moved = self.ledger.move(
            debtor,
            creditor,
            obligation.good,
            requested,
            "loan_repayment",
        )
        if moved <= 0:
            self._missed_payment(obligation, iteration)
            return
        obligation.recovered += moved
        obligation.amount_due -= moved
        if obligation.amount_due <= 0:
            obligation.status = "repaid"
            obligation.amount_due = 0
        trust = getattr(self.world, "trust_settings", {})
        gain = float(trust.get("gain", 0.0)) * min(
            1.0, moved / max(1, obligation.principal)
        )
        self._award_trust(debtor, gain=gain)
        self._record(
            "repayment",
            obligation,
            iteration=iteration,
            amount=moved,
            partial=obligation.amount_due > 0,
        )

    def _missed_payment(self, obligation: Obligation, iteration: int) -> None:
        grace = int(getattr(self.world, "credit_settings", {}).get("grace_days", 0))
        if obligation.grace_used < grace:
            obligation.grace_used += 1
            obligation.due_iteration = iteration + 1
            self._record("late", obligation, iteration=iteration)
            return
        self._default(obligation, iteration, "unable")

    def _default(
        self, obligation: Obligation, iteration: int, reason: str
    ) -> None:
        debtor = self.world._agent_index[obligation.debtor]
        creditor = self.world._agent_index[obligation.creditor]
        remaining = int(obligation.amount_due)
        recovered = 0.0
        collateral_seized = 0
        collateral_value = 0.0

        if obligation.collateral_crates > 0:
            available = sorted(debtor.crates, reverse=True)[
                : obligation.collateral_crates
            ]
            durability = (
                float(sum(available) / len(available))
                if available
                else obligation.collateral_durability
            )
            collateral_seized = self.ledger.move(
                debtor,
                creditor,
                "crates",
                obligation.collateral_crates,
                "collateral",
                crate_preference="high",
            )
            if collateral_seized:
                crate_value = self._crate_value(
                    obligation.good, debtor, durability
                )
                collateral_value = collateral_seized * crate_value
                recovered += collateral_value
                remaining = max(0, remaining - collateral_value)

        if remaining > 0:
            # remaining is in loan-good units but may be fractional after a
            # fractional crate valuation. Goods move in whole units, so seize
            # whole units up to the remaining claim. Berry debt never takes
            # today's meal from a living debtor.
            seize_available = self._available_for_repayment(
                debtor, obligation.good
            )
            request = min(
                int(math.ceil(remaining - 1e-9)), seize_available
            )
            moved = self.ledger.move(
                debtor,
                creditor,
                obligation.good,
                request,
                "default_seizure",
            )
            recovered += moved
            remaining -= moved

        obligation.recovered += recovered
        obligation.loss = max(0.0, float(obligation.amount_due) - recovered)
        obligation.collateral_excess = max(
            0.0, collateral_value - float(obligation.amount_due)
        )
        obligation.amount_due = 0
        obligation.status = "defaulted"
        obligation.reason = reason
        if debtor.is_alive:
            self._penalize_trust(debtor)
        self._record(
            "default",
            obligation,
            iteration=iteration,
            amount=int(round(recovered)),
            collateral_seized=collateral_seized,
            loss=int(round(obligation.loss)),
            reason=reason,
        )

    def _write_off(
        self, obligation: Obligation, iteration: int, reason: str
    ) -> None:
        obligation.loss = int(round(float(obligation.amount_due)))
        obligation.amount_due = 0
        obligation.status = "written_off"
        obligation.reason = reason
        self._record(
            "write_off",
            obligation,
            iteration=iteration,
            amount=0,
            loss=obligation.loss,
            reason=reason,
        )

    def _crate_value(
        self, good: str, agent: Any | None = None, durability: float = 1.0
    ) -> float:
        world = self.world
        if good == "crates":
            return max(0.0, min(1.0, float(durability)))
        if good == "wood":
            return world.crate_wood_value(agent, durability=durability)
        if good == "berries":
            return world.crate_berry_value(agent, durability=durability)
        return 1.0

    def _award_trust(self, agent: Any, gain: float | None = None) -> None:
        if not self.world.features.get("TRUST"):
            return
        layer = getattr(self.world, "trust_layer", None)
        if layer is not None:
            layer.award(agent, gain=gain)

    def _penalize_trust(self, agent: Any) -> None:
        if not self.world.features.get("TRUST"):
            return
        layer = getattr(self.world, "trust_layer", None)
        if layer is not None:
            layer.penalize(agent)

    def _record(
        self,
        event: str,
        obligation: Obligation,
        *,
        iteration: int,
        amount: int = 0,
        partial: bool = False,
        collateral_seized: int = 0,
        loss: int = 0,
        reason: str = "",
    ) -> None:
        record = {
            "iteration": int(iteration),
            "event": event,
            "id": obligation.id,
            "kind": obligation.kind,
            "creditor": obligation.creditor,
            "debtor": obligation.debtor,
            "good": obligation.good,
            "principal": int(obligation.principal),
            "amount_due": int(obligation.amount_due),
            "amount": int(amount),
            "due_iteration": obligation.due_iteration,
            "interest_rate": float(obligation.interest_rate),
            "collateral_crates": int(obligation.collateral_crates),
            "collateral_durability": float(
                obligation.collateral_durability
            ),
            "collateral_seized": int(collateral_seized),
            "collateral_excess": float(obligation.collateral_excess),
            "recovered": float(obligation.recovered),
            "loss": int(loss),
            "partial": bool(partial),
            "reason": reason,
        }
        self.events.append(record)
        if obligation.kind == "loan":
            if event == "origination":
                self.world.loan_log.append(record)
            elif event == "repayment":
                self.world.repayment_log.append(record)
            elif event == "default":
                self.world.default_log.append(record)
            elif event == "write_off":
                self.world.write_off_log.append(record)


def active_obligations(
    obligations: Iterable[Obligation], kind: str | None = None
) -> list[Obligation]:
    return [
        obligation
        for obligation in obligations
        if obligation.active and (kind is None or obligation.kind == kind)
    ]
