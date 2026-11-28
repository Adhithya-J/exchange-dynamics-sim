"""Shared whole-good transfers and deferred-payment obligations.

Gifts, reciprocal gifts, loans, repayments, collateral seizure, and estate
settlement all need to move the same three stocks. Keeping those moves in one
ledger makes the feature layers small and keeps conservation auditable.
"""

from __future__ import annotations

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
    grace_used: int = 0
    status: str = "active"
    recovered: int = 0
    loss: int = 0
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
    ) -> int:
        if source is None or target is None or quantity <= 0:
            return 0
        if good not in GOODS:
            raise ValueError(f"Unknown good {good!r}")
        if good == "crates":
            moved = min(int(quantity), len(source.crates))
            for _ in range(moved):
                target.crates.append(source.crates.pop())
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
            if not debtor_alive:
                self._default(obligation, iteration, "death")
            elif not creditor.is_alive:
                self._write_off(obligation, iteration, "creditor_death")
            elif self._can_pay(debtor, obligation.good, obligation.amount_due):
                self._repay(obligation, iteration)
            else:
                self._missed_payment(obligation, iteration)

    def settle_soft_gifts(self, iteration: int, threshold: int) -> None:
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
            self._award_trust(debtor, gain=moved / max(1, obligation.principal))
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

    def _can_pay(self, debtor: Any, good: str, amount: int) -> bool:
        return self.ledger.amount(debtor, good) >= int(amount)

    def _repay(self, obligation: Obligation, iteration: int) -> None:
        debtor = self.world._agent_index[obligation.debtor]
        creditor = self.world._agent_index[obligation.creditor]
        moved = self.ledger.move(
            debtor,
            creditor,
            obligation.good,
            obligation.amount_due,
            "loan_repayment",
        )
        if moved <= 0:
            self._default(obligation, iteration, "unable")
            return
        obligation.recovered += moved
        obligation.amount_due -= moved
        if obligation.amount_due <= 0:
            obligation.status = "repaid"
            obligation.amount_due = 0
        self._award_trust(debtor)
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
        recovered = 0
        collateral_seized = 0

        if obligation.collateral_crates > 0:
            collateral_seized = self.ledger.move(
                debtor,
                creditor,
                "crates",
                obligation.collateral_crates,
                "collateral",
            )
            if collateral_seized:
                crate_value = self._crate_value(obligation.good)
                recovered += collateral_seized * crate_value
                remaining = max(0, remaining - collateral_seized * crate_value)

        if remaining > 0:
            moved = self.ledger.move(
                debtor,
                creditor,
                obligation.good,
                remaining,
                "default_seizure",
            )
            recovered += moved
            remaining -= moved

        recovered = min(int(obligation.amount_due), recovered)
        obligation.recovered += recovered
        obligation.loss = max(0, int(obligation.amount_due) - recovered)
        obligation.amount_due = 0
        obligation.status = "defaulted"
        obligation.reason = reason
        if debtor.is_alive:
            self._penalize_trust(debtor)
        self._record(
            "default",
            obligation,
            iteration=iteration,
            amount=recovered,
            collateral_seized=collateral_seized,
            loss=obligation.loss,
            reason=reason,
        )

    def _write_off(
        self, obligation: Obligation, iteration: int, reason: str
    ) -> None:
        obligation.loss = int(obligation.amount_due)
        obligation.amount_due = 0
        obligation.status = "written_off"
        obligation.reason = reason
        self._record(
            "default",
            obligation,
            iteration=iteration,
            amount=0,
            loss=obligation.loss,
            reason=reason,
        )

    def _crate_value(self, good: str) -> int:
        storage = self.world.storage_settings
        if good == "crates":
            return 1
        if good == "wood":
            return int(storage["wood_per_container"])
        if good == "berries":
            return int(storage["berries_per_container"])
        return 1

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
            "collateral_seized": int(collateral_seized),
            "recovered": int(obligation.recovered),
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


def active_obligations(
    obligations: Iterable[Obligation], kind: str | None = None
) -> list[Obligation]:
    return [
        obligation
        for obligation in obligations
        if obligation.active and (kind is None or obligation.kind == kind)
    ]
