"""Deferred-payment credit in reciprocal and risk-priced modes."""

from __future__ import annotations

import math
from typing import Any

from .base import FeatureLayer

DEFAULT_CREDIT_SETTINGS: dict[str, Any] = {
    "MODE": "reciprocal",
    "TERM_DAYS": 20,
    "GRACE_DAYS": 2,
    "MAX_WOOD_PRINCIPAL": 8,
    "MAX_BERRY_PRINCIPAL": 6,
    "MAX_CRATE_PRINCIPAL": 1,
    "LENDER_WOOD_BUFFER": 2,
    "LENDER_BERRY_BUFFER": 2,
    "LENDER_CRATE_RESERVE": 1,
    "MIN_TRUST": 0.40,
    "MIN_MARKET_TRUST": 0.30,
    "MAX_LOANS_PER_BORROWER": 2,
    "MAX_LOANS_PER_LENDER": 3,
    "REPAYMENT_CAPACITY_FRACTION": 0.80,
    "BASE_INTEREST": 0.05,
    "RISK_PREMIUM": 0.75,
    "MIN_INTEREST": 0.02,
    "MAX_INTEREST": 0.75,
    "COLLATERAL_DISCOUNT": 0.50,
}

CREDIT_MODES = ("reciprocal", "market")


class CreditLayer(FeatureLayer):
    name = "CREDIT"
    dependencies = ("TRUST",)

    def setup(self) -> None:
        configured = self.world.config.get("CREDIT") or {}
        merged = {**DEFAULT_CREDIT_SETTINGS, **configured}
        mode = str(merged["MODE"]).strip().lower()
        if mode not in CREDIT_MODES:
            raise ValueError(
                f"CREDIT.MODE must be one of {CREDIT_MODES}, got {mode!r}"
            )
        try:
            term_days = int(merged["TERM_DAYS"])
            grace_days = int(merged["GRACE_DAYS"])
            max_wood_principal = int(merged["MAX_WOOD_PRINCIPAL"])
            max_berry_principal = int(merged["MAX_BERRY_PRINCIPAL"])
            max_crate_principal = int(merged["MAX_CRATE_PRINCIPAL"])
            lender_wood_buffer = int(merged["LENDER_WOOD_BUFFER"])
            lender_berry_buffer = int(merged["LENDER_BERRY_BUFFER"])
            lender_crate_reserve = int(merged["LENDER_CRATE_RESERVE"])
            min_trust = float(merged["MIN_TRUST"])
            min_market_trust = float(merged["MIN_MARKET_TRUST"])
            max_loans_per_borrower = int(merged["MAX_LOANS_PER_BORROWER"])
            max_loans_per_lender = int(merged["MAX_LOANS_PER_LENDER"])
            capacity_fraction = float(merged["REPAYMENT_CAPACITY_FRACTION"])
            base_interest = float(merged["BASE_INTEREST"])
            risk_premium = float(merged["RISK_PREMIUM"])
            min_interest = float(merged["MIN_INTEREST"])
            max_interest = float(merged["MAX_INTEREST"])
            collateral_discount = float(merged["COLLATERAL_DISCOUNT"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CREDIT settings must be numeric; principal, term, and loan "
                "limits must be whole numbers"
            ) from exc
        if term_days < 1:
            raise ValueError("CREDIT.TERM_DAYS must be positive")
        if grace_days < 0:
            raise ValueError("CREDIT.GRACE_DAYS must not be negative")
        if max_wood_principal < 0 or max_berry_principal < 0:
            raise ValueError("CREDIT principal limits must not be negative")
        if max_crate_principal < 0:
            raise ValueError("CREDIT.MAX_CRATE_PRINCIPAL must not be negative")
        if lender_wood_buffer < 0 or lender_berry_buffer < 0:
            raise ValueError("CREDIT lender buffers must not be negative")
        if lender_crate_reserve < 0:
            raise ValueError("CREDIT.LENDER_CRATE_RESERVE must not be negative")
        if not 0 <= min_trust <= 1 or not 0 <= min_market_trust <= 1:
            raise ValueError("CREDIT trust thresholds must be between 0 and 1")
        if max_loans_per_borrower < 0 or max_loans_per_lender < 0:
            raise ValueError("CREDIT loan limits must not be negative")
        if not 0 < capacity_fraction <= 1:
            raise ValueError(
                "CREDIT.REPAYMENT_CAPACITY_FRACTION must be in (0, 1]"
            )
        if base_interest < 0 or risk_premium < 0:
            raise ValueError("CREDIT interest settings must not be negative")
        if min_interest < 0 or max_interest < min_interest:
            raise ValueError(
                "CREDIT.MIN_INTEREST must not exceed MAX_INTEREST"
            )
        if not 0 <= collateral_discount <= 1:
            raise ValueError(
                "CREDIT.COLLATERAL_DISCOUNT must be between 0 and 1"
            )
        self.settings = {
            "mode": mode,
            "term_days": term_days,
            "grace_days": grace_days,
            "max_wood_principal": max_wood_principal,
            "max_berry_principal": max_berry_principal,
            "max_crate_principal": max_crate_principal,
            "lender_wood_buffer": lender_wood_buffer,
            "lender_berry_buffer": lender_berry_buffer,
            "lender_crate_reserve": lender_crate_reserve,
            "min_trust": min_trust,
            "min_market_trust": min_market_trust,
            "max_loans_per_borrower": max_loans_per_borrower,
            "max_loans_per_lender": max_loans_per_lender,
            "capacity_fraction": capacity_fraction,
            "base_interest": base_interest,
            "risk_premium": risk_premium,
            "min_interest": min_interest,
            "max_interest": max_interest,
            "collateral_discount": collateral_discount,
        }
        self.world.credit_settings = self.settings
        self.world.credit_layer = self

    def start_of_tick(self) -> None:
        self.world.obligations.settle_due(self.world._current_iteration)

    def before_consume(self) -> None:
        self._originate()

    # ------------------------------------------------------------------
    # Origination
    # ------------------------------------------------------------------
    def _originate(self) -> None:
        world = self.world
        living = [agent for agent in world.agents if agent.is_alive]
        if len(living) < 2:
            return
        borrowers = []
        for agent in living:
            demand = self._demand(agent)
            if demand:
                borrowers.append((agent, demand))
        if not borrowers:
            return

        lenders = []
        for agent in living:
            supply = self._supply(agent)
            if supply:
                lenders.append({"agent": agent, "supply": supply})
        if not lenders:
            return

        world.random.shuffle(borrowers)
        for borrower, demand in borrowers:
            if self._active_count(borrower.id, "debtor") >= int(
                self.settings["max_loans_per_borrower"]
            ):
                continue
            for good, need in demand:
                offers = self._offers(borrower, good, need, lenders)
                if not offers:
                    continue
                if self.settings["mode"] == "market":
                    offers.sort(
                        key=lambda offer: (
                            offer["repayment"],
                            -offer["principal"],
                        )
                    )
                else:
                    offers.sort(
                        key=lambda offer: (-offer["principal"], offer["repayment"])
                    )
                offer = offers[0]
                lender = offer["lender"]
                lender_entry = offer["entry"]
                principal = offer["principal"]
                moved = world.ledger.move(
                    lender, borrower, good, principal, "loan"
                )
                if moved <= 0:
                    continue
                principal = moved
                repayment = offer["repayment"]
                if moved != offer["principal"]:
                    repayment = self._repayment(
                        principal,
                        offer["interest"],
                        fixed=offer["fixed"],
                    )
                world.obligations.add(
                    creditor=lender.id,
                    debtor=borrower.id,
                    good=good,
                    principal=principal,
                    amount_due=repayment,
                    created_iteration=world._current_iteration,
                    due_iteration=world._current_iteration
                    + int(self.settings["term_days"]),
                    kind="loan",
                    interest_rate=(
                        (repayment / principal) - 1.0 if principal > 0 else 0.0
                    ),
                    collateral_crates=offer["collateral"],
                )
                lender.loans_given += 1
                borrower.loans_taken += 1
                lender_entry["supply"][good] -= moved
                if lender_entry["supply"][good] <= 0:
                    lender_entry["supply"].pop(good, None)
                break

    def _demand(self, agent: Any) -> list[tuple[str, int]]:
        world = self.world
        food = int(world.agent_settings["food_per_iteration"])
        storage = world.storage_settings
        requests: list[tuple[str, int]] = []

        if (
            world.features.get("STORAGE")
            and world.features.get("SPOILAGE")
            and agent.crate_count < world._target_crates(agent)
        ):
            need_crates = world._target_crates(agent) - agent.crate_count
            capacity_crates = self._repayment_capacity(agent, "crates")
            outstanding_crates = world.obligations.outstanding(
                agent.id, "crates", kind="loan"
            )
            principal_crates = min(
                need_crates,
                int(self.settings["max_crate_principal"]),
                max(0, capacity_crates - outstanding_crates),
            )
            if principal_crates >= 1:
                requests.append(("crates", principal_crates))

            need = max(0, int(storage["wood_per_container"]) - agent.wood)
            capacity = self._repayment_capacity(agent, "wood")
            outstanding = world.obligations.outstanding(
                agent.id, "wood", kind="loan"
            )
            principal = min(
                need,
                int(self.settings["max_wood_principal"]),
                max(0, capacity - outstanding),
            )
            if principal >= 1:
                requests.append(("wood", principal))

        if agent.berries < food:
            capacity = self._repayment_capacity(agent, "berries")
            outstanding = world.obligations.outstanding(
                agent.id, "berries", kind="loan"
            )
            principal = min(
                food - agent.berries,
                int(self.settings["max_berry_principal"]),
                max(0, capacity - outstanding),
            )
            if principal >= 1:
                requests.append(("berries", principal))
        return requests

    def _repayment_capacity(self, agent: Any, good: str) -> int:
        world = self.world
        if good == "wood":
            rate = world.effective_max_wood_per_day(
                agent, world._current_wood_density
            )
        elif good == "crates":
            rate = world.effective_max_wood_per_day(
                agent, world._current_wood_density
            )
            wood_per_crate = max(
                1, int(world.storage_settings["wood_per_container"])
            )
            rate = rate / wood_per_crate
        else:
            rate = world.effective_max_berries_per_day(
                agent, world._current_berries_density
            )
        if not math.isfinite(rate) or rate <= 0:
            return 0
        capacity = int(
            math.floor(
                rate
                * int(self.settings["term_days"])
                * float(self.settings["capacity_fraction"])
            )
        )
        if good == "crates":
            return max(1, capacity)
        return capacity

    def _supply(self, agent: Any) -> dict[str, int]:
        world = self.world
        food = int(world.agent_settings["food_per_iteration"])
        supply: dict[str, int] = {}
        crates = int(agent.crate_count) - int(
            self.settings["lender_crate_reserve"]
        )
        if crates > 0:
            supply["crates"] = crates
        wood = int(agent.wood) - int(self.settings["lender_wood_buffer"])
        if wood > 0:
            supply["wood"] = wood
        berries = (
            int(agent.berries)
            - food
            - int(self.settings["lender_berry_buffer"])
        )
        if berries > 0:
            supply["berries"] = berries
        return supply

    def _offers(
        self,
        borrower: Any,
        good: str,
        need: int,
        lenders: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        world = self.world
        offers: list[dict[str, Any]] = []
        mode = self.settings["mode"]
        available_collateral = max(
            0,
            borrower.crate_count
            - world.obligations.pledged_crates(borrower.id),
        )
        if mode == "reciprocal":
            if borrower.trust < float(self.settings["min_trust"]):
                if good == "crates":
                    # The lent crate itself is the security.
                    pass
                elif available_collateral <= 0 or borrower.health < 2:
                    return []
        elif borrower.trust < float(self.settings["min_market_trust"]):
            return []

        for entry in lenders:
            lender = entry["agent"]
            if lender.id == borrower.id:
                continue
            if self._active_count(lender.id, "creditor") >= int(
                self.settings["max_loans_per_lender"]
            ):
                continue
            supply = int(entry["supply"].get(good, 0))
            if supply <= 0:
                continue
            principal = min(need, supply)
            if principal < 1:
                continue
            if good == "crates":
                collateral = int(principal)
            else:
                collateral = (
                    1 if available_collateral > 0 and principal >= 2 else 0
                )
            if (
                mode == "reciprocal"
                and good != "crates"
                and borrower.trust < float(self.settings["min_trust"])
                and collateral <= 0
            ):
                continue
            fixed = mode == "reciprocal"
            interest = self._interest(borrower, good, principal, collateral)
            repayment = self._repayment(
                principal, interest, fixed=fixed
            )
            offers.append(
                {
                    "lender": lender,
                    "entry": entry,
                    "principal": principal,
                    "interest": interest,
                    "repayment": repayment,
                    "collateral": collateral,
                    "fixed": fixed,
                }
            )
        return offers

    def _active_count(self, agent_id: str, side: str) -> int:
        if side == "debtor":
            return len(self.world.obligations.for_debtor(agent_id, kind="loan"))
        return len(self.world.obligations.for_creditor(agent_id, kind="loan"))

    def _risk_probability(self, borrower: Any) -> float:
        settings = self.world.agent_settings
        max_health = max(1, int(settings["max_health"]))
        food = max(1, int(settings["food_per_iteration"]))
        health = min(1.0, max(0.0, borrower.health / max_health))
        food_cover = min(1.0, max(0.0, borrower.berries / food))
        probability = (
            0.25
            + 0.35 * health
            + 0.20 * food_cover
            + 0.20 * float(borrower.trust)
        )
        return min(0.98, max(0.15, probability))

    def _interest(
        self,
        borrower: Any,
        good: str,
        principal: int,
        collateral: int,
    ) -> float:
        if self.settings["mode"] == "reciprocal":
            return 0.0
        probability = self._risk_probability(borrower)
        break_even = (1.0 - probability) / max(probability, 1e-6)
        interest = float(self.settings["base_interest"]) + (
            float(self.settings["risk_premium"]) * break_even
        )
        if collateral > 0 and principal > 0:
            coverage = min(1.0, self._crate_value(good) / principal)
            interest *= 1.0 - (
                float(self.settings["collateral_discount"]) * coverage
            )
        return min(
            float(self.settings["max_interest"]),
            max(float(self.settings["min_interest"]), interest),
        )

    def _repayment(
        self, principal: int, interest: float, *, fixed: bool
    ) -> int:
        if fixed:
            return int(principal)
        return max(
            int(principal),
            int(math.ceil(principal * (1.0 + max(0.0, interest)))),
        )

    def _crate_value(self, good: str) -> int:
        storage = self.world.storage_settings
        if good == "crates":
            return 1
        if good == "wood":
            return int(storage["wood_per_container"])
        return int(storage["berries_per_container"])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def default_metrics(self) -> dict[str, Any]:
        return {
            "loans_originated": 0,
            "loan_principal_wood": 0,
            "loan_principal_berries": 0,
            "loan_principal_crates": 0,
            "mean_interest_rate": 0.0,
            "loans_repaid": 0,
            "repayment_wood": 0,
            "repayment_berries": 0,
            "repayment_crates": 0,
            "loans_defaulted": 0,
            "default_loss_wood": 0,
            "default_loss_berries": 0,
            "default_loss_crates": 0,
            "loans_late": 0,
            "collateral_seized": 0,
            "loans_active": 0,
            "credit_outstanding_wood": 0,
            "credit_outstanding_berries": 0,
            "credit_outstanding_crates": 0,
        }

    def metrics(self, iteration: int) -> dict[str, Any]:
        events = [
            event
            for event in self.world.obligations.events
            if event["iteration"] == iteration and event["kind"] == "loan"
        ]
        originated = [event for event in events if event["event"] == "origination"]
        repaid = [event for event in events if event["event"] == "repayment"]
        defaulted = [event for event in events if event["event"] == "default"]
        late = [event for event in events if event["event"] == "late"]
        active = self.world.obligations.loans()
        outstanding_wood = sum(
            obligation.amount_due
            for obligation in active
            if obligation.good == "wood"
        )
        outstanding_berries = sum(
            obligation.amount_due
            for obligation in active
            if obligation.good == "berries"
        )
        outstanding_crates = sum(
            obligation.amount_due
            for obligation in active
            if obligation.good == "crates"
        )
        interest_total = sum(
            float(event["interest_rate"]) for event in originated
        )
        return {
            "loans_originated": len(originated),
            "loan_principal_wood": sum(
                int(event["principal"])
                for event in originated
                if event["good"] == "wood"
            ),
            "loan_principal_berries": sum(
                int(event["principal"])
                for event in originated
                if event["good"] == "berries"
            ),
            "loan_principal_crates": sum(
                int(event["principal"])
                for event in originated
                if event["good"] == "crates"
            ),
            "mean_interest_rate": (
                interest_total / len(originated) if originated else 0.0
            ),
            "loans_repaid": len(repaid),
            "repayment_wood": sum(
                int(event["amount"])
                for event in repaid
                if event["good"] == "wood"
            ),
            "repayment_berries": sum(
                int(event["amount"])
                for event in repaid
                if event["good"] == "berries"
            ),
            "repayment_crates": sum(
                int(event["amount"])
                for event in repaid
                if event["good"] == "crates"
            ),
            "loans_defaulted": len(defaulted),
            "default_loss_wood": sum(
                int(event["loss"])
                for event in defaulted
                if event["good"] == "wood"
            ),
            "default_loss_berries": sum(
                int(event["loss"])
                for event in defaulted
                if event["good"] == "berries"
            ),
            "default_loss_crates": sum(
                int(event["loss"])
                for event in defaulted
                if event["good"] == "crates"
            ),
            "loans_late": len(late),
            "collateral_seized": sum(
                int(event.get("collateral_seized", 0))
                for event in defaulted
            ),
            "loans_active": len(active),
            "credit_outstanding_wood": int(outstanding_wood),
            "credit_outstanding_berries": int(outstanding_berries),
            "credit_outstanding_crates": int(outstanding_crates),
        }
