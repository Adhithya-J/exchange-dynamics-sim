"""Small, world-agnostic social mechanisms.

Worlds compose these helpers instead of embedding social rules in their main
loop. The helpers operate on the minimal agent surface they need, so a world
can keep its own resource model and still reuse the behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Protocol, Sequence


class BerryAgent(Protocol):
    id: str
    berries: int
    health: int

    @property
    def is_alive(self) -> bool: ...


@dataclass(frozen=True)
class GiftSettings:
    """Parameters for need-based whole-berry gifting."""

    probability: float
    amount_range: tuple[int, int]
    min_reserve: int


def validate_gift_settings(settings: GiftSettings) -> None:
    if not 0.0 <= settings.probability <= 1.0:
        raise ValueError("GIFTING.PROBABILITY must be between 0 and 1")
    lower, upper = settings.amount_range
    if lower < 1 or upper < lower:
        raise ValueError(
            "GIFTING.AMOUNT_RANGE must contain positive whole values in "
            "ascending order"
        )
    if settings.min_reserve < 0:
        raise ValueError("GIFTING.MIN_RESERVE must not be negative")


def process_berry_gifts(
    agents: Sequence[BerryAgent],
    *,
    settings: GiftSettings,
    random: Random,
    transfer: Callable[[BerryAgent, BerryAgent, int], int] | None = None,
) -> list[dict[str, Any]]:
    """Move whole berries from agents with a surplus to the neediest agent.

    The receiver is chosen by the lowest health, then the lowest berry stock.
    A giver keeps ``min_reserve`` berries. The returned records are
    observations; the caller is responsible for logging them.

    When ``transfer`` is provided it performs the actual move and returns the
    whole amount moved. This lets a world route every transfer through its own
    goods ledger without changing the selection rule.
    """

    validate_gift_settings(settings)
    living = [agent for agent in agents if agent.is_alive]
    if len(living) < 2:
        return []

    gifts: list[dict[str, Any]] = []
    givers = list(living)
    random.shuffle(givers)
    for giver in givers:
        if random.random() >= settings.probability:
            continue
        surplus = int(giver.berries) - settings.min_reserve
        if surplus < 1:
            continue
        receivers = [agent for agent in living if agent.id != giver.id]
        if not receivers:
            continue
        receiver = min(receivers, key=lambda agent: (agent.health, agent.berries))
        amount = min(random.randint(*settings.amount_range), surplus)
        if amount < 1:
            continue
        if transfer is not None:
            amount = int(transfer(giver, receiver, amount))
            if amount < 1:
                continue
        else:
            giver.berries -= amount
            receiver.berries += amount
        gifts.append(
            {
                "giver": giver.id,
                "receiver": receiver.id,
                "amount": amount,
            }
        )
    return gifts
