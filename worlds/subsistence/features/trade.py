"""Spot barter layer.

The market is still implemented by the core world because it shares target
stock and belief helpers with labour allocation. The layer owns its lifecycle
slot so the main loop no longer branches on the TRADE flag.
"""

from __future__ import annotations

from .base import FeatureLayer


class TradeLayer(FeatureLayer):
    name = "TRADE"

    def market(self) -> None:
        self.world._process_trade()
