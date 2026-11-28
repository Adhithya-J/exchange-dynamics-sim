"""Composable feature layers for the subsistence world.

Each layer owns one optional mechanism and declares the flags it depends on.
The world resolves the dependency graph once, then runs layers through small
lifecycle hooks instead of branching on feature flags throughout the main loop.
"""

from .base import FeatureLayer, FeatureRegistry, PassiveLayer
from .credit import CreditLayer
from .gifting import GiftingLayer
from .reciprocity import ReciprocityLayer
from .shocks import ShockLayer
from .trade import TradeLayer
from .transfers import GoodsLedger, Obligation, ObligationBook
from .trust import TrustLayer

__all__ = [
    "CreditLayer",
    "FeatureLayer",
    "FeatureRegistry",
    "GiftingLayer",
    "GoodsLedger",
    "Obligation",
    "ObligationBook",
    "PassiveLayer",
    "ReciprocityLayer",
    "ShockLayer",
    "TradeLayer",
    "TrustLayer",
]
