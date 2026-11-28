"""Compatibility entry point for the money world.

The implementation lives in :mod:`worlds.money.world`. Existing code that
imports ``main`` keeps working. For the goods-only subsistence world, run
``python -m worlds.subsistence``.
"""

from worlds.money.world import (  # noqa: F401
    CONFIG,
    Agent,
    AffordabilityCalculator,
    MetricsCalculator,
    ResourceSimulation,
    config,
    main,
)

if __name__ == "__main__":
    main()
