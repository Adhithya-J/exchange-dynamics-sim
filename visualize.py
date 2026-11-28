"""Compatibility entry point for the money world visualizer.

For the goods-only subsistence viewer, run ``python -m worlds.subsistence``.
"""

from worlds.money.visualize import *  # noqa: F401,F403
from worlds.money.visualize import main

if __name__ == "__main__":
    main()
