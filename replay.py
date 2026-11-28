"""Compatibility entry point for the money world replay format.

For the subsistence replay format, use :mod:`worlds.subsistence.replay`.
"""

from worlds.money.replay import *  # noqa: F401,F403
from worlds.money.replay import main

if __name__ == "__main__":
    main()
