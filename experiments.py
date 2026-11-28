"""Compatibility entry point for the money world experiments runner.

For subsistence flag and seed sweeps, use
``python -m worlds.subsistence.experiments``.
"""

from utils.experiments import *  # noqa: F401,F403
from utils.experiments import _set_override, main

if __name__ == "__main__":
    main()
