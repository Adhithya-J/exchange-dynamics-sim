"""Command-line entry point for the subsistence world."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

from utils import load_config
from worlds.subsistence.replay import export_replay
from worlds.subsistence.world import CONFIG, SubsistenceWorld


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Custom YAML configuration")
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        metavar="N",
        help="Override SIMULATION.AGENTS for a larger or smaller village",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override the run length"
    )
    parser.add_argument(
        "--replay",
        nargs="?",
        const="replay-subsistence.json.gz",
        default=None,
        help="Write a versioned replay (.gz is compressed)",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="simulation-subsistence.html",
        default=None,
        help="Write a self-contained viewer page",
    )
    return parser


def build_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Apply command-line overrides to the loaded (or default) config."""

    if args.config is None and args.agents is None:
        return None
    config = deepcopy(
        load_config(args.config) if args.config else CONFIG
    )
    if args.agents is not None:
        if args.agents < 1:
            raise SystemExit("--agents must be at least 1")
        settings = config.setdefault("SIMULATION", {})
        settings["AGENTS"] = args.agents
    return config


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = build_config(args)
    simulation = SubsistenceWorld(config)
    final_state = simulation.run(iterations=args.iterations)

    if args.replay:
        path = export_replay(simulation, args.replay)
        print(f"Replay written to {path}")
    if args.html:
        from worlds.subsistence.visualize import create_visualization

        path = create_visualization(simulation, output=args.html)
        print(f"Viewer written to {path}")

    print(final_state.head())
    if simulation.metrics_history:
        print("\nFinal statistics:")
        print(simulation.metrics_history[-1])
    else:
        print("\nNo iterations run; no final statistics.")


if __name__ == "__main__":
    main()
