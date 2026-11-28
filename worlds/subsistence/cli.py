"""Command-line entry point for the subsistence world."""

from __future__ import annotations

import argparse

from utils import load_config
from worlds.subsistence.replay import export_replay
from worlds.subsistence.world import SubsistenceWorld


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Custom YAML configuration")
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config) if args.config else None
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
    print("\nFinal statistics:")
    print(simulation.metrics_history[-1])


if __name__ == "__main__":
    main()
