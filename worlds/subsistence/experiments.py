"""Seed and flag sweeps for the subsistence world.

Example::

    python -m worlds.subsistence.experiments --iterations 300 --seeds 1 2 3 \
        --sweep FEATURES.GIFTING=true,false \
        --sweep FEATURES.STORAGE=true,false \
        --output results/subsistence.csv

Raw per-run results are written to ``--output``. A summary with 95%
confidence intervals is written next to it and as JSON.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Iterable, Sequence

import pandas as pd

from utils import load_config
from utils.experiments import (
    parse_option,
    run_experiment as run_generic_experiment,
    split_top_level,
    summarize_runs,
    write_results,
)
from worlds.subsistence.world import CONFIG, SubsistenceWorld

DEFAULT_METRICS: tuple[str, ...] = (
    "alive",
    "mean_health",
    "total_berries",
    "total_wood",
    "total_crates",
    "crates_crafted",
    "berries_commons_pct",
    "wood_commons_pct",
    "berries_spoiled",
    "trade_berries",
    "crate_trades",
    "barter_price",
    "mean_price_belief",
    "mean_trust",
    "loans_originated",
    "loans_repaid",
    "loans_defaulted",
    "loan_principal_crates",
    "repayment_crates",
    "default_loss_crates",
    "credit_outstanding_wood",
    "credit_outstanding_berries",
    "credit_outstanding_crates",
    "mean_interest_rate",
    "gift_obligations",
    "shock_active",
    "ppf_efficiency",
    "specialisation",
    "gini_berries",
    "gini_wood",
    "unmet_food",
    "extinction_iteration",
)


def run_experiment(
    iterations: int,
    seeds: Iterable[int],
    sweeps: dict[str, Sequence[Any]] | None = None,
    metrics: Iterable[str] = DEFAULT_METRICS,
    base_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run every seed/flag combination and return one row per run."""

    return run_generic_experiment(
        iterations=iterations,
        seeds=seeds,
        sweeps=sweeps,
        metrics=metrics,
        base_config=base_config if base_config is not None else CONFIG,
        world_factory=SubsistenceWorld,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Custom YAML config")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds, one run per seed and flag combination",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help=(
            "Flag sweep as SECTION.KEY=v1,v2 (repeatable). "
            "Values are parsed as JSON when possible. "
            "Example: FEATURES.GIFTING=true,false"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Final metrics to aggregate",
    )
    parser.add_argument(
        "--output",
        default="subsistence-experiments.csv",
        help="Raw per-run CSV path",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Summary JSON path (default: next to --output)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    sweeps: dict[str, list[Any]] = {}
    for specification in args.sweep:
        if "=" not in specification:
            parser.error(f"Invalid --sweep value: {specification}")
        key, raw = specification.split("=", 1)
        options = split_top_level(raw)
        if not options:
            parser.error(f"No values in --sweep {specification}")
        sweeps[key] = [parse_option(option) for option in options]

    base_config = load_config(args.config) if args.config else deepcopy(CONFIG)
    iterations = (
        args.iterations
        if args.iterations is not None
        else int(base_config["SIMULATION"]["ITERATIONS"])
    )
    try:
        runs = run_experiment(
            iterations=iterations,
            seeds=args.seeds,
            sweeps=sweeps,
            metrics=args.metrics,
            base_config=base_config,
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    summary = summarize_runs(runs, list(sweeps), args.metrics)
    paths = write_results(
        runs, summary, args.output, summary_json=args.summary_json
    )
    print(f"Raw runs: {paths[0]}")
    print(f"Summary CSV: {paths[1]}")
    print(f"Summary JSON: {paths[2]}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
