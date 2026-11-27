"""Run controlled experiments across seeds and parameter ranges.

Example::

    python experiments.py --iterations 500 --seeds 1 2 3 \
        --sweep SIMULATION.AGENTS=5,10 --output results/runs.csv

The raw per-run results are written to ``--output``. A summary with 95%
confidence intervals is written next to it and as JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from main import CONFIG, ResourceSimulation
from utils import load_config

DEFAULT_METRICS: tuple[str, ...] = (
    "gini",
    "mean",
    "alive",
    "unmet_demand",
    "reciprocity_rate",
    "mean_risk_tolerance",
)


def split_top_level(value: str) -> list[str]:
    """Split comma-separated sweep options without breaking JSON lists."""

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for character in value:
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
            current.append(character)
        elif character in "[{(":
            depth += 1
            current.append(character)
        elif character in "]})":
            depth -= 1
            current.append(character)
        elif character == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def parse_option(value: str) -> Any:
    """Parse a sweep option as JSON when possible, otherwise as text."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Override '{dotted_key}' must use SECTION.KEY notation"
        )
    target: Any = config
    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            raise KeyError(f"Unknown config section '{part}' in '{dotted_key}'")
        target = target[part]
    if not isinstance(target, dict):
        raise KeyError(f"Config path '{dotted_key}' is not a section")
    target[parts[-1]] = value


def _option_label(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def run_experiment(
    iterations: int,
    seeds: Iterable[int],
    sweeps: dict[str, Sequence[Any]] | None = None,
    metrics: Iterable[str] = DEFAULT_METRICS,
    base_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run every seed/parameter combination and return one row per run."""

    if iterations < 0:
        raise ValueError("iterations must not be negative")
    sweeps = sweeps or {}
    parameter_keys = list(sweeps)
    option_lists = [list(sweeps[key]) for key in parameter_keys]
    combinations = list(itertools.product(*option_lists)) if option_lists else [()]
    metric_names = list(metrics)
    base = deepcopy(base_config or CONFIG)

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for combination in combinations:
            config = deepcopy(base)
            labels = {}
            for key, value in zip(parameter_keys, combination):
                _set_override(config, key, value)
                labels[key] = _option_label(value)
            config["SIMULATION"]["RANDOM_SEED"] = int(seed)
            simulation = ResourceSimulation(config)
            simulation.run(iterations=iterations)
            final = simulation.metrics_history[-1]
            record: dict[str, Any] = {
                "seed": int(seed),
                "iterations": iterations,
                **labels,
            }
            for metric in metric_names:
                record[metric] = final.get(metric)
            record["recovery_time"] = simulation.recovery_time()
            records.append(record)
    return pd.DataFrame(records)


def summarize_runs(
    runs: pd.DataFrame,
    parameter_columns: Sequence[str],
    metrics: Iterable[str],
) -> pd.DataFrame:
    """Aggregate runs into means, standard deviations, and 95% intervals."""

    metric_names = list(metrics)
    parameter_columns = list(parameter_columns)
    if runs.empty:
        return pd.DataFrame(
            columns=[
                *parameter_columns,
                "metric",
                "n",
                "mean",
                "std",
                "ci_low",
                "ci_high",
            ]
        )
    if parameter_columns:
        groups = runs.groupby(parameter_columns, dropna=False)
    else:
        groups = [((), runs)]

    rows: list[dict[str, Any]] = []
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(parameter_columns, key_tuple))
        for metric in metric_names:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            count = int(values.size)
            if count == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            half_width = 1.96 * std / math.sqrt(count) if count else float("nan")
            rows.append(
                {
                    **labels,
                    "metric": metric,
                    "n": count,
                    "mean": mean,
                    "std": std,
                    "ci_low": mean - half_width,
                    "ci_high": mean + half_width,
                }
            )
    return pd.DataFrame(rows)


def write_results(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    output: str | Path,
    summary_json: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write raw runs, a summary CSV, and a summary JSON file."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_path, index=False)

    summary_csv = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_csv, index=False)

    json_path = (
        Path(summary_json)
        if summary_json
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return output_path, summary_csv, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Custom YAML config")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds, one run per seed and parameter combination",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help=(
            "Parameter sweep as SECTION.KEY=v1,v2 (repeatable). "
            "Values are parsed as JSON when possible."
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
        default="experiments.csv",
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
