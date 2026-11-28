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
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_METRICS: tuple[str, ...] = (
    "gini",
    "mean",
    "alive",
    "unmet_demand",
    "reciprocity_rate",
    "mean_risk_tolerance",
)

# Metrics whose per-tick values are flows: experiments should report the run
# total, not the last tick's value. Stock and cumulative metrics stay final.
SUMMED_METRICS = frozenset(
    {
        "altruism_volume",
        "berries_eaten",
        "berries_gathered",
        "berries_spoiled",
        "collateral_excess_value",
        "collateral_seized",
        "commons_depleted",
        "commons_extracted",
        "commons_regenerated",
        "commons_regenerated_berries",
        "commons_regenerated_wood",
        "commons_waste",
        "commons_waste_berries",
        "commons_waste_wood",
        "crate_trades",
        "crates_crafted",
        "crates_worn",
        "default_loss_berries",
        "default_loss_crates",
        "default_loss_wood",
        "defaults",
        "dividend_paid",
        "failed_trades",
        "gift_berries",
        "gift_default_berries",
        "gift_defaults",
        "gift_repayment_berries",
        "gift_repayments",
        "gifts",
        "goods_consumed",
        "goods_produced",
        "goods_traded",
        "living_cost_due",
        "living_cost_paid",
        "loan_principal_berries",
        "loan_principal_crates",
        "loan_principal_wood",
        "loans_defaulted",
        "loans_late",
        "loans_originated",
        "loans_repaid",
        "market_participants",
        "market_traders",
        "repayment_berries",
        "repayment_crates",
        "repayment_volume",
        "repayment_wood",
        "repayments",
        "shocks",
        "trade_berries",
        "trade_crates",
        "trade_volume",
        "trade_wood",
        "transfers",
        "transfer_volume",
        "unmet_demand",
        "unmet_food",
        "wood_crafted",
        "wood_decayed",
        "wood_gathered",
        "written_off_berries",
        "written_off_crates",
        "written_off_wood",
    }
)

MEAN_METRICS = frozenset(
    {
        "average_trade_price",
        "barter_price",
        "berries_commons_pct",
        "gossip_price",
        "mean_accept_probability",
        "mean_give_probability",
        "mean_health",
        "mean_price_belief",
        "mean_reputation",
        "mean_risk_tolerance",
        "mean_trust",
        "ppf_efficiency",
        "price",
        "production_modifier",
        "reputation_brier",
        "reputation_error",
        "reputation_noise",
        "shock_active",
        "specialisation",
        "storage_utilization",
        "trust_observations",
        "wood_commons_pct",
    }
)

NONZERO_MEAN_METRICS = frozenset(
    {"last_crate_price", "mean_interest_rate"}
)


def _aggregate_metric(history: list[dict[str, Any]], metric: str) -> Any:
    """Aggregate a metric over a run using its measurement semantics."""

    final = history[-1].get(metric)
    if metric not in SUMMED_METRICS and metric not in MEAN_METRICS:
        if metric not in NONZERO_MEAN_METRICS:
            return final
    values = []
    for row in history:
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            continue
        values.append(float(value))
    if not values:
        return final
    if metric in SUMMED_METRICS:
        return float(sum(values))
    if metric in NONZERO_MEAN_METRICS:
        nonzero = [value for value in values if value != 0]
        return float(sum(nonzero) / len(nonzero)) if nonzero else 0.0
    return float(sum(values) / len(values))


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
    world_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> pd.DataFrame:
    """Run every seed/parameter combination and return one row per run.

    ``world_factory`` lets a world reuse the seed/sweep machinery. When it is
    omitted, the original money world is used so existing callers keep working.
    """

    if iterations < 0:
        raise ValueError("iterations must not be negative")
    if iterations == 0:
        raise ValueError("iterations must be at least 1")
    if world_factory is None:
        from worlds.money.world import CONFIG as money_config
        from worlds.money.world import ResourceSimulation

        world_factory = ResourceSimulation
        if base_config is None:
            base_config = money_config
    sweeps = sweeps or {}
    parameter_keys = list(sweeps)
    option_lists = [list(sweeps[key]) for key in parameter_keys]
    combinations = list(itertools.product(*option_lists)) if option_lists else [()]
    metric_names = list(metrics)
    base = deepcopy(base_config) if base_config is not None else {}

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for combination in combinations:
            config = deepcopy(base)
            labels = {}
            for key, value in zip(parameter_keys, combination):
                _set_override(config, key, value)
                labels[key] = _option_label(value)
            config["SIMULATION"]["RANDOM_SEED"] = int(seed)
            simulation = world_factory(config)
            simulation.run(iterations=iterations)
            record: dict[str, Any] = {
                "seed": int(seed),
                "iterations": iterations,
                **labels,
            }
            for metric in metric_names:
                record[metric] = _aggregate_metric(
                    simulation.metrics_history, metric
                )
            recovery = getattr(simulation, "recovery_time", None)
            record["recovery_time"] = recovery() if callable(recovery) else None
            records.append(record)
    return pd.DataFrame(records)


# Two-sided 95% Student-t critical values by degrees of freedom. With few
# seeds the normal 1.96 approximation is far too narrow.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def _t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    for key in keys:
        if key >= df:
            return _T95[key]
    return 1.960


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
            if count > 1:
                half_width = _t_critical(count - 1) * std / math.sqrt(count)
            else:
                # A single seed cannot identify a sampling interval.
                half_width = float("nan")
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
    records = []
    for row in summary.to_dict(orient="records"):
        records.append(
            {
                key: (
                    None
                    if isinstance(value, float) and math.isnan(value)
                    else value
                )
                for key, value in row.items()
            }
        )
    json_path.write_text(
        json.dumps(records, indent=2),
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

    from utils import load_config
    from worlds.money.world import CONFIG

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
