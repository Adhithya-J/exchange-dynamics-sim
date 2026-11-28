"""Tests for subsistence flag sweeps."""

from copy import deepcopy

from worlds.subsistence.experiments import (
    main,
    run_experiment,
    summarize_runs,
)
from worlds.subsistence.world import CONFIG


def small_config() -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 3, "RANDOM_SEED": 1})
    return config


def test_sweep_runs_every_seed_and_flag():
    runs = run_experiment(
        iterations=3,
        seeds=[1, 2],
        sweeps={"FEATURES.GIFTING": [True, False]},
        metrics=["alive", "trade_berries", "barter_price"],
        base_config=small_config(),
    )

    assert len(runs) == 4
    assert set(runs["seed"]) == {1, 2}
    assert set(runs["FEATURES.GIFTING"]) == {"true", "false"}
    assert set(runs["alive"]).issubset({0, 3})
    assert "recovery_time" in runs.columns
    assert runs["recovery_time"].isna().all()


def test_summary_groups_by_flag():
    runs = run_experiment(
        iterations=3,
        seeds=[1, 2],
        sweeps={"FEATURES.TRADE": [True, False]},
        metrics=["alive", "trade_berries"],
        base_config=small_config(),
    )

    summary = summarize_runs(runs, ["FEATURES.TRADE"], ["alive", "trade_berries"])

    assert set(summary["metric"]) == {"alive", "trade_berries"}
    assert set(summary["FEATURES.TRADE"]) == {"true", "false"}
    assert (summary["n"] == 2).all()


def test_cli_writes_runs_and_summary(tmp_path):
    output = tmp_path / "runs.csv"
    main(
        [
            "--iterations",
            "2",
            "--seeds",
            "1",
            "--sweep",
            "FEATURES.GIFTING=true,false",
            "--metrics",
            "alive",
            "trade_berries",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    assert output.with_name("runs_summary.csv").exists()
    assert output.with_name("runs_summary.json").exists()
