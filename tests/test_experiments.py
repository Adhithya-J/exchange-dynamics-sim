"""Tests for the seed and parameter sweep experiment harness."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from experiments import (
    _set_override,
    main,
    parse_option,
    run_experiment,
    split_top_level,
    summarize_runs,
    write_results,
)
from main import CONFIG


def make_config() -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update(
        {
            "AGENTS": 2,
            "ITERATIONS": 2,
            "INITIAL_RESOURCES": 100.0,
            "RANDOM_SEED": 1,
            "TRANSFER_AMOUNT_RANGE": (1.0, 1.0),
        }
    )
    config["AGENTS"]["LIVING_COST"] = 0.0
    return config


class TestParsing:
    def test_split_top_level_keeps_json_lists_together(self):
        assert split_top_level("5,10") == ["5", "10"]
        assert split_top_level("[0, 2],[1, 3]") == ["[0, 2]", "[1, 3]"]
        assert split_top_level('"a,b",c') == ['"a,b"', "c"]

    def test_parse_option_uses_json_then_text(self):
        assert parse_option("true") is True
        assert parse_option("5") == 5
        assert parse_option("[1, 2]") == [1, 2]
        assert parse_option("scarcity") == "scarcity"

    def test_override_requires_section_key_notation(self):
        with pytest.raises(ValueError, match="SECTION.KEY"):
            _set_override(make_config(), "AGENTS", 3)

    def test_override_rejects_unknown_sections(self):
        with pytest.raises(KeyError, match="Unknown config section"):
            _set_override(make_config(), "NOPE.KEY", 3)

    def test_override_sets_nested_value(self):
        config = make_config()

        _set_override(config, "SIMULATION.AGENTS", 7)

        assert config["SIMULATION"]["AGENTS"] == 7


class TestRunExperiment:
    def test_one_row_per_seed(self):
        runs = run_experiment(
            iterations=2,
            seeds=[1, 2],
            metrics=["gini"],
            base_config=make_config(),
        )

        assert len(runs) == 2
        assert set(runs["seed"]) == {1, 2}
        assert set(runs["iterations"]) == {2}
        assert runs["gini"].between(0.0, 1.0).all()

    def test_sweep_builds_each_combination(self):
        runs = run_experiment(
            iterations=2,
            seeds=[1],
            sweeps={"SIMULATION.AGENTS": [2, 3]},
            metrics=["alive"],
            base_config=make_config(),
        )

        assert len(runs) == 2
        assert set(runs["SIMULATION.AGENTS"]) == {"2", "3"}
        assert set(runs["alive"]) == {2, 3}

    def test_two_sweeps_produce_the_cartesian_product(self):
        config = make_config()
        config["MARKET"]["ENABLED"] = False
        runs = run_experiment(
            iterations=2,
            seeds=[1],
            sweeps={
                "SIMULATION.AGENTS": [2, 3],
                "AGENTS.LIVING_COST": [0.0, 1.0],
            },
            metrics=["alive"],
            base_config=config,
        )

        assert len(runs) == 4

    def test_negative_iterations_are_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            run_experiment(
                iterations=-1,
                seeds=[1],
                metrics=["gini"],
                base_config=make_config(),
            )


class TestSummaries:
    def test_summary_reports_confidence_interval(self):
        runs = run_experiment(
            iterations=2,
            seeds=[1, 2, 3],
            metrics=["gini"],
            base_config=make_config(),
        )

        summary = summarize_runs(runs, [], ["gini"])

        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["n"] == 3
        assert row["std"] >= 0.0
        assert row["ci_low"] <= row["mean"] <= row["ci_high"]

    def test_summary_groups_by_swept_parameter(self):
        runs = run_experiment(
            iterations=2,
            seeds=[1, 2],
            sweeps={"SIMULATION.AGENTS": [2, 3]},
            metrics=["alive"],
            base_config=make_config(),
        )

        summary = summarize_runs(runs, ["SIMULATION.AGENTS"], ["alive"])

        assert set(summary["SIMULATION.AGENTS"]) == {"2", "3"}
        assert set(summary["n"]) == {2}

    def test_empty_runs_return_an_empty_summary(self):
        empty = pd.DataFrame(columns=["seed", "gini"])

        summary = summarize_runs(empty, [], ["gini"])

        assert summary.empty

    def test_write_results_creates_csv_and_json(self, tmp_path):
        runs = run_experiment(
            iterations=2,
            seeds=[1, 2],
            metrics=["gini"],
            base_config=make_config(),
        )
        summary = summarize_runs(runs, [], ["gini"])

        output, summary_csv, summary_json = write_results(
            runs, summary, tmp_path / "runs.csv"
        )

        assert output.exists()
        assert summary_csv.exists()
        assert summary_json.exists()
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        assert payload
        assert payload[0]["metric"] == "gini"


class TestCommandLine:
    def test_cli_writes_raw_and_summary_files(self, tmp_path):
        output = tmp_path / "results" / "runs.csv"

        main(
            [
                "--iterations",
                "2",
                "--seeds",
                "1",
                "2",
                "--sweep",
                "SIMULATION.AGENTS=2",
                "--metrics",
                "gini",
                "alive",
                "--output",
                str(output),
            ]
        )

        assert output.exists()
        assert output.with_name("runs_summary.csv").exists()
        assert output.with_name("runs_summary.json").exists()
        raw = pd.read_csv(output)
        assert len(raw) == 2
