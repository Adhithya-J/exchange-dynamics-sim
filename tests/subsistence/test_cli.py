"""Command-line override tests for the subsistence world."""

import pytest

from worlds.subsistence.cli import build_config, build_parser


def test_agents_override_uses_the_default_config():
    args = build_parser().parse_args(["--agents", "100"])

    config = build_config(args)

    assert config is not None
    assert config["SIMULATION"]["AGENTS"] == 100
    # The override must not mutate the shared defaults.
    from worlds.subsistence.world import CONFIG

    assert CONFIG["SIMULATION"]["AGENTS"] == 10


def test_agents_override_stacks_on_a_custom_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "SIMULATION:\n  AGENTS: 5\n  ITERATIONS: 7\nSEASONS:\n  ENABLED: false\n",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--config", str(path), "--agents", "50"])

    config = build_config(args)

    assert config["SIMULATION"]["AGENTS"] == 50
    assert config["SIMULATION"]["ITERATIONS"] == 7
    assert config["SEASONS"]["ENABLED"] is False


def test_no_overrides_returns_none():
    args = build_parser().parse_args([])

    assert build_config(args) is None


def test_agents_override_rejects_zero():
    args = build_parser().parse_args(["--agents", "0"])

    with pytest.raises(SystemExit):
        build_config(args)
