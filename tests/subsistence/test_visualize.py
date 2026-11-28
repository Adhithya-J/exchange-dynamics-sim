"""HTML viewer tests for the subsistence world."""

from copy import deepcopy

from worlds.subsistence.replay import build_replay
from worlds.subsistence.visualize import (
    build_viewer_html,
    build_world_html,
    create_visualization,
)
from worlds.subsistence.world import CONFIG, SubsistenceWorld


def small_world() -> SubsistenceWorld:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update({"AGENTS": 4, "RANDOM_SEED": 9})
    world = SubsistenceWorld(config)
    world.run(iterations=4)
    return world


def test_world_html_embeds_a_versioned_replay():
    html = build_world_html(small_world())

    assert "window.__REPLAY__" in html
    assert '"schema_version":3' in html
    assert '"world":"subsistence"' in html
    assert "/*__INLINE_REPLAY__*/" not in html
    assert "<script src=" not in html


def test_world_html_inlines_viewer_and_ppf_panel():
    html = build_world_html(small_world())

    assert "Subsistence" in html
    assert "ppfChart" in html
    assert "Production possibilities" in html
    assert "Barter price" in html
    assert "theme.css" not in html
    assert "<style>" in html


def test_viewer_html_accepts_an_explicit_replay():
    replay = build_replay(small_world())
    html = build_viewer_html(replay, title="A custom run")

    assert "<title>A custom run</title>" in html
    assert '"agent-001"' in html


def test_create_visualization_writes_a_file(tmp_path):
    output = tmp_path / "subsistence.html"
    path = create_visualization(small_world(), output=output)

    assert path == output
    assert path.exists()
    assert path.stat().st_size > 0
    assert "window.__REPLAY__" in path.read_text(encoding="utf-8")
