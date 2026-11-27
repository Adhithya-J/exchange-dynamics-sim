from copy import deepcopy

import pytest
from main import CONFIG, ResourceSimulation
from visualize import (
    build_figure,
    build_world_html,
    create_visualization,
)


def gift_config() -> dict:
    config = deepcopy(CONFIG)
    config["SIMULATION"].update(
        {
            "AGENTS": 3,
            "INITIAL_RESOURCES": 100.0,
            "RESOURCE_FLOOR": 0.0,
            "RANDOM_SEED": 11,
            "TRANSFER_AMOUNT_RANGE": (5.0, 5.0),
        }
    )
    config["AGENTS"].update(
        {
            "GENEROSITY": (1.0, 1.0),
            "ACCEPTANCE": (1.0, 1.0),
            "MIN_RESOURCES_AFTER_GIVING": 0.0,
            "MAX_RESOURCES_AFTER_RECEIVING": 1_000_000.0,
            "LIVING_COST": 0.0,
        }
    )
    config["AFFORDABILITY"].update(
        {"MIN_RESOURCES": 0.0, "MAX_RESOURCES": 1.0}
    )
    return config


def test_build_figure_hover_and_yaxis_range():
    sim = ResourceSimulation()
    sim.run(iterations=5)
    fig = build_figure(sim)

    # Verify customdata is not present on traces to prevent bloat
    for trace in fig.data:
        assert trace.customdata is None or len(trace.customdata) == 0

    # Verify yaxis range matches minimum resources without clamping to 0
    df = sim.agent_history_frame()
    min_res = df["resources"].min()
    max_res = df["resources"].max()
    padding = max((max_res - min_res) * 0.05, 1.0)
    assert fig.layout.yaxis.range[0] == min_res - padding


def test_build_figure_raises_on_uninitialized_simulation():
    sim = ResourceSimulation()
    with pytest.raises(ValueError, match="initialized or run"):
        build_figure(sim)


def test_build_figure_zero_iterations_has_valid_xaxis_range():
    sim = ResourceSimulation()
    sim.initialize_agents()
    fig = build_figure(sim)

    assert fig.layout.xaxis.range[0] == -0.5
    assert fig.layout.xaxis.range[1] == 0.5


def test_hover_includes_risk_tolerance():
    sim = ResourceSimulation()
    sim.run(iterations=2)

    fig = build_figure(sim)

    assert "Risk tolerance" in fig.data[0].hovertemplate
    assert "Unmet demand" not in {trace.name for trace in fig.data}


def test_metrics_row_appears_for_enabled_layers():
    config = deepcopy(CONFIG)
    config["SIMULATION"].update(
        {"AGENTS": 3, "ITERATIONS": 3, "INITIAL_RESOURCES": 100.0}
    )
    config["AGENTS"]["LIVING_COST"] = 0.0
    config["MARKET"].update(
        {
            "ENABLED": True,
            "PRODUCTION_RANGE": (0.0, 0.0),
            "CONSUMPTION_RANGE": (1.0, 1.0),
        }
    )
    config["TRUST"]["ENABLED"] = True
    config["SHOCKS"].update(
        {
            "ENABLED": True,
            "SCHEDULE": [
                {
                    "iteration": 2,
                    "type": "scarcity",
                    "magnitude": 0.5,
                    "duration": 1,
                }
            ],
        }
    )
    sim = ResourceSimulation(config)
    sim.run(iterations=3)

    fig = build_figure(sim)

    names = {trace.name for trace in fig.data}
    assert {"Unmet demand", "Mean reputation", "Shock active"} <= names
    assert fig.layout.yaxis2 is not None


def test_create_visualization(tmp_path):
    output_file = tmp_path / "test_sim.html"
    returned_path = create_visualization(output=output_file, iterations=2)

    assert returned_path.exists()
    assert returned_path.stat().st_size > 0
    html = returned_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "window.commonsReplay" in html


def test_world_html_embeds_a_versioned_replay():
    sim = ResourceSimulation(gift_config())
    sim.run(iterations=4)

    html = build_world_html(sim)

    assert "window.__REPLAY__" in html
    assert '"schema_version":1' in html
    assert "build_world_payload" not in html


def test_world_html_is_self_contained_and_embeds_data():
    sim = ResourceSimulation(gift_config())
    sim.run(iterations=4)

    html = build_world_html(sim)

    assert "__WORLD_PAYLOAD__" not in html
    assert "__PAGE_TITLE__" not in html
    assert "The Commons" in html
    assert "commonsReplay" in html
    assert "https://" not in html
    assert "<script src=" not in html
