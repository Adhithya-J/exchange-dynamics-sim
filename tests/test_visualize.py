import pytest
from main import ResourceSimulation
from visualize import build_figure, create_visualization


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


def test_create_visualization(tmp_path):
    output_file = tmp_path / "test_sim.html"
    returned_path = create_visualization(output=output_file, iterations=2)

    assert returned_path.exists()
    assert returned_path.stat().st_size > 0
