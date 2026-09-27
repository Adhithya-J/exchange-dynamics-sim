import pytest
from main import ResourceSimulation
from visualize import build_figure


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
