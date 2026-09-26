"""Create an interactive Plotly view of a resource simulation.

Run this module directly to create ``simulation.html``::

    python visualize.py

The generated file is self-contained and can be opened in a browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from main import CONFIG, MetricsCalculator, ResourceSimulation


def _summary_for_iteration(
    simulation: ResourceSimulation,
    agent_history: pd.DataFrame,
    iteration: int,
) -> dict[str, float]:
    """Return summary values for a slider position, including iteration zero."""

    if iteration == 0:
        resources = agent_history.loc[
            agent_history["iteration"] == 0, ["resources"]
        ]
        summary = MetricsCalculator.calculate_statistics(resources)
        summary["transfers"] = 0
        return summary

    return next(
        metrics
        for metrics in simulation.metrics_history
        if metrics["iteration"] == iteration
    )


def _summary_annotation(summary: dict[str, float], iteration: int) -> dict[str, object]:
    """Build the in-chart summary card."""

    return {
        "x": 0.99,
        "y": 1.16,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "right",
        "yanchor": "top",
        "align": "left",
        "showarrow": False,
        "text": (
            f"<b>Iteration {iteration:,}</b><br>"
            f"Average: {summary['mean']:,.1f}<br>"
            f"Median: {summary['median']:,.1f}<br>"
            f"Gini: {summary['gini']:.3f}<br>"
            f"Total: {summary['total']:,.1f}<br>"
            f"Transfers: {int(summary['transfers'])}"
        ),
        "bgcolor": "rgba(255, 255, 255, 0.94)",
        "bordercolor": "#9ca3af",
        "borderwidth": 1,
        "borderpad": 8,
        "font": {"size": 12, "color": "#111827"},
    }


def _iteration_marker(iteration: int) -> dict[str, object]:
    return {
        "type": "line",
        "xref": "x",
        "yref": "paper",
        "x0": iteration,
        "x1": iteration,
        "y0": 0,
        "y1": 1,
        "line": {"color": "#6b7280", "width": 1, "dash": "dot"},
    }


def build_figure(simulation: ResourceSimulation) -> go.Figure:
    """Build an animated resource-by-agent Plotly figure."""

    agent_history = simulation.agent_history_frame()
    if agent_history.empty:
        raise ValueError("The simulation must be initialized or run before plotting")

    agent_ids = list(agent_history["id"].drop_duplicates())
    iterations = sorted(agent_history["iteration"].unique().tolist())
    last_iteration = iterations[-1]
    if last_iteration > 500:
        step = max(5, int(last_iteration / 100 / 5 + 0.5) * 5)
        slider_iterations = list(range(iterations[0], last_iteration + 1, step))
        if slider_iterations[-1] != last_iteration:
            slider_iterations.append(last_iteration)
    else:
        slider_iterations = iterations
    colors = qualitative.Plotly
    color_by_agent = {
        agent_id: colors[index % len(colors)]
        for index, agent_id in enumerate(agent_ids)
    }

    def traces() -> list[go.Scatter]:
        traces = []
        for agent_id in agent_ids:
            agent = agent_history[agent_history["id"] == agent_id]
            traces.append(
                go.Scatter(
                    x=agent["iteration"],
                    y=agent["resources"],
                    mode="lines",
                    name=agent_id,
                    line={"color": color_by_agent[agent_id], "width": 2},
                    hovertemplate=(
                        f"{agent_id}<br>Resources: %{{y:,.2f}}<extra></extra>"
                    ),
                )
            )
        return traces

    frames = []
    for iteration in slider_iterations:
        summary = _summary_for_iteration(simulation, agent_history, iteration)
        frames.append(
            go.Frame(
                name=str(iteration),
                layout=go.Layout(
                    annotations=[_summary_annotation(summary, iteration)],
                    shapes=[_iteration_marker(iteration)],
                ),
            )
        )

    first_iteration = iterations[0]
    first_summary = _summary_for_iteration(
        simulation, agent_history, first_iteration
    )
    maximum_resources = float(agent_history["resources"].max())
    minimum_resources = float(agent_history["resources"].min())
    y_padding = max((maximum_resources - minimum_resources) * 0.05, 1.0)

    figure = go.Figure(
        data=traces(),
        frames=frames,
    )
    figure.update_layout(
        title={
            "text": "Resources by Agent Over Time",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        height=700,
        margin={"l": 70, "r": 40, "t": 135, "b": 80},
        xaxis={
            "title": "Iteration",
            "range": [iterations[0], iterations[-1]],
            "fixedrange": False,
        },
        yaxis={
            "title": "Resources",
            "range": [max(0, minimum_resources - y_padding), maximum_resources + y_padding],
        },
        hovermode="x unified",
        legend={"title": {"text": "Agents"}},
        annotations=[_summary_annotation(first_summary, first_iteration)],
        shapes=[_iteration_marker(first_iteration)],
        sliders=[
            {
                "active": 0,
                "x": 0,
                "y": -0.12,
                "len": 1,
                "currentvalue": {"prefix": "Iteration: "},
                "steps": [
                    {
                        "label": f"{iteration:,}",
                        "method": "animate",
                        "args": [
                            [str(iteration)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for iteration in slider_iterations
                ],
            }
        ],
    )
    return figure


def create_visualization(
    output: str | Path = "simulation.html", iterations: int | None = None
) -> Path:
    """Run the configured simulation and write an interactive HTML chart."""

    simulation = ResourceSimulation(CONFIG)
    simulation.run(iterations=iterations)
    output_path = Path(output)
    figure = build_figure(simulation)
    figure.write_html(output_path, include_plotlyjs=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="simulation.html", help="HTML file to create"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the configured number of iterations",
    )
    args = parser.parse_args()
    output_path = create_visualization(args.output, args.iterations)
    print(f"Interactive visualization written to {output_path}")


if __name__ == "__main__":
    main()
