"""Create an interactive view of a resource simulation.

By default this writes a self-contained animated world view. Pass
``--plotly`` to create the original Plotly line chart instead.

Run this module directly to create ``simulation.html``::

    python visualize.py

The generated file is self-contained and can be opened in a browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from worlds.money.replay import build_replay
from worlds.money.world import CONFIG, MetricsCalculator, ResourceSimulation
from utils import load_config

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"


def _viewer_asset(name: str) -> str:
    return (VIEWER_DIR / name).read_text(encoding="utf-8")


def build_world_html(
    simulation: ResourceSimulation,
    *,
    title: str = "Exchange Dynamics - The Commons",
) -> str:
    """Render a self-contained world view from a replay of ``simulation``."""

    return build_viewer_html(build_replay(simulation), title=title)


def build_viewer_html(
    replay: dict[str, object],
    *,
    title: str = "Exchange Dynamics - The Commons",
) -> str:
    """Inline the viewer assets and a replay into one standalone HTML file."""

    payload = json.dumps(
        replay, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    shell = _viewer_asset("index.html")
    html = shell.replace(
        "<title>The Commons - Exchange Dynamics</title>",
        f"<title>{title}</title>",
    )
    html = html.replace(
        '<link rel="stylesheet" href="theme.css">',
        "<style>\n" + _viewer_asset("theme.css") + "\n</style>",
    )
    html = html.replace(
        '<script src="replay-loader.js"></script>',
        "<script>\n" + _viewer_asset("replay-loader.js") + "\n</script>",
    )
    html = html.replace(
        '<script src="viewer.js"></script>',
        "<script>\n" + _viewer_asset("viewer.js") + "\n</script>",
    )
    html = html.replace(
        "/*__INLINE_REPLAY__*/",
        "window.__REPLAY__ = " + payload + ";",
    )
    return html


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
        summary["total_transfers"] = 0
        return summary

    return next(
        metrics
        for metrics in simulation.metrics_history
        if metrics["iteration"] == iteration
    )


def _summary_annotation(
    summary: dict[str, float], iteration: int, simulation: ResourceSimulation
) -> dict[str, object]:
    """Build the in-chart summary card."""

    simulation_settings = simulation.config["SIMULATION"]
    agent_settings = simulation.config["AGENTS"]
    market_enabled = simulation.config.get("MARKET", {}).get("ENABLED", False)
    trust_enabled = simulation.config.get("TRUST", {}).get("ENABLED", False)
    shocks_enabled = simulation.config.get("SHOCKS", {}).get("ENABLED", False)
    optional_lines = ""
    if market_enabled:
        optional_lines += (
            f"<br>Unmet demand: {summary.get('unmet_demand', 0.0):,.1f}"
        )
    if trust_enabled:
        optional_lines += (
            f"<br>Mean reputation: {summary.get('mean_reputation', 0.0):.3f}"
        )
    if shocks_enabled:
        active = "yes" if summary.get("shock_active", 0) else "no"
        optional_lines += f"<br>Shock active: {active}"
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
            f"Total transfers: {int(summary['total_transfers'])}"
            f"{optional_lines}<br><br>"
            f"Initial resources: {simulation_settings['INITIAL_RESOURCES']:,.1f} per agent<br>"
            f"Living cost: {agent_settings['LIVING_COST']:,.1f} per iteration"
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
    """Build the legacy animated resource-by-agent Plotly figure."""

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
            gen_score = float(agent["generosity_score"].iloc[0])
            acc_score = float(agent["acceptance_score"].iloc[0])
            risk_score = float(agent["risk_tolerance"].iloc[0])
            traces.append(
                go.Scatter(
                    x=agent["iteration"],
                    y=agent["resources"],
                    mode="lines",
                    name=agent_id,
                    line={"color": color_by_agent[agent_id], "width": 2},
                    hovertemplate=(
                        f"{agent_id}<br>"
                        "Resources: %{y:,.2f}<br>"
                        f"Altruism / generosity: {gen_score:.2f}<br>"
                        f"Acceptance: {acc_score:.2f}<br>"
                        f"Risk tolerance: {risk_score:.2f}<extra></extra>"
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
                    annotations=[_summary_annotation(summary, iteration, simulation)],
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

    show_metrics = bool(simulation.metrics_history) and any(
        metrics.get(key, 0)
        for metrics in simulation.metrics_history
        for key in ("unmet_demand", "mean_reputation", "shock_active")
    )

    if show_metrics:
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.09,
            specs=[[{}], [{"secondary_y": True}]],
        )
        for trace in traces():
            figure.add_trace(trace, row=1, col=1)
        metric_iterations = [0] + [
            metrics["iteration"] for metrics in simulation.metrics_history
        ]
        unmet_demand = [0.0] + [
            metrics.get("unmet_demand", 0.0)
            for metrics in simulation.metrics_history
        ]
        mean_reputation = [0.0] + [
            metrics.get("mean_reputation", 0.0)
            for metrics in simulation.metrics_history
        ]
        shock_active = [0] + [
            metrics.get("shock_active", 0)
            for metrics in simulation.metrics_history
        ]
        figure.add_trace(
            go.Scatter(
                x=metric_iterations,
                y=unmet_demand,
                mode="lines",
                name="Unmet demand",
                line={"color": "#ef4444", "width": 2},
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=metric_iterations,
                y=mean_reputation,
                mode="lines",
                name="Mean reputation",
                line={"color": "#2563eb", "width": 2},
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        if any(shock_active):
            figure.add_trace(
                go.Scatter(
                    x=metric_iterations,
                    y=shock_active,
                    mode="lines",
                    name="Shock active",
                    line={"color": "#b91c1c", "width": 1, "dash": "dot"},
                    fill="tozeroy",
                    fillcolor="rgba(220, 38, 38, 0.10)",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        figure.update_yaxes(title_text="Unmet demand", row=2, col=1)
        figure.update_yaxes(
            title_text="Trust / shock",
            secondary_y=True,
            row=2,
            col=1,
        )
        figure.update_xaxes(title_text="Iteration", row=2, col=1)
        figure.frames = frames
    else:
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
        height=820 if show_metrics else 700,
        margin={"l": 70, "r": 40, "t": 135, "b": 80},
        xaxis={
            "title": "Iteration",
            "range": (
                [iterations[0] - 0.5, iterations[-1] + 0.5]
                if iterations[0] == iterations[-1]
                else [iterations[0], iterations[-1]]
            ),
            "fixedrange": False,
        },
        yaxis={
            "title": "Resources",
            "range": [minimum_resources - y_padding, maximum_resources + y_padding],
        },
        hovermode="closest",
        legend={"title": {"text": "Agents"}},
        annotations=[
            _summary_annotation(first_summary, first_iteration, simulation)
        ],
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
    output: str | Path = "simulation.html",
    iterations: int | None = None,
    config: dict[str, Any] | None = None,
    renderer: str = "world",
) -> Path:
    """Run the configured simulation and write an interactive HTML view."""

    simulation = ResourceSimulation(config or CONFIG)
    simulation.run(iterations=iterations)
    output_path = Path(output)
    if renderer == "plotly":
        figure = build_figure(simulation)
        figure.write_html(output_path, include_plotlyjs=True)
        return output_path
    if renderer != "world":
        raise ValueError("renderer must be 'world' or 'plotly'")
    output_path.write_text(build_world_html(simulation), encoding="utf-8")
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file",
    )
    parser.add_argument(
        "--plotly",
        action="store_true",
        help="Write the original Plotly line chart instead of the world view",
    )
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else None
    renderer = "plotly" if args.plotly else "world"
    output_path = create_visualization(
        args.output, args.iterations, config=cfg, renderer=renderer
    )
    print(f"Interactive visualization written to {output_path}")


if __name__ == "__main__":
    main()
