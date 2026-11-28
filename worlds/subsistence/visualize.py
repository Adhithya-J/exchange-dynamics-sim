"""Build a self-contained HTML view for a subsistence replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import load_config
from worlds.subsistence.replay import build_replay
from worlds.subsistence.world import SubsistenceWorld

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"


def _viewer_asset(name: str) -> str:
    return (VIEWER_DIR / name).read_text(encoding="utf-8")


def build_world_html(
    simulation: SubsistenceWorld,
    *,
    title: str = "Subsistence - Exchange Dynamics",
) -> str:
    """Render a self-contained village view from a replay of ``simulation``."""

    return build_viewer_html(build_replay(simulation), title=title)


def build_viewer_html(
    replay: dict[str, Any],
    *,
    title: str = "Subsistence - Exchange Dynamics",
) -> str:
    """Inline the viewer assets and a replay into one standalone HTML file."""

    payload = json.dumps(
        replay, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    shell = _viewer_asset("index.html")
    html = shell.replace(
        "<title>Subsistence - Exchange Dynamics</title>",
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


def create_visualization(
    simulation: SubsistenceWorld,
    *,
    output: str | Path = "simulation-subsistence.html",
    title: str = "Subsistence - Exchange Dynamics",
) -> Path:
    """Write a self-contained viewer page for a finished simulation."""

    output_path = Path(output)
    output_path.write_text(
        build_world_html(simulation, title=title), encoding="utf-8"
    )
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="simulation-subsistence.html",
        help="HTML file to create",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else None
    simulation = SubsistenceWorld(config)
    simulation.run(iterations=args.iterations)
    output_path = create_visualization(simulation, output=args.output)
    print(f"Viewer written to {output_path}")


if __name__ == "__main__":
    main()
