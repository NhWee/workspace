"""Build an interactive Plotly viewer from an APIC-MPM particle cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an APIC-MPM particle cache as an interactive 3D animation.")
    parser.add_argument("--input", type=Path, default=Path("outputs/apic_wave_tank_3d.npz"), help="Input particle cache.")
    parser.add_argument("--output", type=Path, default=Path("outputs/apic_wave_tank_3d_viewer.html"), help="Output HTML viewer.")
    parser.add_argument("--render-particles", type=int, default=15_000, help="Maximum coherent particle subset drawn per frame.")
    parser.add_argument("--frame-duration-ms", type=int, default=70, help="Animation frame duration.")
    return parser.parse_args()


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=True) as cache:
        positions = cache["positions"]
        velocities = cache["velocities"]
        if positions.dtype == object:
            positions = np.stack([np.asarray(frame, dtype=np.float32) for frame in positions])
            velocities = np.stack([np.asarray(frame, dtype=np.float32) for frame in velocities])
        metadata = json.loads(str(cache["metadata"].item()))
    return positions.astype(np.float32), velocities.astype(np.float32), metadata


def particle_trace(positions: np.ndarray, velocities: np.ndarray, speed_limit: float) -> go.Scatter3d:
    speed = np.linalg.vector_norm(velocities, axis=1)
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={
            "size": 1.55,
            "color": speed,
            "colorscale": "Turbo",
            "cmin": 0.0,
            "cmax": speed_limit,
            "opacity": 0.72,
            "colorbar": {"title": "speed"},
        },
        hoverinfo="skip",
    )


def tank_edges() -> go.Scatter3d:
    segments = []
    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float32,
    )
    for index, corner in enumerate(corners):
        for axis in range(3):
            neighbor = corner.copy()
            neighbor[axis] = 1.0 - neighbor[axis]
            matches = np.where(np.all(np.isclose(corners, neighbor), axis=1))[0]
            if len(matches) and index < matches[0]:
                segments.extend((corner, neighbor, np.array([np.nan, np.nan, np.nan], dtype=np.float32)))
    values = np.asarray(segments)
    return go.Scatter3d(
        x=values[:, 0], y=values[:, 1], z=values[:, 2],
        mode="lines",
        line={"color": "rgba(155, 215, 240, 0.35)", "width": 2},
        hoverinfo="skip",
        showlegend=False,
    )


def main() -> None:
    args = parse_args()
    positions, velocities, metadata = load_cache(args.input)
    count = min(args.render_particles, positions.shape[1])
    indices = np.linspace(0, positions.shape[1] - 1, count, dtype=np.int64)
    positions = positions[:, indices]
    velocities = velocities[:, indices]
    speed_limit = max(1.0e-6, float(np.quantile(np.linalg.vector_norm(velocities, axis=2), 0.99)))

    first = particle_trace(positions[0], velocities[0], speed_limit)
    frames = [
        go.Frame(data=[particle_trace(frame_positions, frame_velocities, speed_limit)], name=str(index))
        for index, (frame_positions, frame_velocities) in enumerate(zip(positions, velocities, strict=True))
    ]
    figure = go.Figure(data=[first, tank_edges()], frames=frames)
    figure.update_layout(
        title={"text": "CUDA APIC-MPM 3D Free-Surface Particle Simulation", "x": 0.5},
        paper_bgcolor="#061018",
        plot_bgcolor="#061018",
        font={"color": "#d8edf5"},
        scene={
            "xaxis": {"range": [0, 1], "title": "x", "backgroundcolor": "#071721", "gridcolor": "#1a3b4a"},
            "yaxis": {"range": [0, 1], "title": "height", "backgroundcolor": "#071721", "gridcolor": "#1a3b4a"},
            "zaxis": {"range": [0, 1], "title": "z", "backgroundcolor": "#071721", "gridcolor": "#1a3b4a"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.55, "y": 1.15, "z": 1.55}},
        },
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        updatemenus=[{
            "type": "buttons",
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": args.frame_duration_ms, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.14,
            "len": 0.72,
            "y": 0.03,
            "steps": [
                {"label": str(index), "method": "animate", "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                for index in range(len(frames))
            ],
        }],
        annotations=[{
            "text": f"{metadata['n_grid']}^3 grid | {metadata['particles']:,} simulated particles | {count:,} coherent render particles",
            "x": 0.5, "y": 0.01, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 12, "color": "#9cc6d4"},
        }],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved APIC particle viewer: {args.output}")


if __name__ == "__main__":
    main()
