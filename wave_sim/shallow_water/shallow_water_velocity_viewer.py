import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from wave_sim.shallow_water.shallow_water_plotly_viewer import make_surface_trace
from wave_sim.shallow_water.shallow_water_surface_3d import downsample_frame, prepare_surface_grid
from wave_sim.data.wave_dataset import load_wave_dataset_with_velocity


def velocity_magnitude(u_frame, v_frame, max_surface_points: int) -> np.ndarray:
    u = downsample_frame(u_frame, max_surface_points)
    v = downsample_frame(v_frame, max_surface_points)
    return np.sqrt(u * u + v * v)


def make_velocity_water_trace(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    speed_grid: np.ndarray,
    speed_max: float,
    showscale: bool,
) -> go.Surface:
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=speed_grid,
        colorscale="Turbo",
        cmin=0.0,
        cmax=speed_max,
        colorbar={"title": "speed"} if showscale else None,
        opacity=0.78,
        showscale=showscale,
        name="water speed",
        contours_z={"show": False},
        hovertemplate=(
            "x=%{x:.3f}<br>y=%{y:.3f}<br>eta=%{z:.4f}<br>speed=%{surfacecolor:.4f}"
            "<extra>water speed</extra>"
        ),
    )


def build_velocity_figure(
    frames,
    depth,
    u_frames,
    v_frames,
    max_surface_points: int,
) -> go.Figure:
    water_surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    speed_surfaces = [
        velocity_magnitude(u_frame, v_frame, max_surface_points)
        for u_frame, v_frame in zip(u_frames, v_frames)
    ]
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surfaces[0].shape[0])

    eta_limit = max(float(np.max(np.abs(surface))) for surface in water_surfaces)
    speed_max = max(float(np.max(speed)) for speed in speed_surfaces)
    speed_max = max(speed_max, 1.0e-6)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

    bed_trace = make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.90, True)
    water_trace = make_velocity_water_trace(
        x_grid,
        y_grid,
        water_surfaces[0],
        speed_surfaces[0],
        speed_max,
        True,
    )

    figure_frames = [
        go.Frame(
            data=[
                make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.90, True),
                make_velocity_water_trace(x_grid, y_grid, water, speed, speed_max, True),
            ],
            name=str(index),
        )
        for index, (water, speed) in enumerate(zip(water_surfaces, speed_surfaces))
    ]

    slider_steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(water_surfaces))
    ]

    fig = go.Figure(data=[bed_trace, water_trace], frames=figure_frames)
    fig.update_layout(
        title="Interactive bathymetry + wave surface colored by speed",
        scene={
            "xaxis": {"title": "x", "range": [-1, 1]},
            "yaxis": {"title": "y", "range": [-1, 1]},
            "zaxis": {"title": "z", "range": [z_min, z_max]},
            "aspectratio": {"x": 1, "y": 1, "z": 0.55},
            "camera": {"eye": {"x": -1.55, "y": -1.55, "z": 1.05}},
        },
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 1.02,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 90, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame "},
                "pad": {"t": 42},
                "steps": slider_steps,
            }
        ],
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Plotly viewer with water surface color mapped to speed.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("outputs/wave_dataset_velocity.npz"),
        help="NPZ dataset created with export_wave_dataset.py --store-velocity.",
    )
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, depth, metadata, u_frames, v_frames = load_wave_dataset_with_velocity(args.input_npz)
    if u_frames is None or v_frames is None:
        raise SystemExit(
            "Dataset does not include velocity fields. Recreate it with "
            "export_wave_dataset.py --store-velocity."
        )

    print(f"Loaded wave dataset: {args.input_npz}")
    print(f"Dataset metadata: {metadata}")
    fig = build_velocity_figure(frames, depth, u_frames, v_frames, args.max_surface_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shallow_water_velocity_viewer.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved velocity viewer: {output_path}")


if __name__ == "__main__":
    main()
