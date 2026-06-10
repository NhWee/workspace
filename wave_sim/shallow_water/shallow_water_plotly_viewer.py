import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
import torch

from wave_sim.shallow_water.shallow_water_bathymetry_3d import simulate_bathymetry
from wave_sim.shallow_water.shallow_water_surface_3d import downsample_frame, prepare_surface_grid
from wave_sim.data.wave_dataset import load_wave_dataset


def make_surface_trace(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    name: str,
    colorscale: str,
    opacity: float,
    showscale: bool = False,
) -> go.Surface:
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        name=name,
        colorscale=colorscale,
        opacity=opacity,
        showscale=showscale,
        contours_z={"show": False},
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.4f}<extra>" + name + "</extra>",
    )


def build_interactive_figure(
    frames: list[torch.Tensor],
    depth: torch.Tensor,
    max_surface_points: int,
) -> go.Figure:
    water_surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surfaces[0].shape[0])

    eta_limit = max(float(np.max(np.abs(surface))) for surface in water_surfaces)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

    bed_trace = make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.94, True)
    water_trace = make_surface_trace(x_grid, y_grid, water_surfaces[0], "water", "Blues", 0.68)

    figure_frames = [
        go.Frame(
            data=[
                make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.94, True),
                make_surface_trace(x_grid, y_grid, surface, "water", "Blues", 0.68),
            ],
            name=str(index),
        )
        for index, surface in enumerate(water_surfaces)
    ]

    steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(water_surfaces))
    ]

    fig = go.Figure(data=[bed_trace, water_trace], frames=figure_frames)
    fig.update_layout(
        title="Interactive bathymetry + GPU wave surface",
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
                "steps": steps,
            }
        ],
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive Plotly HTML viewer for the GPU wave scene.")
    parser.add_argument("--size", type=int, default=192, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=18, help="Save one viewer frame every N simulation steps.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravity coefficient g.")
    parser.add_argument("--dt", default="auto", help="Time step, or 'auto' to use a CFL-based value.")
    parser.add_argument("--cfl", type=float, default=0.35, help="CFL factor used when --dt auto.")
    parser.add_argument("--damping", type=float, default=0.9994, help="Global damping per step.")
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional NPZ dataset from export_wave_dataset.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_npz:
        frames, depth, metadata = load_wave_dataset(args.input_npz)
        print(f"Loaded wave dataset: {args.input_npz}")
        print(f"Dataset metadata: {metadata}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        dt = None if str(args.dt).lower() == "auto" else float(args.dt)

        frames, depth = simulate_bathymetry(
            size=args.size,
            steps=args.steps,
            frame_every=args.frame_every,
            gravity=args.gravity,
            dt=dt,
            damping=args.damping,
            device=device,
            cfl=args.cfl,
        )
    fig = build_interactive_figure(frames, depth, args.max_surface_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shallow_water_plotly_viewer.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved interactive viewer: {output_path}")


if __name__ == "__main__":
    main()
