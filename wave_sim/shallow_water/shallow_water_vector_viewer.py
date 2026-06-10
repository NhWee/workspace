import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from wave_sim.shallow_water.shallow_water_plotly_viewer import make_surface_trace
from wave_sim.shallow_water.shallow_water_surface_3d import downsample_frame, prepare_surface_grid
from wave_sim.shallow_water.shallow_water_velocity_viewer import make_velocity_water_trace, velocity_magnitude
from wave_sim.data.wave_dataset import load_wave_dataset_with_velocity


def sample_vector_field(eta_frame, u_frame, v_frame, vector_step: int):
    eta = eta_frame.detach().cpu().numpy()
    u = u_frame.detach().cpu().numpy()
    v = v_frame.detach().cpu().numpy()
    size = eta.shape[0]
    axis = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(axis, axis)
    sample = slice(vector_step // 2, size, vector_step)
    return (
        xx[sample, sample].reshape(-1),
        yy[sample, sample].reshape(-1),
        eta[sample, sample].reshape(-1),
        u[sample, sample].reshape(-1),
        v[sample, sample].reshape(-1),
    )


def make_vector_trace(eta_frame, u_frame, v_frame, vector_step: int, scale: float, sizeref: float) -> go.Cone:
    x, y, z, u, v = sample_vector_field(eta_frame, u_frame, v_frame, vector_step)
    speed = np.sqrt(u * u + v * v)
    return go.Cone(
        x=x,
        y=y,
        z=z + 0.015,
        u=u * scale,
        v=v * scale,
        w=np.zeros_like(u),
        anchor="tail",
        colorscale="Turbo",
        cmin=0.0,
        cmax=max(float(speed.max()), 1.0e-6),
        sizemode="absolute",
        sizeref=sizeref,
        showscale=False,
        name="velocity vectors",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>speed=%{customdata:.4f}<extra>velocity</extra>",
        customdata=speed,
    )


def build_vector_figure(
    frames,
    depth,
    u_frames,
    v_frames,
    max_surface_points: int,
    vector_step: int,
    vector_scale: float,
    vector_size: float,
) -> go.Figure:
    water_surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    speed_surfaces = [
        velocity_magnitude(u_frame, v_frame, max_surface_points)
        for u_frame, v_frame in zip(u_frames, v_frames)
    ]
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surfaces[0].shape[0])

    eta_limit = max(float(np.max(np.abs(surface))) for surface in water_surfaces)
    speed_max = max(max(float(np.max(speed)), 1.0e-6) for speed in speed_surfaces)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

    bed_trace = make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.88, True)
    water_trace = make_velocity_water_trace(
        x_grid,
        y_grid,
        water_surfaces[0],
        speed_surfaces[0],
        speed_max,
        True,
    )
    vector_trace = make_vector_trace(frames[0], u_frames[0], v_frames[0], vector_step, vector_scale, vector_size)

    figure_frames = [
        go.Frame(
            data=[
                make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.88, True),
                make_velocity_water_trace(x_grid, y_grid, water, speed, speed_max, True),
                make_vector_trace(eta, u, v, vector_step, vector_scale, vector_size),
            ],
            name=str(index),
        )
        for index, (eta, water, speed, u, v) in enumerate(
            zip(frames, water_surfaces, speed_surfaces, u_frames, v_frames)
        )
    ]

    slider_steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(water_surfaces))
    ]

    fig = go.Figure(data=[bed_trace, water_trace, vector_trace], frames=figure_frames)
    fig.update_layout(
        title="Interactive wave surface with speed color and velocity vectors",
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
                                "frame": {"duration": 110, "redraw": True},
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
    parser = argparse.ArgumentParser(description="Create a Plotly viewer with speed color and velocity cones.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("outputs/wave_dataset_velocity.npz"),
        help="NPZ dataset created with export_wave_dataset.py --store-velocity.",
    )
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--vector-step", type=int, default=16, help="Sample every N grid cells for velocity cones.")
    parser.add_argument("--vector-scale", type=float, default=1.8, help="Scale factor for u/v cone direction vectors.")
    parser.add_argument("--vector-size", type=float, default=0.035, help="Absolute cone size reference.")
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
    fig = build_vector_figure(
        frames,
        depth,
        u_frames,
        v_frames,
        args.max_surface_points,
        args.vector_step,
        args.vector_scale,
        args.vector_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shallow_water_vector_viewer.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved vector viewer: {output_path}")


if __name__ == "__main__":
    main()
