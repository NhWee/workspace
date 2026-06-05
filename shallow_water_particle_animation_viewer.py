import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from shallow_water_particle_viewer import make_particle_seeds, trace_particles
from shallow_water_plotly_viewer import make_surface_trace
from shallow_water_surface_3d import downsample_frame, prepare_surface_grid
from shallow_water_velocity_viewer import make_velocity_water_trace, velocity_magnitude
from wave_dataset import load_wave_dataset_with_velocity


def make_particle_trail_trace(
    paths_x: np.ndarray,
    paths_y: np.ndarray,
    paths_z: np.ndarray,
    frame_index: int,
    trail_length: int,
) -> go.Scatter3d:
    start_index = max(0, frame_index - trail_length + 1)
    line_x = []
    line_y = []
    line_z = []
    for particle_index in range(paths_x.shape[1]):
        line_x.extend(paths_x[start_index : frame_index + 1, particle_index].tolist())
        line_y.extend(paths_y[start_index : frame_index + 1, particle_index].tolist())
        line_z.extend((paths_z[start_index : frame_index + 1, particle_index] + 0.025).tolist())
        line_x.append(None)
        line_y.append(None)
        line_z.append(None)

    return go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        name="particle trails",
        line={"color": "#111827", "width": 4},
        hoverinfo="skip",
    )


def make_particle_marker_trace(paths_x: np.ndarray, paths_y: np.ndarray, paths_z: np.ndarray, frame_index: int):
    return go.Scatter3d(
        x=paths_x[frame_index],
        y=paths_y[frame_index],
        z=paths_z[frame_index] + 0.04,
        mode="markers",
        name="particles",
        marker={
            "size": 4,
            "color": paths_z[frame_index],
            "colorscale": "Viridis",
            "line": {"color": "#f8fafc", "width": 1},
            "opacity": 0.96,
        },
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>eta=%{z:.4f}<extra>particle</extra>",
    )


def make_animation_frame(
    frame_index: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bed_surface: np.ndarray,
    water_surface: np.ndarray,
    speed_surface: np.ndarray,
    speed_max: float,
    paths_x: np.ndarray,
    paths_y: np.ndarray,
    paths_z: np.ndarray,
    trail_length: int,
) -> go.Frame:
    return go.Frame(
        data=[
            make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.86, True),
            make_velocity_water_trace(x_grid, y_grid, water_surface, speed_surface, speed_max, True),
            make_particle_trail_trace(paths_x, paths_y, paths_z, frame_index, trail_length),
            make_particle_marker_trace(paths_x, paths_y, paths_z, frame_index),
        ],
        name=str(frame_index),
    )


def build_particle_animation_figure(
    frames,
    depth,
    u_frames,
    v_frames,
    max_surface_points: int,
    seed_count_x: int,
    seed_count_y: int,
    seed_x_min: float,
    seed_x_max: float,
    seed_y_min: float,
    seed_y_max: float,
    particle_step_scale: float,
    wet_depth_threshold: float,
    block_dry_cells: bool,
    particle_integrator: str,
    trail_length: int,
    frame_duration_ms: int,
) -> go.Figure:
    water_surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    speed_surfaces = [
        velocity_magnitude(u_frame, v_frame, max_surface_points)
        for u_frame, v_frame in zip(u_frames, v_frames)
    ]
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surfaces[0].shape[0])

    seed_x, seed_y = make_particle_seeds(
        seed_count_x,
        seed_count_y,
        seed_x_min,
        seed_x_max,
        seed_y_min,
        seed_y_max,
    )
    paths_x, paths_y, paths_z = trace_particles(
        frames,
        u_frames,
        v_frames,
        seed_x,
        seed_y,
        particle_step_scale,
        depth=depth,
        wet_depth_threshold=wet_depth_threshold,
        block_dry_cells=block_dry_cells,
        integrator=particle_integrator,
    )

    eta_limit = max(float(np.max(np.abs(surface))) for surface in water_surfaces)
    speed_max = max(max(float(np.max(speed)), 1.0e-6) for speed in speed_surfaces)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

    figure_frames = [
        make_animation_frame(
            index,
            x_grid,
            y_grid,
            bed_surface,
            water_surface,
            speed_surface,
            speed_max,
            paths_x,
            paths_y,
            paths_z,
            trail_length,
        )
        for index, (water_surface, speed_surface) in enumerate(zip(water_surfaces, speed_surfaces))
    ]

    slider_steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(water_surfaces))
    ]

    fig = go.Figure(data=figure_frames[0].data, frames=figure_frames)
    fig.update_layout(
        title="Animated particles over speed-colored 3D wave surface",
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
                                "frame": {"duration": frame_duration_ms, "redraw": True},
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
    parser = argparse.ArgumentParser(description="Create an animated Plotly particle viewer for the GPU wave scene.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("outputs/wave_dataset_velocity.npz"),
        help="NPZ dataset created with export_wave_dataset.py --store-velocity.",
    )
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--seed-count-x", type=int, default=6, help="Number of particle seeds along x.")
    parser.add_argument("--seed-count-y", type=int, default=8, help="Number of particle seeds along y.")
    parser.add_argument("--seed-x-min", type=float, default=-0.88, help="Minimum seed x coordinate.")
    parser.add_argument("--seed-x-max", type=float, default=-0.58, help="Maximum seed x coordinate.")
    parser.add_argument("--seed-y-min", type=float, default=-0.62, help="Minimum seed y coordinate.")
    parser.add_argument("--seed-y-max", type=float, default=0.62, help="Maximum seed y coordinate.")
    parser.add_argument("--particle-step-scale", type=float, default=0.55, help="Scale factor for particle advection.")
    parser.add_argument("--wet-depth-threshold", type=float, default=0.055, help="Minimum depth treated as wet.")
    parser.add_argument("--allow-dry-particles", action="store_true", help="Allow particles to move into dry cells.")
    parser.add_argument(
        "--particle-integrator",
        choices=("euler", "rk2"),
        default="rk2",
        help="Particle advection integrator.",
    )
    parser.add_argument("--trail-length", type=int, default=8, help="Number of prior frames kept in each particle trail.")
    parser.add_argument("--frame-duration-ms", type=int, default=110, help="Animation frame duration in milliseconds.")
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
    fig = build_particle_animation_figure(
        frames,
        depth,
        u_frames,
        v_frames,
        args.max_surface_points,
        args.seed_count_x,
        args.seed_count_y,
        args.seed_x_min,
        args.seed_x_max,
        args.seed_y_min,
        args.seed_y_max,
        args.particle_step_scale,
        args.wet_depth_threshold,
        not args.allow_dry_particles,
        args.particle_integrator,
        args.trail_length,
        args.frame_duration_ms,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shallow_water_particle_animation_viewer.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved particle animation viewer: {output_path}")


if __name__ == "__main__":
    main()
