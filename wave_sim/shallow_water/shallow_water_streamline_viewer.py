import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go

from wave_sim.shallow_water.shallow_water_particle_viewer import bilinear_sample, make_particle_seeds, make_wet_mask, particle_step
from wave_sim.shallow_water.shallow_water_plotly_viewer import make_surface_trace
from wave_sim.shallow_water.shallow_water_surface_3d import downsample_frame, prepare_surface_grid
from wave_sim.shallow_water.shallow_water_velocity_viewer import make_velocity_water_trace, velocity_magnitude
from wave_sim.data.wave_dataset import load_wave_dataset_with_velocity


def resolve_frame_index(frame_index: int, frame_count: int) -> int:
    resolved = frame_index if frame_index >= 0 else frame_count + frame_index
    if resolved < 0 or resolved >= frame_count:
        raise ValueError(f"Frame index {frame_index} is out of range for {frame_count} frames.")
    return resolved


def trace_streamlines(
    eta_frame,
    depth,
    u_frame,
    v_frame,
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    streamline_steps: int,
    step_scale: float,
    wet_depth_threshold: float,
    block_dry_cells: bool,
    integrator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = eta_frame.detach().cpu().numpy()
    u = u_frame.detach().cpu().numpy()
    v = v_frame.detach().cpu().numpy()
    wet_mask = make_wet_mask(depth, wet_depth_threshold) if block_dry_cells else None

    x = seed_x.copy()
    y = seed_y.copy()
    lines_x = [x.copy()]
    lines_y = [y.copy()]
    lines_z = [bilinear_sample(eta, x, y)]

    for _ in range(streamline_steps):
        proposed_x, proposed_y = particle_step(x, y, u, v, step_scale, integrator)
        if wet_mask is not None:
            proposed_is_wet = bilinear_sample(wet_mask, proposed_x, proposed_y) >= 0.5
            x = np.where(proposed_is_wet, proposed_x, x)
            y = np.where(proposed_is_wet, proposed_y, y)
        else:
            x = proposed_x
            y = proposed_y
        lines_x.append(x.copy())
        lines_y.append(y.copy())
        lines_z.append(bilinear_sample(eta, x, y))

    return np.stack(lines_x), np.stack(lines_y), np.stack(lines_z)


def make_streamline_trace(lines_x: np.ndarray, lines_y: np.ndarray, lines_z: np.ndarray) -> go.Scatter3d:
    trace_x = []
    trace_y = []
    trace_z = []
    for line_index in range(lines_x.shape[1]):
        trace_x.extend(lines_x[:, line_index].tolist())
        trace_y.extend(lines_y[:, line_index].tolist())
        trace_z.extend((lines_z[:, line_index] + 0.035).tolist())
        trace_x.append(None)
        trace_y.append(None)
        trace_z.append(None)

    return go.Scatter3d(
        x=trace_x,
        y=trace_y,
        z=trace_z,
        mode="lines",
        name="streamlines",
        line={"color": "#0f172a", "width": 5},
        hoverinfo="skip",
    )


def build_streamline_figure(
    frames,
    depth,
    u_frames,
    v_frames,
    max_surface_points: int,
    frame_index: int,
    seed_count_x: int,
    seed_count_y: int,
    seed_x_min: float,
    seed_x_max: float,
    seed_y_min: float,
    seed_y_max: float,
    streamline_steps: int,
    streamline_step_scale: float,
    wet_depth_threshold: float,
    block_dry_cells: bool,
    particle_integrator: str,
) -> go.Figure:
    resolved_frame_index = resolve_frame_index(frame_index, len(frames))
    water_surface = downsample_frame(frames[resolved_frame_index], max_surface_points)
    speed_surface = velocity_magnitude(u_frames[resolved_frame_index], v_frames[resolved_frame_index], max_surface_points)
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surface.shape[0])

    seed_x, seed_y = make_particle_seeds(
        seed_count_x,
        seed_count_y,
        seed_x_min,
        seed_x_max,
        seed_y_min,
        seed_y_max,
    )
    lines_x, lines_y, lines_z = trace_streamlines(
        frames[resolved_frame_index],
        depth,
        u_frames[resolved_frame_index],
        v_frames[resolved_frame_index],
        seed_x,
        seed_y,
        streamline_steps,
        streamline_step_scale,
        wet_depth_threshold,
        block_dry_cells,
        particle_integrator,
    )

    eta_limit = float(np.max(np.abs(water_surface)))
    speed_max = max(float(np.max(speed_surface)), 1.0e-6)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

    fig = go.Figure(
        data=[
            make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.86, True),
            make_velocity_water_trace(x_grid, y_grid, water_surface, speed_surface, speed_max, True),
            make_streamline_trace(lines_x, lines_y, lines_z),
        ]
    )
    fig.update_layout(
        title=f"Streamlines over speed-colored 3D wave surface, frame {resolved_frame_index + 1}",
        scene={
            "xaxis": {"title": "x", "range": [-1, 1]},
            "yaxis": {"title": "y", "range": [-1, 1]},
            "zaxis": {"title": "z", "range": [z_min, z_max]},
            "aspectratio": {"x": 1, "y": 1, "z": 0.55},
            "camera": {"eye": {"x": -1.55, "y": -1.55, "z": 1.05}},
        },
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Plotly streamline viewer from one velocity frame.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("outputs/wave_dataset_velocity.npz"),
        help="NPZ dataset created with export_wave_dataset.py --store-velocity.",
    )
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--frame-index", type=int, default=-1, help="Velocity frame index to visualize. Negative values count from the end.")
    parser.add_argument("--seed-count-x", type=int, default=8, help="Number of streamline seeds along x.")
    parser.add_argument("--seed-count-y", type=int, default=8, help="Number of streamline seeds along y.")
    parser.add_argument("--seed-x-min", type=float, default=-0.88, help="Minimum seed x coordinate.")
    parser.add_argument("--seed-x-max", type=float, default=0.68, help="Maximum seed x coordinate.")
    parser.add_argument("--seed-y-min", type=float, default=-0.70, help="Minimum seed y coordinate.")
    parser.add_argument("--seed-y-max", type=float, default=0.70, help="Maximum seed y coordinate.")
    parser.add_argument("--streamline-steps", type=int, default=36, help="Number of integration steps per streamline.")
    parser.add_argument("--streamline-step-scale", type=float, default=0.18, help="Scale factor per streamline integration step.")
    parser.add_argument("--wet-depth-threshold", type=float, default=0.055, help="Minimum depth treated as wet.")
    parser.add_argument("--allow-dry-streamlines", action="store_true", help="Allow streamlines to move into dry cells.")
    parser.add_argument(
        "--particle-integrator",
        choices=("euler", "rk2"),
        default="rk2",
        help="Streamline integration method.",
    )
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
    fig = build_streamline_figure(
        frames,
        depth,
        u_frames,
        v_frames,
        args.max_surface_points,
        args.frame_index,
        args.seed_count_x,
        args.seed_count_y,
        args.seed_x_min,
        args.seed_x_max,
        args.seed_y_min,
        args.seed_y_max,
        args.streamline_steps,
        args.streamline_step_scale,
        args.wet_depth_threshold,
        not args.allow_dry_streamlines,
        args.particle_integrator,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shallow_water_streamline_viewer.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved streamline viewer: {output_path}")


if __name__ == "__main__":
    main()
