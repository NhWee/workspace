import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from wave_sim.shallow_water.shallow_water_plotly_viewer import make_surface_trace
from wave_sim.shallow_water.shallow_water_surface_3d import downsample_frame, prepare_surface_grid
from wave_sim.shallow_water.shallow_water_velocity_viewer import make_velocity_water_trace, velocity_magnitude
from wave_sim.data.wave_dataset import load_wave_dataset_with_velocity


def bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    size = field.shape[0]
    gx = (np.clip(x, -1.0, 1.0) + 1.0) * 0.5 * (size - 1)
    gy = (np.clip(y, -1.0, 1.0) + 1.0) * 0.5 * (size - 1)

    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, size - 1)
    y1 = np.clip(y0 + 1, 0, size - 1)
    x0 = np.clip(x0, 0, size - 1)
    y0 = np.clip(y0, 0, size - 1)

    wx = gx - x0
    wy = gy - y0

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]
    return (1.0 - wx) * (1.0 - wy) * f00 + wx * (1.0 - wy) * f10 + (1.0 - wx) * wy * f01 + wx * wy * f11


def make_particle_seeds(
    count_x: int,
    count_y: int,
    x_min: float = -0.88,
    x_max: float = -0.58,
    y_min: float = -0.62,
    y_max: float = 0.62,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, count_x)
    y = np.linspace(y_min, y_max, count_y)
    xx, yy = np.meshgrid(x, y)
    return xx.reshape(-1), yy.reshape(-1)


def make_wet_mask(depth, wet_depth_threshold: float) -> np.ndarray:
    return (depth.detach().cpu().numpy() > wet_depth_threshold).astype(np.float32)


def particle_step(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    step_scale: float,
    integrator: str,
) -> tuple[np.ndarray, np.ndarray]:
    if integrator == "euler":
        proposed_x = x + step_scale * bilinear_sample(u, x, y)
        proposed_y = y + step_scale * bilinear_sample(v, x, y)
    elif integrator == "rk2":
        velocity_x = bilinear_sample(u, x, y)
        velocity_y = bilinear_sample(v, x, y)
        midpoint_x = np.clip(x + 0.5 * step_scale * velocity_x, -1.0, 1.0)
        midpoint_y = np.clip(y + 0.5 * step_scale * velocity_y, -1.0, 1.0)
        proposed_x = x + step_scale * bilinear_sample(u, midpoint_x, midpoint_y)
        proposed_y = y + step_scale * bilinear_sample(v, midpoint_x, midpoint_y)
    else:
        raise ValueError(f"Unknown particle integrator: {integrator}")

    return np.clip(proposed_x, -1.0, 1.0), np.clip(proposed_y, -1.0, 1.0)


def trace_particles(
    frames,
    u_frames,
    v_frames,
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    step_scale: float,
    depth=None,
    wet_depth_threshold: float = 0.055,
    block_dry_cells: bool = True,
    integrator: str = "rk2",
):
    x = seed_x.copy()
    y = seed_y.copy()
    wet_mask = make_wet_mask(depth, wet_depth_threshold) if block_dry_cells and depth is not None else None

    paths_x = [x.copy()]
    paths_y = [y.copy()]
    paths_z = [bilinear_sample(frames[0].detach().cpu().numpy(), x, y)]

    for eta_frame, u_frame, v_frame in zip(frames[1:], u_frames[1:], v_frames[1:]):
        u = u_frame.detach().cpu().numpy()
        v = v_frame.detach().cpu().numpy()
        proposed_x, proposed_y = particle_step(x, y, u, v, step_scale, integrator)
        if wet_mask is not None:
            proposed_is_wet = bilinear_sample(wet_mask, proposed_x, proposed_y) >= 0.5
            x = np.where(proposed_is_wet, proposed_x, x)
            y = np.where(proposed_is_wet, proposed_y, y)
        else:
            x = proposed_x
            y = proposed_y
        z = bilinear_sample(eta_frame.detach().cpu().numpy(), x, y)
        paths_x.append(x.copy())
        paths_y.append(y.copy())
        paths_z.append(z)

    return np.stack(paths_x), np.stack(paths_y), np.stack(paths_z)


def make_particle_trace(paths_x: np.ndarray, paths_y: np.ndarray, paths_z: np.ndarray) -> go.Scatter3d:
    line_x = []
    line_y = []
    line_z = []
    for particle_index in range(paths_x.shape[1]):
        line_x.extend(paths_x[:, particle_index].tolist())
        line_y.extend(paths_y[:, particle_index].tolist())
        line_z.extend((paths_z[:, particle_index] + 0.025).tolist())
        line_x.append(None)
        line_y.append(None)
        line_z.append(None)

    return go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        name="particle paths",
        line={"color": "#111827", "width": 4},
        hoverinfo="skip",
    )


def build_particle_figure(
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
) -> go.Figure:
    water_surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    speed_surfaces = [
        velocity_magnitude(u_frame, v_frame, max_surface_points)
        for u_frame, v_frame in zip(u_frames, v_frames)
    ]
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surfaces[-1].shape[0])

    eta_limit = max(float(np.max(np.abs(surface))) for surface in water_surfaces)
    speed_max = max(max(float(np.max(speed)), 1.0e-6) for speed in speed_surfaces)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

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

    fig = go.Figure(
        data=[
            make_surface_trace(x_grid, y_grid, bed_surface, "bathymetry", "Earth", 0.86, True),
            make_velocity_water_trace(
                x_grid,
                y_grid,
                water_surfaces[-1],
                speed_surfaces[-1],
                speed_max,
                True,
            ),
            make_particle_trace(paths_x, paths_y, paths_z),
        ]
    )
    fig.update_layout(
        title="Particle traces over speed-colored 3D wave surface",
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
    parser = argparse.ArgumentParser(description="Create a Plotly viewer with particle traces advected by u/v fields.")
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
    fig = build_particle_figure(
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
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shallow_water_particle_viewer.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved particle viewer: {output_path}")


if __name__ == "__main__":
    main()
