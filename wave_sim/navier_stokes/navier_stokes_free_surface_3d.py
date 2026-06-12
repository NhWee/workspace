# 2D Navier-Stokes velocity field coupled to a 3D free-surface height field.
# du/dt + (u * grad)u = -grad(p) + viscosity * laplacian(u) + force
# div(u) = 0
# d2 eta/dt2 = wave_speed^2 * laplacian(eta) + vortex_surface_coupling - surface_damping * d eta/dt
# foam is generated from high vorticity, high speed, and elevated crests.
# You can handle the parameters
# size, steps, frame_every, dt, viscosity, pressure_iters, force_strength,
# force_radius, wave_speed, surface_coupling, surface_damping, eta_scale,
# foam_vorticity_threshold, foam_speed_threshold, foam_birth, foam_decay,
# max_surface_points, fps, frame_duration_ms, output
import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
import torch

from wave_sim.navier_stokes.navier_stokes_foam_2d import (
    add_vortex_forces,
    advect,
    curl_z,
    initial_velocity,
    laplacian,
    make_grid,
    project_velocity,
    update_foam,
)


def downsample(array: np.ndarray, max_points: int) -> np.ndarray:
    stride = max(1, int(np.ceil(array.shape[0] / max_points)))
    return array[::stride, ::stride]


def normalize01(field: torch.Tensor) -> torch.Tensor:
    field_min = torch.min(field)
    field_max = torch.max(field)
    return (field - field_min) / torch.clamp(field_max - field_min, min=1.0e-6)


def make_initial_eta(size: int, device: torch.device) -> torch.Tensor:
    xx, yy = make_grid(size, device)
    ring_a = torch.exp(-((xx - 0.35) ** 2 + (yy - 0.48) ** 2) / 0.018)
    ring_b = -0.75 * torch.exp(-((xx - 0.66) ** 2 + (yy - 0.54) ** 2) / 0.014)
    ripple = 0.15 * torch.sin(8.0 * np.pi * xx + 3.0 * torch.sin(2.0 * np.pi * yy))
    return 0.035 * (ring_a + ring_b + ripple)


def update_surface(
    eta: torch.Tensor,
    eta_velocity: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dt: float,
    wave_speed: float,
    surface_coupling: float,
    surface_damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eta = advect(eta, u, v, dt)
    eta_velocity = advect(eta_velocity, u, v, dt)

    vorticity = torch.abs(curl_z(u, v, dx))
    speed = torch.sqrt(u * u + v * v)
    swirl_source = normalize01(vorticity) * normalize01(speed)
    swirl_source = swirl_source - torch.mean(swirl_source)

    acceleration = wave_speed * wave_speed * laplacian(eta, dx)
    acceleration = acceleration + surface_coupling * swirl_source
    acceleration = acceleration - surface_damping * eta_velocity
    eta_velocity = eta_velocity + acceleration * dt
    eta = eta + eta_velocity * dt
    eta = eta - torch.mean(eta)
    return eta, eta_velocity


def simulate_free_surface(
    size: int,
    steps: int,
    frame_every: int,
    dt: float,
    viscosity: float,
    pressure_iters: int,
    force_strength: float,
    force_radius: float,
    wave_speed: float,
    surface_coupling: float,
    surface_damping: float,
    eta_scale: float,
    foam_vorticity_threshold: float,
    foam_speed_threshold: float,
    foam_birth: float,
    foam_decay: float,
    max_surface_points: int,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    dx = 1.0 / size
    xx, yy = make_grid(size, device)
    u, v = initial_velocity(size, device)
    eta = make_initial_eta(size, device)
    eta_velocity = torch.zeros_like(eta)
    foam = torch.zeros_like(eta)
    frames = []

    for step in range(steps):
        u = advect(u, u, v, dt)
        v = advect(v, u, v, dt)
        u, v = add_vortex_forces(u, v, xx, yy, step, dt, force_strength, force_radius)

        if viscosity > 0.0:
            u = u + viscosity * laplacian(u, dx) * dt
            v = v + viscosity * laplacian(v, dx) * dt

        u, v = project_velocity(u, v, dx, pressure_iters)
        eta, eta_velocity = update_surface(
            eta,
            eta_velocity,
            u,
            v,
            dx,
            dt,
            wave_speed,
            surface_coupling,
            surface_damping,
        )
        foam = update_foam(
            foam,
            u,
            v,
            dx,
            dt,
            foam_vorticity_threshold,
            foam_speed_threshold,
            foam_birth,
            foam_decay,
        )
        crest = torch.relu((eta - torch.mean(eta)) / torch.clamp(torch.std(eta), min=1.0e-6))
        foam = torch.clamp(torch.maximum(foam, 0.12 * normalize01(crest)), 0.0, 1.0)

        if step % frame_every == 0:
            x = (xx - 0.5).detach().cpu().numpy().astype(np.float32)
            y = (yy - 0.5).detach().cpu().numpy().astype(np.float32)
            z = (eta * eta_scale).detach().cpu().numpy().astype(np.float32)
            foam_np = foam.detach().cpu().numpy().astype(np.float32)
            frames.append((
                downsample(x, max_surface_points),
                downsample(y, max_surface_points),
                downsample(z, max_surface_points),
                downsample(foam_np, max_surface_points),
            ))

    return frames


def make_surface_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    foam: np.ndarray,
    z_limit: float,
    showscale: bool,
) -> go.Surface:
    z_norm = (z + z_limit) / max(2.0 * z_limit, 1.0e-6)
    surface_color = np.clip(0.65 * z_norm + 0.7 * foam, 0.0, 1.0)
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surface_color,
        cmin=0.0,
        cmax=1.0,
        colorscale=[
            [0.0, "#07111f"],
            [0.24, "#0f4c81"],
            [0.52, "#2aa7c8"],
            [0.76, "#b9e6f2"],
            [1.0, "#ffffff"],
        ],
        showscale=showscale,
        colorbar={"title": "eta + foam"} if showscale else None,
        opacity=0.92,
        contours_z={"show": False},
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>eta=%{z:.4f}<extra>free surface</extra>",
    )


def build_figure(frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], frame_duration_ms: int) -> go.Figure:
    z_limit = max(float(np.max(np.abs(z))) for _, _, z, _ in frames)
    z_limit = max(z_limit * 1.25, 0.02)
    first_x, first_y, first_z, first_foam = frames[0]
    fig = go.Figure(data=[make_surface_trace(first_x, first_y, first_z, first_foam, z_limit, True)])
    fig.frames = [
        go.Frame(
            data=[make_surface_trace(x, y, z, foam, z_limit, False)],
            name=str(index),
        )
        for index, (x, y, z, foam) in enumerate(frames)
    ]
    fig.update_layout(
        title="Navier-Stokes driven 3D free-surface foam experiment",
        width=980,
        height=780,
        paper_bgcolor="#06101d",
        plot_bgcolor="#06101d",
        font={"color": "#e5edf7"},
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        scene={
            "xaxis": {"title": "x", "range": [-0.55, 0.55], "backgroundcolor": "#06101d", "gridcolor": "#17324a"},
            "yaxis": {"title": "y", "range": [-0.55, 0.55], "backgroundcolor": "#06101d", "gridcolor": "#17324a"},
            "zaxis": {"title": "eta", "range": [-z_limit, z_limit], "backgroundcolor": "#06101d", "gridcolor": "#17324a"},
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.26},
            "camera": {"eye": {"x": 1.35, "y": 1.25, "z": 0.78}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "x": 0.5,
                "y": 0.02,
                "xanchor": "center",
                "direction": "left",
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
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Navier-Stokes driven 3D free-surface foam experiment.")
    parser.add_argument("--size", type=int, default=144, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=720, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=3, help="Save one viewer frame every N simulation steps.")
    parser.add_argument("--dt", type=float, default=0.0045, help="Simulation time step.")
    parser.add_argument("--viscosity", type=float, default=0.00008, help="Kinematic viscosity.")
    parser.add_argument("--pressure-iters", type=int, default=60, help="Jacobi pressure projection iterations.")
    parser.add_argument("--force-strength", type=float, default=4.4, help="Strength of rotating vortex force sources.")
    parser.add_argument("--force-radius", type=float, default=0.12, help="Radius of rotating vortex force sources.")
    parser.add_argument("--wave-speed", type=float, default=0.18, help="Free-surface wave propagation speed.")
    parser.add_argument("--surface-coupling", type=float, default=0.55, help="How strongly vortices disturb the surface.")
    parser.add_argument("--surface-damping", type=float, default=0.85, help="Damping applied to vertical surface velocity.")
    parser.add_argument("--eta-scale", type=float, default=1.0, help="Vertical display scale for eta.")
    parser.add_argument("--foam-vorticity-threshold", type=float, default=8.0, help="Curl magnitude needed to generate foam.")
    parser.add_argument("--foam-speed-threshold", type=float, default=0.18, help="Speed needed to generate foam.")
    parser.add_argument("--foam-birth", type=float, default=2.2, help="Foam generation rate.")
    parser.add_argument("--foam-decay", type=float, default=0.48, help="Foam fade-out rate.")
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max rendered points per surface axis.")
    parser.add_argument("--fps", type=float, default=None, help="Viewer playback FPS. Overrides --frame-duration-ms when set.")
    parser.add_argument("--frame-duration-ms", type=int, default=30, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/navier_stokes_free_surface_3d.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames = simulate_free_surface(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        dt=args.dt,
        viscosity=args.viscosity,
        pressure_iters=args.pressure_iters,
        force_strength=args.force_strength,
        force_radius=args.force_radius,
        wave_speed=args.wave_speed,
        surface_coupling=args.surface_coupling,
        surface_damping=args.surface_damping,
        eta_scale=args.eta_scale,
        foam_vorticity_threshold=args.foam_vorticity_threshold,
        foam_speed_threshold=args.foam_speed_threshold,
        foam_birth=args.foam_birth,
        foam_decay=args.foam_decay,
        max_surface_points=args.max_surface_points,
        device=device,
    )
    frame_duration_ms = args.frame_duration_ms
    if args.fps is not None:
        frame_duration_ms = max(1, int(round(1000.0 / max(args.fps, 1.0e-6))))

    fig = build_figure(frames, frame_duration_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved Navier-Stokes free-surface viewer: {args.output}")


if __name__ == "__main__":
    main()
