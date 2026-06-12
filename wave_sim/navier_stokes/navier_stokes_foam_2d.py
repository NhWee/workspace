# 2D incompressible Navier-Stokes vortex and foam experiment using PyTorch GPU acceleration.
# du/dt + (u * grad)u = -grad(p) + viscosity * laplacian(u) + force
# div(u) = 0
# foam is a visual scalar generated from high vorticity and speed, then advected and decayed.
# You can handle the parameters
# size, steps, frame_every, dt, viscosity, pressure_iters, force_strength,
# force_radius, foam_vorticity_threshold, foam_speed_threshold, foam_birth,
# foam_decay, output
import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F


def gradient_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=1) - torch.roll(field, shifts=1, dims=1)) / (2.0 * dx)


def gradient_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=0) - torch.roll(field, shifts=1, dims=0)) / (2.0 * dx)


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(u, dx) + gradient_y(v, dx)


def curl_z(u: torch.Tensor, v: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(v, dx) - gradient_y(u, dx)


def laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        torch.roll(field, shifts=1, dims=0)
        + torch.roll(field, shifts=-1, dims=0)
        + torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
        - 4.0 * field
    ) / (dx * dx)


def make_grid(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, size + 1, device=device)[:-1]
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def advect(field: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    size = field.shape[0]
    yy, xx = make_grid(size, field.device)
    x_prev = torch.remainder(xx - u * dt, 1.0)
    y_prev = torch.remainder(yy - v * dt, 1.0)
    sample_grid = torch.stack((2.0 * x_prev - 1.0, 2.0 * y_prev - 1.0), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, 0]


def project_velocity(u: torch.Tensor, v: torch.Tensor, dx: float, pressure_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    div = divergence(u, v, dx)
    pressure = torch.zeros_like(div)
    for _ in range(pressure_iters):
        pressure = (
            torch.roll(pressure, shifts=1, dims=0)
            + torch.roll(pressure, shifts=-1, dims=0)
            + torch.roll(pressure, shifts=1, dims=1)
            + torch.roll(pressure, shifts=-1, dims=1)
            - div * dx * dx
        ) * 0.25

    u = u - gradient_x(pressure, dx)
    v = v - gradient_y(pressure, dx)
    return u, v


def add_vortex_forces(
    u: torch.Tensor,
    v: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    step: int,
    dt: float,
    force_strength: float,
    force_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = [
        (0.32 + 0.08 * np.sin(step * 0.021), 0.48 + 0.10 * np.cos(step * 0.017), 1.0),
        (0.68 + 0.07 * np.cos(step * 0.019), 0.53 + 0.09 * np.sin(step * 0.023), -1.0),
    ]
    force_u = torch.zeros_like(u)
    force_v = torch.zeros_like(v)
    for cx, cy, spin in centers:
        dx = xx - cx
        dy = yy - cy
        r2 = dx * dx + dy * dy
        weight = torch.exp(-r2 / max(force_radius * force_radius, 1.0e-6))
        force_u = force_u + spin * (-dy) * weight * force_strength
        force_v = force_v + spin * dx * weight * force_strength
    return u + force_u * dt, v + force_v * dt


def initial_velocity(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xx, yy = make_grid(size, device)
    u = torch.zeros((size, size), device=device)
    v = torch.zeros((size, size), device=device)
    u, v = add_vortex_forces(u, v, xx, yy, step=0, dt=1.0, force_strength=0.65, force_radius=0.16)
    return u, v


def update_foam(
    foam: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dt: float,
    foam_vorticity_threshold: float,
    foam_speed_threshold: float,
    foam_birth: float,
    foam_decay: float,
) -> torch.Tensor:
    foam = advect(foam, u, v, dt)
    vorticity = torch.abs(curl_z(u, v, dx))
    speed = torch.sqrt(u * u + v * v)
    vortex_source = torch.relu((vorticity - foam_vorticity_threshold) / max(foam_vorticity_threshold, 1.0e-6))
    speed_source = torch.relu((speed - foam_speed_threshold) / max(foam_speed_threshold, 1.0e-6))
    source = torch.clamp(vortex_source * speed_source, 0.0, 1.0)
    foam = foam * np.exp(-foam_decay * dt) + foam_birth * source * dt
    return torch.clamp(foam, 0.0, 1.0)


def simulate(
    size: int,
    steps: int,
    frame_every: int,
    dt: float,
    viscosity: float,
    pressure_iters: int,
    force_strength: float,
    force_radius: float,
    foam_vorticity_threshold: float,
    foam_speed_threshold: float,
    foam_birth: float,
    foam_decay: float,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dx = 1.0 / size
    xx, yy = make_grid(size, device)
    u, v = initial_velocity(size, device)
    foam = torch.zeros_like(u)
    frames = []

    for step in range(steps):
        u = advect(u, u, v, dt)
        v = advect(v, u, v, dt)
        u, v = add_vortex_forces(u, v, xx, yy, step, dt, force_strength, force_radius)

        if viscosity > 0.0:
            u = u + viscosity * laplacian(u, dx) * dt
            v = v + viscosity * laplacian(v, dx) * dt

        u, v = project_velocity(u, v, dx, pressure_iters)
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

        if step % frame_every == 0:
            vort = curl_z(u, v, dx)
            speed = torch.sqrt(u * u + v * v)
            frames.append((
                vort.detach().cpu().numpy().astype(np.float32),
                foam.detach().cpu().numpy().astype(np.float32),
                speed.detach().cpu().numpy().astype(np.float32),
            ))

    return frames


def build_figure(frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]], frame_duration_ms: int) -> go.Figure:
    first_vort, first_foam, first_speed = frames[0]
    vort_limit = max(float(np.max(np.abs(vort))) for vort, _, _ in frames)
    speed_limit = max(float(np.max(speed)) for _, _, speed in frames)
    vort_limit = max(vort_limit, 1.0e-6)
    speed_limit = max(speed_limit, 1.0e-6)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("vorticity / swirl", "foam scalar"),
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Heatmap(
            z=first_vort,
            colorscale="RdBu",
            zmin=-vort_limit,
            zmax=vort_limit,
            colorbar={"title": "curl"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=first_foam,
            colorscale=[[0.0, "#07111f"], [0.25, "#155e75"], [0.65, "#bae6fd"], [1.0, "#ffffff"]],
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "foam"},
        ),
        row=1,
        col=2,
    )

    fig.frames = [
        go.Frame(
            data=[
                go.Heatmap(z=vort, colorscale="RdBu", zmin=-vort_limit, zmax=vort_limit),
                go.Heatmap(
                    z=foam,
                    colorscale=[[0.0, "#07111f"], [0.25, "#155e75"], [0.65, "#bae6fd"], [1.0, "#ffffff"]],
                    zmin=0.0,
                    zmax=1.0,
                ),
            ],
            name=str(index),
        )
        for index, (vort, foam, _speed) in enumerate(frames)
    ]

    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"2D Navier-Stokes vortex foam experiment | max speed {speed_limit:.3f}",
        width=1120,
        height=560,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        plot_bgcolor="#06101d",
        paper_bgcolor="#06101d",
        font={"color": "#e5edf7"},
        updatemenus=[
            {
                "type": "buttons",
                "x": 0.5,
                "y": -0.03,
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
    parser = argparse.ArgumentParser(description="Run a 2D incompressible Navier-Stokes vortex foam experiment.")
    parser.add_argument("--size", type=int, default=160, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=900, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=4, help="Save one viewer frame every N simulation steps.")
    parser.add_argument("--dt", type=float, default=0.006, help="Simulation time step.")
    parser.add_argument("--viscosity", type=float, default=0.00008, help="Kinematic viscosity.")
    parser.add_argument("--pressure-iters", type=int, default=60, help="Jacobi pressure projection iterations.")
    parser.add_argument("--force-strength", type=float, default=4.5, help="Strength of rotating vortex force sources.")
    parser.add_argument("--force-radius", type=float, default=0.12, help="Radius of rotating vortex force sources.")
    parser.add_argument("--foam-vorticity-threshold", type=float, default=8.0, help="Curl magnitude needed to generate foam.")
    parser.add_argument("--foam-speed-threshold", type=float, default=0.18, help="Speed needed to generate foam.")
    parser.add_argument("--foam-birth", type=float, default=2.4, help="Foam generation rate.")
    parser.add_argument("--foam-decay", type=float, default=0.55, help="Foam decay rate.")
    parser.add_argument("--frame-duration-ms", type=int, default=45, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/navier_stokes_foam_2d.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames = simulate(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        dt=args.dt,
        viscosity=args.viscosity,
        pressure_iters=args.pressure_iters,
        force_strength=args.force_strength,
        force_radius=args.force_radius,
        foam_vorticity_threshold=args.foam_vorticity_threshold,
        foam_speed_threshold=args.foam_speed_threshold,
        foam_birth=args.foam_birth,
        foam_decay=args.foam_decay,
        device=device,
    )
    fig = build_figure(frames, args.frame_duration_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved Navier-Stokes foam viewer: {args.output}")


if __name__ == "__main__":
    main()
