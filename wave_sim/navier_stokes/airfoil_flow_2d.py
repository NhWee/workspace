# 2D incompressible airfoil flow experiment using PyTorch GPU acceleration.
# du/dt + (u * grad)u = -grad(p) + viscosity * laplacian(u) + force
# div(u) = 0
# A NACA 4-digit airfoil is represented as a solid obstacle mask.
# You can handle the parameters
# size_x, size_y, steps, frame_every, dt, viscosity, pressure_iters,
# inlet_velocity, angle_of_attack, chord, naca, output
import argparse
import sys
from pathlib import Path

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


def laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        torch.roll(field, shifts=1, dims=0)
        + torch.roll(field, shifts=-1, dims=0)
        + torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
        - 4.0 * field
    ) / (dx * dx)


def curl_z(u: torch.Tensor, v: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(v, dx) - gradient_y(u, dx)


def make_grid(size_x: int, size_y: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, size_x, device=device)
    y = torch.linspace(0.0, 1.0, size_y, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def naca4_mask(
    xx: torch.Tensor,
    yy: torch.Tensor,
    naca: str,
    chord: float,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> torch.Tensor:
    digits = "".join(ch for ch in naca if ch.isdigit())
    if len(digits) != 4:
        raise ValueError("Only NACA 4-digit airfoils are supported, for example NACA0012 or 2412.")

    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    thickness = int(digits[2:]) / 100.0

    angle = np.deg2rad(angle_deg)
    dx = xx - center_x
    dy = yy - center_y
    x_local = (torch.cos(torch.tensor(angle, device=xx.device)) * dx + torch.sin(torch.tensor(angle, device=xx.device)) * dy) / chord
    y_local = (-torch.sin(torch.tensor(angle, device=xx.device)) * dx + torch.cos(torch.tensor(angle, device=xx.device)) * dy) / chord

    x_clamped = torch.clamp(x_local, 0.0, 1.0)
    yt = 5.0 * thickness * (
        0.2969 * torch.sqrt(torch.clamp(x_clamped, min=1.0e-6))
        - 0.1260 * x_clamped
        - 0.3516 * x_clamped**2
        + 0.2843 * x_clamped**3
        - 0.1015 * x_clamped**4
    )

    if m > 0.0 and p > 0.0:
        yc_left = m / (p * p) * (2.0 * p * x_clamped - x_clamped**2)
        yc_right = m / ((1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x_clamped - x_clamped**2)
        yc = torch.where(x_clamped < p, yc_left, yc_right)
    else:
        yc = torch.zeros_like(x_clamped)

    inside_chord = (x_local >= 0.0) & (x_local <= 1.0)
    return inside_chord & (torch.abs(y_local - yc) <= yt)


def apply_boundaries(
    u: torch.Tensor,
    v: torch.Tensor,
    obstacle: torch.Tensor,
    inlet_velocity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = u.clone()
    v = v.clone()
    u[:, 0] = inlet_velocity
    v[:, 0] = 0.0
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[0, :] = 0.0
    v[-1, :] = 0.0
    u = torch.where(obstacle, torch.zeros_like(u), u)
    v = torch.where(obstacle, torch.zeros_like(v), v)
    return u, v


def advect(field: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float, obstacle: torch.Tensor) -> torch.Tensor:
    size_y, size_x = field.shape
    yy, xx = make_grid(size_x, size_y, field.device)
    x_prev = torch.clamp(xx - u * dt, 0.0, 1.0)
    y_prev = torch.clamp(yy - v * dt, 0.0, 1.0)
    sample_grid = torch.stack((2.0 * x_prev - 1.0, 2.0 * y_prev - 1.0), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0]
    return torch.where(obstacle, torch.zeros_like(sampled), sampled)


def project_velocity(
    u: torch.Tensor,
    v: torch.Tensor,
    obstacle: torch.Tensor,
    dx: float,
    pressure_iters: int,
    inlet_velocity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    div = gradient_x(u, dx) + gradient_y(v, dx)
    div = torch.where(obstacle, torch.zeros_like(div), div)
    pressure = torch.zeros_like(div)
    fluid = ~obstacle

    for _ in range(pressure_iters):
        neighbor_sum = (
            torch.roll(pressure, shifts=1, dims=0)
            + torch.roll(pressure, shifts=-1, dims=0)
            + torch.roll(pressure, shifts=1, dims=1)
            + torch.roll(pressure, shifts=-1, dims=1)
        )
        pressure_next = (neighbor_sum - div * dx * dx) * 0.25
        pressure = torch.where(fluid, pressure_next, torch.zeros_like(pressure_next))
        pressure[:, 0] = pressure[:, 1]
        pressure[:, -1] = pressure[:, -2]
        pressure[0, :] = pressure[1, :]
        pressure[-1, :] = pressure[-2, :]

    u = u - gradient_x(pressure, dx)
    v = v - gradient_y(pressure, dx)
    u, v = apply_boundaries(u, v, obstacle, inlet_velocity)
    return u, v, pressure


def add_inlet_perturbation(
    u: torch.Tensor,
    v: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    step: int,
    dt: float,
    inlet_velocity: float,
    perturbation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inlet_weight = torch.exp(-((xx - 0.06) ** 2) / 0.0025)
    shear = torch.sin(2.0 * np.pi * (yy * 3.0 + step * 0.012))
    u = u + inlet_weight * inlet_velocity * 0.14 * perturbation * shear * dt
    v = v + inlet_weight * inlet_velocity * 0.18 * perturbation * torch.cos(2.0 * np.pi * (yy * 2.0 - step * 0.010)) * dt
    return u, v


def simulate(
    size_x: int,
    size_y: int,
    steps: int,
    frame_every: int,
    dt: float,
    viscosity: float,
    pressure_iters: int,
    inlet_velocity: float,
    angle_of_attack: float,
    chord: float,
    naca: str,
    perturbation: float,
    device: torch.device,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    dx = 1.0 / max(size_x, size_y)
    xx, yy = make_grid(size_x, size_y, device)
    obstacle = naca4_mask(xx, yy, naca, chord, center_x=0.38, center_y=0.52, angle_deg=angle_of_attack)
    u = torch.full((size_y, size_x), inlet_velocity, device=device)
    v = torch.zeros_like(u)
    u, v = apply_boundaries(u, v, obstacle, inlet_velocity)
    pressure = torch.zeros_like(u)
    frames = []

    for step in range(steps):
        u = advect(u, u, v, dt, obstacle)
        v = advect(v, u, v, dt, obstacle)

        u, v = add_inlet_perturbation(u, v, xx, yy, step, dt, inlet_velocity, perturbation)

        if viscosity > 0.0:
            u = u + viscosity * laplacian(u, dx) * dt
            v = v + viscosity * laplacian(v, dx) * dt

        u, v = apply_boundaries(u, v, obstacle, inlet_velocity)
        u, v, pressure = project_velocity(u, v, obstacle, dx, pressure_iters, inlet_velocity)

        if step % frame_every == 0:
            vort = torch.where(obstacle, torch.zeros_like(u), curl_z(u, v, dx))
            speed = torch.where(obstacle, torch.zeros_like(u), torch.sqrt(u * u + v * v))
            pressure_view = torch.where(obstacle, torch.zeros_like(pressure), pressure)
            frames.append((
                vort.detach().cpu().numpy().astype(np.float32),
                speed.detach().cpu().numpy().astype(np.float32),
                pressure_view.detach().cpu().numpy().astype(np.float32),
            ))

    return frames, obstacle.detach().cpu().numpy()


def build_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    obstacle: np.ndarray,
    frame_duration_ms: int,
    title: str,
) -> go.Figure:
    first_vort, first_speed, first_pressure = frames[0]
    vort_limit = max(float(np.percentile(np.abs(vort), 99.4)) for vort, _, _ in frames)
    speed_limit = max(float(np.percentile(speed, 99.5)) for _, speed, _ in frames)
    pressure_limit = max(float(np.percentile(np.abs(pressure), 99.0)) for _, _, pressure in frames)
    vort_limit = max(vort_limit, 1.0e-6)
    speed_limit = max(speed_limit, 1.0e-6)
    pressure_limit = max(pressure_limit, 1.0e-6)

    obstacle_z = np.where(obstacle, 1.0, np.nan)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("vorticity / wake", "speed field", "pressure-like field"),
        horizontal_spacing=0.035,
    )
    traces = [
        go.Heatmap(z=first_vort, colorscale="RdBu", zmin=-vort_limit, zmax=vort_limit, colorbar={"title": "curl"}),
        go.Heatmap(z=first_speed, colorscale="Turbo", zmin=0.0, zmax=speed_limit, colorbar={"title": "speed"}),
        go.Heatmap(z=first_pressure, colorscale="RdBu", zmin=-pressure_limit, zmax=pressure_limit, colorbar={"title": "p"}),
    ]
    for col, trace in enumerate(traces, start=1):
        fig.add_trace(trace, row=1, col=col)
        fig.add_trace(
            go.Heatmap(
                z=obstacle_z,
                colorscale=[[0.0, "#111827"], [1.0, "#111827"]],
                showscale=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    fig.frames = [
        go.Frame(
            data=[
                go.Heatmap(z=vort, colorscale="RdBu", zmin=-vort_limit, zmax=vort_limit),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#111827"], [1.0, "#111827"]], showscale=False),
                go.Heatmap(z=speed, colorscale="Turbo", zmin=0.0, zmax=speed_limit),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#111827"], [1.0, "#111827"]], showscale=False),
                go.Heatmap(z=pressure, colorscale="RdBu", zmin=-pressure_limit, zmax=pressure_limit),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#111827"], [1.0, "#111827"]], showscale=False),
            ],
            name=str(index),
        )
        for index, (vort, speed, pressure) in enumerate(frames)
    ]

    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title,
        width=1320,
        height=520,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        plot_bgcolor="#07111f",
        paper_bgcolor="#07111f",
        font={"color": "#e5edf7"},
        updatemenus=[
            {
                "type": "buttons",
                "x": 0.5,
                "y": -0.04,
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
    parser = argparse.ArgumentParser(description="Run a 2D NACA airfoil flow experiment.")
    parser.add_argument("--size-x", type=int, default=180, help="Grid width.")
    parser.add_argument("--size-y", type=int, default=90, help="Grid height.")
    parser.add_argument("--steps", type=int, default=600, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=10, help="Save one viewer frame every N simulation steps.")
    parser.add_argument("--dt", type=float, default=0.0025, help="Simulation time step.")
    parser.add_argument("--viscosity", type=float, default=0.000035, help="Kinematic viscosity.")
    parser.add_argument("--pressure-iters", type=int, default=70, help="Jacobi pressure projection iterations.")
    parser.add_argument("--inlet-velocity", type=float, default=0.46, help="Left boundary flow speed.")
    parser.add_argument("--angle-of-attack", type=float, default=8.0, help="Airfoil angle of attack in degrees.")
    parser.add_argument("--chord", type=float, default=0.34, help="Airfoil chord length as a fraction of domain width.")
    parser.add_argument("--naca", type=str, default="NACA2412", help="NACA 4-digit airfoil, for example NACA0012 or NACA2412.")
    parser.add_argument("--perturbation", type=float, default=1.0, help="Small inlet disturbance strength.")
    parser.add_argument("--frame-duration-ms", type=int, default=35, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/airfoil_flow_2d.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames, obstacle = simulate(
        size_x=args.size_x,
        size_y=args.size_y,
        steps=args.steps,
        frame_every=args.frame_every,
        dt=args.dt,
        viscosity=args.viscosity,
        pressure_iters=args.pressure_iters,
        inlet_velocity=args.inlet_velocity,
        angle_of_attack=args.angle_of_attack,
        chord=args.chord,
        naca=args.naca,
        perturbation=args.perturbation,
        device=device,
    )
    fig = build_figure(
        frames,
        obstacle,
        args.frame_duration_ms,
        title=f"2D airfoil flow | {args.naca} | angle {args.angle_of_attack:.1f} deg | frames {len(frames)}",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved airfoil flow viewer: {args.output}")


if __name__ == "__main__":
    main()
