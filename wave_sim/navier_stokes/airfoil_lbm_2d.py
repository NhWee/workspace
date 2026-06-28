# 2D airfoil flow using a D2Q9 Lattice Boltzmann Method solver with PyTorch GPU acceleration.
# This is a better first airfoil solver than the previous stable-fluids demo:
# it uses distribution functions, collision, streaming, and bounce-back
# boundary handling around a NACA 4-digit airfoil obstacle.
# You can handle the parameters
# size_x, size_y, steps, frame_every, tau, inlet_velocity, angle_of_attack,
# chord, naca, tracer_decay, particles, output
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


C = torch.tensor(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=torch.float32,
)
W = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=torch.float32)
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)


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
        raise ValueError("Only NACA 4-digit airfoils are supported, for example NACA0012 or NACA2412.")

    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    thickness = int(digits[2:]) / 100.0
    angle = torch.tensor(np.deg2rad(angle_deg), device=xx.device)

    dx = xx - center_x
    dy = yy - center_y
    x_local = (torch.cos(angle) * dx + torch.sin(angle) * dy) / chord
    y_local = (-torch.sin(angle) * dx + torch.cos(angle) * dy) / chord
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

    return (x_local >= 0.0) & (x_local <= 1.0) & (torch.abs(y_local - yc) <= yt)


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    cu = c[:, 0, None, None] * ux[None, :, :] + c[:, 1, None, None] * uy[None, :, :]
    u2 = ux * ux + uy * uy
    return w[:, None, None] * rho[None, :, :] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None, :, :])


def macroscopic(f: torch.Tensor, c: torch.Tensor, obstacle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = torch.clamp(torch.sum(f, dim=0), min=1.0e-6)
    ux = torch.sum(f * c[:, 0, None, None], dim=0) / rho
    uy = torch.sum(f * c[:, 1, None, None], dim=0) / rho
    ux = torch.where(obstacle, torch.zeros_like(ux), ux)
    uy = torch.where(obstacle, torch.zeros_like(uy), uy)
    return rho, ux, uy


def curl_z(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        torch.roll(uy, shifts=-1, dims=1)
        - torch.roll(uy, shifts=1, dims=1)
        - torch.roll(ux, shifts=-1, dims=0)
        + torch.roll(ux, shifts=1, dims=0)
    )


def stream(f: torch.Tensor, c_int: torch.Tensor) -> torch.Tensor:
    streamed = torch.empty_like(f)
    for i in range(9):
        cx = int(c_int[i, 0].item())
        cy = int(c_int[i, 1].item())
        streamed[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return streamed


def impose_open_boundaries(f: torch.Tensor, inlet_velocity: float, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    height, width = f.shape[1:]
    rho_left = torch.ones((height,), device=f.device)
    ux_left = torch.full((height,), inlet_velocity, device=f.device)
    uy_left = torch.zeros((height,), device=f.device)
    feq_left = equilibrium(rho_left[:, None], ux_left[:, None], uy_left[:, None], c, w)[:, :, 0]
    f[:, :, 0] = feq_left
    f[:, :, 1] = feq_left
    f[:, :, -1] = f[:, :, -2]
    f[:, 0, :] = f[:, 1, :]
    f[:, -1, :] = f[:, -2, :]
    return f


def advect_tracer(tracer: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, obstacle: torch.Tensor, decay: float) -> torch.Tensor:
    height, width = tracer.shape
    yy, xx = make_grid(width, height, tracer.device)
    x_prev = torch.clamp(xx - ux / max(width - 1, 1), 0.0, 1.0)
    y_prev = torch.clamp(yy - uy / max(height - 1, 1), 0.0, 1.0)
    sample_grid = torch.stack((2.0 * x_prev - 1.0, 2.0 * y_prev - 1.0), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        tracer[None, None, :, :],
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0]
    stripe = 0.5 + 0.5 * torch.sin(2.0 * np.pi * yy * 9.0)
    sampled[:, :3] = torch.maximum(sampled[:, :3], stripe[:, :3])
    return torch.where(obstacle, torch.zeros_like(sampled), torch.clamp(sampled * decay, 0.0, 1.0))


def sample_field(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    grid = torch.stack((2.0 * x - 1.0, 2.0 * y - 1.0), dim=-1)[None, None, :, :]
    return F.grid_sample(
        field[None, None, :, :],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0, 0]


def make_particles(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        return torch.empty(0, device=device), torch.empty(0, device=device)
    y = torch.linspace(0.08, 0.92, count, device=device)
    x = torch.full_like(y, 0.025)
    return x, y


def advance_particles(
    particle_x: torch.Tensor,
    particle_y: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    obstacle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if particle_x.numel() == 0:
        return particle_x, particle_y

    sampled_u = sample_field(ux, particle_x, particle_y)
    sampled_v = sample_field(uy, particle_x, particle_y)
    height, width = ux.shape
    particle_x = particle_x + sampled_u / max(width - 1, 1)
    particle_y = particle_y + sampled_v / max(height - 1, 1)

    obstacle_hit = sample_field(obstacle.to(torch.float32), particle_x, particle_y) > 0.15
    out = (particle_x > 0.99) | (particle_x < 0.0) | (particle_y < 0.02) | (particle_y > 0.98) | obstacle_hit
    if torch.any(out):
        count = int(torch.sum(out).item())
        particle_x[out] = 0.025
        particle_y[out] = torch.linspace(0.08, 0.92, count, device=particle_x.device)
    return particle_x, particle_y


def estimate_momentum_exchange_force(
    collided: torch.Tensor,
    obstacle: torch.Tensor,
    c: torch.Tensor,
    opp: torch.Tensor,
    inlet_velocity: float,
    chord: float,
) -> tuple[float, float]:
    force_x = torch.zeros((), device=collided.device)
    force_y = torch.zeros((), device=collided.device)
    for i in range(1, 9):
        incoming = collided[int(opp[i].item()), obstacle]
        force_x = force_x + torch.sum(2.0 * c[i, 0] * incoming)
        force_y = force_y + torch.sum(2.0 * c[i, 1] * incoming)
    scale = 0.5 * max(inlet_velocity * inlet_velocity, 1.0e-6) * max(chord, 1.0e-6) * 100.0
    return float(force_y.detach().cpu() / scale), float(force_x.detach().cpu() / scale)


def simulate(
    size_x: int,
    size_y: int,
    steps: int,
    frame_every: int,
    tau: float,
    inlet_velocity: float,
    angle_of_attack: float,
    chord: float,
    naca: str,
    tracer_decay: float,
    particles: int,
    device: torch.device,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], np.ndarray, list[tuple[float, float]]]:
    c = C.to(device)
    c_int = C.to(torch.int64).to(device)
    w = W.to(device)
    opp = OPP.to(device)
    xx, yy = make_grid(size_x, size_y, device)
    obstacle = naca4_mask(xx, yy, naca, chord, center_x=0.33, center_y=0.50, angle_deg=angle_of_attack)

    rho = torch.ones((size_y, size_x), device=device)
    ux = torch.full_like(rho, inlet_velocity)
    uy = torch.zeros_like(rho)
    ux = torch.where(obstacle, torch.zeros_like(ux), ux)
    f = equilibrium(rho, ux, uy, c, w)
    tracer = torch.zeros_like(rho)
    particle_x, particle_y = make_particles(particles, device)
    omega = 1.0 / tau
    frames = []
    force_history = []
    current_force = (0.0, 0.0)

    for step in range(steps):
        rho, ux, uy = macroscopic(f, c, obstacle)
        feq = equilibrium(rho, ux, uy, c, w)
        collided = f - omega * (f - feq)
        streamed = stream(collided, c_int)

        bounced = streamed.clone()
        for i in range(9):
            bounced[i, obstacle] = collided[int(opp[i].item()), obstacle]
        current_force = estimate_momentum_exchange_force(collided, obstacle, c, opp, inlet_velocity, chord)
        f = impose_open_boundaries(bounced, inlet_velocity, c, w)

        rho, ux, uy = macroscopic(f, c, obstacle)
        tracer = advect_tracer(tracer, ux, uy, obstacle, tracer_decay)
        particle_x, particle_y = advance_particles(particle_x, particle_y, ux, uy, obstacle)

        if step % frame_every == 0:
            vort = torch.where(obstacle, torch.zeros_like(rho), curl_z(ux, uy))
            speed = torch.where(obstacle, torch.zeros_like(rho), torch.sqrt(ux * ux + uy * uy))
            pressure = torch.where(obstacle, torch.zeros_like(rho), (rho - 1.0) / 3.0)
            frames.append(
                (
                    vort.detach().cpu().numpy().astype(np.float32),
                    speed.detach().cpu().numpy().astype(np.float32),
                    pressure.detach().cpu().numpy().astype(np.float32),
                    tracer.detach().cpu().numpy().astype(np.float32),
                    (particle_x * (size_x - 1)).detach().cpu().numpy().astype(np.float32),
                    (particle_y * (size_y - 1)).detach().cpu().numpy().astype(np.float32),
                )
            )
            force_history.append(current_force)

    return frames, obstacle.detach().cpu().numpy(), force_history


def overlay_airfoil(fig: go.Figure, obstacle: np.ndarray, row: int, col: int) -> None:
    obstacle_z = np.where(obstacle, 1.0, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=obstacle_z,
            colorscale=[[0.0, "#0f172a"], [1.0, "#0f172a"]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def build_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    obstacle: np.ndarray,
    force_history: list[tuple[float, float]],
    frame_duration_ms: int,
    title: str,
) -> go.Figure:
    first_vort, first_speed, first_pressure, first_tracer, first_particle_x, first_particle_y = frames[0]
    lift = np.array([x for x, _ in force_history], dtype=np.float32)
    drag = np.array([x for _, x in force_history], dtype=np.float32)
    vort_limit = max(1.0e-6, max(float(np.percentile(np.abs(v), 99.2)) for v, _, _, _, _, _ in frames))
    speed_limit = max(1.0e-6, max(float(np.percentile(s, 99.5)) for _, s, _, _, _, _ in frames))
    pressure_limit = max(1.0e-6, max(float(np.percentile(np.abs(p), 99.2)) for _, _, p, _, _, _ in frames))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("vorticity / wake", "speed", "pressure", "inlet tracer"),
        horizontal_spacing=0.055,
        vertical_spacing=0.090,
    )
    base_traces = [
        (go.Heatmap(z=first_vort, colorscale="RdBu", zmin=-vort_limit, zmax=vort_limit, colorbar={"title": "curl"}), 1, 1),
        (go.Heatmap(z=first_speed, colorscale="Turbo", zmin=0.0, zmax=speed_limit, colorbar={"title": "speed"}), 1, 2),
        (go.Heatmap(z=first_pressure, colorscale="RdBu", zmin=-pressure_limit, zmax=pressure_limit, colorbar={"title": "p"}), 2, 1),
        (go.Heatmap(z=first_tracer, colorscale="Viridis", zmin=0.0, zmax=1.0, colorbar={"title": "dye"}), 2, 2),
    ]
    for trace, row, col in base_traces:
        fig.add_trace(trace, row=row, col=col)
        overlay_airfoil(fig, obstacle, row, col)
    fig.add_trace(
        go.Scatter(
            x=first_particle_x,
            y=first_particle_y,
            mode="markers",
            marker={"size": 3, "color": "#f8fafc", "opacity": 0.74},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )

    obstacle_z = np.where(obstacle, 1.0, np.nan)
    fig.frames = [
        go.Frame(
            data=[
                go.Heatmap(z=vort, colorscale="RdBu", zmin=-vort_limit, zmax=vort_limit),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#0f172a"], [1.0, "#0f172a"]], showscale=False),
                go.Heatmap(z=speed, colorscale="Turbo", zmin=0.0, zmax=speed_limit),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#0f172a"], [1.0, "#0f172a"]], showscale=False),
                go.Heatmap(z=pressure, colorscale="RdBu", zmin=-pressure_limit, zmax=pressure_limit),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#0f172a"], [1.0, "#0f172a"]], showscale=False),
                go.Heatmap(z=tracer, colorscale="Viridis", zmin=0.0, zmax=1.0),
                go.Heatmap(z=obstacle_z, colorscale=[[0.0, "#0f172a"], [1.0, "#0f172a"]], showscale=False),
                go.Scatter(
                    x=particle_x,
                    y=particle_y,
                    mode="markers",
                    marker={"size": 3, "color": "#f8fafc", "opacity": 0.74},
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ],
            name=str(index),
            layout=go.Layout(title_text=f"{title} | CL~{lift[index]:+.3f}, CD~{drag[index]:+.3f}"),
        )
        for index, (vort, speed, pressure, tracer, particle_x, particle_y) in enumerate(frames)
    ]

    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"{title} | CL~{lift[-1]:+.3f}, CD~{drag[-1]:+.3f}",
        width=1180,
        height=820,
        margin={"l": 20, "r": 20, "t": 72, "b": 20},
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
    parser = argparse.ArgumentParser(description="Run a 2D D2Q9 LBM airfoil flow experiment.")
    parser.add_argument("--size-x", type=int, default=200, help="Grid width.")
    parser.add_argument("--size-y", type=int, default=100, help="Grid height.")
    parser.add_argument("--steps", type=int, default=720, help="LBM time steps.")
    parser.add_argument("--frame-every", type=int, default=16, help="Save one viewer frame every N steps.")
    parser.add_argument("--tau", type=float, default=0.58, help="LBM relaxation time. Must be greater than 0.5.")
    parser.add_argument("--inlet-velocity", type=float, default=0.055, help="Lattice-unit inlet speed.")
    parser.add_argument("--angle-of-attack", type=float, default=8.0, help="Airfoil angle of attack in degrees.")
    parser.add_argument("--chord", type=float, default=0.30, help="Airfoil chord as a fraction of domain width.")
    parser.add_argument("--naca", type=str, default="NACA2412", help="NACA 4-digit airfoil.")
    parser.add_argument("--tracer-decay", type=float, default=0.996, help="Passive tracer decay per step.")
    parser.add_argument("--particles", type=int, default=280, help="Number of passive pathline particles.")
    parser.add_argument("--frame-duration-ms", type=int, default=38, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/airfoil_lbm_2d.html"), help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tau <= 0.5:
        raise ValueError("--tau must be greater than 0.5 for LBM stability.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames, obstacle, force_history = simulate(
        size_x=args.size_x,
        size_y=args.size_y,
        steps=args.steps,
        frame_every=args.frame_every,
        tau=args.tau,
        inlet_velocity=args.inlet_velocity,
        angle_of_attack=args.angle_of_attack,
        chord=args.chord,
        naca=args.naca,
        tracer_decay=args.tracer_decay,
        particles=args.particles,
        device=device,
    )
    fig = build_figure(
        frames,
        obstacle,
        force_history,
        args.frame_duration_ms,
        title=f"D2Q9 LBM airfoil flow | {args.naca} | angle {args.angle_of_attack:.1f} deg",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved LBM airfoil viewer: {args.output}")


if __name__ == "__main__":
    main()
