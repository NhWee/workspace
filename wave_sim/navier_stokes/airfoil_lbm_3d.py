# 3D finite-wing flow using a D3Q19 Lattice Boltzmann Method solver with PyTorch GPU acceleration.
# This runs without Blender: PyTorch computes the flow and Plotly writes an
# interactive 3D HTML viewer with wing geometry and passive tracer particles.
# You can handle the parameters
# size_x, size_y, size_z, steps, frame_every, tau, inlet_velocity,
# angle_of_attack, chord, span, thickness, particles, output
import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F


C19_LIST = [
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, -1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, -1),
    (1, 0, -1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, -1),
    (0, 1, -1),
    (0, -1, 1),
]
C = torch.tensor(C19_LIST, dtype=torch.float32)
W = torch.tensor([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=torch.float32)
OPP = torch.tensor([C19_LIST.index((-cx, -cy, -cz)) for cx, cy, cz in C19_LIST], dtype=torch.long)


def make_grid(size_x: int, size_y: int, size_z: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, size_x, device=device)
    y = torch.linspace(0.0, 1.0, size_y, device=device)
    z = torch.linspace(0.0, 1.0, size_z, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


def finite_wing_mask(
    xx: torch.Tensor,
    yy: torch.Tensor,
    zz: torch.Tensor,
    chord: float,
    span: float,
    thickness: float,
    angle_deg: float,
) -> torch.Tensor:
    center_x = 0.34
    center_y = 0.50
    center_z = 0.50
    angle = torch.tensor(np.deg2rad(angle_deg), device=xx.device)

    dx = xx - center_x
    dz = zz - center_z
    x_local = (torch.cos(angle) * dx + torch.sin(angle) * dz) / chord
    z_local = (-torch.sin(angle) * dx + torch.cos(angle) * dz) / chord
    y_local = (yy - center_y) / span
    x_clamped = torch.clamp(x_local, 0.0, 1.0)

    naca_like_thickness = 5.0 * thickness * (
        0.2969 * torch.sqrt(torch.clamp(x_clamped, min=1.0e-6))
        - 0.1260 * x_clamped
        - 0.3516 * x_clamped**2
        + 0.2843 * x_clamped**3
        - 0.1015 * x_clamped**4
    )
    taper = torch.clamp(1.0 - 0.72 * torch.abs(y_local) ** 2.0, min=0.28)
    rounded_tip = torch.abs(y_local) <= 0.5
    inside_chord = (x_local >= 0.0) & (x_local <= 1.0)
    inside_thickness = torch.abs(z_local) <= naca_like_thickness * taper
    return inside_chord & rounded_tip & inside_thickness


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    cu = (
        c[:, 0, None, None, None] * ux[None, :, :, :]
        + c[:, 1, None, None, None] * uy[None, :, :, :]
        + c[:, 2, None, None, None] * uz[None, :, :, :]
    )
    u2 = ux * ux + uy * uy + uz * uz
    return w[:, None, None, None] * rho[None, :, :, :] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[None, :, :, :])


def macroscopic(f: torch.Tensor, c: torch.Tensor, obstacle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = torch.clamp(torch.sum(f, dim=0), min=1.0e-6)
    ux = torch.sum(f * c[:, 0, None, None, None], dim=0) / rho
    uy = torch.sum(f * c[:, 1, None, None, None], dim=0) / rho
    uz = torch.sum(f * c[:, 2, None, None, None], dim=0) / rho
    ux = torch.where(obstacle, torch.zeros_like(ux), ux)
    uy = torch.where(obstacle, torch.zeros_like(uy), uy)
    uz = torch.where(obstacle, torch.zeros_like(uz), uz)
    return rho, ux, uy, uz


def curl_magnitude(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
    d_uz_dy = 0.5 * (torch.roll(uz, shifts=-1, dims=1) - torch.roll(uz, shifts=1, dims=1))
    d_uy_dz = 0.5 * (torch.roll(uy, shifts=-1, dims=0) - torch.roll(uy, shifts=1, dims=0))
    d_ux_dz = 0.5 * (torch.roll(ux, shifts=-1, dims=0) - torch.roll(ux, shifts=1, dims=0))
    d_uz_dx = 0.5 * (torch.roll(uz, shifts=-1, dims=2) - torch.roll(uz, shifts=1, dims=2))
    d_uy_dx = 0.5 * (torch.roll(uy, shifts=-1, dims=2) - torch.roll(uy, shifts=1, dims=2))
    d_ux_dy = 0.5 * (torch.roll(ux, shifts=-1, dims=1) - torch.roll(ux, shifts=1, dims=1))
    wx = d_uz_dy - d_uy_dz
    wy = d_ux_dz - d_uz_dx
    wz = d_uy_dx - d_ux_dy
    return torch.sqrt(wx * wx + wy * wy + wz * wz)


def stream(f: torch.Tensor, c_int: torch.Tensor) -> torch.Tensor:
    streamed = torch.empty_like(f)
    for i in range(19):
        cx = int(c_int[i, 0].item())
        cy = int(c_int[i, 1].item())
        cz = int(c_int[i, 2].item())
        streamed[i] = torch.roll(f[i], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return streamed


def impose_boundaries(f: torch.Tensor, inlet_velocity: float, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    depth, height, width = f.shape[1:]
    rho_left = torch.ones((depth, height), device=f.device)
    ux_left = torch.full((depth, height), inlet_velocity, device=f.device)
    uy_left = torch.zeros((depth, height), device=f.device)
    uz_left = torch.zeros((depth, height), device=f.device)
    feq_left = equilibrium(rho_left[:, :, None], ux_left[:, :, None], uy_left[:, :, None], uz_left[:, :, None], c, w)[:, :, :, 0]
    f[:, :, :, 0] = feq_left
    f[:, :, :, 1] = feq_left
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, :, 0, :] = f[:, :, 1, :]
    f[:, :, -1, :] = f[:, :, -2, :]
    f[:, 0, :, :] = f[:, 1, :, :]
    f[:, -1, :, :] = f[:, -2, :, :]
    return f


def make_particles(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        return torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, device=device)
    side = int(np.ceil(np.sqrt(count)))
    y = torch.linspace(0.13, 0.87, side, device=device)
    z = torch.linspace(0.24, 0.76, side, device=device)
    zz, yy = torch.meshgrid(z, y, indexing="ij")
    py = yy.reshape(-1)[:count]
    pz = zz.reshape(-1)[:count]
    px = torch.full_like(py, 0.035)
    return px, py, pz


def sample_field(field: torch.Tensor, px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
    if px.numel() == 0:
        return torch.empty(0, device=field.device)
    grid = torch.stack((2.0 * px - 1.0, 2.0 * py - 1.0, 2.0 * pz - 1.0), dim=-1)[None, :, None, None, :]
    return F.grid_sample(
        field[None, None, :, :, :],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0, :, 0, 0]


def advance_particles(
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    obstacle: torch.Tensor,
    step_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if px.numel() == 0:
        return px, py, pz
    depth, height, width = ux.shape
    sx = sample_field(ux, px, py, pz)
    sy = sample_field(uy, px, py, pz)
    sz = sample_field(uz, px, py, pz)
    px = px + step_scale * sx / max(width - 1, 1)
    py = py + step_scale * sy / max(height - 1, 1)
    pz = pz + step_scale * sz / max(depth - 1, 1)

    hit = sample_field(obstacle.to(torch.float32), px, py, pz) > 0.12
    out = (px > 0.985) | (px < 0.0) | (py < 0.03) | (py > 0.97) | (pz < 0.05) | (pz > 0.95) | hit
    if torch.any(out):
        count = int(torch.sum(out).item())
        reset_y = torch.linspace(0.13, 0.87, count, device=px.device)
        reset_z = 0.50 + 0.26 * torch.sin(torch.linspace(0.0, np.pi * 2.0, count, device=px.device))
        px[out] = 0.035
        py[out] = reset_y
        pz[out] = torch.clamp(reset_z, 0.18, 0.82)
    return px, py, pz


def boundary_points(obstacle: torch.Tensor, max_points: int) -> np.ndarray:
    exposed = obstacle & (
        ~torch.roll(obstacle, shifts=1, dims=0)
        | ~torch.roll(obstacle, shifts=-1, dims=0)
        | ~torch.roll(obstacle, shifts=1, dims=1)
        | ~torch.roll(obstacle, shifts=-1, dims=1)
        | ~torch.roll(obstacle, shifts=1, dims=2)
        | ~torch.roll(obstacle, shifts=-1, dims=2)
    )
    coords = torch.nonzero(exposed, as_tuple=False).detach().cpu().numpy()
    if coords.shape[0] > max_points:
        indices = np.linspace(0, coords.shape[0] - 1, max_points).astype(np.int64)
        coords = coords[indices]
    return coords


def simulate(
    size_x: int,
    size_y: int,
    size_z: int,
    steps: int,
    frame_every: int,
    tau: float,
    inlet_velocity: float,
    angle_of_attack: float,
    chord: float,
    span: float,
    thickness: float,
    particles: int,
    particle_step_scale: float,
    device: torch.device,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], np.ndarray, list[float]]:
    c = C.to(device)
    c_int = C.to(torch.int64).to(device)
    w = W.to(device)
    opp = OPP.to(device)
    xx, yy, zz = make_grid(size_x, size_y, size_z, device)
    obstacle = finite_wing_mask(xx, yy, zz, chord, span, thickness, angle_of_attack)

    rho = torch.ones((size_z, size_y, size_x), device=device)
    ux = torch.full_like(rho, inlet_velocity)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    ux = torch.where(obstacle, torch.zeros_like(ux), ux)
    f = equilibrium(rho, ux, uy, uz, c, w)
    px, py, pz = make_particles(particles, device)
    omega = 1.0 / tau
    frames = []
    vort_history = []

    for step in range(steps):
        rho, ux, uy, uz = macroscopic(f, c, obstacle)
        feq = equilibrium(rho, ux, uy, uz, c, w)
        collided = f - omega * (f - feq)
        streamed = stream(collided, c_int)
        bounced = streamed.clone()
        for i in range(19):
            bounced[i, obstacle] = collided[int(opp[i].item()), obstacle]
        f = impose_boundaries(bounced, inlet_velocity, c, w)

        rho, ux, uy, uz = macroscopic(f, c, obstacle)
        px, py, pz = advance_particles(px, py, pz, ux, uy, uz, obstacle, particle_step_scale)

        if step % frame_every == 0:
            vort = torch.where(obstacle, torch.zeros_like(rho), curl_magnitude(ux, uy, uz))
            particle_vort = sample_field(vort, px, py, pz)
            frames.append(
                (
                    (px * (size_x - 1)).detach().cpu().numpy().astype(np.float32),
                    (py * (size_y - 1)).detach().cpu().numpy().astype(np.float32),
                    (pz * (size_z - 1)).detach().cpu().numpy().astype(np.float32),
                    particle_vort.detach().cpu().numpy().astype(np.float32),
                )
            )
            vort_history.append(float(torch.max(vort).detach().cpu()))

    return frames, boundary_points(obstacle, max_points=3500), vort_history


def build_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    wing_points: np.ndarray,
    vort_history: list[float],
    size_x: int,
    size_y: int,
    size_z: int,
    frame_duration_ms: int,
    title: str,
) -> go.Figure:
    first_x, first_y, first_z, first_vort = frames[0]
    vort_limit = max(1.0e-6, max(float(np.percentile(vort, 98.0)) for *_xyz, vort in frames))
    wing_z, wing_y, wing_x = wing_points[:, 0], wing_points[:, 1], wing_points[:, 2]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=wing_x,
            y=wing_y,
            z=wing_z,
            mode="markers",
            marker={"size": 2.5, "color": "#1f2937", "opacity": 0.82},
            name="finite wing",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=first_x,
            y=first_y,
            z=first_z,
            mode="markers",
            marker={
                "size": 3.0,
                "color": first_vort,
                "colorscale": "Turbo",
                "cmin": 0.0,
                "cmax": vort_limit,
                "opacity": 0.78,
                "colorbar": {"title": "vort"},
            },
            name="flow particles",
            hoverinfo="skip",
        )
    )
    fig.frames = [
        go.Frame(
            data=[
                go.Scatter3d(x=wing_x, y=wing_y, z=wing_z, mode="markers", marker={"size": 2.5, "color": "#1f2937", "opacity": 0.82}),
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker={"size": 3.0, "color": vort, "colorscale": "Turbo", "cmin": 0.0, "cmax": vort_limit, "opacity": 0.78},
                ),
            ],
            name=str(index),
            layout=go.Layout(title_text=f"{title} | max vorticity {vort_history[index]:.4f}"),
        )
        for index, (x, y, z, vort) in enumerate(frames)
    ]
    fig.update_layout(
        title=f"{title} | frames {len(frames)}",
        width=1120,
        height=760,
        paper_bgcolor="#07111f",
        plot_bgcolor="#07111f",
        font={"color": "#e5edf7"},
        scene={
            "xaxis": {"range": [0, size_x - 1], "title": "x / flow", "backgroundcolor": "#07111f", "gridcolor": "#1f3347"},
            "yaxis": {"range": [0, size_y - 1], "title": "span", "backgroundcolor": "#07111f", "gridcolor": "#1f3347"},
            "zaxis": {"range": [0, size_z - 1], "title": "z", "backgroundcolor": "#07111f", "gridcolor": "#1f3347"},
            "aspectmode": "manual",
            "aspectratio": {"x": 2.2, "y": 1.0, "z": 0.7},
            "camera": {"eye": {"x": 1.75, "y": -1.65, "z": 0.92}},
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
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
    parser = argparse.ArgumentParser(description="Run a 3D D3Q19 LBM finite-wing flow prototype.")
    parser.add_argument("--size-x", type=int, default=80, help="Grid width along the flow.")
    parser.add_argument("--size-y", type=int, default=42, help="Grid span direction.")
    parser.add_argument("--size-z", type=int, default=30, help="Grid vertical direction.")
    parser.add_argument("--steps", type=int, default=520, help="LBM time steps.")
    parser.add_argument("--frame-every", type=int, default=13, help="Save one viewer frame every N steps.")
    parser.add_argument("--tau", type=float, default=0.60, help="LBM relaxation time. Must be greater than 0.5.")
    parser.add_argument("--inlet-velocity", type=float, default=0.045, help="Lattice-unit inlet speed.")
    parser.add_argument("--angle-of-attack", type=float, default=8.0, help="Wing angle of attack in degrees.")
    parser.add_argument("--chord", type=float, default=0.28, help="Wing chord as a fraction of domain length.")
    parser.add_argument("--span", type=float, default=0.58, help="Wing span as a fraction of domain width.")
    parser.add_argument("--thickness", type=float, default=0.13, help="NACA-like relative thickness.")
    parser.add_argument("--particles", type=int, default=650, help="Passive tracer particles.")
    parser.add_argument("--particle-step-scale", type=float, default=4.0, help="Particle advection visual speed multiplier.")
    parser.add_argument("--frame-duration-ms", type=int, default=45, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/airfoil_lbm_3d.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tau <= 0.5:
        raise ValueError("--tau must be greater than 0.5 for LBM stability.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames, wing_points, vort_history = simulate(
        size_x=args.size_x,
        size_y=args.size_y,
        size_z=args.size_z,
        steps=args.steps,
        frame_every=args.frame_every,
        tau=args.tau,
        inlet_velocity=args.inlet_velocity,
        angle_of_attack=args.angle_of_attack,
        chord=args.chord,
        span=args.span,
        thickness=args.thickness,
        particles=args.particles,
        particle_step_scale=args.particle_step_scale,
        device=device,
    )
    fig = build_figure(
        frames,
        wing_points,
        vort_history,
        args.size_x,
        args.size_y,
        args.size_z,
        args.frame_duration_ms,
        title=f"3D D3Q19 LBM finite-wing flow | angle {args.angle_of_attack:.1f} deg",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved 3D LBM airfoil viewer: {args.output}")


if __name__ == "__main__":
    main()
