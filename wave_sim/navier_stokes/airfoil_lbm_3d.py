# 3D finite-wing flow using a D3Q19 Lattice Boltzmann Method solver with PyTorch GPU acceleration.
# This runs without Blender: PyTorch computes the flow and Plotly writes an
# interactive 3D HTML viewer with a smooth wing mesh and smoke-like pathline curves.
# You can handle the parameters
# size_x, size_y, size_z, steps, frame_every, tau, inlet_velocity,
# angle_of_attack, chord, span, thickness, particles, trail_length, output
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


def make_wing_mesh(
    size_x: int,
    size_y: int,
    size_z: int,
    chord: float,
    span: float,
    thickness: float,
    angle_deg: float,
    chord_segments: int = 46,
    span_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_x = 0.34
    center_y = 0.50
    center_z = 0.50
    angle = np.deg2rad(angle_deg)
    x = np.linspace(0.0, 1.0, chord_segments)
    y_local = np.linspace(-0.5, 0.5, span_segments)
    xx, yy = np.meshgrid(x, y_local, indexing="xy")
    taper = np.clip(1.0 - 0.72 * np.abs(yy) ** 2.0, 0.28, None)
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(np.clip(xx, 1.0e-6, None))
        - 0.1260 * xx
        - 0.3516 * xx**2
        + 0.2843 * xx**3
        - 0.1015 * xx**4
    ) * taper

    vertices = []
    for sign in (1.0, -1.0):
        x_world = center_x + chord * xx * np.cos(angle) - sign * chord * yt * np.sin(angle)
        z_world = center_z + chord * xx * np.sin(angle) + sign * chord * yt * np.cos(angle)
        y_world = center_y + span * yy
        vertices.append(np.column_stack((
            (x_world.reshape(-1) * (size_x - 1)),
            (y_world.reshape(-1) * (size_y - 1)),
            (z_world.reshape(-1) * (size_z - 1)),
        )))
    vertices_np = np.vstack(vertices)

    faces_i = []
    faces_j = []
    faces_k = []
    sheet_size = chord_segments * span_segments
    for sheet in range(2):
        offset = sheet * sheet_size
        for row in range(span_segments - 1):
            for col in range(chord_segments - 1):
                a = offset + row * chord_segments + col
                b = a + 1
                c = a + chord_segments
                d = c + 1
                faces_i.extend([a, b])
                faces_j.extend([b, d])
                faces_k.extend([c, c])
    return (
        vertices_np[:, 0],
        vertices_np[:, 1],
        vertices_np[:, 2],
        np.array(faces_i, dtype=np.int32),
        np.array(faces_j, dtype=np.int32),
        np.array(faces_k, dtype=np.int32),
    )


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


def make_trails(
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
    trail_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if trail_length <= 1 or px.numel() == 0:
        empty = torch.empty((0, px.numel()), device=px.device)
        return empty, empty, empty
    return (
        px.repeat(trail_length, 1),
        py.repeat(trail_length, 1),
        pz.repeat(trail_length, 1),
    )


def update_trails(
    trail_x: torch.Tensor,
    trail_y: torch.Tensor,
    trail_z: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if trail_x.numel() == 0:
        return trail_x, trail_y, trail_z
    trail_x = torch.roll(trail_x, shifts=-1, dims=0)
    trail_y = torch.roll(trail_y, shifts=-1, dims=0)
    trail_z = torch.roll(trail_z, shifts=-1, dims=0)
    trail_x[-1] = px
    trail_y[-1] = py
    trail_z[-1] = pz
    return trail_x, trail_y, trail_z


def trails_to_plot_arrays(
    trail_x: np.ndarray,
    trail_y: np.ndarray,
    trail_z: np.ndarray,
    max_lines: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if trail_x.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    particle_count = trail_x.shape[1]
    if particle_count > max_lines:
        indices = np.linspace(0, particle_count - 1, max_lines).astype(np.int64)
        trail_x = trail_x[:, indices]
        trail_y = trail_y[:, indices]
        trail_z = trail_z[:, indices]
    line_x = []
    line_y = []
    line_z = []
    for index in range(trail_x.shape[1]):
        line_x.extend(trail_x[:, index].tolist())
        line_y.extend(trail_y[:, index].tolist())
        line_z.extend(trail_z[:, index].tolist())
        line_x.append(np.nan)
        line_y.append(np.nan)
        line_z.append(np.nan)
    return np.array(line_x, dtype=np.float32), np.array(line_y, dtype=np.float32), np.array(line_z, dtype=np.float32)


def wingtip_vortex_markers(
    particle_x: np.ndarray,
    particle_y: np.ndarray,
    particle_z: np.ndarray,
    particle_vort: np.ndarray,
    wing_y: np.ndarray,
    size_x: int,
    size_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if particle_x.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    left_tip = float(np.min(wing_y))
    right_tip = float(np.max(wing_y))
    tip_band = max(2.5, size_y * 0.11)
    wake_start = size_x * 0.32
    vort_threshold = max(float(np.percentile(particle_vort, 70.0)), 1.0e-8)
    near_tip = (np.abs(particle_y - left_tip) < tip_band) | (np.abs(particle_y - right_tip) < tip_band)
    in_wake = particle_x > wake_start
    strong = particle_vort >= vort_threshold
    mask = near_tip & in_wake & strong
    return particle_x[mask], particle_y[mask], particle_z[mask], particle_vort[mask]


def wingtip_vortex_core_curves(
    particle_x: np.ndarray,
    particle_y: np.ndarray,
    particle_z: np.ndarray,
    particle_vort: np.ndarray,
    wing_y: np.ndarray,
    size_x: int,
    size_y: int,
    bins: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if particle_x.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty, empty, empty

    left_tip = float(np.min(wing_y))
    right_tip = float(np.max(wing_y))
    tip_band = max(3.0, size_y * 0.14)
    wake_min = size_x * 0.34
    wake_max = size_x * 0.95
    edges = np.linspace(wake_min, wake_max, bins + 1)

    curves = []
    for tip in (left_tip, right_tip):
        near_tip = np.abs(particle_y - tip) < tip_band
        in_wake = (particle_x >= wake_min) & (particle_x <= wake_max)
        threshold = max(float(np.percentile(particle_vort, 62.0)), 1.0e-8)
        selected = near_tip & in_wake & (particle_vort >= threshold)
        cx = []
        cy = []
        cz = []
        for start, end in zip(edges[:-1], edges[1:]):
            bin_mask = selected & (particle_x >= start) & (particle_x < end)
            if not np.any(bin_mask):
                continue
            weight = particle_vort[bin_mask] + 1.0e-6
            cx.append(float(np.average(particle_x[bin_mask], weights=weight)))
            cy.append(float(np.average(particle_y[bin_mask], weights=weight)))
            cz.append(float(np.average(particle_z[bin_mask], weights=weight)))
        curves.append((
            np.array(cx, dtype=np.float32),
            np.array(cy, dtype=np.float32),
            np.array(cz, dtype=np.float32),
        ))
    return (*curves[0], *curves[1])


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
    trail_length: int,
    device: torch.device,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[float]]:
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
    trail_x, trail_y, trail_z = make_trails(px, py, pz, trail_length)
    omega = 1.0 / tau
    frames = []
    vort_history = []
    wing_mesh = make_wing_mesh(size_x, size_y, size_z, chord, span, thickness, angle_of_attack)

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
        trail_x, trail_y, trail_z = update_trails(trail_x, trail_y, trail_z, px, py, pz)

        if step % frame_every == 0:
            vort = torch.where(obstacle, torch.zeros_like(rho), curl_magnitude(ux, uy, uz))
            particle_vort = sample_field(vort, px, py, pz)
            scaled_trail_x = (trail_x * (size_x - 1)).detach().cpu().numpy().astype(np.float32)
            scaled_trail_y = (trail_y * (size_y - 1)).detach().cpu().numpy().astype(np.float32)
            scaled_trail_z = (trail_z * (size_z - 1)).detach().cpu().numpy().astype(np.float32)
            line_x, line_y, line_z = trails_to_plot_arrays(scaled_trail_x, scaled_trail_y, scaled_trail_z, max_lines=95)
            frames.append(
                (
                    (px * (size_x - 1)).detach().cpu().numpy().astype(np.float32),
                    (py * (size_y - 1)).detach().cpu().numpy().astype(np.float32),
                    (pz * (size_z - 1)).detach().cpu().numpy().astype(np.float32),
                    particle_vort.detach().cpu().numpy().astype(np.float32),
                    line_x,
                    line_y,
                    line_z,
                )
            )
            vort_history.append(float(torch.max(vort).detach().cpu()))

    return frames, wing_mesh, vort_history


def build_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    wing_mesh: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    vort_history: list[float],
    size_x: int,
    size_y: int,
    size_z: int,
    frame_duration_ms: int,
    title: str,
) -> go.Figure:
    first_x, first_y, first_z, first_vort, first_line_x, first_line_y, first_line_z = frames[0]
    vort_limit = max(1.0e-6, max(float(np.percentile(vort, 98.0)) for _x, _y, _z, vort, _lx, _ly, _lz in frames))
    wing_x, wing_y, wing_z, wing_i, wing_j, wing_k = wing_mesh
    first_tip_x, first_tip_y, first_tip_z, first_tip_vort = wingtip_vortex_markers(
        first_x,
        first_y,
        first_z,
        first_vort,
        wing_y,
        size_x,
        size_y,
    )
    first_left_core_x, first_left_core_y, first_left_core_z, first_right_core_x, first_right_core_y, first_right_core_z = (
        wingtip_vortex_core_curves(first_x, first_y, first_z, first_vort, wing_y, size_x, size_y)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=wing_x,
            y=wing_y,
            z=wing_z,
            i=wing_i,
            j=wing_j,
            k=wing_k,
            color="#9ca3af",
            opacity=0.78,
            flatshading=False,
            name="finite wing",
            hoverinfo="skip",
        )
    )
    smoke_line = {"width": 5.0, "color": "rgba(219, 234, 254, 0.68)"}
    fig.add_trace(
        go.Scatter3d(
            x=first_line_x,
            y=first_line_y,
            z=first_line_z,
            mode="lines",
            line=smoke_line,
            name="smoke pathlines",
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
                "size": 1.2,
                "color": first_vort,
                "colorscale": "Turbo",
                "cmin": 0.0,
                "cmax": vort_limit,
                "opacity": 0.16,
                "colorbar": {"title": "vort"},
            },
            name="particle heads (debug)",
            hoverinfo="skip",
            visible="legendonly",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=first_tip_x,
            y=first_tip_y,
            z=first_tip_z,
            mode="markers",
            marker={
                "size": 2.5,
                "color": first_tip_vort,
                "colorscale": [[0.0, "#fde047"], [0.55, "#fb923c"], [1.0, "#ef4444"]],
                "cmin": 0.0,
                "cmax": vort_limit,
                "opacity": 0.22,
            },
            name="wingtip marker samples",
            hoverinfo="skip",
            visible="legendonly",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=first_left_core_x,
            y=first_left_core_y,
            z=first_left_core_z,
            mode="lines+markers",
            line={"width": 9.0, "color": "#f97316"},
            marker={"size": 2.0, "color": "#fed7aa", "opacity": 0.5},
            name="left vortex core",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=first_right_core_x,
            y=first_right_core_y,
            z=first_right_core_z,
            mode="lines+markers",
            line={"width": 9.0, "color": "#facc15"},
            marker={"size": 2.0, "color": "#fef08a", "opacity": 0.5},
            name="right vortex core",
            hoverinfo="skip",
        )
    )
    plotly_frames = []
    for index, (x, y, z, vort, line_x, line_y, line_z) in enumerate(frames):
        tip_x, tip_y, tip_z, tip_vort = wingtip_vortex_markers(x, y, z, vort, wing_y, size_x, size_y)
        left_core_x, left_core_y, left_core_z, right_core_x, right_core_y, right_core_z = wingtip_vortex_core_curves(
            x,
            y,
            z,
            vort,
            wing_y,
            size_x,
            size_y,
        )
        plotly_frames.append(
            go.Frame(
                data=[
                    go.Mesh3d(x=wing_x, y=wing_y, z=wing_z, i=wing_i, j=wing_j, k=wing_k, color="#9ca3af", opacity=0.78, flatshading=False),
                    go.Scatter3d(
                        x=line_x,
                        y=line_y,
                        z=line_z,
                        mode="lines",
                        line=smoke_line,
                    ),
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker={"size": 1.2, "color": vort, "colorscale": "Turbo", "cmin": 0.0, "cmax": vort_limit, "opacity": 0.16},
                        visible="legendonly",
                    ),
                    go.Scatter3d(
                        x=tip_x,
                        y=tip_y,
                        z=tip_z,
                        mode="markers",
                        marker={
                            "size": 2.5,
                            "color": tip_vort,
                            "colorscale": [[0.0, "#fde047"], [0.55, "#fb923c"], [1.0, "#ef4444"]],
                            "cmin": 0.0,
                            "cmax": vort_limit,
                            "opacity": 0.22,
                        },
                        visible="legendonly",
                    ),
                    go.Scatter3d(
                        x=left_core_x,
                        y=left_core_y,
                        z=left_core_z,
                        mode="lines+markers",
                        line={"width": 9.0, "color": "#f97316"},
                        marker={"size": 2.0, "color": "#fed7aa", "opacity": 0.5},
                    ),
                    go.Scatter3d(
                        x=right_core_x,
                        y=right_core_y,
                        z=right_core_z,
                        mode="lines+markers",
                        line={"width": 9.0, "color": "#facc15"},
                        marker={"size": 2.0, "color": "#fef08a", "opacity": 0.5},
                    ),
                ],
                name=str(index),
                layout=go.Layout(title_text=f"{title} | max vorticity {vort_history[index]:.4f}"),
            )
        )
    fig.frames = plotly_frames
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
    parser.add_argument("--steps", type=int, default=650, help="LBM time steps.")
    parser.add_argument("--frame-every", type=int, default=13, help="Save one viewer frame every N steps.")
    parser.add_argument("--tau", type=float, default=0.60, help="LBM relaxation time. Must be greater than 0.5.")
    parser.add_argument("--inlet-velocity", type=float, default=0.045, help="Lattice-unit inlet speed.")
    parser.add_argument("--angle-of-attack", type=float, default=8.0, help="Wing angle of attack in degrees.")
    parser.add_argument("--chord", type=float, default=0.28, help="Wing chord as a fraction of domain length.")
    parser.add_argument("--span", type=float, default=0.58, help="Wing span as a fraction of domain width.")
    parser.add_argument("--thickness", type=float, default=0.13, help="NACA-like relative thickness.")
    parser.add_argument("--particles", type=int, default=760, help="Passive tracer particles.")
    parser.add_argument("--particle-step-scale", type=float, default=4.0, help="Particle advection visual speed multiplier.")
    parser.add_argument("--trail-length", type=int, default=28, help="Number of recent particle positions shown as pathline curves.")
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

    frames, wing_mesh, vort_history = simulate(
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
        trail_length=args.trail_length,
        device=device,
    )
    fig = build_figure(
        frames,
        wing_mesh,
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
