# 2D Navier-Stokes velocity field coupled to a 3D free-surface height field.
# du/dt + (u * grad)u = -grad(p) + viscosity * laplacian(u) + force
# div(u) = 0
# d2 eta/dt2 = wave_speed^2 * laplacian(eta) + vortex_surface_coupling - surface_damping * d eta/dt
# foam is generated from high vorticity, high speed, and elevated crests.
# You can handle the parameters
# size, steps, frame_every, dt, viscosity, pressure_iters, force_strength,
# force_radius, wave_speed, surface_coupling, surface_damping, eta_scale,
# periodic_force, periodic_force_strength, periodic_surface_strength,
# periodic_wavelength, periodic_period, periodic_direction_degrees,
# foam_vorticity_threshold, foam_speed_threshold, foam_birth, foam_decay,
# surface_smoothing, max_eta_velocity, foam_particles, particle_life,
# particle_spawn_per_frame, max_particles, splash_particles,
# splash_spawn_per_frame, splash_life, vortex_markers, quality,
# scene, max_surface_points, fps, frame_duration_ms, output
import argparse
from dataclasses import dataclass
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
    laplacian,
    make_grid,
    project_velocity,
    update_foam,
)


SCENE_PRESETS = {
    "none": {},
    "smooth_long": {
        "size": 64,
        "steps": 720,
        "frame_every": 2,
        "pressure_iters": 15,
        "fps": 24.0,
        "max_surface_points": 64,
        "particle_spawn_per_frame": 20,
        "max_particles": 600,
        "splash_spawn_per_frame": 10,
        "max_splash_particles": 260,
        "vortex_markers": False,
        "vortex_spirals": False,
        "force_strength": 4.8,
        "force_radius": 0.13,
        "wave_speed": 0.10,
        "surface_coupling": 0.35,
        "surface_damping": 1.20,
        "surface_smoothing": 0.12,
        "periodic_force": True,
        "periodic_force_strength": 0.18,
        "periodic_surface_strength": 0.030,
        "periodic_wavelength": 0.62,
        "periodic_period": 1.65,
        "output": Path("outputs/navier_stokes_free_surface_3d_smooth_long_scene.html"),
    },
    "dynamic_splash": {
        "size": 64,
        "steps": 720,
        "frame_every": 2,
        "pressure_iters": 18,
        "fps": 24.0,
        "max_surface_points": 64,
        "force_strength": 5.8,
        "force_radius": 0.15,
        "wave_speed": 0.12,
        "surface_coupling": 0.48,
        "surface_damping": 1.05,
        "surface_smoothing": 0.10,
        "periodic_force": True,
        "periodic_force_strength": 0.22,
        "periodic_surface_strength": 0.040,
        "periodic_wavelength": 0.58,
        "periodic_period": 1.45,
        "particle_spawn_per_frame": 28,
        "max_particles": 700,
        "splash_spawn_per_frame": 28,
        "max_splash_particles": 650,
        "splash_life": 0.50,
        "splash_gravity": 2.40,
        "splash_burst_max": 1.80,
        "splash_spread": 0.085,
        "vortex_markers": False,
        "vortex_spirals": False,
        "output": Path("outputs/navier_stokes_free_surface_3d_dynamic_splash_scene.html"),
    },
    "vortex_focus": {
        "size": 72,
        "steps": 720,
        "frame_every": 2,
        "pressure_iters": 22,
        "fps": 24.0,
        "max_surface_points": 72,
        "force_strength": 6.2,
        "force_radius": 0.17,
        "wave_speed": 0.11,
        "surface_coupling": 0.55,
        "surface_damping": 1.00,
        "surface_smoothing": 0.09,
        "periodic_force": True,
        "periodic_force_strength": 0.20,
        "periodic_surface_strength": 0.034,
        "periodic_wavelength": 0.55,
        "periodic_period": 1.50,
        "particle_spawn_per_frame": 24,
        "splash_spawn_per_frame": 18,
        "max_splash_particles": 480,
        "vortex_spiral_count": 7,
        "vortex_spiral_points": 64,
        "vortex_spiral_radius": 0.15,
        "vortex_line_width": 8.0,
        "max_vortex_markers": 130,
        "output": Path("outputs/navier_stokes_free_surface_3d_vortex_focus_scene.html"),
    },
}


def downsample(array: np.ndarray, max_points: int) -> np.ndarray:
    stride = max(1, int(np.ceil(array.shape[0] / max_points)))
    return array[::stride, ::stride]


def normalize01(field: torch.Tensor) -> torch.Tensor:
    field_min = torch.min(field)
    field_max = torch.max(field)
    return (field - field_min) / torch.clamp(field_max - field_min, min=1.0e-6)


@dataclass
class FoamParticles:
    x: torch.Tensor
    y: torch.Tensor
    age: torch.Tensor
    life: torch.Tensor


@dataclass
class SplashParticles:
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    age: torch.Tensor
    life: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor
    vz: torch.Tensor


def empty_particles(device: torch.device) -> FoamParticles:
    empty = torch.empty(0, device=device)
    return FoamParticles(empty, empty, empty, empty)


def empty_splash_particles(device: torch.device) -> SplashParticles:
    empty = torch.empty(0, device=device)
    return SplashParticles(empty, empty, empty, empty, empty, empty, empty, empty)


def smooth_field(field: torch.Tensor, amount: float) -> torch.Tensor:
    if amount <= 0.0:
        return field
    neighbor_average = (
        torch.roll(field, shifts=1, dims=0)
        + torch.roll(field, shifts=-1, dims=0)
        + torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
    ) * 0.25
    return (1.0 - amount) * field + amount * neighbor_average


def make_initial_eta(size: int, device: torch.device) -> torch.Tensor:
    xx, yy = make_grid(size, device)
    crest_a = 1.25 * torch.exp(-((xx - 0.30) ** 2 + (yy - 0.44) ** 2) / 0.010)
    trough_a = -1.05 * torch.exp(-((xx - 0.42) ** 2 + (yy - 0.56) ** 2) / 0.013)
    crest_b = 0.95 * torch.exp(-((xx - 0.70) ** 2 + (yy - 0.57) ** 2) / 0.012)
    trough_b = -0.85 * torch.exp(-((xx - 0.59) ** 2 + (yy - 0.43) ** 2) / 0.015)
    crossing_ripples = 0.22 * torch.sin(10.0 * np.pi * xx + 2.5 * torch.sin(2.0 * np.pi * yy))
    diagonal_ripples = 0.14 * torch.sin(7.0 * np.pi * (xx + yy))
    return 0.045 * (crest_a + trough_a + crest_b + trough_b + crossing_ripples + diagonal_ripples)


def make_initial_velocity(size: int, device: torch.device, strength: float, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    xx, yy = make_grid(size, device)
    u = torch.zeros((size, size), device=device)
    v = torch.zeros((size, size), device=device)
    vortices = [
        (0.32, 0.46, 1.35),
        (0.68, 0.54, -1.25),
        (0.50, 0.32, 0.70),
    ]
    for cx, cy, spin in vortices:
        dx = xx - cx
        dy = yy - cy
        r2 = dx * dx + dy * dy
        weight = torch.exp(-r2 / max(radius * radius, 1.0e-6))
        u = u + spin * (-dy) * weight * strength
        v = v + spin * dx * weight * strength
    shear = torch.exp(-((yy - 0.5) ** 2) / 0.05)
    u = u + 0.28 * strength * shear * torch.sin(2.0 * np.pi * yy)
    return u, v


def periodic_wave_phase(
    xx: torch.Tensor,
    yy: torch.Tensor,
    step: int,
    dt: float,
    wavelength: float,
    period: float,
    direction_degrees: float,
) -> torch.Tensor:
    theta = np.deg2rad(direction_degrees)
    direction_x = np.cos(theta)
    direction_y = np.sin(theta)
    coordinate = xx * direction_x + yy * direction_y
    return 2.0 * np.pi * (coordinate / max(wavelength, 1.0e-6) - (step * dt) / max(period, 1.0e-6))


def add_periodic_wave_forcing(
    u: torch.Tensor,
    v: torch.Tensor,
    eta_velocity: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    step: int,
    dt: float,
    force_strength: float,
    surface_strength: float,
    wavelength: float,
    period: float,
    direction_degrees: float,
    inlet_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if force_strength <= 0.0 and surface_strength <= 0.0:
        return u, v, eta_velocity

    theta = np.deg2rad(direction_degrees)
    direction_x = np.cos(theta)
    direction_y = np.sin(theta)
    phase = periodic_wave_phase(xx, yy, step, dt, wavelength, period, direction_degrees)
    coordinate = xx * direction_x + yy * direction_y
    inlet = torch.exp(-((coordinate - 0.08) ** 2) / max(inlet_width * inlet_width, 1.0e-6))
    lateral = torch.exp(-((yy - 0.5) ** 2) / 0.28)
    envelope = inlet * (0.35 + 0.65 * lateral)
    wave = torch.sin(phase)

    u = u + direction_x * force_strength * wave * envelope * dt
    v = v + direction_y * force_strength * wave * envelope * dt
    eta_velocity = eta_velocity + surface_strength * wave * envelope * dt
    return u, v, eta_velocity


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
    surface_smoothing: float,
    max_eta_velocity: float,
    periodic_acceleration: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    eta = advect(eta, u, v, dt)
    eta_velocity = advect(eta_velocity, u, v, dt)

    vorticity = torch.abs(curl_z(u, v, dx))
    speed = torch.sqrt(u * u + v * v)
    swirl_source = normalize01(vorticity) * normalize01(speed)
    swirl_source = swirl_source - torch.mean(swirl_source)

    acceleration = wave_speed * wave_speed * laplacian(eta, dx)
    acceleration = acceleration + surface_coupling * swirl_source
    if periodic_acceleration is not None:
        acceleration = acceleration + periodic_acceleration
    acceleration = acceleration - surface_damping * eta_velocity
    eta_velocity = eta_velocity + acceleration * dt
    if max_eta_velocity > 0.0:
        eta_velocity = torch.clamp(eta_velocity, -max_eta_velocity, max_eta_velocity)
    eta = eta + eta_velocity * dt
    eta = smooth_field(eta, surface_smoothing)
    eta = eta - torch.mean(eta)
    return eta, eta_velocity


def sample_field(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if len(x) == 0:
        return torch.empty(0, device=field.device)
    sample_grid = torch.stack((2.0 * torch.remainder(x, 1.0) - 1.0, 2.0 * torch.remainder(y, 1.0) - 1.0), dim=-1)
    sampled = torch.nn.functional.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        sample_grid.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, 0, :, 0]


def step_foam_particles(
    particles: FoamParticles,
    foam: torch.Tensor,
    eta: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    particle_life: float,
    particle_spawn_per_frame: int,
    max_particles: int,
    frame_every: int,
    step: int,
) -> FoamParticles:
    if len(particles.x) > 0:
        vx = sample_field(u, particles.x, particles.y)
        vy = sample_field(v, particles.x, particles.y)
        age = particles.age + dt
        x = torch.remainder(particles.x + vx * dt, 1.0)
        y = torch.remainder(particles.y + vy * dt, 1.0)
        alive = age < particles.life
        particles = FoamParticles(x[alive], y[alive], age[alive], particles.life[alive])

    spawn_count = max(0, int(np.ceil(particle_spawn_per_frame / max(frame_every, 1))))
    if spawn_count > 0 and len(particles.x) < max_particles:
        source = torch.flatten(foam)
        source = torch.clamp(source - 0.18, min=0.0)
        source_sum = torch.sum(source)
        if float(source_sum.detach().cpu()) > 0.0:
            spawn_count = min(spawn_count, max_particles - len(particles.x))
            chosen = torch.multinomial(source / source_sum, spawn_count, replacement=True)
            size = foam.shape[0]
            new_y = (torch.div(chosen, size, rounding_mode="floor").float() + torch.rand(spawn_count, device=foam.device)) / size
            new_x = ((chosen % size).float() + torch.rand(spawn_count, device=foam.device)) / size
            new_life = torch.empty(spawn_count, device=foam.device).uniform_(0.65 * particle_life, 1.25 * particle_life)
            particles = FoamParticles(
                torch.cat([particles.x, new_x]),
                torch.cat([particles.y, new_y]),
                torch.cat([particles.age, torch.zeros(spawn_count, device=foam.device)]),
                torch.cat([particles.life, new_life]),
            )

    return particles


def step_splash_particles(
    particles: SplashParticles,
    eta: torch.Tensor,
    foam: torch.Tensor,
    vorticity: torch.Tensor,
    speed: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    gravity: float,
    splash_spawn_per_frame: int,
    frame_every: int,
    max_splash_particles: int,
    splash_life: float,
    splash_threshold: float,
    splash_burst_min: float,
    splash_burst_max: float,
    splash_spread: float,
) -> SplashParticles:
    if len(particles.x) > 0:
        age = particles.age + dt
        vx = particles.vx
        vy = particles.vy
        vz = particles.vz - gravity * dt
        x = torch.remainder(particles.x + vx * dt, 1.0)
        y = torch.remainder(particles.y + vy * dt, 1.0)
        z = particles.z + vz * dt
        surface_z = sample_field(eta, x, y)
        alive = (age < particles.life) & (z > surface_z + 0.002)
        particles = SplashParticles(x[alive], y[alive], z[alive], age[alive], particles.life[alive], vx[alive], vy[alive], vz[alive])

    spawn_count = max(0, int(np.ceil(splash_spawn_per_frame / max(frame_every, 1))))
    if spawn_count > 0 and len(particles.x) < max_splash_particles:
        source = normalize01(torch.abs(vorticity)) * normalize01(speed) * torch.clamp(foam + 0.35, 0.0, 1.0)
        source = torch.flatten(torch.clamp(source - splash_threshold, min=0.0))
        source_sum = torch.sum(source)
        if float(source_sum.detach().cpu()) > 0.0:
            spawn_count = min(spawn_count, max_splash_particles - len(particles.x))
            chosen = torch.multinomial(source / source_sum, spawn_count, replacement=True)
            size = eta.shape[0]
            new_y = (torch.div(chosen, size, rounding_mode="floor").float() + torch.rand(spawn_count, device=eta.device)) / size
            new_x = ((chosen % size).float() + torch.rand(spawn_count, device=eta.device)) / size
            base_z = sample_field(eta, new_x, new_y)
            local_u = sample_field(u, new_x, new_y)
            local_v = sample_field(v, new_x, new_y)
            burst = torch.empty(spawn_count, device=eta.device).uniform_(splash_burst_min, splash_burst_max)
            new_life = torch.empty(spawn_count, device=eta.device).uniform_(0.65 * splash_life, 1.25 * splash_life)
            particles = SplashParticles(
                torch.cat([particles.x, new_x]),
                torch.cat([particles.y, new_y]),
                torch.cat([particles.z, base_z + 0.018]),
                torch.cat([particles.age, torch.zeros(spawn_count, device=eta.device)]),
                torch.cat([particles.life, new_life]),
                torch.cat([particles.vx, local_u * 0.45 + torch.empty(spawn_count, device=eta.device).uniform_(-splash_spread, splash_spread)]),
                torch.cat([particles.vy, local_v * 0.45 + torch.empty(spawn_count, device=eta.device).uniform_(-splash_spread, splash_spread)]),
                torch.cat([particles.vz, burst]),
            )

    return particles


def particle_frame(
    particles: FoamParticles,
    eta: torch.Tensor,
    eta_scale: float,
    particle_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(particles.x) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    z = sample_field(eta, particles.x, particles.y) * eta_scale + particle_height
    alpha = torch.clamp(1.0 - particles.age / torch.clamp(particles.life, min=1.0e-6), 0.0, 1.0)
    return (
        (particles.x - 0.5).detach().cpu().numpy().astype(np.float32),
        (particles.y - 0.5).detach().cpu().numpy().astype(np.float32),
        z.detach().cpu().numpy().astype(np.float32),
        alpha.detach().cpu().numpy().astype(np.float32),
    )


def splash_frame(
    particles: SplashParticles,
    eta_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(particles.x) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    alpha = torch.clamp(1.0 - particles.age / torch.clamp(particles.life, min=1.0e-6), 0.0, 1.0)
    return (
        (particles.x - 0.5).detach().cpu().numpy().astype(np.float32),
        (particles.y - 0.5).detach().cpu().numpy().astype(np.float32),
        (particles.z * eta_scale).detach().cpu().numpy().astype(np.float32),
        alpha.detach().cpu().numpy().astype(np.float32),
    )


def vortex_marker_frame(
    xx: torch.Tensor,
    yy: torch.Tensor,
    eta: torch.Tensor,
    vorticity: torch.Tensor,
    speed: torch.Tensor,
    eta_scale: float,
    max_markers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_markers <= 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    score = torch.flatten(normalize01(torch.abs(vorticity)) * normalize01(speed))
    count = min(max_markers, score.numel())
    values, indices = torch.topk(score, count)
    keep = values > 0.55
    if int(torch.sum(keep).detach().cpu()) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    indices = indices[keep]
    values = values[keep]
    size = eta.shape[0]
    y_idx = torch.div(indices, size, rounding_mode="floor")
    x_idx = indices % size
    return (
        (xx[y_idx, x_idx] - 0.5).detach().cpu().numpy().astype(np.float32),
        (yy[y_idx, x_idx] - 0.5).detach().cpu().numpy().astype(np.float32),
        (eta[y_idx, x_idx] * eta_scale + 0.018).detach().cpu().numpy().astype(np.float32),
        values.detach().cpu().numpy().astype(np.float32),
    )


def vortex_spiral_frame(
    eta: torch.Tensor,
    vorticity: torch.Tensor,
    speed: torch.Tensor,
    eta_scale: float,
    vortex_count: int,
    points_per_vortex: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if vortex_count <= 0 or points_per_vortex <= 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    score = torch.flatten(normalize01(torch.abs(vorticity)) * normalize01(speed))
    values, indices = torch.topk(score, min(vortex_count, score.numel()))
    keep = values > 0.58
    if int(torch.sum(keep).detach().cpu()) == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty

    values = values[keep]
    indices = indices[keep]
    size = eta.shape[0]
    centers_y = (torch.div(indices, size, rounding_mode="floor").float() + 0.5) / size
    centers_x = ((indices % size).float() + 0.5) / size
    signs = torch.sign(torch.flatten(vorticity)[indices])

    angles = torch.linspace(0.0, 2.5 * np.pi, points_per_vortex, device=eta.device)
    radial = torch.linspace(0.18 * radius, radius, points_per_vortex, device=eta.device)
    xs = []
    ys = []
    zs = []
    strengths = []
    for cx, cy, sign, strength in zip(centers_x, centers_y, signs, values):
        theta = sign * angles
        x = torch.remainder(cx + radial * torch.cos(theta), 1.0)
        y = torch.remainder(cy + radial * torch.sin(theta), 1.0)
        z = sample_field(eta, x, y) * eta_scale + 0.024
        xs.append((x - 0.5).detach().cpu().numpy())
        ys.append((y - 0.5).detach().cpu().numpy())
        zs.append(z.detach().cpu().numpy())
        strengths.append(torch.full_like(x, strength).detach().cpu().numpy())
        xs.append(np.array([np.nan], dtype=np.float32))
        ys.append(np.array([np.nan], dtype=np.float32))
        zs.append(np.array([np.nan], dtype=np.float32))
        strengths.append(np.array([np.nan], dtype=np.float32))

    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(zs).astype(np.float32),
        np.concatenate(strengths).astype(np.float32),
    )


def simulate_free_surface(
    size: int,
    steps: int,
    frame_every: int,
    dt: float,
    viscosity: float,
    pressure_iters: int,
    force_strength: float,
    force_radius: float,
    periodic_force: bool,
    periodic_force_strength: float,
    periodic_surface_strength: float,
    periodic_wavelength: float,
    periodic_period: float,
    periodic_direction_degrees: float,
    periodic_inlet_width: float,
    wave_speed: float,
    surface_coupling: float,
    surface_damping: float,
    surface_smoothing: float,
    max_eta_velocity: float,
    eta_scale: float,
    foam_vorticity_threshold: float,
    foam_speed_threshold: float,
    foam_birth: float,
    foam_decay: float,
    foam_particles: bool,
    particle_life: float,
    particle_spawn_per_frame: int,
    max_particles: int,
    particle_height: float,
    splash_particles: bool,
    splash_spawn_per_frame: int,
    max_splash_particles: int,
    splash_life: float,
    splash_gravity: float,
    splash_threshold: float,
    splash_burst_min: float,
    splash_burst_max: float,
    splash_spread: float,
    vortex_markers: bool,
    max_vortex_markers: int,
    vortex_spirals: bool,
    vortex_spiral_count: int,
    vortex_spiral_points: int,
    vortex_spiral_radius: float,
    max_surface_points: int,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    dx = 1.0 / size
    xx, yy = make_grid(size, device)
    u, v = make_initial_velocity(size, device, force_strength, force_radius * 1.35)
    eta = make_initial_eta(size, device)
    eta_velocity = torch.zeros_like(eta)
    foam = torch.zeros_like(eta)
    particles = empty_particles(device)
    splash = empty_splash_particles(device)
    frames = []

    for step in range(steps):
        u = advect(u, u, v, dt)
        v = advect(v, u, v, dt)
        u, v = add_vortex_forces(u, v, xx, yy, step, dt, force_strength, force_radius)
        periodic_acceleration = None
        if periodic_force:
            phase = periodic_wave_phase(xx, yy, step, dt, periodic_wavelength, periodic_period, periodic_direction_degrees)
            theta = np.deg2rad(periodic_direction_degrees)
            coordinate = xx * np.cos(theta) + yy * np.sin(theta)
            inlet = torch.exp(-((coordinate - 0.08) ** 2) / max(periodic_inlet_width * periodic_inlet_width, 1.0e-6))
            periodic_acceleration = periodic_surface_strength * torch.sin(phase) * inlet
            u, v, eta_velocity = add_periodic_wave_forcing(
                u,
                v,
                eta_velocity,
                xx,
                yy,
                step,
                dt,
                periodic_force_strength,
                periodic_surface_strength,
                periodic_wavelength,
                periodic_period,
                periodic_direction_degrees,
                periodic_inlet_width,
            )

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
            surface_smoothing,
            max_eta_velocity,
            periodic_acceleration,
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
        vorticity = curl_z(u, v, dx)
        speed = torch.sqrt(u * u + v * v)
        if foam_particles:
            particles = step_foam_particles(
                particles,
                foam,
                eta,
                u,
                v,
                dt,
                particle_life,
                particle_spawn_per_frame,
                max_particles,
                frame_every,
                step,
            )
        if splash_particles:
            splash = step_splash_particles(
                splash,
                eta,
                foam,
                vorticity,
                speed,
                u,
                v,
                dt,
                splash_gravity,
                splash_spawn_per_frame,
                frame_every,
                max_splash_particles,
                splash_life,
                splash_threshold,
                splash_burst_min,
                splash_burst_max,
                splash_spread,
            )

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
                particle_frame(particles, eta, eta_scale, particle_height) if foam_particles else particle_frame(empty_particles(device), eta, eta_scale, particle_height),
                splash_frame(splash, eta_scale) if splash_particles else splash_frame(empty_splash_particles(device), eta_scale),
                vortex_marker_frame(xx, yy, eta, vorticity, speed, eta_scale, max_vortex_markers) if vortex_markers else vortex_marker_frame(xx, yy, eta, vorticity, speed, eta_scale, 0),
                vortex_spiral_frame(eta, vorticity, speed, eta_scale, vortex_spiral_count, vortex_spiral_points, vortex_spiral_radius) if vortex_spirals else vortex_spiral_frame(eta, vorticity, speed, eta_scale, 0, 0, vortex_spiral_radius),
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
        showlegend=False,
        uid="free_surface",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>eta=%{z:.4f}<extra>free surface</extra>",
    )


def make_particle_trace(
    particles: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    particle_size: float,
) -> go.Scatter3d:
    x, y, z, alpha = particles
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={
            "size": np.clip((2.0 + 4.0 * alpha) * particle_size, 1.0, 7.0),
            "color": alpha,
            "colorscale": [[0.0, "#bfdbfe"], [0.45, "#e0f2fe"], [1.0, "#ffffff"]],
            "opacity": 0.72,
            "showscale": False,
        },
        name="foam particles",
        showlegend=False,
        uid="foam_particles",
        hovertemplate="foam alpha=%{marker.color:.2f}<extra></extra>",
    )


def make_splash_trace(
    particles: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    splash_size: float,
) -> go.Scatter3d:
    x, y, z, alpha = particles
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={
            "size": np.clip((1.5 + 4.0 * alpha) * splash_size, 1.0, 6.0),
            "color": alpha,
            "colorscale": [[0.0, "#93c5fd"], [0.45, "#dbeafe"], [1.0, "#ffffff"]],
            "opacity": 0.78,
            "showscale": False,
        },
        name="splash",
        showlegend=False,
        uid="splash_particles",
        hovertemplate="splash alpha=%{marker.color:.2f}<extra></extra>",
    )


def make_vortex_trace(
    markers: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    vortex_size: float,
) -> go.Scatter3d:
    x, y, z, strength = markers
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={
            "size": np.clip((2.0 + 5.0 * strength) * vortex_size, 1.0, 8.0),
            "color": strength,
            "colorscale": [[0.0, "#38bdf8"], [0.55, "#facc15"], [1.0, "#fb7185"]],
            "opacity": 0.56,
            "showscale": False,
        },
        name="vortex markers",
        showlegend=False,
        uid="vortex_markers",
        hovertemplate="vortex=%{marker.color:.2f}<extra></extra>",
    )


def make_vortex_spiral_trace(
    spiral: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    line_width: float,
) -> go.Scatter3d:
    x, y, z, strength = spiral
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={
            "width": line_width,
            "color": strength,
            "colorscale": [[0.0, "#22d3ee"], [0.5, "#fef08a"], [1.0, "#fb7185"]],
        },
        opacity=0.82,
        name="vortex spirals",
        showlegend=False,
        uid="vortex_spirals",
        hoverinfo="skip",
    )


def frame_z_limit(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
) -> float:
    z_values = []
    for _x, _y, z, _foam, particles, splash, vortex, spiral in frames:
        z_values.append(np.abs(z).ravel())
        for layer in (particles, splash, vortex, spiral):
            if len(layer[2]) > 0:
                z_values.append(np.abs(layer[2][np.isfinite(layer[2])]).ravel())
    finite = np.concatenate([values for values in z_values if len(values) > 0])
    if len(finite) == 0:
        return 0.02
    return max(float(np.max(finite)) * 1.20, 0.02)


def build_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    frame_duration_ms: int,
    particle_size: float,
    splash_size: float,
    vortex_size: float,
    vortex_line_width: float,
) -> go.Figure:
    z_limit = frame_z_limit(frames)
    first_x, first_y, first_z, first_foam, first_particles, first_splash, first_vortex, first_spiral = frames[0]
    fig = go.Figure(data=[
        make_surface_trace(first_x, first_y, first_z, first_foam, z_limit, True),
        make_particle_trace(first_particles, particle_size),
        make_splash_trace(first_splash, splash_size),
        make_vortex_trace(first_vortex, vortex_size),
        make_vortex_spiral_trace(first_spiral, vortex_line_width),
    ])
    fig.frames = [
        go.Frame(
            data=[
                make_surface_trace(x, y, z, foam, z_limit, False),
                make_particle_trace(particles, particle_size),
                make_splash_trace(splash, splash_size),
                make_vortex_trace(vortex, vortex_size),
                make_vortex_spiral_trace(spiral, vortex_line_width),
            ],
            name=str(index),
        )
        for index, (x, y, z, foam, particles, splash, vortex, spiral) in enumerate(frames)
    ]
    frame_names = [frame.name for frame in fig.frames]
    animation_options = {
        "frame": {"duration": frame_duration_ms, "redraw": True},
        "transition": {"duration": 0},
        "mode": "immediate",
    }
    fig.update_layout(
        title="Navier-Stokes driven 3D free-surface foam experiment",
        width=980,
        height=780,
        paper_bgcolor="#06101d",
        plot_bgcolor="#06101d",
        font={"color": "#e5edf7"},
        showlegend=False,
        uirevision="fixed_camera",
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        scene={
            "xaxis": {"title": "x", "range": [-0.55, 0.55], "autorange": False, "backgroundcolor": "#06101d", "gridcolor": "#17324a"},
            "yaxis": {"title": "y", "range": [-0.55, 0.55], "autorange": False, "backgroundcolor": "#06101d", "gridcolor": "#17324a"},
            "zaxis": {"title": "eta", "range": [-z_limit, z_limit], "autorange": False, "backgroundcolor": "#06101d", "gridcolor": "#17324a"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.26},
            "camera": {"eye": {"x": 1.35, "y": 1.25, "z": 0.78}},
            "uirevision": "fixed_camera",
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
                        "label": "Replay",
                        "method": "animate",
                        "args": [
                            frame_names,
                            {
                                **animation_options,
                                "fromcurrent": False,
                            },
                        ],
                    },
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                **animation_options,
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "y": 0.0,
                "len": 0.76,
                "currentvalue": {"prefix": "frame ", "font": {"size": 12}},
                "steps": [
                    {
                        "label": str(index),
                        "method": "animate",
                        "args": [
                            [name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for index, name in enumerate(frame_names)
                ],
            }
        ],
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Navier-Stokes driven 3D free-surface foam experiment.")
    parser.add_argument("--scene", choices=tuple(SCENE_PRESETS), default="none", help="Reusable scene preset.")
    parser.add_argument("--quality", choices=("preview", "balanced", "final"), default="balanced", help="Output preset.")
    parser.add_argument("--size", type=int, default=288, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=1080, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=3, help="Save one viewer frame every N simulation steps.")
    parser.add_argument("--dt", type=float, default=0.0045, help="Simulation time step.")
    parser.add_argument("--viscosity", type=float, default=0.00008, help="Kinematic viscosity.")
    parser.add_argument("--pressure-iters", type=int, default=60, help="Jacobi pressure projection iterations.")
    parser.add_argument("--force-strength", type=float, default=4.4, help="Strength of rotating vortex force sources.")
    parser.add_argument("--force-radius", type=float, default=0.12, help="Radius of rotating vortex force sources.")
    parser.add_argument("--periodic-force", action=argparse.BooleanOptionalAction, default=False, help="Apply repeating external wave forcing from one side.")
    parser.add_argument("--periodic-force-strength", type=float, default=0.18, help="Horizontal periodic forcing strength.")
    parser.add_argument("--periodic-surface-strength", type=float, default=0.030, help="Vertical free-surface periodic forcing strength.")
    parser.add_argument("--periodic-wavelength", type=float, default=0.62, help="Wavelength of the external forcing pattern.")
    parser.add_argument("--periodic-period", type=float, default=1.60, help="Temporal period of the external forcing pattern.")
    parser.add_argument("--periodic-direction-degrees", type=float, default=0.0, help="Direction of periodic forcing in degrees.")
    parser.add_argument("--periodic-inlet-width", type=float, default=0.20, help="Width of the side inlet region for periodic forcing.")
    parser.add_argument("--wave-speed", type=float, default=0.18, help="Free-surface wave propagation speed.")
    parser.add_argument("--surface-coupling", type=float, default=0.55, help="How strongly vortices disturb the surface.")
    parser.add_argument("--surface-damping", type=float, default=0.85, help="Damping applied to vertical surface velocity.")
    parser.add_argument("--surface-smoothing", type=float, default=0.08, help="Per-step surface smoothing amount.")
    parser.add_argument("--max-eta-velocity", type=float, default=0.18, help="Clamp for vertical surface velocity. Use 0 to disable.")
    parser.add_argument("--eta-scale", type=float, default=1.0, help="Vertical display scale for eta.")
    parser.add_argument("--foam-vorticity-threshold", type=float, default=8.0, help="Curl magnitude needed to generate foam.")
    parser.add_argument("--foam-speed-threshold", type=float, default=0.18, help="Speed needed to generate foam.")
    parser.add_argument("--foam-birth", type=float, default=2.2, help="Foam generation rate.")
    parser.add_argument("--foam-decay", type=float, default=0.48, help="Foam fade-out rate.")
    parser.add_argument("--foam-particles", action=argparse.BooleanOptionalAction, default=True, help="Render foam as particles above the surface.")
    parser.add_argument("--particle-life", type=float, default=0.9, help="Average foam particle lifetime.")
    parser.add_argument("--particle-spawn-per-frame", type=int, default=45, help="Approximate foam particles spawned per rendered frame.")
    parser.add_argument("--max-particles", type=int, default=1600, help="Maximum active foam particles.")
    parser.add_argument("--particle-height", type=float, default=0.006, help="Vertical offset above the surface for foam particles.")
    parser.add_argument("--particle-size", type=float, default=0.85, help="Rendered foam particle size scale.")
    parser.add_argument("--splash-particles", action=argparse.BooleanOptionalAction, default=True, help="Render upward splash particles.")
    parser.add_argument("--splash-spawn-per-frame", type=int, default=10, help="Approximate splash particles spawned per rendered frame.")
    parser.add_argument("--max-splash-particles", type=int, default=420, help="Maximum active splash particles.")
    parser.add_argument("--splash-life", type=float, default=0.42, help="Average splash particle lifetime.")
    parser.add_argument("--splash-gravity", type=float, default=2.8, help="Downward acceleration applied to splash particles.")
    parser.add_argument("--splash-threshold", type=float, default=0.16, help="Minimum energetic source value for splash emission.")
    parser.add_argument("--splash-burst-min", type=float, default=0.80, help="Minimum upward splash velocity.")
    parser.add_argument("--splash-burst-max", type=float, default=1.45, help="Maximum upward splash velocity.")
    parser.add_argument("--splash-spread", type=float, default=0.065, help="Horizontal random spread for splash particles.")
    parser.add_argument("--splash-size", type=float, default=0.8, help="Rendered splash particle size scale.")
    parser.add_argument("--vortex-markers", action=argparse.BooleanOptionalAction, default=False, help="Render analysis markers on high-vorticity regions.")
    parser.add_argument("--max-vortex-markers", type=int, default=90, help="Maximum vortex markers per frame.")
    parser.add_argument("--vortex-size", type=float, default=0.75, help="Rendered vortex marker size scale.")
    parser.add_argument("--vortex-spirals", action=argparse.BooleanOptionalAction, default=False, help="Render analysis spiral lines over high-vorticity regions.")
    parser.add_argument("--vortex-spiral-count", type=int, default=4, help="Number of visible vortex spirals.")
    parser.add_argument("--vortex-spiral-points", type=int, default=48, help="Points per vortex spiral.")
    parser.add_argument("--vortex-spiral-radius", type=float, default=0.105, help="Radius of each rendered vortex spiral.")
    parser.add_argument("--vortex-line-width", type=float, default=6.0, help="Rendered vortex spiral line width.")
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max rendered points per surface axis.")
    parser.add_argument("--fps", type=float, default=None, help="Viewer playback FPS. Overrides --frame-duration-ms when set.")
    parser.add_argument("--frame-duration-ms", type=int, default=30, help="Animation frame duration in milliseconds.")
    parser.add_argument("--output", type=Path, default=Path("outputs/navier_stokes_free_surface_3d.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def arg_was_supplied(name: str, argv: list[str]) -> bool:
    option = f"--{name.replace('_', '-')}"
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def apply_scene_preset(args: argparse.Namespace, argv: list[str]) -> None:
    preset = SCENE_PRESETS[args.scene]
    for key, value in preset.items():
        if not arg_was_supplied(key, argv):
            setattr(args, key, value)


def apply_quality_preset(args: argparse.Namespace) -> None:
    if args.quality == "preview":
        args.size = min(args.size, 128)
        args.steps = min(args.steps, 420)
        args.frame_every = max(args.frame_every, 4)
        args.pressure_iters = min(args.pressure_iters, 35)
        args.max_surface_points = min(args.max_surface_points, 96)
        args.max_particles = min(args.max_particles, 700)
        args.max_splash_particles = min(args.max_splash_particles, 180)
        args.max_vortex_markers = min(args.max_vortex_markers, 45)
        args.vortex_spiral_points = min(args.vortex_spiral_points, 32)
    elif args.quality == "final":
        args.frame_every = min(args.frame_every, 2)
        args.pressure_iters = max(args.pressure_iters, 80)
        args.max_surface_points = max(args.max_surface_points, 160)
        args.max_particles = max(args.max_particles, 2400)
        args.max_splash_particles = max(args.max_splash_particles, 650)
        args.max_vortex_markers = max(args.max_vortex_markers, 120)
        args.vortex_spiral_points = max(args.vortex_spiral_points, 64)


def main() -> None:
    args = parse_args()
    apply_scene_preset(args, sys.argv[1:])
    apply_quality_preset(args)
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
        periodic_force=args.periodic_force,
        periodic_force_strength=args.periodic_force_strength,
        periodic_surface_strength=args.periodic_surface_strength,
        periodic_wavelength=args.periodic_wavelength,
        periodic_period=args.periodic_period,
        periodic_direction_degrees=args.periodic_direction_degrees,
        periodic_inlet_width=args.periodic_inlet_width,
        wave_speed=args.wave_speed,
        surface_coupling=args.surface_coupling,
        surface_damping=args.surface_damping,
        surface_smoothing=args.surface_smoothing,
        max_eta_velocity=args.max_eta_velocity,
        eta_scale=args.eta_scale,
        foam_vorticity_threshold=args.foam_vorticity_threshold,
        foam_speed_threshold=args.foam_speed_threshold,
        foam_birth=args.foam_birth,
        foam_decay=args.foam_decay,
        foam_particles=args.foam_particles,
        particle_life=args.particle_life,
        particle_spawn_per_frame=args.particle_spawn_per_frame,
        max_particles=args.max_particles,
        particle_height=args.particle_height,
        splash_particles=args.splash_particles,
        splash_spawn_per_frame=args.splash_spawn_per_frame,
        max_splash_particles=args.max_splash_particles,
        splash_life=args.splash_life,
        splash_gravity=args.splash_gravity,
        splash_threshold=args.splash_threshold,
        splash_burst_min=args.splash_burst_min,
        splash_burst_max=args.splash_burst_max,
        splash_spread=args.splash_spread,
        vortex_markers=args.vortex_markers,
        max_vortex_markers=args.max_vortex_markers,
        vortex_spirals=args.vortex_spirals,
        vortex_spiral_count=args.vortex_spiral_count,
        vortex_spiral_points=args.vortex_spiral_points,
        vortex_spiral_radius=args.vortex_spiral_radius,
        max_surface_points=args.max_surface_points,
        device=device,
    )
    frame_duration_ms = args.frame_duration_ms
    if args.fps is not None:
        frame_duration_ms = max(1, int(round(1000.0 / max(args.fps, 1.0e-6))))

    fig = build_figure(frames, frame_duration_ms, args.particle_size, args.splash_size, args.vortex_size, args.vortex_line_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved Navier-Stokes free-surface viewer: {args.output}")


if __name__ == "__main__":
    main()
