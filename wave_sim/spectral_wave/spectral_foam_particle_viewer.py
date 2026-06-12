# GPU FFT choppy wave with foam particle simulation.
# foam_source = steepness_source * crest_weight
# particles spawn on high foam_source crests, drift along wind direction,
# spread with random jitter, follow the nearest eta surface, and fade by age.
# You can handle the parameters
# size, steps, frame_every, domain_size, gravity, dt, wave_amplitude,
# peak_wavelength, bandwidth, wind_direction_degrees, directional_spread,
# damping, seed, choppiness, foam_threshold, foam_softness, foam_crest_bias,
# particle_life, particle_drift, particle_spread, max_particles,
# spawn_per_frame, trail_length, frame_duration_ms, max_surface_points, output
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
import torch

from wave_sim.spectral_wave.spectral_choppy_wave_viewer import (
    foam_intensity,
    make_surface_trace,
    simulate_choppy_frames,
)


@dataclass
class FoamParticles:
    x: np.ndarray
    y: np.ndarray
    age: np.ndarray
    life: np.ndarray
    drift_x: np.ndarray
    drift_y: np.ndarray


def empty_particles() -> FoamParticles:
    return FoamParticles(
        x=np.empty(0, dtype=np.float32),
        y=np.empty(0, dtype=np.float32),
        age=np.empty(0, dtype=np.float32),
        life=np.empty(0, dtype=np.float32),
        drift_x=np.empty(0, dtype=np.float32),
        drift_y=np.empty(0, dtype=np.float32),
    )


def bilinear_sample_grid(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.empty(0, dtype=np.float32)

    rows, cols = z_grid.shape
    x_axis = x_grid[0, :]
    y_axis = y_grid[:, 0]
    x_min, x_max = float(x_axis.min()), float(x_axis.max())
    y_min, y_max = float(y_axis.min()), float(y_axis.max())
    x_norm = np.clip((x - x_min) / max(x_max - x_min, 1.0e-6) * (cols - 1), 0.0, cols - 1.001)
    y_norm = np.clip((y - y_min) / max(y_max - y_min, 1.0e-6) * (rows - 1), 0.0, rows - 1.001)

    x0 = np.floor(x_norm).astype(np.int32)
    y0 = np.floor(y_norm).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)
    sx = x_norm - x0
    sy = y_norm - y0

    z00 = z_grid[y0, x0]
    z10 = z_grid[y0, x1]
    z01 = z_grid[y1, x0]
    z11 = z_grid[y1, x1]
    return ((1.0 - sx) * (1.0 - sy) * z00 + sx * (1.0 - sy) * z10 + (1.0 - sx) * sy * z01 + sx * sy * z11).astype(np.float32)


def spawn_particles(
    particles: FoamParticles,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    source: np.ndarray,
    wind_direction_degrees: float,
    particle_life: float,
    particle_drift: float,
    particle_spread: float,
    spawn_per_frame: int,
    max_particles: int,
    rng: np.random.Generator,
) -> FoamParticles:
    candidate_y, candidate_x = np.nonzero(source > 0.05)
    if len(candidate_x) == 0 or spawn_per_frame <= 0:
        return particles

    weights = source[candidate_y, candidate_x].astype(np.float64)
    weights_sum = float(weights.sum())
    if weights_sum <= 0.0:
        return particles
    weights /= weights_sum
    spawn_count = min(spawn_per_frame, len(candidate_x), max_particles)
    chosen = rng.choice(len(candidate_x), size=spawn_count, replace=len(candidate_x) < spawn_count, p=weights)

    theta = np.deg2rad(wind_direction_degrees)
    base_dx = np.cos(theta) * particle_drift
    base_dy = np.sin(theta) * particle_drift
    spread = rng.normal(0.0, particle_spread, size=(spawn_count, 2)).astype(np.float32)
    new_x = x_grid[candidate_y[chosen], candidate_x[chosen]].astype(np.float32)
    new_y = y_grid[candidate_y[chosen], candidate_x[chosen]].astype(np.float32)
    new_life = rng.uniform(0.65 * particle_life, 1.25 * particle_life, spawn_count).astype(np.float32)

    particles = FoamParticles(
        x=np.concatenate([particles.x, new_x]),
        y=np.concatenate([particles.y, new_y]),
        age=np.concatenate([particles.age, np.zeros(spawn_count, dtype=np.float32)]),
        life=np.concatenate([particles.life, new_life]),
        drift_x=np.concatenate([particles.drift_x, np.full(spawn_count, base_dx, dtype=np.float32) + spread[:, 0]]),
        drift_y=np.concatenate([particles.drift_y, np.full(spawn_count, base_dy, dtype=np.float32) + spread[:, 1]]),
    )

    if len(particles.x) > max_particles:
        keep = np.argsort(particles.age / np.maximum(particles.life, 1.0e-6))[:max_particles]
        particles = FoamParticles(
            x=particles.x[keep],
            y=particles.y[keep],
            age=particles.age[keep],
            life=particles.life[keep],
            drift_x=particles.drift_x[keep],
            drift_y=particles.drift_y[keep],
        )
    return particles


def step_particles(
    particles: FoamParticles,
    dt: float,
    domain_size: float,
) -> FoamParticles:
    if len(particles.x) == 0:
        return particles

    age = particles.age + dt
    x = particles.x + particles.drift_x * dt
    y = particles.y + particles.drift_y * dt
    half_domain = 0.55 * domain_size
    alive = (age < particles.life) & (np.abs(x) <= half_domain) & (np.abs(y) <= half_domain)
    return FoamParticles(
        x=x[alive],
        y=y[alive],
        age=age[alive],
        life=particles.life[alive],
        drift_x=particles.drift_x[alive],
        drift_y=particles.drift_y[alive],
    )


def simulate_foam_particles(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    wind_direction_degrees: float,
    dt: float,
    frame_every: int,
    domain_size: float,
    foam_threshold: float,
    foam_softness: float,
    foam_crest_bias: float,
    particle_life: float,
    particle_drift: float,
    particle_spread: float,
    spawn_per_frame: int,
    max_particles: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    particles = empty_particles()
    particle_frames = []
    frame_dt = dt * frame_every

    for x_grid, y_grid, z_grid in frames:
        source = foam_intensity(z_grid, foam_threshold, foam_softness, foam_crest_bias)
        particles = step_particles(particles, frame_dt, domain_size)
        particles = spawn_particles(
            particles,
            x_grid,
            y_grid,
            source,
            wind_direction_degrees,
            particle_life,
            particle_drift,
            particle_spread,
            spawn_per_frame,
            max_particles,
            rng,
        )
        z = bilinear_sample_grid(x_grid, y_grid, z_grid, particles.x, particles.y) + 0.018
        alpha = np.clip(1.0 - particles.age / np.maximum(particles.life, 1.0e-6), 0.0, 1.0)
        particle_frames.append((particles.x.copy(), particles.y.copy(), z, alpha.astype(np.float32)))

    return particle_frames


def make_particle_trace(
    particle_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    name: str = "foam particles",
    size_scale: float = 1.0,
    opacity: float = 0.82,
) -> go.Scatter3d:
    x, y, z, alpha = particle_frame
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={
            "size": np.clip((2.5 + 4.5 * alpha) * size_scale, 1.0, 7.0),
            "color": alpha,
            "colorscale": [[0.0, "#dbeafe"], [0.35, "#eff6ff"], [1.0, "#ffffff"]],
            "opacity": opacity,
            "showscale": False,
        },
        name=name,
        hovertemplate="foam age opacity=%{marker.color:.2f}<extra></extra>",
    )


def make_trail_particle_frame(
    particle_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    frame_index: int,
    trail_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if trail_length <= 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty

    xs = []
    ys = []
    zs = []
    alphas = []
    start = max(0, frame_index - trail_length)
    for offset, source_index in enumerate(range(start, frame_index)):
        x, y, z, alpha = particle_frames[source_index]
        fade = (offset + 1) / max(frame_index - start, 1)
        xs.append(x)
        ys.append(y)
        zs.append(z - 0.006)
        alphas.append(alpha * 0.35 * fade)
    if not xs:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(alphas)


def build_foam_particle_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    particle_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    domain_size: float,
    foam_threshold: float,
    foam_softness: float,
    foam_crest_bias: float,
    trail_length: int,
    frame_duration_ms: int,
) -> go.Figure:
    z_limit = max(float(np.max(np.abs(z))) for _, _, z in frames)
    z_limit = max(z_limit * 1.4, 0.08)
    half_domain = 0.55 * domain_size

    def traces_for(index: int, showscale: bool) -> list:
        x, y, z = frames[index]
        trail_frame = make_trail_particle_frame(particle_frames, index, trail_length)
        return [
            make_surface_trace(x, y, z, showscale, "overlay", foam_threshold, foam_softness, foam_crest_bias),
            make_particle_trace(trail_frame, name="foam trail", size_scale=0.8, opacity=0.35),
            make_particle_trace(particle_frames[index]),
        ]

    figure_frames = [
        go.Frame(data=traces_for(index, True), name=str(index))
        for index in range(len(frames))
    ]
    slider_steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(frames))
    ]

    fig = go.Figure(data=traces_for(0, True), frames=figure_frames)
    fig.update_layout(
        title="Interactive GPU FFT choppy wave with foam particles",
        scene={
            "xaxis": {"title": "x", "range": [-half_domain, half_domain]},
            "yaxis": {"title": "y", "range": [-half_domain, half_domain]},
            "zaxis": {"title": "eta", "range": [-z_limit, z_limit]},
            "aspectratio": {"x": 1, "y": 1, "z": 0.22},
            "camera": {"eye": {"x": -1.55, "y": -1.55, "z": 0.92}},
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
        sliders=[{"active": 0, "currentvalue": {"prefix": "Frame "}, "pad": {"t": 42}, "steps": slider_steps}],
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive choppy wave viewer with foam particles.")
    parser.add_argument("--size", type=int, default=384, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=2160, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=2, help="Save one frame every N simulation steps.")
    parser.add_argument("--domain-size", type=float, default=8.0, help="Physical width of the periodic domain.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity coefficient.")
    parser.add_argument("--dt", type=float, default=0.025, help="Time step.")
    parser.add_argument("--wave-amplitude", type=float, default=0.08, help="Target initial standard deviation of eta.")
    parser.add_argument("--peak-wavelength", type=float, default=1.8, help="Dominant wavelength.")
    parser.add_argument("--bandwidth", type=float, default=0.18, help="Relative spectral bandwidth around the peak.")
    parser.add_argument("--wind-direction-degrees", type=float, default=25.0, help="Dominant propagation direction.")
    parser.add_argument("--directional-spread", type=float, default=10.0, help="Higher values narrow the directional spectrum.")
    parser.add_argument("--damping", type=float, default=0.9995, help="Global spectral amplitude damping per step.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the initial spectrum.")
    parser.add_argument("--choppiness", type=float, default=0.75, help="Horizontal displacement multiplier.")
    parser.add_argument("--foam-threshold", type=float, default=0.035, help="Minimum downsampled eta steepness for foam.")
    parser.add_argument("--foam-softness", type=float, default=1.5, help="Soft transition width for foam source.")
    parser.add_argument("--foam-crest-bias", type=float, default=0.85, help="Lower values restrict foam more strongly to crests.")
    parser.add_argument("--particle-life", type=float, default=1.0, help="Average foam particle lifetime in seconds.")
    parser.add_argument("--particle-drift", type=float, default=0.62, help="Foam particle drift speed along wind direction.")
    parser.add_argument("--particle-spread", type=float, default=0.06, help="Random drift spread per particle.")
    parser.add_argument("--spawn-per-frame", type=int, default=55, help="Maximum new foam particles per rendered frame.")
    parser.add_argument("--max-particles", type=int, default=2400, help="Maximum active foam particles.")
    parser.add_argument("--trail-length", type=int, default=5, help="Number of previous frames drawn as a soft foam trail.")
    parser.add_argument("--frame-duration-ms", type=int, default=45, help="Animation frame duration in milliseconds.")
    parser.add_argument("--max-surface-points", type=int, default=160, help="Max rendered points per surface axis.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_foam_particle_viewer.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames = simulate_choppy_frames(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        domain_size=args.domain_size,
        gravity=args.gravity,
        dt=args.dt,
        wave_amplitude=args.wave_amplitude,
        peak_wavelength=args.peak_wavelength,
        bandwidth=args.bandwidth,
        wind_direction_degrees=args.wind_direction_degrees,
        directional_spread=args.directional_spread,
        damping=args.damping,
        seed=args.seed,
        choppiness=args.choppiness,
        max_surface_points=args.max_surface_points,
        device=device,
    )
    particle_frames = simulate_foam_particles(
        frames,
        wind_direction_degrees=args.wind_direction_degrees,
        dt=args.dt,
        frame_every=args.frame_every,
        domain_size=args.domain_size,
        foam_threshold=args.foam_threshold,
        foam_softness=args.foam_softness,
        foam_crest_bias=args.foam_crest_bias,
        particle_life=args.particle_life,
        particle_drift=args.particle_drift,
        particle_spread=args.particle_spread,
        spawn_per_frame=args.spawn_per_frame,
        max_particles=args.max_particles,
        seed=args.seed + 101,
    )
    fig = build_foam_particle_figure(
        frames,
        particle_frames,
        domain_size=args.domain_size,
        foam_threshold=args.foam_threshold,
        foam_softness=args.foam_softness,
        foam_crest_bias=args.foam_crest_bias,
        trail_length=args.trail_length,
        frame_duration_ms=args.frame_duration_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved foam particle viewer: {args.output}")


if __name__ == "__main__":
    main()
