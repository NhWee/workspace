import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from wave_sim.spectral_wave.spectral_wave_surface_3d import make_initial_spectrum, make_wave_numbers


def downsample_array(array: np.ndarray, max_points: int) -> np.ndarray:
    stride = max(1, int(np.ceil(array.shape[0] / max_points)))
    return array[::stride, ::stride]


def make_base_grid(size: int, domain_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-0.5 * domain_size, 0.5 * domain_size, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def simulate_choppy_frames(
    size: int,
    steps: int,
    frame_every: int,
    domain_size: float,
    gravity: float,
    dt: float,
    wave_amplitude: float,
    peak_wavelength: float,
    bandwidth: float,
    wind_direction_degrees: float,
    directional_spread: float,
    damping: float,
    seed: int,
    choppiness: float,
    max_surface_points: int,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    spectrum, k_mag = make_initial_spectrum(
        size=size,
        domain_size=domain_size,
        peak_wavelength=peak_wavelength,
        bandwidth=bandwidth,
        wind_direction_degrees=wind_direction_degrees,
        directional_spread=directional_spread,
        seed=seed,
        device=device,
    )
    kx_grid, ky_grid, _ = make_wave_numbers(size, domain_size, device)
    omega = torch.sqrt(torch.clamp(gravity * k_mag, min=0.0))
    safe_k = torch.clamp(k_mag, min=1.0e-6)
    base_x, base_y = make_base_grid(size, domain_size)
    base_x = base_x.to(device)
    base_y = base_y.to(device)

    initial_eta = torch.fft.irfft2(spectrum, s=(size, size))
    normalization = torch.clamp(initial_eta.std(), min=1.0e-6)
    frames = []
    for step in range(steps):
        if step % frame_every != 0:
            continue
        elapsed = step * dt
        phase = torch.exp(1j * omega * elapsed)
        decay = damping ** step
        eta_spectrum = spectrum * phase
        scale = wave_amplitude * decay / normalization
        eta = torch.fft.irfft2(eta_spectrum, s=(size, size)) * scale
        displacement_x = torch.fft.irfft2(-1j * kx_grid / safe_k * eta_spectrum, s=(size, size)) * scale * choppiness
        displacement_y = torch.fft.irfft2(-1j * ky_grid / safe_k * eta_spectrum, s=(size, size)) * scale * choppiness

        x = (base_x + displacement_x).detach().cpu().numpy()
        y = (base_y + displacement_y).detach().cpu().numpy()
        z = eta.detach().cpu().numpy()
        frames.append(
            (
                downsample_array(x, max_surface_points),
                downsample_array(y, max_surface_points),
                downsample_array(z, max_surface_points),
            )
        )
    return frames


def make_surface_trace(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray, showscale: bool) -> go.Surface:
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=z_grid,
        colorscale="Blues",
        opacity=0.82,
        showscale=showscale,
        colorbar={"title": "eta"} if showscale else None,
        contours_z={"show": False},
        name="choppy water",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>eta=%{z:.4f}<extra>choppy water</extra>",
    )


def sample_foam_points(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    foam_threshold: float,
    max_foam_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gradient_y, gradient_x = np.gradient(z_grid)
    steepness = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    mask = steepness >= foam_threshold
    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    foam_x = x_grid[mask]
    foam_y = y_grid[mask]
    foam_z = z_grid[mask]
    foam_steepness = steepness[mask]
    if len(foam_x) > max_foam_points:
        order = np.argsort(foam_steepness)[-max_foam_points:]
        foam_x = foam_x[order]
        foam_y = foam_y[order]
        foam_z = foam_z[order]
        foam_steepness = foam_steepness[order]
    return foam_x, foam_y, foam_z, foam_steepness


def make_foam_trace(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    foam_threshold: float,
    max_foam_points: int,
) -> go.Scatter3d:
    foam_x, foam_y, foam_z, foam_steepness = sample_foam_points(
        x_grid,
        y_grid,
        z_grid,
        foam_threshold,
        max_foam_points,
    )
    return go.Scatter3d(
        x=foam_x,
        y=foam_y,
        z=foam_z + 0.01,
        mode="markers",
        marker={
            "size": 3,
            "color": foam_steepness,
            "colorscale": [[0.0, "#f8fbff"], [1.0, "#ffffff"]],
            "opacity": 0.92,
            "showscale": False,
        },
        name="foam highlights",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>eta=%{z:.4f}<extra>foam highlights</extra>",
    )


def build_choppy_figure(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    domain_size: float,
    show_foam: bool = True,
    foam_threshold: float = 0.018,
    max_foam_points: int = 900,
) -> go.Figure:
    z_limit = max(float(np.max(np.abs(z))) for _, _, z in frames)
    z_limit = max(z_limit * 1.4, 0.08)
    half_domain = 0.55 * domain_size

    def frame_traces(x: np.ndarray, y: np.ndarray, z: np.ndarray, showscale: bool) -> list:
        traces = [make_surface_trace(x, y, z, showscale)]
        if show_foam:
            traces.append(make_foam_trace(x, y, z, foam_threshold, max_foam_points))
        return traces

    figure_frames = [
        go.Frame(
            data=frame_traces(x, y, z, True),
            name=str(index),
        )
        for index, (x, y, z) in enumerate(frames)
    ]
    slider_steps = [
        {
            "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(frames))
    ]

    fig = go.Figure(data=frame_traces(*frames[0], showscale=True), frames=figure_frames)
    fig.update_layout(
        title="Interactive GPU FFT choppy wave surface",
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
                                "frame": {"duration": 90, "redraw": True},
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
    parser = argparse.ArgumentParser(description="Create an interactive GPU FFT choppy wave surface viewer.")
    parser.add_argument("--size", type=int, default=256, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=12, help="Save one frame every N simulation steps.")
    parser.add_argument("--domain-size", type=float, default=8.0, help="Physical width of the periodic domain.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity coefficient.")
    parser.add_argument("--dt", type=float, default=0.04, help="Time step.")
    parser.add_argument("--wave-amplitude", type=float, default=0.08, help="Target initial standard deviation of eta.")
    parser.add_argument("--peak-wavelength", type=float, default=1.2, help="Dominant wavelength.")
    parser.add_argument("--bandwidth", type=float, default=0.32, help="Relative spectral bandwidth around the peak.")
    parser.add_argument("--wind-direction-degrees", type=float, default=25.0, help="Dominant propagation direction.")
    parser.add_argument("--directional-spread", type=float, default=6.0, help="Higher values narrow the directional spectrum.")
    parser.add_argument("--damping", type=float, default=0.9995, help="Global spectral amplitude damping per step.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the initial spectrum.")
    parser.add_argument("--choppiness", type=float, default=0.75, help="Horizontal displacement multiplier.")
    parser.add_argument("--hide-foam", action="store_true", help="Disable steepness-based foam marker overlay.")
    parser.add_argument("--foam-threshold", type=float, default=0.018, help="Minimum downsampled eta steepness for foam markers.")
    parser.add_argument("--max-foam-points", type=int, default=900, help="Maximum foam markers per frame.")
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_choppy_wave_viewer.html"), help="Output Plotly HTML path.")
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
    fig = build_choppy_figure(
        frames,
        args.domain_size,
        show_foam=not args.hide_foam,
        foam_threshold=args.foam_threshold,
        max_foam_points=args.max_foam_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Saved choppy wave viewer: {args.output}")


if __name__ == "__main__":
    main()
