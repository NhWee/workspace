import argparse
from pathlib import Path

import torch

from shallow_water_plotly_viewer import build_interactive_figure
from wave_dataset import save_wave_dataset


def make_wave_numbers(size: int, domain_size: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spacing = domain_size / size
    ky = torch.fft.fftfreq(size, d=spacing, device=device) * (2.0 * torch.pi)
    kx = torch.fft.rfftfreq(size, d=spacing, device=device) * (2.0 * torch.pi)
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = torch.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
    return kx_grid, ky_grid, k_mag


def make_initial_spectrum(
    size: int,
    domain_size: float,
    peak_wavelength: float,
    bandwidth: float,
    wind_direction_degrees: float,
    directional_spread: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    kx_grid, ky_grid, k_mag = make_wave_numbers(size, domain_size, device)
    peak_k = 2.0 * torch.pi / peak_wavelength
    width = max(bandwidth * peak_k, 1.0e-6)

    wind_direction = torch.deg2rad(torch.tensor(wind_direction_degrees, device=device))
    wind_x = torch.cos(wind_direction)
    wind_y = torch.sin(wind_direction)
    direction_cosine = (kx_grid * wind_x + ky_grid * wind_y) / torch.clamp(k_mag, min=1.0e-6)
    direction_weight = torch.clamp(direction_cosine, min=0.0) ** directional_spread
    radial_weight = torch.exp(-0.5 * ((k_mag - peak_k) / width) ** 2)
    amplitude = radial_weight * direction_weight
    amplitude = torch.where(k_mag > 0.0, amplitude, torch.zeros_like(amplitude))

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise_real = torch.randn((size, size // 2 + 1), generator=generator, device=device)
    noise_imag = torch.randn((size, size // 2 + 1), generator=generator, device=device)
    spectrum = torch.complex(noise_real, noise_imag) * amplitude
    return spectrum, k_mag


def simulate_spectral_wave(
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
    device: torch.device,
    store_velocity: bool = False,
) -> tuple[list[torch.Tensor], torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
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
    omega = torch.sqrt(torch.clamp(gravity * k_mag, min=0.0))
    kx_grid, ky_grid, _ = make_wave_numbers(size, domain_size, device)
    safe_omega = torch.clamp(omega, min=1.0e-6)
    initial_eta = torch.fft.irfft2(spectrum, s=(size, size))
    normalization = torch.clamp(initial_eta.std(), min=1.0e-6)

    frames = []
    u_frames = []
    v_frames = []
    for step in range(steps):
        if step % frame_every != 0:
            continue
        elapsed = step * dt
        phase = torch.exp(1j * omega * elapsed)
        decay = damping ** step
        eta_spectrum = spectrum * phase
        eta = torch.fft.irfft2(eta_spectrum, s=(size, size))
        eta = eta * (wave_amplitude * decay / normalization)
        frames.append(eta.detach().cpu())
        if store_velocity:
            velocity_scale = wave_amplitude * decay / normalization
            u_spectrum = -1j * gravity * kx_grid * eta_spectrum / safe_omega
            v_spectrum = -1j * gravity * ky_grid * eta_spectrum / safe_omega
            u = torch.fft.irfft2(u_spectrum, s=(size, size)) * velocity_scale
            v = torch.fft.irfft2(v_spectrum, s=(size, size)) * velocity_scale
            u_frames.append(u.detach().cpu())
            v_frames.append(v.detach().cpu())

    depth = torch.ones((size, size), dtype=torch.float32) * 0.6
    if store_velocity:
        return frames, depth, u_frames, v_frames
    return frames, depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GPU FFT spectral wave surface dataset and Plotly viewer.")
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
    parser.add_argument("--store-velocity", action="store_true", help="Store approximate surface orbital u/v frames.")
    parser.add_argument("--max-surface-points", type=int, default=96, help="Max rendered points per surface axis.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_wave_dataset.npz"), help="Output NPZ path.")
    parser.add_argument("--viewer-output", type=Path, default=Path("outputs/spectral_wave_viewer.html"), help="Output Plotly HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    result = simulate_spectral_wave(
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
        device=device,
        store_velocity=args.store_velocity,
    )
    if args.store_velocity:
        frames, depth, u_frames, v_frames = result
    else:
        frames, depth = result
        u_frames = None
        v_frames = None
    metadata = {
        "solver": "spectral_wave_surface",
        "size": args.size,
        "steps": args.steps,
        "frame_every": args.frame_every,
        "domain_size": args.domain_size,
        "gravity": args.gravity,
        "dt": args.dt,
        "wave_amplitude": args.wave_amplitude,
        "peak_wavelength": args.peak_wavelength,
        "bandwidth": args.bandwidth,
        "wind_direction_degrees": args.wind_direction_degrees,
        "directional_spread": args.directional_spread,
        "damping": args.damping,
        "seed": args.seed,
        "device": str(device),
        "frame_count": len(frames),
        "stores_velocity": args.store_velocity,
    }
    save_wave_dataset(args.output, frames, depth, metadata, u_frames=u_frames, v_frames=v_frames)

    args.viewer_output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_interactive_figure(frames, depth, args.max_surface_points)
    fig.update_layout(title="Interactive GPU FFT spectral wave surface")
    fig.write_html(args.viewer_output, include_plotlyjs=True, full_html=True)
    print(f"Saved spectral wave viewer: {args.viewer_output}")


if __name__ == "__main__":
    main()
