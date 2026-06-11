# 1D foam probe for eta(x, t).
# foam_source = slope_source * crest_weight
# slope_source = smoothstep((|d eta / dx| - foam_threshold) / transition)
# foam_t = max(foam_{t-1} * foam_decay, foam_source)
# You can handle the parameters
# points, frames, domain_size, dt, wave_speed, amplitude, wavelength,
# steepening, foam_threshold, foam_softness, crest_bias, foam_decay, output_dir
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_eta_series(
    points: int,
    frames: int,
    domain_size: float,
    dt: float,
    wave_speed: float,
    amplitude: float,
    wavelength: float,
    steepening: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    x = np.linspace(-0.5 * domain_size, 0.5 * domain_size, points)
    k = 2.0 * np.pi / wavelength
    eta_frames = []
    for frame in range(frames):
        phase = k * (x - wave_speed * frame * dt)
        carrier = np.sin(phase) + 0.35 * np.sin(2.0 * phase)
        envelope = np.exp(-0.5 * (x / (0.28 * domain_size)) ** 2)
        eta = amplitude * envelope * carrier
        eta += steepening * amplitude * envelope * np.maximum(np.sin(phase), 0.0) ** 2
        eta_frames.append(eta)
    return x, eta_frames


def foam_source_1d(
    eta: np.ndarray,
    dx: float,
    foam_threshold: float,
    foam_softness: float,
    crest_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = np.abs(np.gradient(eta, dx))
    transition = max(foam_threshold * foam_softness, 1.0e-6)
    slope_source = smoothstep((slope - foam_threshold) / transition)
    eta_std = max(float(np.std(eta)), 1.0e-6)
    crest_weight = smoothstep((eta - float(np.mean(eta))) / (eta_std * max(crest_bias, 1.0e-6)))
    return slope_source * crest_weight, slope, crest_weight


def persistent_foam_1d(
    eta_frames: list[np.ndarray],
    dx: float,
    foam_threshold: float,
    foam_softness: float,
    crest_bias: float,
    foam_decay: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    foam_frames = []
    source_frames = []
    slope_frames = []
    crest_weight_frames = []
    previous = None
    decay = float(np.clip(foam_decay, 0.0, 1.0))
    for eta in eta_frames:
        source, slope, crest_weight = foam_source_1d(eta, dx, foam_threshold, foam_softness, crest_bias)
        foam = source if previous is None else np.maximum(previous * decay, source)
        source_frames.append(source)
        slope_frames.append(slope)
        crest_weight_frames.append(crest_weight)
        foam_frames.append(foam)
        previous = foam
    return foam_frames, source_frames, slope_frames, crest_weight_frames


def save_probe_plot(
    x: np.ndarray,
    eta_frames: list[np.ndarray],
    foam_frames: list[np.ndarray],
    source_frames: list[np.ndarray],
    slope_frames: list[np.ndarray],
    crest_weight_frames: list[np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "foam_1d_probe_preview.png"
    heatmap_path = output_dir / "foam_1d_probe_heatmap.png"

    sample_indices = np.linspace(0, len(eta_frames) - 1, 4).round().astype(int)
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(11, 9), dpi=130, sharex=True)
    for axis, frame_index in zip(axes, sample_indices):
        eta = eta_frames[frame_index]
        foam = foam_frames[frame_index]
        source = source_frames[frame_index]
        slope = slope_frames[frame_index]
        crest_weight = crest_weight_frames[frame_index]
        foam_band = 0.09 * max(float(eta.max() - eta.min()), 1.0e-6)
        axis.plot(x, eta, color="#155e75", linewidth=1.8, label="eta")
        axis.fill_between(x, eta, eta + foam * foam_band, color="#e0f2fe", alpha=0.95, label="persistent foam")
        axis.plot(x, eta + source * foam_band * 1.25, color="#f97316", linewidth=1.0, alpha=0.75, label="foam source")
        axis.plot(x, eta.min() + slope / max(float(slope.max()), 1.0e-6) * foam_band * 2.5, color="#334155", linewidth=0.9, alpha=0.55, label="normalized |d eta/dx|")
        axis.plot(x, eta.min() + crest_weight * foam_band * 1.8, color="#64748b", linewidth=0.8, alpha=0.55, label="crest weight")
        axis.set_title(f"frame {frame_index}")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(preview_path, bbox_inches="tight")
    plt.close(fig)

    foam_array = np.stack(foam_frames)
    fig, ax = plt.subplots(figsize=(11, 4), dpi=130)
    image = ax.imshow(foam_array, aspect="auto", origin="lower", extent=[x.min(), x.max(), 0, len(foam_frames) - 1], cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Persistent foam density over time")
    ax.set_xlabel("x")
    ax.set_ylabel("frame")
    fig.colorbar(image, ax=ax, label="foam")
    fig.tight_layout()
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved preview: {preview_path}")
    print(f"Saved heatmap: {heatmap_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe foam generation on a 1D eta wave.")
    parser.add_argument("--points", type=int, default=512, help="Number of 1D samples.")
    parser.add_argument("--frames", type=int, default=80, help="Number of eta frames.")
    parser.add_argument("--domain-size", type=float, default=8.0, help="1D domain width.")
    parser.add_argument("--dt", type=float, default=0.035, help="Time step.")
    parser.add_argument("--wave-speed", type=float, default=1.2, help="Travel speed of the test wave.")
    parser.add_argument("--amplitude", type=float, default=0.18, help="Eta amplitude.")
    parser.add_argument("--wavelength", type=float, default=1.35, help="Dominant wavelength.")
    parser.add_argument("--steepening", type=float, default=0.55, help="Extra crest sharpening amount.")
    parser.add_argument("--foam-threshold", type=float, default=0.18, help="Slope threshold that starts foam.")
    parser.add_argument("--foam-softness", type=float, default=1.2, help="Soft transition width for foam source.")
    parser.add_argument("--crest-bias", type=float, default=0.85, help="Lower values restrict foam more strongly to crests.")
    parser.add_argument("--foam-decay", type=float, default=0.88, help="How much foam remains between frames.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/foam_1d_probe"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, eta_frames = make_eta_series(
        points=args.points,
        frames=args.frames,
        domain_size=args.domain_size,
        dt=args.dt,
        wave_speed=args.wave_speed,
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        steepening=args.steepening,
    )
    dx = args.domain_size / max(args.points - 1, 1)
    foam_frames, source_frames, slope_frames, crest_weight_frames = persistent_foam_1d(
        eta_frames,
        dx=dx,
        foam_threshold=args.foam_threshold,
        foam_softness=args.foam_softness,
        crest_bias=args.crest_bias,
        foam_decay=args.foam_decay,
    )
    save_probe_plot(x, eta_frames, foam_frames, source_frames, slope_frames, crest_weight_frames, args.output_dir)


if __name__ == "__main__":
    main()
