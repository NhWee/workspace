import argparse
import csv
import time
from pathlib import Path

import torch
import plotly.graph_objects as go

from spectral_wave_surface_3d import simulate_spectral_wave


def parse_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def benchmark_size(
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
    store_velocity: bool,
    device: torch.device,
) -> dict[str, float | int | str | bool]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = simulate_spectral_wave(
        size=size,
        steps=steps,
        frame_every=frame_every,
        domain_size=domain_size,
        gravity=gravity,
        dt=dt,
        wave_amplitude=wave_amplitude,
        peak_wavelength=peak_wavelength,
        bandwidth=bandwidth,
        wind_direction_degrees=wind_direction_degrees,
        directional_spread=directional_spread,
        damping=damping,
        seed=seed,
        device=device,
        store_velocity=store_velocity,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if store_velocity:
        frames, _, u_frames, v_frames = result
        max_speed = float(torch.sqrt(u_frames[-1] * u_frames[-1] + v_frames[-1] * v_frames[-1]).max())
    else:
        frames, _ = result
        max_speed = 0.0

    cells = size * size
    cell_steps = cells * steps
    frame_count = len(frames)
    max_abs_eta = float(frames[-1].abs().max())
    peak_vram_gib = float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

    return {
        "device": str(device),
        "size": size,
        "steps": steps,
        "frame_every": frame_every,
        "frame_count": frame_count,
        "store_velocity": store_velocity,
        "dt": dt,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps / elapsed,
        "million_cell_steps_per_sec": cell_steps / elapsed / 1_000_000,
        "max_abs_eta": max_abs_eta,
        "max_speed": max_speed,
        "peak_vram_gib": peak_vram_gib,
    }


def make_benchmark_chart(rows: list[dict[str, float | int | str | bool]]) -> go.Figure:
    sizes = [row["size"] for row in rows]
    labels = [f"{row['size']} x {row['size']}" for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=[row["million_cell_steps_per_sec"] for row in rows],
            mode="lines+markers",
            name="Throughput",
            customdata=labels,
            hovertemplate="grid=%{customdata}<br>throughput=%{y:.4g}M cell-steps/s<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=[row["elapsed_sec"] for row in rows],
            mode="lines+markers",
            name="Elapsed seconds",
            yaxis="y2",
            customdata=labels,
            hovertemplate="grid=%{customdata}<br>elapsed=%{y:.4g}s<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=[row["peak_vram_gib"] for row in rows],
            mode="lines+markers",
            name="Peak VRAM GiB",
            yaxis="y3",
            customdata=labels,
            hovertemplate="grid=%{customdata}<br>peak VRAM=%{y:.4g}GiB<extra></extra>",
        )
    )
    fig.update_layout(
        title="Spectral wave benchmark",
        xaxis={"title": "Grid size"},
        yaxis={"title": "M cell-steps/s"},
        yaxis2={
            "title": "Elapsed seconds",
            "overlaying": "y",
            "side": "right",
        },
        yaxis3={
            "title": "Peak VRAM GiB",
            "overlaying": "y",
            "side": "right",
            "anchor": "free",
            "position": 0.94,
        },
        hovermode="x unified",
        margin={"l": 72, "r": 120, "t": 56, "b": 48},
    )
    return fig


def save_benchmark_chart(rows: list[dict[str, float | int | str | bool]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_benchmark_chart(rows)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved spectral benchmark chart: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the GPU FFT spectral wave surface solver.")
    parser.add_argument("--sizes", default="256,512,1024", help="Comma-separated grid sizes.")
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps per grid size.")
    parser.add_argument("--frame-every", type=int, default=12, help="Store one frame every N simulation steps.")
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
    parser.add_argument("--store-velocity", action="store_true", help="Also benchmark u/v velocity frame generation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        benchmark_size(
            size=size,
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
            store_velocity=args.store_velocity,
            device=device,
        )
        for size in parse_sizes(args.sizes)
    ]

    csv_path = args.output_dir / "spectral_wave_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    chart_path = args.output_dir / "spectral_wave_benchmark.html"
    save_benchmark_chart(rows, chart_path)

    for row in rows:
        print(
            "size={size} steps={steps} frames={frame_count} velocity={store_velocity} "
            "elapsed={elapsed_sec:.3f}s throughput={million_cell_steps_per_sec:.1f}M cell-steps/s "
            "peak_vram={peak_vram_gib:.3f}GiB max_eta={max_abs_eta:.4f} max_speed={max_speed:.4f}".format(**row)
        )
    print(f"Saved spectral benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
