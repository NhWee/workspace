import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go
import torch

from benchmark_bathymetry import benchmark_size as benchmark_bathymetry_size
from benchmark_bathymetry import parse_sizes
from benchmark_spectral_wave import benchmark_size as benchmark_spectral_size


def add_solver_name(row: dict, solver: str) -> dict:
    result = dict(row)
    result["solver"] = solver
    return result


def make_comparison_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    solvers = list(dict.fromkeys(row["solver"] for row in rows))
    for solver in solvers:
        solver_rows = [row for row in rows if row["solver"] == solver]
        sizes = [row["size"] for row in solver_rows]
        labels = [f"{solver} {row['size']} x {row['size']}" for row in solver_rows]
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[row["million_cell_steps_per_sec"] for row in solver_rows],
                mode="lines+markers",
                name=f"{solver} throughput",
                customdata=labels,
                hovertemplate="%{customdata}<br>throughput=%{y:.4g}M cell-steps/s<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[row["elapsed_sec"] for row in solver_rows],
                mode="lines+markers",
                name=f"{solver} elapsed",
                yaxis="y2",
                customdata=labels,
                hovertemplate="%{customdata}<br>elapsed=%{y:.4g}s<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[row["peak_vram_gib"] for row in solver_rows],
                mode="lines+markers",
                name=f"{solver} VRAM",
                yaxis="y3",
                customdata=labels,
                hovertemplate="%{customdata}<br>peak VRAM=%{y:.4g}GiB<extra></extra>",
            )
        )

    fig.update_layout(
        title="Wave solver benchmark comparison",
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
        margin={"l": 72, "r": 130, "t": 56, "b": 48},
    )
    return fig


def write_rows_csv(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solver",
        "device",
        "size",
        "steps",
        "frame_every",
        "frame_count",
        "store_velocity",
        "dt",
        "elapsed_sec",
        "steps_per_sec",
        "million_cell_steps_per_sec",
        "max_abs_eta",
        "max_speed",
        "peak_vram_gib",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark comparison CSV: {output_path}")
    return output_path


def save_comparison_chart(rows: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_comparison_chart(rows)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved benchmark comparison chart: {output_path}")
    return output_path


def run_comparison(args: argparse.Namespace) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for size in parse_sizes(args.sizes):
        bathymetry_row = benchmark_bathymetry_size(
            size=size,
            steps=args.bathymetry_steps,
            gravity=args.bathymetry_gravity,
            damping=args.bathymetry_damping,
            cfl=args.bathymetry_cfl,
            device=device,
        )
        rows.append(add_solver_name(bathymetry_row, "bathymetry"))

        spectral_row = benchmark_spectral_size(
            size=size,
            steps=args.spectral_steps,
            frame_every=args.spectral_frame_every,
            domain_size=args.spectral_domain_size,
            gravity=args.spectral_gravity,
            dt=args.spectral_dt,
            wave_amplitude=args.spectral_wave_amplitude,
            peak_wavelength=args.spectral_peak_wavelength,
            bandwidth=args.spectral_bandwidth,
            wind_direction_degrees=args.spectral_wind_direction_degrees,
            directional_spread=args.spectral_directional_spread,
            damping=args.spectral_damping,
            seed=args.spectral_seed,
            store_velocity=args.spectral_store_velocity,
            device=device,
        )
        rows.append(add_solver_name(spectral_row, "spectral"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, args.output_dir / "wave_solver_benchmark_comparison.csv")
    save_comparison_chart(rows, args.output_dir / "wave_solver_benchmark_comparison.html")

    for row in rows:
        print(
            "solver={solver} size={size} steps={steps} elapsed={elapsed_sec:.3f}s "
            "throughput={million_cell_steps_per_sec:.1f}M cell-steps/s "
            "peak_vram={peak_vram_gib:.3f}GiB".format(**row)
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare bathymetry and spectral wave solver GPU benchmarks.")
    parser.add_argument("--sizes", default="256,512,1024", help="Comma-separated grid sizes.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")

    parser.add_argument("--bathymetry-steps", type=int, default=600, help="Bathymetry simulation steps.")
    parser.add_argument("--bathymetry-gravity", type=float, default=1.0, help="Bathymetry gravity coefficient.")
    parser.add_argument("--bathymetry-damping", type=float, default=0.9994, help="Bathymetry damping.")
    parser.add_argument("--bathymetry-cfl", type=float, default=0.35, help="Bathymetry CFL factor.")

    parser.add_argument("--spectral-steps", type=int, default=360, help="Spectral simulation steps.")
    parser.add_argument("--spectral-frame-every", type=int, default=12, help="Spectral frame interval.")
    parser.add_argument("--spectral-domain-size", type=float, default=8.0, help="Spectral periodic domain width.")
    parser.add_argument("--spectral-gravity", type=float, default=9.81, help="Spectral gravity coefficient.")
    parser.add_argument("--spectral-dt", type=float, default=0.04, help="Spectral time step.")
    parser.add_argument("--spectral-wave-amplitude", type=float, default=0.08, help="Spectral target eta scale.")
    parser.add_argument("--spectral-peak-wavelength", type=float, default=1.2, help="Spectral dominant wavelength.")
    parser.add_argument("--spectral-bandwidth", type=float, default=0.32, help="Spectral bandwidth.")
    parser.add_argument("--spectral-wind-direction-degrees", type=float, default=25.0, help="Spectral wind direction.")
    parser.add_argument("--spectral-directional-spread", type=float, default=6.0, help="Spectral directional spread.")
    parser.add_argument("--spectral-damping", type=float, default=0.9995, help="Spectral damping.")
    parser.add_argument("--spectral-seed", type=int, default=7, help="Spectral random seed.")
    parser.add_argument("--spectral-store-velocity", action="store_true", help="Benchmark spectral velocity generation.")
    return parser.parse_args()


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
