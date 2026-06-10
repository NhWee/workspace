import argparse
import csv
import time
from pathlib import Path

import torch

from wave_sim.shallow_water.shallow_water_bathymetry_3d import compute_cfl_dt, make_bathymetry, simulate_bathymetry


def benchmark_size(
    size: int,
    steps: int,
    gravity: float,
    damping: float,
    cfl: float,
    device: torch.device,
) -> dict[str, float | int | str]:
    dx = 2.0 / (size - 1)
    depth, _ = make_bathymetry(size, device)
    dt = compute_cfl_dt(depth, gravity, dx, cfl)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    frames, _ = simulate_bathymetry(
        size=size,
        steps=steps,
        frame_every=steps,
        gravity=gravity,
        dt=dt,
        damping=damping,
        device=device,
        cfl=cfl,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    cells = size * size
    cell_steps = cells * steps
    max_abs_eta = float(frames[-1].abs().max())
    peak_vram_gib = (
        float(torch.cuda.max_memory_allocated() / 1024**3)
        if device.type == "cuda"
        else 0.0
    )

    return {
        "device": str(device),
        "size": size,
        "steps": steps,
        "dt": dt,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps / elapsed,
        "million_cell_steps_per_sec": cell_steps / elapsed / 1_000_000,
        "max_abs_eta": max_abs_eta,
        "peak_vram_gib": peak_vram_gib,
    }


def parse_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the GPU bathymetry shallow-water solver.")
    parser.add_argument("--sizes", default="256,512,1024", help="Comma-separated grid sizes.")
    parser.add_argument("--steps", type=int, default=600, help="Simulation steps per grid size.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravity coefficient g.")
    parser.add_argument("--damping", type=float, default=0.9994, help="Global damping per step.")
    parser.add_argument("--cfl", type=float, default=0.35, help="CFL factor for automatic dt.")
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
        benchmark_size(size, args.steps, args.gravity, args.damping, args.cfl, device)
        for size in parse_sizes(args.sizes)
    ]

    csv_path = args.output_dir / "bathymetry_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            "size={size} steps={steps} dt={dt:.6g} elapsed={elapsed_sec:.3f}s "
            "throughput={million_cell_steps_per_sec:.1f}M cell-steps/s "
            "peak_vram={peak_vram_gib:.3f}GiB max_eta={max_abs_eta:.4f}".format(**row)
        )
    print(f"Saved benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
