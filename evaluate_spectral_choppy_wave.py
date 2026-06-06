import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from spectral_choppy_wave_viewer import simulate_choppy_frames


def make_reference_grid(shape: tuple[int, int], domain_size: float) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = shape
    x_axis = np.linspace(-0.5 * domain_size, 0.5 * domain_size, columns)
    y_axis = np.linspace(-0.5 * domain_size, 0.5 * domain_size, rows)
    reference_x, reference_y = np.meshgrid(x_axis, y_axis, indexing="xy")
    return reference_x, reference_y


def signed_triangle_areas_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    return 0.5 * (ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0])


def mesh_fold_metrics(x_grid: np.ndarray, y_grid: np.ndarray) -> dict:
    points = np.stack((x_grid, y_grid), axis=-1)
    p00 = points[:-1, :-1]
    p01 = points[:-1, 1:]
    p10 = points[1:, :-1]
    p11 = points[1:, 1:]
    triangle_areas = np.concatenate(
        (
            signed_triangle_areas_2d(p00, p01, p11).reshape(-1),
            signed_triangle_areas_2d(p00, p11, p10).reshape(-1),
        )
    )
    folded = triangle_areas <= 0.0
    return {
        "signed_triangle_area_min": float(np.min(triangle_areas)),
        "signed_triangle_area_mean": float(np.mean(triangle_areas)),
        "folded_triangle_count": int(np.count_nonzero(folded)),
        "folded_triangle_ratio": float(np.count_nonzero(folded) / len(triangle_areas)),
    }


def frame_metrics(
    frame_index: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    domain_size: float,
    foam_threshold: float,
) -> dict:
    gradient_y, gradient_x = np.gradient(z_grid)
    steepness = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    foam_mask = steepness >= foam_threshold
    reference_x, reference_y = make_reference_grid(z_grid.shape, domain_size)
    horizontal_displacement = np.sqrt((x_grid - reference_x) ** 2 + (y_grid - reference_y) ** 2)
    metrics = {
        "frame_index": frame_index,
        "eta_min": float(np.min(z_grid)),
        "eta_max": float(np.max(z_grid)),
        "eta_mean": float(np.mean(z_grid)),
        "eta_std": float(np.std(z_grid)),
        "eta_range": float(np.max(z_grid) - np.min(z_grid)),
        "steepness_mean": float(np.mean(steepness)),
        "steepness_p95": float(np.percentile(steepness, 95.0)),
        "steepness_max": float(np.max(steepness)),
        "foam_point_count": int(np.count_nonzero(foam_mask)),
        "foam_ratio": float(np.count_nonzero(foam_mask) / foam_mask.size),
        "horizontal_displacement_mean": float(np.mean(horizontal_displacement)),
        "horizontal_displacement_p95": float(np.percentile(horizontal_displacement, 95.0)),
        "horizontal_displacement_max": float(np.max(horizontal_displacement)),
    }
    metrics.update(mesh_fold_metrics(x_grid, y_grid))
    return metrics


def summarize_frame_metrics(frame_rows: list[dict]) -> dict:
    if not frame_rows:
        raise RuntimeError("No frame metrics were provided.")
    numeric_keys = [key for key in frame_rows[0] if key != "frame_index"]
    summary = {"frame_count": len(frame_rows)}
    for key in numeric_keys:
        values = np.array([row[key] for row in frame_rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_max"] = float(np.max(values))
        summary[f"{key}_min"] = float(np.min(values))
    return summary


def evaluate_choppy_frames(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    domain_size: float,
    foam_threshold: float,
) -> dict:
    frame_rows = [
        frame_metrics(index, x_grid, y_grid, z_grid, domain_size, foam_threshold)
        for index, (x_grid, y_grid, z_grid) in enumerate(frames)
    ]
    return {
        "frame_metrics": frame_rows,
        "summary": summarize_frame_metrics(frame_rows),
    }


def write_metric_csv(rows: list[dict], output: Path) -> Path:
    if not rows:
        raise RuntimeError("No metric rows to write.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_metric_report(metrics: dict) -> str:
    summary = metrics["summary"]
    lines = [
        "# Spectral Choppy Wave Metrics",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| frame_count | {summary['frame_count']} |",
        f"| eta_range_max | {summary['eta_range_max']:.6g} |",
        f"| eta_std_mean | {summary['eta_std_mean']:.6g} |",
        f"| steepness_p95_max | {summary['steepness_p95_max']:.6g} |",
        f"| steepness_max_max | {summary['steepness_max_max']:.6g} |",
        f"| foam_ratio_mean | {summary['foam_ratio_mean']:.6g} |",
        f"| foam_point_count_max | {summary['foam_point_count_max']:.0f} |",
        f"| horizontal_displacement_p95_max | {summary['horizontal_displacement_p95_max']:.6g} |",
        f"| horizontal_displacement_max_max | {summary['horizontal_displacement_max_max']:.6g} |",
        f"| folded_triangle_ratio_max | {summary['folded_triangle_ratio_max']:.6g} |",
        f"| folded_triangle_count_max | {summary['folded_triangle_count_max']:.0f} |",
        f"| signed_triangle_area_min_min | {summary['signed_triangle_area_min_min']:.6g} |",
    ]
    return "\n".join(lines) + "\n"


def write_metric_outputs(metrics: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_metric_csv(metrics["frame_metrics"], output_dir / "choppy_wave_metrics.csv")
    json_path = output_dir / "choppy_wave_metrics.json"
    report_path = output_dir / "choppy_wave_metrics.md"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_metric_report(metrics), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "report": str(report_path),
        "summary": metrics["summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPU FFT choppy wave surface quality metrics.")
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
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max evaluated surface points per axis.")
    parser.add_argument("--foam-threshold", type=float, default=0.018, help="Minimum downsampled eta steepness for foam points.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spectral_choppy_wave_metrics"), help="Metric output directory.")
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
    outputs = write_metric_outputs(evaluate_choppy_frames(frames, args.domain_size, args.foam_threshold), args.output_dir)
    print(f"Frames: {outputs['summary']['frame_count']}")
    print(f"CSV: {outputs['csv']}")
    print(f"Report: {outputs['report']}")


if __name__ == "__main__":
    main()
