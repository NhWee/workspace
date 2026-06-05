import argparse
import json
from pathlib import Path
from typing import Any

import torch

from compare_wave_datasets import (
    load_dataset_summary,
    make_markdown_table,
    save_final_frame_difference_heatmaps,
    save_frame_metric_chart,
    save_frame_metric_series,
    write_summary,
)
from shallow_water_bathymetry_3d import simulate_bathymetry
from wave_dataset import enrich_metadata, save_wave_dataset


SWEEP_PARAMETERS = ("damping", "gravity", "cfl")


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def format_value_for_name(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def build_run_parameters(args: argparse.Namespace, value: float) -> dict[str, Any]:
    parameters = {
        "size": args.size,
        "steps": args.steps,
        "frame_every": args.frame_every,
        "gravity": args.gravity,
        "dt": None if str(args.dt).lower() == "auto" else float(args.dt),
        "dt_label": args.dt,
        "damping": args.damping,
        "cfl": args.cfl,
        "store_velocity": args.store_velocity,
    }
    parameters[args.parameter] = value
    return parameters


def run_single_experiment(
    output_path: Path,
    run_index: int,
    experiment_name: str,
    sweep_parameter: str,
    sweep_value: float,
    parameters: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    result = simulate_bathymetry(
        size=parameters["size"],
        steps=parameters["steps"],
        frame_every=parameters["frame_every"],
        gravity=parameters["gravity"],
        dt=parameters["dt"],
        damping=parameters["damping"],
        device=device,
        cfl=parameters["cfl"],
        store_velocity=parameters["store_velocity"],
    )
    if parameters["store_velocity"]:
        frames, depth, u_frames, v_frames = result
    else:
        frames, depth = result
        u_frames = None
        v_frames = None

    metadata = {
        "solver": "bathymetry_shallow_water_sweep",
        "experiment_name": experiment_name,
        "run_index": run_index,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
        "size": parameters["size"],
        "steps": parameters["steps"],
        "frame_every": parameters["frame_every"],
        "gravity": parameters["gravity"],
        "dt": parameters["dt_label"],
        "cfl": parameters["cfl"],
        "damping": parameters["damping"],
        "device": str(device),
        "frame_count": len(frames),
        "stores_velocity": parameters["store_velocity"],
    }
    save_wave_dataset(output_path, frames, depth, metadata, u_frames=u_frames, v_frames=v_frames)
    return enrich_metadata(metadata) | {"dataset_path": str(output_path)}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved sweep manifest: {path}")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    experiment_dir = args.output_dir / args.experiment_name
    dataset_dir = experiment_dir / "datasets"
    dataset_paths = []
    runs = []

    for run_index, value in enumerate(args.values):
        run_name = f"{run_index:02d}_{args.parameter}_{format_value_for_name(value)}"
        output_path = dataset_dir / f"{run_name}.npz"
        parameters = build_run_parameters(args, value)
        print(f"Running {run_name}: {args.parameter}={value:g}")
        run_metadata = run_single_experiment(
            output_path=output_path,
            run_index=run_index,
            experiment_name=args.experiment_name,
            sweep_parameter=args.parameter,
            sweep_value=value,
            parameters=parameters,
            device=device,
        )
        dataset_paths.append(output_path)
        runs.append(run_metadata)

    summaries = [load_dataset_summary(path) for path in dataset_paths]
    heatmap_dir = experiment_dir / "diff_heatmaps"
    frame_metrics_dir = experiment_dir / "frame_metrics"
    comparison_path = experiment_dir / "comparison.md"
    frame_metrics_chart_path = experiment_dir / "frame_metrics.html"

    heatmap_paths = save_final_frame_difference_heatmaps(summaries, heatmap_dir)
    frame_metric_paths = save_frame_metric_series(summaries, frame_metrics_dir)
    save_frame_metric_chart(summaries, frame_metrics_chart_path)
    comparison_table = make_markdown_table(summaries)
    write_summary(comparison_path, comparison_table)

    manifest = {
        "experiment_name": args.experiment_name,
        "sweep_parameter": args.parameter,
        "sweep_values": args.values,
        "device": str(device),
        "outputs": {
            "experiment_dir": str(experiment_dir),
            "comparison": str(comparison_path),
            "frame_metrics_chart": str(frame_metrics_chart_path),
            "diff_heatmaps": [str(path) for path in heatmap_paths],
            "frame_metrics_csv": [str(path) for path in frame_metric_paths],
        },
        "runs": runs,
    }
    write_manifest(experiment_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small wave parameter sweep and compare the generated datasets.")
    parser.add_argument("--experiment-name", default="bathymetry_damping_sweep", help="Name of the output experiment folder.")
    parser.add_argument("--parameter", choices=SWEEP_PARAMETERS, default="damping", help="Parameter to sweep.")
    parser.add_argument("--values", type=parse_float_list, default=parse_float_list("0.9992,0.9994,0.9996"), help="Comma-separated sweep values.")
    parser.add_argument("--size", type=int, default=192, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps per run.")
    parser.add_argument("--frame-every", type=int, default=18, help="Save one frame every N simulation steps.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Base gravity coefficient.")
    parser.add_argument("--dt", default="auto", help="Time step, or 'auto' to use a CFL-based value.")
    parser.add_argument("--cfl", type=float, default=0.35, help="Base CFL factor used when --dt auto.")
    parser.add_argument("--damping", type=float, default=0.9994, help="Base global damping per step.")
    parser.add_argument("--store-velocity", action="store_true", help="Store u/v velocity frames in every dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweeps"), help="Root output directory.")
    return parser.parse_args()


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
