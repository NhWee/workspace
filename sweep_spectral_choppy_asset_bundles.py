import argparse
import json
from pathlib import Path

import torch

from compare_spectral_choppy_asset_bundles import write_comparison
from export_spectral_choppy_asset_bundle import export_choppy_asset_bundle
from report_spectral_choppy_asset_bundle import write_report


SWEEP_PARAMETERS = ("choppiness", "foam_threshold")


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def format_value_for_name(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def run_asset_bundle_sweep(args: argparse.Namespace, device: torch.device) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    runs = []
    for index, value in enumerate(args.values):
        choppiness = value if args.parameter == "choppiness" else args.choppiness
        foam_threshold = value if args.parameter == "foam_threshold" else args.foam_threshold
        run_name = f"{index:02d}_{args.parameter}_{format_value_for_name(value)}"
        run_dir = args.output_dir / run_name
        print(f"Running {run_name}")
        manifest = export_choppy_asset_bundle(
            output_dir=run_dir,
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
            choppiness=choppiness,
            max_surface_points=args.max_surface_points,
            foam_threshold=foam_threshold,
            max_foam_points=args.max_foam_points,
            foam_z_offset=args.foam_z_offset,
            device=device,
        )
        report_path = write_report(run_dir)
        manifest_path = run_dir / "bundle_manifest.json"
        manifests.append(manifest_path)
        runs.append(
            {
                "run_index": index,
                "run_name": run_name,
                "sweep_parameter": args.parameter,
                "sweep_value": value,
                "bundle_dir": str(run_dir),
                "manifest": str(manifest_path),
                "report": str(report_path),
                "frame_count": manifest["frame_count"],
            }
        )

    comparison_path = args.output_dir / "bundle_comparison.md"
    write_comparison(manifests, comparison_path)
    sweep_manifest = {
        "sweep": "spectral_choppy_asset_bundle_sweep",
        "sweep_parameter": args.parameter,
        "values": args.values,
        "device": str(device),
        "runs": runs,
        "outputs": {
            "comparison": str(comparison_path),
        },
    }
    sweep_manifest_path = args.output_dir / "sweep_manifest.json"
    sweep_manifest_path.write_text(json.dumps(sweep_manifest, indent=2), encoding="utf-8")
    print(f"Saved sweep manifest: {sweep_manifest_path}")
    return sweep_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameter sweep for spectral choppy wave asset bundles.")
    parser.add_argument("--parameter", choices=SWEEP_PARAMETERS, default="choppiness", help="Parameter to sweep.")
    parser.add_argument("--values", type=parse_float_list, default=parse_float_list("0.45,0.75"), help="Comma-separated sweep values.")
    parser.add_argument("--size", type=int, default=128, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=120, help="Simulation steps.")
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
    parser.add_argument("--choppiness", type=float, default=0.75, help="Base horizontal displacement multiplier.")
    parser.add_argument("--max-surface-points", type=int, default=64, help="Max exported/rendered surface points per axis.")
    parser.add_argument("--foam-threshold", type=float, default=0.018, help="Base minimum eta steepness for exported foam points.")
    parser.add_argument("--max-foam-points", type=int, default=300, help="Maximum exported foam points per frame.")
    parser.add_argument("--foam-z-offset", type=float, default=0.01, help="Vertical offset applied to exported foam points.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spectral_choppy_asset_bundle_sweep"), help="Sweep output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    manifest = run_asset_bundle_sweep(args, device)
    print(f"Runs: {len(manifest['runs'])}")
    print(f"Comparison: {manifest['outputs']['comparison']}")


if __name__ == "__main__":
    main()
