import argparse
import json
from pathlib import Path

import torch

from export_spectral_choppy_gltf import write_glb_scene, write_glb_sequence, write_gltf_scene, write_gltf_sequence
from export_spectral_choppy_mesh import write_foam_ply, write_foam_sequence, write_obj_mesh, write_obj_sequence
from spectral_choppy_wave_viewer import build_choppy_figure, simulate_choppy_frames


def write_asset_bundle(
    output_dir: Path,
    frames: list[tuple],
    domain_size: float,
    simulation_parameters: dict,
    device: torch.device,
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
) -> dict:
    if not frames:
        raise RuntimeError("No choppy wave frames were provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    x_grid, y_grid, z_grid = frames[-1]

    viewer_path = output_dir / "viewer.html"
    figure = build_choppy_figure(
        frames,
        domain_size=domain_size,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
    )
    figure.write_html(viewer_path, include_plotlyjs=True, full_html=True)

    obj_summary = write_obj_mesh(output_dir / "final.obj", x_grid, y_grid, z_grid)
    obj_sequence_summary = write_obj_sequence(output_dir / "obj_sequence", frames)
    foam_summary = write_foam_ply(
        output_dir / "foam.ply",
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        z_offset=foam_z_offset,
    )
    foam_sequence_summary = write_foam_sequence(
        output_dir / "foam_sequence",
        frames,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        z_offset=foam_z_offset,
    )
    gltf_summary = write_gltf_scene(
        output_dir / "final.gltf",
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
    )
    glb_summary = write_glb_scene(
        output_dir / "final.glb",
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
    )
    gltf_sequence_summary = write_gltf_sequence(
        output_dir / "gltf_sequence",
        frames,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
    )
    glb_sequence_summary = write_glb_sequence(
        output_dir / "glb_sequence",
        frames,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
    )

    manifest = {
        "bundle": "spectral_choppy_wave_asset_bundle",
        "device": str(device),
        "frame_count": len(frames),
        "simulation": simulation_parameters,
        "viewer": {"path": str(viewer_path)},
        "obj": obj_summary,
        "obj_sequence": obj_sequence_summary,
        "foam": foam_summary,
        "foam_sequence": foam_sequence_summary,
        "gltf": gltf_summary,
        "glb": glb_summary,
        "gltf_sequence": gltf_sequence_summary,
        "glb_sequence": glb_sequence_summary,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def export_choppy_asset_bundle(
    output_dir: Path,
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
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
    device: torch.device,
) -> dict:
    frames = simulate_choppy_frames(
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
        choppiness=choppiness,
        max_surface_points=max_surface_points,
        device=device,
    )
    simulation_parameters = {
        "size": size,
        "steps": steps,
        "frame_every": frame_every,
        "domain_size": domain_size,
        "gravity": gravity,
        "dt": dt,
        "wave_amplitude": wave_amplitude,
        "peak_wavelength": peak_wavelength,
        "bandwidth": bandwidth,
        "wind_direction_degrees": wind_direction_degrees,
        "directional_spread": directional_spread,
        "damping": damping,
        "seed": seed,
        "choppiness": choppiness,
        "max_surface_points": max_surface_points,
        "foam_threshold": foam_threshold,
        "max_foam_points": max_foam_points,
        "foam_z_offset": foam_z_offset,
    }
    return write_asset_bundle(
        output_dir,
        frames,
        domain_size,
        simulation_parameters,
        device,
        foam_threshold,
        max_foam_points,
        foam_z_offset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a complete GPU FFT choppy wave 3D asset bundle.")
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
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max exported/rendered surface points per axis.")
    parser.add_argument("--foam-threshold", type=float, default=0.018, help="Minimum eta steepness for exported foam points.")
    parser.add_argument("--max-foam-points", type=int, default=900, help="Maximum exported foam points per frame.")
    parser.add_argument("--foam-z-offset", type=float, default=0.01, help="Vertical offset applied to exported foam points.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spectral_choppy_asset_bundle"), help="Output bundle directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    manifest = export_choppy_asset_bundle(
        output_dir=args.output_dir,
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
        foam_threshold=args.foam_threshold,
        max_foam_points=args.max_foam_points,
        foam_z_offset=args.foam_z_offset,
        device=device,
    )
    print(f"Saved bundle: {args.output_dir}")
    print(f"Frames: {manifest['frame_count']}")
    print(f"Manifest: {manifest['manifest_path']}")
    print(f"Final GLB: {manifest['glb']['path']}")


if __name__ == "__main__":
    main()
