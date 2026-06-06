import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectral_choppy_wave_viewer import simulate_choppy_frames


def write_obj_mesh(path: Path, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> dict:
    if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
        raise ValueError("x_grid, y_grid, and z_grid must have the same shape.")
    if x_grid.ndim != 2:
        raise ValueError("OBJ export expects 2D grid arrays.")

    rows, cols = x_grid.shape
    vertex_count = rows * cols
    face_count = max(0, rows - 1) * max(0, cols - 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write("# Spectral choppy wave final-frame mesh\n")
        obj_file.write(f"# vertices {vertex_count}\n")
        obj_file.write(f"# quad_faces {face_count}\n")
        for row in range(rows):
            for col in range(cols):
                obj_file.write(f"v {x_grid[row, col]:.9g} {y_grid[row, col]:.9g} {z_grid[row, col]:.9g}\n")

        for row in range(rows - 1):
            for col in range(cols - 1):
                v00 = row * cols + col + 1
                v01 = v00 + 1
                v10 = (row + 1) * cols + col + 1
                v11 = v10 + 1
                obj_file.write(f"f {v00} {v01} {v11} {v10}\n")

    return {
        "path": str(path),
        "rows": rows,
        "cols": cols,
        "vertex_count": vertex_count,
        "quad_face_count": face_count,
    }


def write_metadata(path: Path, mesh_summary: dict, simulation_parameters: dict, device: torch.device) -> None:
    metadata = {
        "solver": "spectral_choppy_mesh",
        "mesh": mesh_summary,
        "simulation": simulation_parameters,
        "device": str(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def export_final_choppy_mesh(
    output: Path,
    metadata_output: Path,
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
    max_mesh_points: int,
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
        max_surface_points=max_mesh_points,
        device=device,
    )
    if not frames:
        raise RuntimeError("No choppy wave frames were produced.")

    x_grid, y_grid, z_grid = frames[-1]
    mesh_summary = write_obj_mesh(output, x_grid, y_grid, z_grid)
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
        "max_mesh_points": max_mesh_points,
        "exported_frame_index": len(frames) - 1,
    }
    write_metadata(metadata_output, mesh_summary, simulation_parameters, device)
    return {"mesh": mesh_summary, "metadata_path": str(metadata_output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the final GPU FFT choppy wave frame as an OBJ mesh.")
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
    parser.add_argument("--max-mesh-points", type=int, default=128, help="Max exported mesh points per axis.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_choppy_wave_final.obj"), help="Output OBJ path.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Output metadata JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_output = args.metadata_output if args.metadata_output is not None else args.output.with_suffix(".json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    result = export_final_choppy_mesh(
        output=args.output,
        metadata_output=metadata_output,
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
        max_mesh_points=args.max_mesh_points,
        device=device,
    )
    print(f"Saved OBJ mesh: {result['mesh']['path']}")
    print(f"Vertices: {result['mesh']['vertex_count']}")
    print(f"Quad faces: {result['mesh']['quad_face_count']}")
    print(f"Saved metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
