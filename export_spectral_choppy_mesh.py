import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectral_choppy_wave_viewer import simulate_choppy_frames


def compute_vertex_normals(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    positions = np.stack((x_grid, y_grid, z_grid), axis=-1)
    tangent_y = np.gradient(positions, axis=0)
    tangent_x = np.gradient(positions, axis=1)
    normals = np.cross(tangent_x, tangent_y)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(lengths, 1.0e-12, None)
    downward = normals[..., 2] < 0.0
    normals[downward] *= -1.0
    return normals


def write_obj_mesh(path: Path, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> dict:
    if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
        raise ValueError("x_grid, y_grid, and z_grid must have the same shape.")
    if x_grid.ndim != 2:
        raise ValueError("OBJ export expects 2D grid arrays.")

    rows, cols = x_grid.shape
    vertex_count = rows * cols
    normal_count = vertex_count
    face_count = max(0, rows - 1) * max(0, cols - 1)
    normals = compute_vertex_normals(x_grid, y_grid, z_grid)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as obj_file:
        obj_file.write("# Spectral choppy wave final-frame mesh\n")
        obj_file.write(f"# vertices {vertex_count}\n")
        obj_file.write(f"# vertex_normals {normal_count}\n")
        obj_file.write(f"# quad_faces {face_count}\n")
        for row in range(rows):
            for col in range(cols):
                obj_file.write(f"v {x_grid[row, col]:.9g} {y_grid[row, col]:.9g} {z_grid[row, col]:.9g}\n")

        for row in range(rows):
            for col in range(cols):
                nx, ny, nz = normals[row, col]
                obj_file.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")

        for row in range(rows - 1):
            for col in range(cols - 1):
                v00 = row * cols + col + 1
                v01 = v00 + 1
                v10 = (row + 1) * cols + col + 1
                v11 = v10 + 1
                obj_file.write(f"f {v00}//{v00} {v01}//{v01} {v11}//{v11} {v10}//{v10}\n")

    return {
        "path": str(path),
        "rows": rows,
        "cols": cols,
        "vertex_count": vertex_count,
        "normal_count": normal_count,
        "quad_face_count": face_count,
    }


def write_metadata(
    path: Path,
    mesh_summary: dict | None,
    simulation_parameters: dict,
    device: torch.device,
    sequence_summary: dict | None = None,
) -> None:
    metadata = {
        "solver": "spectral_choppy_mesh",
        "simulation": simulation_parameters,
        "device": str(device),
    }
    if mesh_summary is not None:
        metadata["mesh"] = mesh_summary
    if sequence_summary is not None:
        metadata["sequence"] = sequence_summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_obj_sequence(output_dir: Path, frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_summaries = []
    for index, (x_grid, y_grid, z_grid) in enumerate(frames):
        frame_path = output_dir / f"frame_{index:04d}.obj"
        frame_summaries.append(write_obj_mesh(frame_path, x_grid, y_grid, z_grid))

    return {
        "directory": str(output_dir),
        "frame_count": len(frame_summaries),
        "frames": frame_summaries,
    }


def export_final_choppy_mesh(
    output: Path,
    metadata_output: Path,
    sequence_output_dir: Path | None,
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
    sequence_summary = write_obj_sequence(sequence_output_dir, frames) if sequence_output_dir is not None else None
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
    write_metadata(metadata_output, mesh_summary, simulation_parameters, device, sequence_summary)
    return {"mesh": mesh_summary, "metadata_path": str(metadata_output), "sequence": sequence_summary}


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
    parser.add_argument("--sequence-output-dir", type=Path, default=None, help="Optional directory for OBJ files for every saved frame.")
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
        sequence_output_dir=args.sequence_output_dir,
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
    if result["sequence"] is not None:
        print(f"Saved OBJ sequence: {result['sequence']['directory']}")
        print(f"Sequence frames: {result['sequence']['frame_count']}")


if __name__ == "__main__":
    main()
