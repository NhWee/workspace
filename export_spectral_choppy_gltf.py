import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from export_spectral_choppy_mesh import compute_vertex_normals
from spectral_choppy_wave_viewer import sample_foam_points, simulate_choppy_frames


GL_ARRAY_BUFFER = 34962
GL_ELEMENT_ARRAY_BUFFER = 34963
GL_FLOAT = 5126
GL_UNSIGNED_INT = 5125
GL_POINTS = 0
GL_TRIANGLES = 4


def make_triangle_indices(rows: int, cols: int) -> np.ndarray:
    triangles = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            v00 = row * cols + col
            v01 = v00 + 1
            v10 = (row + 1) * cols + col
            v11 = v10 + 1
            triangles.append((v00, v01, v11))
            triangles.append((v00, v11, v10))
    return np.asarray(triangles, dtype=np.uint32).reshape(-1)


def append_buffer(
    chunks: list[bytes],
    buffer_views: list[dict],
    accessors: list[dict],
    array: np.ndarray,
    component_type: int,
    accessor_type: str,
    target: int | None,
) -> int:
    while sum(len(chunk) for chunk in chunks) % 4:
        chunks.append(b"\x00")

    byte_offset = sum(len(chunk) for chunk in chunks)
    contiguous = np.ascontiguousarray(array)
    data = contiguous.tobytes()
    chunks.append(data)

    buffer_view = {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(data)}
    if target is not None:
        buffer_view["target"] = target
    buffer_views.append(buffer_view)
    buffer_view_index = len(buffer_views) - 1

    accessor = {
        "bufferView": buffer_view_index,
        "componentType": component_type,
        "count": int(contiguous.shape[0]),
        "type": accessor_type,
    }
    if accessor_type == "VEC3":
        accessor["min"] = contiguous.min(axis=0).astype(float).tolist()
        accessor["max"] = contiguous.max(axis=0).astype(float).tolist()
    accessors.append(accessor)
    return len(accessors) - 1


def write_gltf_scene(
    path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
) -> dict:
    if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
        raise ValueError("x_grid, y_grid, and z_grid must have the same shape.")
    if x_grid.ndim != 2:
        raise ValueError("glTF export expects 2D grid arrays.")

    rows, cols = x_grid.shape
    positions = np.stack((x_grid, y_grid, z_grid), axis=-1).reshape(-1, 3).astype(np.float32)
    normals = compute_vertex_normals(x_grid, y_grid, z_grid).reshape(-1, 3).astype(np.float32)
    indices = make_triangle_indices(rows, cols)

    foam_x, foam_y, foam_z, foam_steepness = sample_foam_points(
        x_grid,
        y_grid,
        z_grid,
        foam_threshold,
        max_foam_points,
    )
    foam_positions = np.stack((foam_x, foam_y, foam_z + foam_z_offset), axis=-1).astype(np.float32)
    foam_steepness = foam_steepness.reshape(-1, 1).astype(np.float32)

    chunks: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []

    position_accessor = append_buffer(chunks, buffer_views, accessors, positions, GL_FLOAT, "VEC3", GL_ARRAY_BUFFER)
    normal_accessor = append_buffer(chunks, buffer_views, accessors, normals, GL_FLOAT, "VEC3", GL_ARRAY_BUFFER)
    index_accessor = append_buffer(chunks, buffer_views, accessors, indices, GL_UNSIGNED_INT, "SCALAR", GL_ELEMENT_ARRAY_BUFFER)

    primitives = [
        {
            "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
            "indices": index_accessor,
            "material": 0,
            "mode": GL_TRIANGLES,
        }
    ]

    foam_point_count = int(len(foam_positions))
    if foam_point_count > 0:
        foam_position_accessor = append_buffer(
            chunks,
            buffer_views,
            accessors,
            foam_positions,
            GL_FLOAT,
            "VEC3",
            GL_ARRAY_BUFFER,
        )
        foam_steepness_accessor = append_buffer(
            chunks,
            buffer_views,
            accessors,
            foam_steepness,
            GL_FLOAT,
            "SCALAR",
            GL_ARRAY_BUFFER,
        )
        primitives.append(
            {
                "attributes": {"POSITION": foam_position_accessor, "_STEEPNESS": foam_steepness_accessor},
                "material": 1,
                "mode": GL_POINTS,
            }
        )

    buffer_data = b"".join(chunks)
    data_uri = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode("ascii")
    gltf = {
        "asset": {"version": "2.0", "generator": "export_spectral_choppy_gltf.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "spectral_choppy_wave"}],
        "meshes": [{"name": "spectral_choppy_wave_final_frame", "primitives": primitives}],
        "materials": [
            {
                "name": "water_surface",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.08, 0.32, 0.72, 0.82],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.35,
                },
                "alphaMode": "BLEND",
            },
            {
                "name": "foam_points",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.75,
                },
            },
        ],
        "buffers": [{"uri": data_uri, "byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gltf, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "rows": rows,
        "cols": cols,
        "vertex_count": int(len(positions)),
        "normal_count": int(len(normals)),
        "triangle_count": int(len(indices) // 3),
        "foam_point_count": foam_point_count,
        "byte_length": len(buffer_data),
    }


def export_final_choppy_gltf(
    output: Path,
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
    if not frames:
        raise RuntimeError("No choppy wave frames were produced.")

    x_grid, y_grid, z_grid = frames[-1]
    return write_gltf_scene(
        output,
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the final GPU FFT choppy wave frame as embedded glTF.")
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
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max exported surface points per axis.")
    parser.add_argument("--foam-threshold", type=float, default=0.018, help="Minimum eta steepness for exported foam points.")
    parser.add_argument("--max-foam-points", type=int, default=900, help="Maximum exported foam points.")
    parser.add_argument("--foam-z-offset", type=float, default=0.01, help="Vertical offset applied to exported foam points.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_choppy_wave_final.gltf"), help="Output glTF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    summary = export_final_choppy_gltf(
        output=args.output,
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
    print(f"Saved glTF: {summary['path']}")
    print(f"Vertices: {summary['vertex_count']}")
    print(f"Triangles: {summary['triangle_count']}")
    print(f"Foam points: {summary['foam_point_count']}")


if __name__ == "__main__":
    main()
