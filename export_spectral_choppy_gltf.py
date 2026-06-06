import argparse
import base64
import json
import struct
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
GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942


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


def build_gltf_document(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
    embed_buffer: bool,
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
    buffer = {"byteLength": len(buffer_data)}
    if embed_buffer:
        buffer["uri"] = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode("ascii")
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
        "buffers": [buffer],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    summary = {
        "rows": rows,
        "cols": cols,
        "vertex_count": int(len(positions)),
        "normal_count": int(len(normals)),
        "triangle_count": int(len(indices) // 3),
        "foam_point_count": foam_point_count,
        "byte_length": len(buffer_data),
    }
    return {"gltf": gltf, "buffer_data": buffer_data, "summary": summary}


def write_gltf_scene(
    path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
) -> dict:
    document = build_gltf_document(
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
        embed_buffer=True,
    )
    summary = document["summary"]
    summary["path"] = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document["gltf"], indent=2), encoding="utf-8")
    return summary


def pad_bytes(data: bytes, multiple: int, pad_byte: bytes) -> bytes:
    remainder = len(data) % multiple
    if remainder == 0:
        return data
    return data + pad_byte * (multiple - remainder)


def write_glb_scene(
    path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
) -> dict:
    document = build_gltf_document(
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
        embed_buffer=False,
    )
    gltf_json = json.dumps(document["gltf"], separators=(",", ":")).encode("utf-8")
    json_chunk = pad_bytes(gltf_json, 4, b" ")
    bin_chunk = pad_bytes(document["buffer_data"], 4, b"\x00")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = b"".join(
        [
            struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_length),
            struct.pack("<II", len(json_chunk), GLB_JSON_CHUNK),
            json_chunk,
            struct.pack("<II", len(bin_chunk), GLB_BIN_CHUNK),
            bin_chunk,
        ]
    )

    summary = document["summary"]
    summary["path"] = str(path)
    summary["glb_length"] = len(glb)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)
    return summary


def build_glb_bytes(gltf: dict, buffer_data: bytes) -> bytes:
    gltf_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = pad_bytes(gltf_json, 4, b" ")
    bin_chunk = pad_bytes(buffer_data, 4, b"\x00")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return b"".join(
        [
            struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_length),
            struct.pack("<II", len(json_chunk), GLB_JSON_CHUNK),
            json_chunk,
            struct.pack("<II", len(bin_chunk), GLB_BIN_CHUNK),
            bin_chunk,
        ]
    )


def build_animated_gltf_document(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_duration: float,
    embed_buffer: bool,
) -> dict:
    if len(frames) < 2:
        raise RuntimeError("Animated glTF export requires at least two frames.")

    x0, y0, z0 = frames[0]
    if x0.shape != y0.shape or x0.shape != z0.shape:
        raise ValueError("Frame grids must have matching shapes.")
    if x0.ndim != 2:
        raise ValueError("Animated glTF export expects 2D grid arrays.")

    rows, cols = x0.shape
    base_positions = np.stack((x0, y0, z0), axis=-1).reshape(-1, 3).astype(np.float32)
    base_normals = compute_vertex_normals(x0, y0, z0).reshape(-1, 3).astype(np.float32)
    indices = make_triangle_indices(rows, cols)

    chunks: list[bytes] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    position_accessor = append_buffer(chunks, buffer_views, accessors, base_positions, GL_FLOAT, "VEC3", GL_ARRAY_BUFFER)
    normal_accessor = append_buffer(chunks, buffer_views, accessors, base_normals, GL_FLOAT, "VEC3", GL_ARRAY_BUFFER)
    index_accessor = append_buffer(chunks, buffer_views, accessors, indices, GL_UNSIGNED_INT, "SCALAR", GL_ELEMENT_ARRAY_BUFFER)

    targets = []
    for frame_index, (x_grid, y_grid, z_grid) in enumerate(frames[1:], start=1):
        if x_grid.shape != x0.shape or y_grid.shape != x0.shape or z_grid.shape != x0.shape:
            raise ValueError(f"Frame {frame_index} shape does not match the first frame.")
        positions = np.stack((x_grid, y_grid, z_grid), axis=-1).reshape(-1, 3).astype(np.float32)
        delta_positions = positions - base_positions
        delta_accessor = append_buffer(chunks, buffer_views, accessors, delta_positions, GL_FLOAT, "VEC3", GL_ARRAY_BUFFER)
        targets.append({"POSITION": delta_accessor})

    times = (np.arange(len(frames), dtype=np.float32) * np.float32(frame_duration)).reshape(-1)
    target_count = len(targets)
    weights = np.zeros((len(frames), target_count), dtype=np.float32)
    for frame_index in range(1, len(frames)):
        weights[frame_index, frame_index - 1] = 1.0
    time_accessor = append_buffer(chunks, buffer_views, accessors, times, GL_FLOAT, "SCALAR", None)
    weight_accessor = append_buffer(chunks, buffer_views, accessors, weights.reshape(-1), GL_FLOAT, "SCALAR", None)

    buffer_data = b"".join(chunks)
    buffer = {"byteLength": len(buffer_data)}
    if embed_buffer:
        buffer["uri"] = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode("ascii")

    gltf = {
        "asset": {"version": "2.0", "generator": "export_spectral_choppy_gltf.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "animated_spectral_choppy_wave", "weights": [0.0] * target_count}],
        "meshes": [
            {
                "name": "animated_spectral_choppy_wave_surface",
                "weights": [0.0] * target_count,
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
                        "indices": index_accessor,
                        "targets": targets,
                        "material": 0,
                        "mode": GL_TRIANGLES,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "animated_water_surface",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.08, 0.32, 0.72, 0.88],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.35,
                },
                "alphaMode": "BLEND",
            }
        ],
        "animations": [
            {
                "name": "spectral_choppy_wave_surface_weights",
                "samplers": [
                    {
                        "input": time_accessor,
                        "output": weight_accessor,
                        "interpolation": "LINEAR",
                    }
                ],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
            }
        ],
        "buffers": [buffer],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    summary = {
        "rows": rows,
        "cols": cols,
        "frame_count": len(frames),
        "duration_seconds": float(times[-1]) if len(times) else 0.0,
        "frame_duration_seconds": frame_duration,
        "vertex_count": int(len(base_positions)),
        "triangle_count": int(len(indices) // 3),
        "morph_target_count": target_count,
        "byte_length": len(buffer_data),
    }
    return {"gltf": gltf, "buffer_data": buffer_data, "summary": summary}


def write_animated_gltf_scene(
    path: Path,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_duration: float,
) -> dict:
    document = build_animated_gltf_document(frames, frame_duration=frame_duration, embed_buffer=True)
    summary = document["summary"]
    summary["path"] = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document["gltf"], indent=2), encoding="utf-8")
    return summary


def write_animated_glb_scene(
    path: Path,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_duration: float,
) -> dict:
    document = build_animated_gltf_document(frames, frame_duration=frame_duration, embed_buffer=False)
    glb = build_glb_bytes(document["gltf"], document["buffer_data"])
    summary = document["summary"]
    summary["path"] = str(path)
    summary["glb_length"] = len(glb)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb)
    return summary


def write_gltf_sequence(
    output_dir: Path,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_summaries = []
    for index, (x_grid, y_grid, z_grid) in enumerate(frames):
        frame_path = output_dir / f"frame_{index:04d}.gltf"
        frame_summaries.append(
            write_gltf_scene(
                frame_path,
                x_grid,
                y_grid,
                z_grid,
                foam_threshold=foam_threshold,
                max_foam_points=max_foam_points,
                foam_z_offset=foam_z_offset,
            )
        )

    manifest = {
        "exporter": "export_spectral_choppy_gltf.py",
        "format": "embedded_gltf_sequence",
        "frame_count": len(frame_summaries),
        "frames": frame_summaries,
    }
    manifest_path = output_dir / "sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "directory": str(output_dir),
        "manifest_path": str(manifest_path),
        "frame_count": len(frame_summaries),
        "frames": frame_summaries,
    }


def write_glb_sequence(
    output_dir: Path,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    foam_threshold: float,
    max_foam_points: int,
    foam_z_offset: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_summaries = []
    for index, (x_grid, y_grid, z_grid) in enumerate(frames):
        frame_path = output_dir / f"frame_{index:04d}.glb"
        frame_summaries.append(
            write_glb_scene(
                frame_path,
                x_grid,
                y_grid,
                z_grid,
                foam_threshold=foam_threshold,
                max_foam_points=max_foam_points,
                foam_z_offset=foam_z_offset,
            )
        )

    manifest = {
        "exporter": "export_spectral_choppy_gltf.py",
        "format": "glb_sequence",
        "frame_count": len(frame_summaries),
        "frames": frame_summaries,
    }
    manifest_path = output_dir / "sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "directory": str(output_dir),
        "manifest_path": str(manifest_path),
        "frame_count": len(frame_summaries),
        "frames": frame_summaries,
    }


def export_final_choppy_gltf(
    output: Path,
    glb_output: Path | None,
    sequence_output_dir: Path | None,
    glb_sequence_output_dir: Path | None,
    animated_output: Path | None,
    animated_glb_output: Path | None,
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
    animation_frame_duration: float,
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
    final_summary = write_gltf_scene(
        output,
        x_grid,
        y_grid,
        z_grid,
        foam_threshold=foam_threshold,
        max_foam_points=max_foam_points,
        foam_z_offset=foam_z_offset,
    )
    glb_summary = (
        write_glb_scene(
            glb_output,
            x_grid,
            y_grid,
            z_grid,
            foam_threshold=foam_threshold,
            max_foam_points=max_foam_points,
            foam_z_offset=foam_z_offset,
        )
        if glb_output is not None
        else None
    )
    sequence_summary = (
        write_gltf_sequence(
            sequence_output_dir,
            frames,
            foam_threshold=foam_threshold,
            max_foam_points=max_foam_points,
            foam_z_offset=foam_z_offset,
        )
        if sequence_output_dir is not None
        else None
    )
    glb_sequence_summary = (
        write_glb_sequence(
            glb_sequence_output_dir,
            frames,
            foam_threshold=foam_threshold,
            max_foam_points=max_foam_points,
            foam_z_offset=foam_z_offset,
        )
        if glb_sequence_output_dir is not None
        else None
    )
    animated_summary = (
        write_animated_gltf_scene(
            animated_output,
            frames,
            frame_duration=animation_frame_duration,
        )
        if animated_output is not None
        else None
    )
    animated_glb_summary = (
        write_animated_glb_scene(
            animated_glb_output,
            frames,
            frame_duration=animation_frame_duration,
        )
        if animated_glb_output is not None
        else None
    )
    return {
        "final": final_summary,
        "glb": glb_summary,
        "sequence": sequence_summary,
        "glb_sequence": glb_sequence_summary,
        "animated": animated_summary,
        "animated_glb": animated_glb_summary,
    }


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
    parser.add_argument("--animation-frame-duration", type=float, default=None, help="Seconds between saved frames in animated glTF/GLB. Defaults to frame_every * dt.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_choppy_wave_final.gltf"), help="Output glTF path.")
    parser.add_argument("--glb-output", type=Path, default=None, help="Optional output binary GLB path for the final frame.")
    parser.add_argument("--sequence-output-dir", type=Path, default=None, help="Optional directory for glTF files for every saved frame.")
    parser.add_argument("--glb-sequence-output-dir", type=Path, default=None, help="Optional directory for GLB files for every saved frame.")
    parser.add_argument("--animated-output", type=Path, default=None, help="Optional output animated glTF path using morph targets.")
    parser.add_argument("--animated-glb-output", type=Path, default=None, help="Optional output animated binary GLB path using morph targets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    summary = export_final_choppy_gltf(
        output=args.output,
        glb_output=args.glb_output,
        sequence_output_dir=args.sequence_output_dir,
        glb_sequence_output_dir=args.glb_sequence_output_dir,
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
        animation_frame_duration=args.animation_frame_duration if args.animation_frame_duration is not None else args.frame_every * args.dt,
        animated_output=args.animated_output,
        animated_glb_output=args.animated_glb_output,
        device=device,
    )
    print(f"Saved glTF: {summary['final']['path']}")
    print(f"Vertices: {summary['final']['vertex_count']}")
    print(f"Triangles: {summary['final']['triangle_count']}")
    print(f"Foam points: {summary['final']['foam_point_count']}")
    if summary["glb"] is not None:
        print(f"Saved GLB: {summary['glb']['path']}")
        print(f"GLB bytes: {summary['glb']['glb_length']}")
    if summary["sequence"] is not None:
        print(f"Saved glTF sequence: {summary['sequence']['directory']}")
        print(f"Sequence frames: {summary['sequence']['frame_count']}")
        print(f"Saved sequence manifest: {summary['sequence']['manifest_path']}")
    if summary["glb_sequence"] is not None:
        print(f"Saved GLB sequence: {summary['glb_sequence']['directory']}")
        print(f"GLB sequence frames: {summary['glb_sequence']['frame_count']}")
        print(f"Saved GLB sequence manifest: {summary['glb_sequence']['manifest_path']}")
    if summary["animated"] is not None:
        print(f"Saved animated glTF: {summary['animated']['path']}")
        print(f"Animated frames: {summary['animated']['frame_count']}")
        print(f"Morph targets: {summary['animated']['morph_target_count']}")
    if summary["animated_glb"] is not None:
        print(f"Saved animated GLB: {summary['animated_glb']['path']}")
        print(f"Animated GLB bytes: {summary['animated_glb']['glb_length']}")


if __name__ == "__main__":
    main()
