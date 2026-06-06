import argparse
import json
from pathlib import Path

from report_spectral_choppy_asset_bundle import format_bytes, make_asset_row


ASSET_KEYS = [
    ("viewer", "path"),
    ("obj", "path"),
    ("obj_sequence", "directory"),
    ("foam", "path"),
    ("foam_sequence", "directory"),
    ("gltf", "path"),
    ("glb", "path"),
    ("gltf_sequence", "directory"),
    ("glb_sequence", "directory"),
    ("animated_gltf", "path"),
    ("animated_glb", "path"),
]


def resolve_manifest(path: Path) -> Path:
    return path / "bundle_manifest.json" if path.is_dir() else path


def summarize_bundle(path: Path) -> dict:
    manifest_path = resolve_manifest(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_dir = manifest_path.parent
    rows = [make_asset_row(bundle_dir, key, manifest.get(key, {}), path_key) for key, path_key in ASSET_KEYS]
    missing = [row["asset"] for row in rows if not row["exists"]]
    total_size = sum(row["size_bytes"] for row in rows)
    simulation = manifest.get("simulation", {})
    glb = manifest.get("glb", {})
    glb_sequence = manifest.get("glb_sequence", {})
    animated_glb = manifest.get("animated_glb", {})
    metric_summary = manifest.get("metrics", {}).get("summary", {})
    return {
        "name": bundle_dir.name,
        "manifest": str(manifest_path),
        "device": manifest.get("device", ""),
        "frame_count": manifest.get("frame_count", ""),
        "size": simulation.get("size", ""),
        "steps": simulation.get("steps", ""),
        "frame_every": simulation.get("frame_every", ""),
        "max_surface_points": simulation.get("max_surface_points", ""),
        "choppiness": simulation.get("choppiness", ""),
        "foam_threshold": simulation.get("foam_threshold", ""),
        "missing_assets": len(missing),
        "total_size_bytes": total_size,
        "glb_size_bytes": glb.get("glb_length", 0),
        "glb_vertices": glb.get("vertex_count", ""),
        "glb_triangles": glb.get("triangle_count", ""),
        "glb_foam_points": glb.get("foam_point_count", ""),
        "glb_sequence_frames": glb_sequence.get("frame_count", ""),
        "animated_glb_size_bytes": animated_glb.get("glb_length", 0),
        "animated_frames": animated_glb.get("frame_count", ""),
        "animated_morph_targets": animated_glb.get("morph_target_count", ""),
        "eta_range_max": metric_summary.get("eta_range_max", ""),
        "steepness_p95_max": metric_summary.get("steepness_p95_max", ""),
        "foam_ratio_mean": metric_summary.get("foam_ratio_mean", ""),
        "horizontal_displacement_p95_max": metric_summary.get("horizontal_displacement_p95_max", ""),
        "folded_triangle_ratio_max": metric_summary.get("folded_triangle_ratio_max", ""),
    }


def build_comparison_table(summaries: list[dict]) -> str:
    lines = [
        "# Spectral Choppy Wave Asset Bundle Comparison",
        "",
        "| Bundle | Frames | Grid | Steps | Frame Every | Surface Points | Choppiness | Foam Threshold | Eta Range Max | Steepness P95 Max | Foam Ratio Mean | Disp P95 Max | Fold Ratio Max | Total Size | GLB Size | Animated GLB Size | Animated Frames | Morph Targets | Vertices | Triangles | Foam Points | GLB Seq Frames | Missing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {name} | {frame_count} | {size} | {steps} | {frame_every} | {max_surface_points} | {choppiness} | {foam_threshold} | {eta_range_max} | {steepness_p95_max} | {foam_ratio_mean} | {horizontal_displacement_p95_max} | {folded_triangle_ratio_max} | {total_size} | {glb_size} | {animated_glb_size} | {animated_frames} | {animated_morph_targets} | {vertices} | {triangles} | {foam_points} | {glb_sequence_frames} | {missing_assets} |".format(
                name=summary["name"],
                frame_count=summary["frame_count"],
                size=summary["size"],
                steps=summary["steps"],
                frame_every=summary["frame_every"],
                max_surface_points=summary["max_surface_points"],
                choppiness=summary["choppiness"],
                foam_threshold=summary["foam_threshold"],
                eta_range_max=summary["eta_range_max"],
                steepness_p95_max=summary["steepness_p95_max"],
                foam_ratio_mean=summary["foam_ratio_mean"],
                horizontal_displacement_p95_max=summary["horizontal_displacement_p95_max"],
                folded_triangle_ratio_max=summary["folded_triangle_ratio_max"],
                total_size=format_bytes(summary["total_size_bytes"]),
                glb_size=format_bytes(int(summary["glb_size_bytes"] or 0)),
                animated_glb_size=format_bytes(int(summary["animated_glb_size_bytes"] or 0)),
                animated_frames=summary["animated_frames"],
                animated_morph_targets=summary["animated_morph_targets"],
                vertices=summary["glb_vertices"],
                triangles=summary["glb_triangles"],
                foam_points=summary["glb_foam_points"],
                glb_sequence_frames=summary["glb_sequence_frames"],
                missing_assets=summary["missing_assets"],
            )
        )
    return "\n".join(lines) + "\n"


def write_comparison(paths: list[Path], output: Path) -> Path:
    summaries = [summarize_bundle(path) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_comparison_table(summaries), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare spectral choppy wave asset bundle manifests.")
    parser.add_argument("bundles", nargs="+", type=Path, help="Bundle directories or bundle_manifest.json paths.")
    parser.add_argument("--output", type=Path, default=Path("outputs/spectral_choppy_asset_bundle_comparison.md"), help="Output Markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_comparison(args.bundles, args.output)
    print(f"Saved bundle comparison: {output}")


if __name__ == "__main__":
    main()
