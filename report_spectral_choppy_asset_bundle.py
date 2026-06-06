import argparse
import json
from pathlib import Path


def resolve_manifest_path(bundle_dir: Path, manifest_path: Path | None) -> Path:
    return manifest_path if manifest_path is not None else bundle_dir / "bundle_manifest.json"


def format_bytes(byte_count: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(byte_count)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{byte_count} B"
        value /= 1024.0
    return f"{byte_count} B"


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def make_asset_row(bundle_dir: Path, label: str, manifest_entry: dict, path_key: str = "path") -> dict:
    raw_path = manifest_entry.get(path_key, "")
    path = Path(raw_path)
    if raw_path and not path.exists() and not path.is_absolute():
        path = bundle_dir / path
    exists = path.exists()
    return {
        "asset": label,
        "path": str(path),
        "exists": exists,
        "size_bytes": path_size(path) if exists else 0,
        "frame_count": manifest_entry.get("frame_count", ""),
        "vertices": manifest_entry.get("vertex_count", ""),
        "triangles": manifest_entry.get("triangle_count", ""),
        "foam_points": manifest_entry.get("foam_point_count", manifest_entry.get("point_count", "")),
    }


def build_report(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_dir = manifest_path.parent
    rows = [
        make_asset_row(bundle_dir, "viewer", manifest.get("viewer", {})),
        make_asset_row(bundle_dir, "final OBJ", manifest.get("obj", {})),
        make_asset_row(bundle_dir, "OBJ sequence", manifest.get("obj_sequence", {}), "directory"),
        make_asset_row(bundle_dir, "foam PLY", manifest.get("foam", {})),
        make_asset_row(bundle_dir, "foam sequence", manifest.get("foam_sequence", {}), "directory"),
        make_asset_row(bundle_dir, "final glTF", manifest.get("gltf", {})),
        make_asset_row(bundle_dir, "final GLB", manifest.get("glb", {})),
        make_asset_row(bundle_dir, "glTF sequence", manifest.get("gltf_sequence", {}), "directory"),
        make_asset_row(bundle_dir, "GLB sequence", manifest.get("glb_sequence", {}), "directory"),
    ]
    missing = [row["asset"] for row in rows if not row["exists"]]
    total_size = sum(row["size_bytes"] for row in rows)
    simulation = manifest.get("simulation", {})
    metric_summary = manifest.get("metrics", {}).get("summary", {})

    lines = [
        "# Spectral Choppy Wave Asset Bundle Report",
        "",
        "## Summary",
        "",
        f"- bundle: `{manifest.get('bundle', '')}`",
        f"- device: `{manifest.get('device', '')}`",
        f"- frame_count: `{manifest.get('frame_count', '')}`",
        f"- total_asset_size: `{format_bytes(total_size)}`",
        f"- missing_assets: `{len(missing)}`",
        "",
        "## Simulation",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
    ]
    for key in sorted(simulation):
        lines.append(f"| `{key}` | `{simulation[key]}` |")

    if metric_summary:
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| eta_range_max | `{metric_summary.get('eta_range_max', '')}` |",
                f"| eta_std_mean | `{metric_summary.get('eta_std_mean', '')}` |",
                f"| steepness_p95_max | `{metric_summary.get('steepness_p95_max', '')}` |",
                f"| foam_ratio_mean | `{metric_summary.get('foam_ratio_mean', '')}` |",
                f"| foam_point_count_max | `{metric_summary.get('foam_point_count_max', '')}` |",
                f"| horizontal_displacement_p95_max | `{metric_summary.get('horizontal_displacement_p95_max', '')}` |",
            ]
        )

    lines.extend(
        [
            "",
            "## Assets",
            "",
            "| Asset | Exists | Size | Frames | Vertices | Triangles | Foam Points | Path |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {asset} | {exists} | {size} | {frames} | {vertices} | {triangles} | {foam_points} | `{path}` |".format(
                asset=row["asset"],
                exists="yes" if row["exists"] else "no",
                size=format_bytes(row["size_bytes"]),
                frames=row["frame_count"],
                vertices=row["vertices"],
                triangles=row["triangles"],
                foam_points=row["foam_points"],
                path=row["path"],
            )
        )

    if missing:
        lines.extend(["", "## Missing Assets", ""])
        for asset in missing:
            lines.append(f"- {asset}")

    return "\n".join(lines) + "\n"


def write_report(bundle_dir: Path, output: Path | None = None, manifest_path: Path | None = None) -> Path:
    resolved_manifest_path = resolve_manifest_path(bundle_dir, manifest_path)
    report = build_report(resolved_manifest_path)
    output_path = output if output is not None else bundle_dir / "bundle_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown report from a spectral choppy wave asset bundle manifest.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("outputs/spectral_choppy_asset_bundle"), help="Bundle directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit bundle_manifest.json path.")
    parser.add_argument("--output", type=Path, default=None, help="Output Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_report(args.bundle_dir, args.output, args.manifest)
    print(f"Saved bundle report: {output_path}")


if __name__ == "__main__":
    main()
