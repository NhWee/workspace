import argparse
import json
import struct
from pathlib import Path


GLB_MAGIC = 0x46546C67
GLB_VERSION = 2
GLB_JSON_CHUNK = 0x4E4F534A


ASSET_PATHS = [
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


def resolve_manifest_path(bundle_dir: Path, manifest_path: Path | None = None) -> Path:
    return manifest_path if manifest_path is not None else bundle_dir / "bundle_manifest.json"


def resolve_asset_path(bundle_dir: Path, manifest_entry: dict, path_key: str) -> Path:
    raw_path = manifest_entry.get(path_key, "")
    path = Path(raw_path)
    if raw_path and not path.exists() and not path.is_absolute():
        path = bundle_dir / path
    return path


def add_check(checks: list[dict], name: str, passed: bool, detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def read_gltf(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_glb_json(path: Path) -> tuple[dict, dict]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("GLB file is too small.")
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("GLB magic header mismatch.")
    if version != GLB_VERSION:
        raise ValueError("GLB version mismatch.")
    if total_length != len(data):
        raise ValueError("GLB total length does not match file size.")

    json_chunk_length, json_chunk_type = struct.unpack_from("<II", data, 12)
    if json_chunk_type != GLB_JSON_CHUNK:
        raise ValueError("First GLB chunk is not JSON.")
    json_start = 20
    json_end = json_start + json_chunk_length
    gltf = json.loads(data[json_start:json_end].decode("utf-8"))
    return gltf, {"version": version, "total_length": total_length, "json_chunk_length": json_chunk_length}


def has_weight_animation(gltf: dict) -> bool:
    for animation in gltf.get("animations", []):
        for channel in animation.get("channels", []):
            if channel.get("target", {}).get("path") == "weights":
                return True
    return False


def first_primitive(gltf: dict) -> dict:
    return gltf.get("meshes", [{}])[0].get("primitives", [{}])[0]


def primitive_count(gltf: dict) -> int:
    return len(gltf.get("meshes", [{}])[0].get("primitives", []))


def validate_bundle(bundle_dir: Path, manifest_path: Path | None = None) -> dict:
    resolved_manifest_path = resolve_manifest_path(bundle_dir, manifest_path)
    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    bundle_root = resolved_manifest_path.parent
    checks: list[dict] = []

    add_check(checks, "manifest exists", resolved_manifest_path.exists(), str(resolved_manifest_path))
    add_check(checks, "bundle marker", manifest.get("bundle") == "spectral_choppy_wave_asset_bundle", str(manifest.get("bundle", "")))
    add_check(checks, "frame count positive", int(manifest.get("frame_count", 0)) > 0, str(manifest.get("frame_count", "")))

    for key, path_key in ASSET_PATHS:
        path = resolve_asset_path(bundle_root, manifest.get(key, {}), path_key)
        add_check(checks, f"{key} exists", path.exists(), str(path))

    metrics = manifest.get("metrics", {})
    for key in ("csv", "json", "report"):
        path = Path(metrics.get(key, ""))
        if str(path) and not path.exists() and not path.is_absolute():
            path = bundle_root / path
        add_check(checks, f"metrics {key} exists", path.exists(), str(path))
    add_check(checks, "metrics summary frame count", metrics.get("summary", {}).get("frame_count") == manifest.get("frame_count"), "")

    gltf_path = resolve_asset_path(bundle_root, manifest.get("gltf", {}), "path")
    if gltf_path.exists():
        try:
            gltf = read_gltf(gltf_path)
            add_check(checks, "final glTF version", gltf.get("asset", {}).get("version") == "2.0", str(gltf_path))
            add_check(checks, "final glTF has water and foam primitives", len(gltf.get("meshes", [{}])[0].get("primitives", [])) >= 2, str(gltf_path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add_check(checks, "final glTF parse", False, str(error))

    glb_path = resolve_asset_path(bundle_root, manifest.get("glb", {}), "path")
    if glb_path.exists():
        try:
            _, glb_header = read_glb_json(glb_path)
            add_check(checks, "final GLB header", glb_header["version"] == 2, str(glb_path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add_check(checks, "final GLB parse", False, str(error))

    animated_gltf_path = resolve_asset_path(bundle_root, manifest.get("animated_gltf", {}), "path")
    if animated_gltf_path.exists():
        try:
            animated_gltf = read_gltf(animated_gltf_path)
            primitive = first_primitive(animated_gltf)
            add_check(checks, "animated glTF weight animation", has_weight_animation(animated_gltf), str(animated_gltf_path))
            add_check(checks, "animated glTF water and foam primitives", primitive_count(animated_gltf) >= 2, str(animated_gltf_path))
            add_check(checks, "animated glTF morph targets", bool(primitive.get("targets")), str(animated_gltf_path))
            add_check(
                checks,
                "animated glTF morph target count",
                len(primitive.get("targets", [])) == manifest.get("frame_count", 0) - 1,
                str(len(primitive.get("targets", []))),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add_check(checks, "animated glTF parse", False, str(error))

    animated_glb_path = resolve_asset_path(bundle_root, manifest.get("animated_glb", {}), "path")
    if animated_glb_path.exists():
        try:
            animated_glb, animated_glb_header = read_glb_json(animated_glb_path)
            primitive = first_primitive(animated_glb)
            add_check(checks, "animated GLB header", animated_glb_header["version"] == 2, str(animated_glb_path))
            add_check(checks, "animated GLB weight animation", has_weight_animation(animated_glb), str(animated_glb_path))
            add_check(checks, "animated GLB water and foam primitives", primitive_count(animated_glb) >= 2, str(animated_glb_path))
            add_check(
                checks,
                "animated GLB morph target count",
                len(primitive.get("targets", [])) == manifest.get("frame_count", 0) - 1,
                str(len(primitive.get("targets", []))),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add_check(checks, "animated GLB parse", False, str(error))

    passed_count = sum(1 for check in checks if check["passed"])
    failed_count = len(checks) - passed_count
    return {
        "bundle_dir": str(bundle_root),
        "manifest": str(resolved_manifest_path),
        "passed": failed_count == 0,
        "passed_checks": passed_count,
        "failed_checks": failed_count,
        "checks": checks,
    }


def build_validation_report(result: dict) -> str:
    lines = [
        "# Spectral Choppy Wave Asset Bundle Validation",
        "",
        "## Summary",
        "",
        f"- bundle_dir: `{result['bundle_dir']}`",
        f"- manifest: `{result['manifest']}`",
        f"- passed: `{result['passed']}`",
        f"- passed_checks: `{result['passed_checks']}`",
        f"- failed_checks: `{result['failed_checks']}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Detail |",
        "| --- | ---: | --- |",
    ]
    for check in result["checks"]:
        lines.append(
            "| {name} | {passed} | `{detail}` |".format(
                name=check["name"],
                passed="yes" if check["passed"] else "no",
                detail=check["detail"],
            )
        )
    return "\n".join(lines) + "\n"


def write_validation_outputs(result: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bundle_validation.json"
    report_path = output_dir / "bundle_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(build_validation_report(result), encoding="utf-8")
    return {"json": str(json_path), "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a spectral choppy wave asset bundle.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("outputs/spectral_choppy_asset_bundle"), help="Bundle directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit bundle_manifest.json path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for validation JSON/Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_bundle(args.bundle_dir, args.manifest)
    output_dir = args.output_dir if args.output_dir is not None else args.bundle_dir
    outputs = write_validation_outputs(result, output_dir)
    print(f"Passed: {result['passed']}")
    print(f"Passed checks: {result['passed_checks']}")
    print(f"Failed checks: {result['failed_checks']}")
    print(f"Report: {outputs['report']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
