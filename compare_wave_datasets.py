import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_FIELDS = [
    "path",
    "size",
    "steps",
    "frame_every",
    "frame_count",
    "stores_velocity",
    "device",
    "created_at_utc",
    "git_commit",
    "git_is_dirty",
    "eta_min",
    "eta_max",
    "depth_min",
    "depth_max",
    "speed_max",
]


def compact_commit(commit: Any) -> str:
    if not commit:
        return "None"
    return str(commit)[:8]


def load_dataset_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"].astype(np.float32)
        depth = data["depth"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))
        u_frames = data["u_frames"].astype(np.float32) if "u_frames" in data.files else None
        v_frames = data["v_frames"].astype(np.float32) if "v_frames" in data.files else None

    speed_max = None
    if u_frames is not None and v_frames is not None:
        speed_max = float(np.sqrt(u_frames * u_frames + v_frames * v_frames).max())

    return {
        "path": str(path),
        "size": metadata.get("size", frames.shape[-1]),
        "steps": metadata.get("steps"),
        "frame_every": metadata.get("frame_every"),
        "frame_count": metadata.get("frame_count", frames.shape[0]),
        "stores_velocity": bool(metadata.get("stores_velocity", u_frames is not None and v_frames is not None)),
        "device": metadata.get("device"),
        "created_at_utc": metadata.get("created_at_utc"),
        "git_commit": compact_commit(metadata.get("git_commit")),
        "git_is_dirty": metadata.get("git_is_dirty"),
        "eta_min": float(frames.min()),
        "eta_max": float(frames.max()),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "speed_max": speed_max,
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def make_markdown_table(summaries: list[dict[str, Any]]) -> str:
    rows = [[format_value(summary.get(field)) for field in SUMMARY_FIELDS] for summary in summaries]
    widths = [
        max(len(field), *(len(row[index]) for row in rows))
        for index, field in enumerate(SUMMARY_FIELDS)
    ]
    header = "| " + " | ".join(field.ljust(widths[index]) for index, field in enumerate(SUMMARY_FIELDS)) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(SUMMARY_FIELDS))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(SUMMARY_FIELDS))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_summary(output_path: Path, table_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text + "\n", encoding="utf-8")
    print(f"Saved dataset comparison: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reusable wave NPZ datasets.")
    parser.add_argument("datasets", nargs="+", type=Path, help="NPZ dataset paths to compare.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/wave_dataset_comparison.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [load_dataset_summary(path) for path in args.datasets]
    table_text = make_markdown_table(summaries)
    print(table_text)
    write_summary(args.output, table_text)


if __name__ == "__main__":
    main()
