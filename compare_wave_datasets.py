import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


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
    "final_l2_vs_baseline",
    "final_linf_vs_baseline",
    "final_diff_heatmap",
    "frames_l2_mean_vs_baseline",
    "frames_linf_max_vs_baseline",
    "frame_metrics_csv",
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
        "_frames": frames,
        "_final_frame": frames[-1],
    }


def add_baseline_difference_metrics(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return

    baseline = summaries[0]["_final_frame"]
    baseline_frames = summaries[0]["_frames"]
    for summary in summaries:
        final_frame = summary["_final_frame"]
        if final_frame.shape != baseline.shape:
            summary["final_l2_vs_baseline"] = None
            summary["final_linf_vs_baseline"] = None
            summary["frames_l2_mean_vs_baseline"] = None
            summary["frames_linf_max_vs_baseline"] = None
            continue

        difference = final_frame - baseline
        summary["final_l2_vs_baseline"] = float(np.sqrt(np.mean(difference * difference)))
        summary["final_linf_vs_baseline"] = float(np.max(np.abs(difference)))

        frames = summary["_frames"]
        if frames.shape != baseline_frames.shape:
            summary["frames_l2_mean_vs_baseline"] = None
            summary["frames_linf_max_vs_baseline"] = None
            continue

        frame_difference = frames - baseline_frames
        frame_l2 = np.sqrt(np.mean(frame_difference * frame_difference, axis=(1, 2)))
        frame_linf = np.max(np.abs(frame_difference), axis=(1, 2))
        summary["frames_l2_mean_vs_baseline"] = float(frame_l2.mean())
        summary["frames_linf_max_vs_baseline"] = float(frame_linf.max())


def safe_stem(path_text: str) -> str:
    return Path(path_text).stem.replace(" ", "_")


def compute_frame_metric_series(summary: dict[str, Any], baseline_summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    frames = summary["_frames"]
    baseline_frames = baseline_summary["_frames"]
    if frames.shape != baseline_frames.shape:
        return None

    frame_difference = frames - baseline_frames
    frame_l2 = np.sqrt(np.mean(frame_difference * frame_difference, axis=(1, 2)))
    frame_linf = np.max(np.abs(frame_difference), axis=(1, 2))
    return frame_l2, frame_linf


def save_frame_metric_series(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    add_baseline_difference_metrics(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return []

    baseline_summary = summaries[0]
    saved_paths = []
    for index, summary in enumerate(summaries):
        summary["frame_metrics_csv"] = None
        if index == 0:
            continue

        series = compute_frame_metric_series(summary, baseline_summary)
        if series is None:
            continue

        frame_l2, frame_linf = series
        output_path = output_dir / f"frame_metrics_{index:02d}_{safe_stem(summary['path'])}.csv"
        lines = ["frame_index,l2_vs_baseline,linf_vs_baseline"]
        lines.extend(
            f"{frame_index},{frame_l2[frame_index]:.9g},{frame_linf[frame_index]:.9g}"
            for frame_index in range(len(frame_l2))
        )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["frame_metrics_csv"] = str(output_path)
        saved_paths.append(output_path)

    return saved_paths


def make_frame_metric_chart(summaries: list[dict[str, Any]]) -> go.Figure:
    add_baseline_difference_metrics(summaries)
    fig = go.Figure()
    if not summaries:
        return fig

    baseline_summary = summaries[0]
    for index, summary in enumerate(summaries):
        if index == 0:
            continue

        series = compute_frame_metric_series(summary, baseline_summary)
        if series is None:
            continue

        frame_l2, frame_linf = series
        frame_index = np.arange(len(frame_l2))
        dataset_name = Path(summary["path"]).name
        fig.add_trace(
            go.Scatter(
                x=frame_index,
                y=frame_l2,
                mode="lines+markers",
                name=f"{dataset_name} L2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame_index,
                y=frame_linf,
                mode="lines+markers",
                name=f"{dataset_name} Linf",
            )
        )

    fig.update_layout(
        title="Frame-wise wave dataset difference vs baseline",
        xaxis={"title": "Frame index"},
        yaxis={"title": "Difference"},
        hovermode="x unified",
        margin={"l": 64, "r": 24, "t": 56, "b": 48},
    )
    return fig


def save_frame_metric_chart(summaries: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_frame_metric_chart(summaries)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Saved frame metric chart: {output_path}")
    return output_path


def save_final_frame_difference_heatmaps(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    add_baseline_difference_metrics(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return []

    baseline = summaries[0]["_final_frame"]
    saved_paths = []
    for index, summary in enumerate(summaries):
        summary["final_diff_heatmap"] = None
        if index == 0:
            continue
        final_frame = summary["_final_frame"]
        if final_frame.shape != baseline.shape:
            continue

        difference = np.abs(final_frame - baseline)
        output_path = output_dir / f"final_diff_{index:02d}_{safe_stem(summary['path'])}.png"
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        image = ax.imshow(difference, cmap="magma", origin="lower", interpolation="nearest")
        ax.set_title(f"Final frame abs difference vs baseline: {Path(summary['path']).name}")
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="abs eta difference")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        summary["final_diff_heatmap"] = str(output_path)
        saved_paths.append(output_path)

    return saved_paths


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def make_markdown_table(summaries: list[dict[str, Any]]) -> str:
    add_baseline_difference_metrics(summaries)
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
    parser.add_argument(
        "--diff-heatmap-dir",
        type=Path,
        default=None,
        help="Optional directory for final-frame absolute difference heatmaps.",
    )
    parser.add_argument(
        "--frame-metrics-dir",
        type=Path,
        default=None,
        help="Optional directory for per-frame L2/Linf metric CSV files.",
    )
    parser.add_argument(
        "--frame-metrics-chart",
        type=Path,
        default=None,
        help="Optional Plotly HTML output for per-frame L2/Linf metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [load_dataset_summary(path) for path in args.datasets]
    if args.diff_heatmap_dir:
        saved_paths = save_final_frame_difference_heatmaps(summaries, args.diff_heatmap_dir)
        for path in saved_paths:
            print(f"Saved final-frame difference heatmap: {path}")
    if args.frame_metrics_dir:
        saved_paths = save_frame_metric_series(summaries, args.frame_metrics_dir)
        for path in saved_paths:
            print(f"Saved frame metric series: {path}")
    if args.frame_metrics_chart:
        save_frame_metric_chart(summaries, args.frame_metrics_chart)
    table_text = make_markdown_table(summaries)
    print(table_text)
    write_summary(args.output, table_text)


if __name__ == "__main__":
    main()
