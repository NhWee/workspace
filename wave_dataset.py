import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def frames_to_numpy(frames: list[torch.Tensor]) -> np.ndarray:
    return np.stack([frame.detach().cpu().numpy().astype(np.float32) for frame in frames])


def get_git_metadata(repo_dir: Path | None = None) -> dict[str, Any]:
    repo_dir = repo_dir or Path(__file__).resolve().parent
    metadata: dict[str, Any] = {
        "git_commit": None,
        "git_is_dirty": None,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return metadata

    metadata["git_commit"] = commit
    metadata["git_is_dirty"] = bool(status.strip())
    return metadata


def enrich_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for key, value in get_git_metadata().items():
        enriched.setdefault(key, value)
    return enriched


def save_wave_dataset(
    path: Path,
    frames: list[torch.Tensor],
    depth: torch.Tensor,
    metadata: dict[str, Any],
    u_frames: list[torch.Tensor] | None = None,
    v_frames: list[torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched_metadata = enrich_metadata(metadata)
    arrays = {
        "frames": frames_to_numpy(frames),
        "depth": depth.detach().cpu().numpy().astype(np.float32),
        "metadata": json.dumps(enriched_metadata, ensure_ascii=False),
    }
    if u_frames is not None:
        arrays["u_frames"] = frames_to_numpy(u_frames)
    if v_frames is not None:
        arrays["v_frames"] = frames_to_numpy(v_frames)
    np.savez_compressed(path, **arrays)
    print(f"Saved wave dataset: {path}")


def load_wave_dataset(path: Path) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        frames_array = data["frames"].astype(np.float32)
        depth_array = data["depth"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))

    frames = [torch.from_numpy(frame) for frame in frames_array]
    depth = torch.from_numpy(depth_array)
    return frames, depth, metadata


def load_wave_dataset_with_velocity(
    path: Path,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any], list[torch.Tensor] | None, list[torch.Tensor] | None]:
    with np.load(path, allow_pickle=False) as data:
        frames_array = data["frames"].astype(np.float32)
        depth_array = data["depth"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))
        u_frames_array = data["u_frames"].astype(np.float32) if "u_frames" in data.files else None
        v_frames_array = data["v_frames"].astype(np.float32) if "v_frames" in data.files else None

    frames = [torch.from_numpy(frame) for frame in frames_array]
    depth = torch.from_numpy(depth_array)
    u_frames = [torch.from_numpy(frame) for frame in u_frames_array] if u_frames_array is not None else None
    v_frames = [torch.from_numpy(frame) for frame in v_frames_array] if v_frames_array is not None else None
    return frames, depth, metadata, u_frames, v_frames
