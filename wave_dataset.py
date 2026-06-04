import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def frames_to_numpy(frames: list[torch.Tensor]) -> np.ndarray:
    return np.stack([frame.detach().cpu().numpy().astype(np.float32) for frame in frames])


def save_wave_dataset(
    path: Path,
    frames: list[torch.Tensor],
    depth: torch.Tensor,
    metadata: dict[str, Any],
    u_frames: list[torch.Tensor] | None = None,
    v_frames: list[torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frames": frames_to_numpy(frames),
        "depth": depth.detach().cpu().numpy().astype(np.float32),
        "metadata": json.dumps(metadata, ensure_ascii=False),
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
