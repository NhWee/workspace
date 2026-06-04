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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames_to_numpy(frames),
        depth=depth.detach().cpu().numpy().astype(np.float32),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"Saved wave dataset: {path}")


def load_wave_dataset(path: Path) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        frames_array = data["frames"].astype(np.float32)
        depth_array = data["depth"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))

    frames = [torch.from_numpy(frame) for frame in frames_array]
    depth = torch.from_numpy(depth_array)
    return frames, depth, metadata
