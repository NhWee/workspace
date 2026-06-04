import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from shallow_water_2d import simulate


def downsample_frame(frame: torch.Tensor, max_points: int) -> np.ndarray:
    stride = max(1, int(np.ceil(frame.shape[0] / max_points)))
    return frame[::stride, ::stride].numpy()


def prepare_surface_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(axis, axis)


def set_axis_style(ax, z_limit: float, compact: bool = False) -> None:
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-z_limit, z_limit)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.view_init(elev=35, azim=-135)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("height")


def save_3d_outputs(frames: list[torch.Tensor], output_dir: Path, fps: int, max_surface_points: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "shallow_water_surface_3d.gif"
    png_path = output_dir / "shallow_water_surface_3d_final.png"
    preview_path = output_dir / "shallow_water_surface_3d_preview.png"

    surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    x_grid, y_grid = prepare_surface_grid(surfaces[0].shape[0])
    z_limit = max(float(np.max(np.abs(surface))) for surface in surfaces) * 1.05
    z_limit = max(z_limit, 0.1)

    fig = plt.figure(figsize=(8, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    set_axis_style(ax, z_limit)

    surface = [ax.plot_surface(x_grid, y_grid, surfaces[0], cmap="turbo", linewidth=0, antialiased=True)]

    def update(index: int):
        surface[0].remove()
        surface[0] = ax.plot_surface(
            x_grid,
            y_grid,
            surfaces[index],
            cmap="turbo",
            linewidth=0,
            antialiased=True,
            vmin=-z_limit,
            vmax=z_limit,
        )
        ax.set_title(f"3D wave surface | frame {index + 1}/{len(surfaces)}")
        return surface

    animation = FuncAnimation(fig, update, frames=len(surfaces), blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=fps))

    update(len(surfaces) - 1)
    ax.set_title("3D wave surface | final frame")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    preview_indices = np.linspace(0, len(surfaces) - 1, 6).round().astype(int)
    preview_fig = plt.figure(figsize=(13, 8), dpi=110)
    for plot_index, frame_index in enumerate(preview_indices, start=1):
        preview_ax = preview_fig.add_subplot(2, 3, plot_index, projection="3d")
        preview_ax.plot_surface(
            x_grid,
            y_grid,
            surfaces[frame_index],
            cmap="turbo",
            linewidth=0,
            antialiased=True,
            vmin=-z_limit,
            vmax=z_limit,
        )
        set_axis_style(preview_ax, z_limit, compact=True)
        preview_ax.set_title(f"frame {frame_index + 1}")
    preview_fig.suptitle("3D wave surface preview")
    preview_fig.tight_layout()
    preview_fig.savefig(preview_path, bbox_inches="tight")
    plt.close(preview_fig)

    print(f"Saved GIF: {gif_path}")
    print(f"Saved final frame: {png_path}")
    print(f"Saved preview: {preview_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the GPU height-field wave simulation as a 3D surface.")
    parser.add_argument("--size", type=int, default=256, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=420, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=7, help="Save one render frame every N simulation steps.")
    parser.add_argument("--wave-speed", type=float, default=0.42, help="Wave speed coefficient.")
    parser.add_argument("--dt", type=float, default=0.9, help="Time step.")
    parser.add_argument("--damping", type=float, default=0.999, help="Global damping per step.")
    parser.add_argument("--fps", type=int, default=18, help="Output GIF frames per second.")
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max rendered points per surface axis.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frames = simulate(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        wave_speed=args.wave_speed,
        dt=args.dt,
        damping=args.damping,
        device=device,
    )
    save_3d_outputs(frames, args.output_dir, args.fps, args.max_surface_points)


if __name__ == "__main__":
    main()
