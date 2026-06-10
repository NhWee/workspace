import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from wave_sim.shallow_water.shallow_water_bathymetry_3d import save_bathymetry_map, simulate_bathymetry
from wave_sim.shallow_water.shallow_water_surface_3d import downsample_frame, prepare_surface_grid


def set_scene_axis(ax, z_min: float, z_max: float, compact: bool = False) -> None:
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((1.0, 1.0, 0.55))
    ax.view_init(elev=33, azim=-135)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


def save_scene_outputs(
    frames: list[torch.Tensor],
    depth: torch.Tensor,
    output_dir: Path,
    fps: int,
    max_surface_points: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "shallow_water_bathymetry_scene_3d.gif"
    png_path = output_dir / "shallow_water_bathymetry_scene_3d_final.png"
    preview_path = output_dir / "shallow_water_bathymetry_scene_3d_preview.png"

    water_surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    bed_surface = -downsample_frame(depth, max_surface_points)
    x_grid, y_grid = prepare_surface_grid(water_surfaces[0].shape[0])

    eta_limit = max(float(np.max(np.abs(surface))) for surface in water_surfaces)
    z_min = float(bed_surface.min()) * 1.05
    z_max = max(eta_limit * 1.4, 0.08)

    fig = plt.figure(figsize=(8.5, 7.2), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    set_scene_axis(ax, z_min, z_max)
    bed = ax.plot_surface(x_grid, y_grid, bed_surface, cmap="terrain", linewidth=0, alpha=0.92)
    water = [
        ax.plot_surface(
            x_grid,
            y_grid,
            water_surfaces[0],
            cmap="Blues_r",
            linewidth=0,
            alpha=0.72,
            vmin=-z_max,
            vmax=z_max,
        )
    ]

    def update(index: int):
        water[0].remove()
        water[0] = ax.plot_surface(
            x_grid,
            y_grid,
            water_surfaces[index],
            cmap="Blues_r",
            linewidth=0,
            alpha=0.72,
            vmin=-z_max,
            vmax=z_max,
        )
        ax.set_title(f"Bathymetry + 3D wave surface | frame {index + 1}/{len(water_surfaces)}")
        return [bed, water[0]]

    animation = FuncAnimation(fig, update, frames=len(water_surfaces), blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=fps))

    update(len(water_surfaces) - 1)
    ax.set_title("Bathymetry + 3D wave surface | final frame")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    preview_indices = np.linspace(0, len(water_surfaces) - 1, 6).round().astype(int)
    preview_fig = plt.figure(figsize=(13, 8), dpi=110)
    for plot_index, frame_index in enumerate(preview_indices, start=1):
        preview_ax = preview_fig.add_subplot(2, 3, plot_index, projection="3d")
        preview_ax.plot_surface(x_grid, y_grid, bed_surface, cmap="terrain", linewidth=0, alpha=0.92)
        preview_ax.plot_surface(
            x_grid,
            y_grid,
            water_surfaces[frame_index],
            cmap="Blues_r",
            linewidth=0,
            alpha=0.72,
            vmin=-z_max,
            vmax=z_max,
        )
        set_scene_axis(preview_ax, z_min, z_max, compact=True)
        preview_ax.set_title(f"frame {frame_index + 1}")
    preview_fig.suptitle("Bathymetry + 3D wave surface preview")
    preview_fig.tight_layout()
    preview_fig.savefig(preview_path, bbox_inches="tight")
    plt.close(preview_fig)

    print(f"Saved GIF: {gif_path}")
    print(f"Saved final frame: {png_path}")
    print(f"Saved preview: {preview_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render bathymetry and the GPU wave surface in one 3D scene.")
    parser.add_argument("--size", type=int, default=256, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=480, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=12, help="Save one render frame every N simulation steps.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravity coefficient g.")
    parser.add_argument("--dt", default="auto", help="Time step, or 'auto' to use a CFL-based value.")
    parser.add_argument("--cfl", type=float, default=0.35, help="CFL factor used when --dt auto.")
    parser.add_argument("--damping", type=float, default=0.9994, help="Global damping per step.")
    parser.add_argument("--fps", type=int, default=20, help="Output GIF frames per second.")
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max rendered points per surface axis.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    dt = None if str(args.dt).lower() == "auto" else float(args.dt)

    frames, depth = simulate_bathymetry(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        gravity=args.gravity,
        dt=dt,
        damping=args.damping,
        device=device,
        cfl=args.cfl,
    )
    save_scene_outputs(frames, depth, args.output_dir, args.fps, args.max_surface_points)
    save_bathymetry_map(depth, args.output_dir)


if __name__ == "__main__":
    main()
