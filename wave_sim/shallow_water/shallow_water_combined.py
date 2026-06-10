import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter


def laplacian(field: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(field, shifts=1, dims=0)
        + torch.roll(field, shifts=-1, dims=0)
        + torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
        - 4.0 * field
    )


def gradient_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=1) - torch.roll(field, shifts=1, dims=1)) / (2.0 * dx)


def gradient_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=0) - torch.roll(field, shifts=1, dims=0)) / (2.0 * dx)


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(u, dx) + gradient_y(v, dx)


def make_edge_damping(size: int, width: int, strength: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device)
    edge_dist = torch.minimum(coords, size - 1 - coords)
    dist = torch.minimum(edge_dist[:, None], edge_dist[None, :])
    edge = torch.clamp((width - dist.float()) / max(width, 1), min=0.0, max=1.0)
    return 1.0 - strength * edge**2


def make_height_initial_condition(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    center_pulse = 0.85 * torch.exp(-80.0 * ((xx + 0.25) ** 2 + (yy + 0.05) ** 2))
    side_pulse = -0.35 * torch.exp(-120.0 * ((xx - 0.35) ** 2 + (yy - 0.25) ** 2))
    return center_pulse + side_pulse


def make_uv_initial_eta(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    main = 0.38 * torch.exp(-90.0 * ((xx + 0.35) ** 2 + (yy + 0.10) ** 2))
    trough = -0.24 * torch.exp(-110.0 * ((xx - 0.25) ** 2 + (yy - 0.15) ** 2))
    ridge = 0.10 * torch.exp(-35.0 * ((xx + 0.05) ** 2 + (yy - 0.45) ** 2))
    return main + trough + ridge


def make_bathymetry(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    shelf = 0.18 * torch.sigmoid(10.0 * (xx + 0.15))
    shoal = 0.22 * torch.exp(-24.0 * ((xx - 0.35) ** 2 + (yy + 0.10) ** 2))
    island = 0.70 * torch.exp(-42.0 * ((xx + 0.35) ** 2 + (yy - 0.25) ** 2))

    depth = 0.58 - shelf - shoal - island
    wet_mask = depth > 0.055
    depth = torch.clamp(depth, min=0.0)
    return depth, wet_mask.float()


def make_incoming_wave(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    packet = torch.exp(-90.0 * (xx + 0.72) ** 2)
    transverse = torch.exp(-1.8 * yy**2)
    ripple = torch.cos(24.0 * (xx + 0.72))
    return 0.08 * packet * transverse * ripple


def simulate_height_field(
    size: int,
    steps: int,
    frame_every: int,
    wave_speed: float,
    dt: float,
    damping: float,
    device: torch.device,
) -> list[torch.Tensor]:
    h = make_height_initial_condition(size, device)
    h_prev = h.clone()
    edge_damping = make_edge_damping(size, width=max(8, size // 12), strength=0.06, device=device)
    frames = []
    c2dt2 = (wave_speed * dt) ** 2

    for step in range(steps):
        h_next = (2.0 * h - h_prev) + c2dt2 * laplacian(h)
        h_next = damping * h_next * edge_damping
        h_prev, h = h, h_next
        if step % frame_every == 0:
            frames.append(h.detach().cpu())

    return frames


def simulate_uv_shallow_water(
    size: int,
    steps: int,
    frame_every: int,
    mean_depth: float,
    gravity: float,
    dt: float,
    damping: float,
    device: torch.device,
) -> list[torch.Tensor]:
    dx = 2.0 / (size - 1)
    eta = make_uv_initial_eta(size, device)
    u = torch.zeros_like(eta)
    v = torch.zeros_like(eta)
    edge_damping = make_edge_damping(size, width=max(8, size // 12), strength=0.08, device=device)
    frames = []

    for step in range(steps):
        u_next = damping * (u - gravity * dt * gradient_x(eta, dx))
        v_next = damping * (v - gravity * dt * gradient_y(eta, dx))
        eta_next = damping * (eta - mean_depth * dt * divergence(u_next, v_next, dx))

        u_next = u_next * edge_damping
        v_next = v_next * edge_damping
        eta_next = eta_next * edge_damping
        eta, u, v = eta_next, u_next, v_next

        if step % frame_every == 0:
            frames.append(eta.detach().cpu())

    return frames


def compute_cfl_dt(depth: torch.Tensor, gravity: float, dx: float, cfl: float) -> float:
    max_depth = float(depth.max())
    if max_depth <= 0.0:
        raise ValueError("Bathymetry depth must contain at least one wet cell.")
    max_wave_speed = (gravity * max_depth) ** 0.5
    return cfl * dx / max_wave_speed


def simulate_bathymetry(
    size: int,
    steps: int,
    frame_every: int,
    gravity: float,
    dt: float | None,
    damping: float,
    device: torch.device,
    cfl: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    dx = 2.0 / (size - 1)
    depth, wet_mask = make_bathymetry(size, device)
    if dt is None:
        dt = compute_cfl_dt(depth, gravity, dx, cfl)

    eta = make_incoming_wave(size, device) * wet_mask
    u = torch.zeros_like(eta)
    v = torch.zeros_like(eta)
    edge_damping = make_edge_damping(size, width=max(8, size // 12), strength=0.10, device=device)
    frames = []

    for step in range(steps):
        u_next = damping * (u - gravity * dt * gradient_x(eta, dx))
        v_next = damping * (v - gravity * dt * gradient_y(eta, dx))
        u_next = u_next * wet_mask
        v_next = v_next * wet_mask

        flux_x = depth * u_next
        flux_y = depth * v_next
        eta_next = damping * (eta - dt * divergence(flux_x, flux_y, dx))

        u_next = u_next * edge_damping
        v_next = v_next * edge_damping
        eta_next = eta_next * edge_damping * wet_mask
        eta, u, v = eta_next, u_next, v_next

        if step % frame_every == 0:
            frames.append(eta.detach().cpu())

    return frames, depth.detach().cpu()


def downsample_frame(frame: torch.Tensor, max_points: int) -> np.ndarray:
    stride = max(1, int(np.ceil(frame.shape[0] / max_points)))
    return frame[::stride, ::stride].numpy()


def prepare_surface_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(axis, axis)


def save_2d_outputs(frames: list[torch.Tensor], output_dir: Path, fps: int, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{prefix}.gif"
    png_path = output_dir / f"{prefix}_final.png"
    preview_path = output_dir / f"{prefix}_preview.png"
    vmin = min(float(frame.min()) for frame in frames)
    vmax = max(float(frame.max()) for frame in frames)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    image = ax.imshow(frames[0], cmap="turbo", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="height")

    def update(index: int):
        image.set_data(frames[index])
        ax.set_title(f"2D height field | frame {index + 1}/{len(frames)}")
        return (image,)

    animation = FuncAnimation(fig, update, frames=len(frames), blit=True)
    animation.save(gif_path, writer=PillowWriter(fps=fps))

    image.set_data(frames[-1])
    ax.set_title("2D height field | final frame")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    preview_indices = torch.linspace(0, len(frames) - 1, 6).round().int().tolist()
    preview_fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=120)
    for axis, frame_index in zip(axes.flat, preview_indices):
        axis.imshow(frames[frame_index], cmap="turbo", vmin=vmin, vmax=vmax, interpolation="bilinear")
        axis.set_title(f"frame {frame_index + 1}")
        axis.set_axis_off()
    preview_fig.suptitle("2D height-field preview")
    preview_fig.tight_layout()
    preview_fig.savefig(preview_path, bbox_inches="tight")
    plt.close(preview_fig)

    print(f"Saved GIF: {gif_path}")
    print(f"Saved final frame: {png_path}")
    print(f"Saved preview: {preview_path}")


def set_surface_axis(ax, z_limit: float, compact: bool = False) -> None:
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


def save_3d_surface_outputs(
    frames: list[torch.Tensor],
    output_dir: Path,
    fps: int,
    max_surface_points: int,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{prefix}.gif"
    png_path = output_dir / f"{prefix}_final.png"
    preview_path = output_dir / f"{prefix}_preview.png"
    surfaces = [downsample_frame(frame, max_surface_points) for frame in frames]
    x_grid, y_grid = prepare_surface_grid(surfaces[0].shape[0])
    z_limit = max(float(np.max(np.abs(surface))) for surface in surfaces) * 1.05
    z_limit = max(z_limit, 0.1)

    fig = plt.figure(figsize=(8, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    set_surface_axis(ax, z_limit)
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
        set_surface_axis(preview_ax, z_limit, compact=True)
        preview_ax.set_title(f"frame {frame_index + 1}")
    preview_fig.suptitle("3D wave surface preview")
    preview_fig.tight_layout()
    preview_fig.savefig(preview_path, bbox_inches="tight")
    plt.close(preview_fig)

    print(f"Saved GIF: {gif_path}")
    print(f"Saved final frame: {png_path}")
    print(f"Saved preview: {preview_path}")


def save_bathymetry_map(depth: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "combined_bathymetry_depth.png"
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    image = ax.imshow(depth, cmap="viridis", origin="lower", interpolation="bilinear")
    ax.set_title("Bathymetry depth map")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="depth")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved depth map: {path}")


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


def save_bathymetry_scene_outputs(
    frames: list[torch.Tensor],
    depth: torch.Tensor,
    output_dir: Path,
    fps: int,
    max_surface_points: int,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{prefix}.gif"
    png_path = output_dir / f"{prefix}_final.png"
    preview_path = output_dir / f"{prefix}_preview.png"
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


def parse_dt(value: str) -> float | None:
    return None if str(value).lower() == "auto" else float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combined GPU shallow-water experiments in one file.")
    parser.add_argument(
        "--mode",
        choices=["height2d", "height3d", "uv3d", "bathymetry3d", "bathymetry-scene", "all"],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument("--size", type=int, default=192, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=360, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=12, help="Save one frame every N simulation steps.")
    parser.add_argument("--wave-speed", type=float, default=0.42, help="Height-field wave speed coefficient.")
    parser.add_argument("--mean-depth", type=float, default=0.35, help="Mean water depth H for the u/v model.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravity coefficient g.")
    parser.add_argument("--height-dt", type=float, default=0.9, help="Time step for the height-only model.")
    parser.add_argument("--uv-dt", type=float, default=0.0025, help="Time step for the u/v model.")
    parser.add_argument("--bathymetry-dt", default="auto", help="Bathymetry time step, or 'auto' for CFL.")
    parser.add_argument("--cfl", type=float, default=0.35, help="CFL factor used when --bathymetry-dt auto.")
    parser.add_argument("--damping", type=float, default=0.9994, help="Global damping per step.")
    parser.add_argument("--fps", type=int, default=20, help="Output GIF frames per second.")
    parser.add_argument("--max-surface-points", type=int, default=128, help="Max rendered points per surface axis.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/combined"), help="Output directory.")
    return parser


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def run_height2d(args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    frames = simulate_height_field(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        wave_speed=args.wave_speed,
        dt=args.height_dt,
        damping=args.damping,
        device=device,
    )
    save_2d_outputs(frames, args.output_dir, args.fps, "combined_height2d")
    return frames


def run_height3d(args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    frames = simulate_height_field(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        wave_speed=args.wave_speed,
        dt=args.height_dt,
        damping=args.damping,
        device=device,
    )
    save_3d_surface_outputs(frames, args.output_dir, args.fps, args.max_surface_points, "combined_height3d")
    return frames


def run_uv3d(args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    frames = simulate_uv_shallow_water(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        mean_depth=args.mean_depth,
        gravity=args.gravity,
        dt=args.uv_dt,
        damping=args.damping,
        device=device,
    )
    save_3d_surface_outputs(frames, args.output_dir, args.fps, args.max_surface_points, "combined_uv3d")
    return frames


def run_bathymetry3d(args: argparse.Namespace, device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor]:
    frames, depth = simulate_bathymetry(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        gravity=args.gravity,
        dt=parse_dt(args.bathymetry_dt),
        damping=args.damping,
        device=device,
        cfl=args.cfl,
    )
    save_3d_surface_outputs(frames, args.output_dir, args.fps, args.max_surface_points, "combined_bathymetry3d")
    save_bathymetry_map(depth, args.output_dir)
    return frames, depth


def run_bathymetry_scene(args: argparse.Namespace, device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor]:
    frames, depth = simulate_bathymetry(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        gravity=args.gravity,
        dt=parse_dt(args.bathymetry_dt),
        damping=args.damping,
        device=device,
        cfl=args.cfl,
    )
    save_bathymetry_scene_outputs(
        frames,
        depth,
        args.output_dir,
        args.fps,
        args.max_surface_points,
        "combined_bathymetry_scene",
    )
    save_bathymetry_map(depth, args.output_dir)
    return frames, depth


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = get_device()

    if args.mode in ("height2d", "all"):
        run_height2d(args, device)
    if args.mode in ("height3d", "all"):
        run_height3d(args, device)
    if args.mode in ("uv3d", "all"):
        run_uv3d(args, device)
    if args.mode in ("bathymetry3d", "all"):
        run_bathymetry3d(args, device)
    if args.mode in ("bathymetry-scene", "all"):
        run_bathymetry_scene(args, device)


if __name__ == "__main__":
    main()
