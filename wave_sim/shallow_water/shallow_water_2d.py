# Shallow water wave simulation in 2D using PyTorch for GPU acceleration.
# d²h/dt² = c² ∇²h
# You can handle the parameters
# size, steps, frame_every, wave_speed, dt, damping, fps
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter


def laplacian(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(x, shifts=1, dims=0)
        + torch.roll(x, shifts=-1, dims=0)
        + torch.roll(x, shifts=1, dims=1)
        + torch.roll(x, shifts=-1, dims=1)
        - 4.0 * x
    )


def make_initial_height(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    center_pulse = 0.85 * torch.exp(-80.0 * ((xx + 0.25) ** 2 + (yy + 0.05) ** 2))
    side_pulse = -0.35 * torch.exp(-120.0 * ((xx - 0.35) ** 2 + (yy - 0.25) ** 2))
    return center_pulse + side_pulse


def damp_edges(field: torch.Tensor, damping_mask: torch.Tensor) -> torch.Tensor:
    return field * damping_mask


def make_edge_damping(size: int, width: int, strength: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device)
    dist = torch.minimum(torch.minimum(coords, size - 1 - coords)[:, None], torch.minimum(coords, size - 1 - coords)[None, :])
    edge = torch.clamp((width - dist.float()) / max(width, 1), min=0.0, max=1.0)
    return 1.0 - strength * edge**2


def simulate(
    size: int,
    steps: int,
    frame_every: int,
    wave_speed: float,
    dt: float,
    damping: float,
    device: torch.device,
) -> list[torch.Tensor]:
    h = make_initial_height(size, device)
    h_prev = h.clone()
    edge_damping = make_edge_damping(size, width=max(8, size // 12), strength=0.06, device=device)
    frames = []

    c2dt2 = (wave_speed * dt) ** 2
    for step in range(steps):
        h_next = (2.0 * h - h_prev) + c2dt2 * laplacian(h)
        h_next = damping * damp_edges(h_next, edge_damping)

        h_prev, h = h, h_next

        if step % frame_every == 0:
            frames.append(h.detach().cpu())

    return frames


def save_outputs(frames: list[torch.Tensor], output_dir: Path, fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "shallow_water_2d.gif"
    png_path = output_dir / "shallow_water_2d_final.png"
    preview_path = output_dir / "shallow_water_2d_preview.png"

    vmin = min(float(frame.min()) for frame in frames)
    vmax = max(float(frame.max()) for frame in frames)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    image = ax.imshow(frames[0], cmap="turbo", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="height")

    def update(index: int):
        image.set_data(frames[index])
        ax.set_title(f"2D shallow-water height field | frame {index + 1}/{len(frames)}")
        return (image,)

    animation = FuncAnimation(fig, update, frames=len(frames), blit=True)
    animation.save(gif_path, writer=PillowWriter(fps=fps))

    image.set_data(frames[-1])
    ax.set_title("2D shallow-water height field | final frame")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    preview_indices = torch.linspace(0, len(frames) - 1, 6).round().int().tolist()
    preview_fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=120)
    for axis, frame_index in zip(axes.flat, preview_indices):
        axis.imshow(frames[frame_index], cmap="turbo", vmin=vmin, vmax=vmax, interpolation="bilinear")
        axis.set_title(f"frame {frame_index + 1}")
        axis.set_axis_off()
    preview_fig.suptitle("2D shallow-water height field preview")
    preview_fig.tight_layout()
    preview_fig.savefig(preview_path, bbox_inches="tight")
    plt.close(preview_fig)

    print(f"Saved GIF: {gif_path}")
    print(f"Saved final frame: {png_path}")
    print(f"Saved preview: {preview_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-accelerated 2D height-field wave simulation.")
    parser.add_argument("--size", type=int, default=256, help="Square grid size.")
    parser.add_argument("--steps", type=int, default=540, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=6, help="Save one frame every N steps.")
    parser.add_argument("--wave-speed", type=float, default=0.42, help="Wave speed coefficient.")
    parser.add_argument("--dt", type=float, default=0.9, help="Time step.")
    parser.add_argument("--damping", type=float, default=0.999, help="Global damping per step.")
    parser.add_argument("--fps", type=int, default=24, help="Output GIF frames per second.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shallow_water_2d"), help="Output directory.")
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
    save_outputs(frames, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
