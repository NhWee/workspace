# Shallow water wave simulation considering velocity fields (u/v) in 2D using PyTorch for GPU acceleration, rendered in 3D.
# Linear Shallow Water Equations
# You can handle the parameters
# size, steps, frame_every, mean_depth, gravity, dt, damping, fps
import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from wave_sim.shallow_water.shallow_water_2d import make_edge_damping
from wave_sim.shallow_water.shallow_water_surface_3d import save_3d_outputs


def gradient_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=1) - torch.roll(field, shifts=1, dims=1)) / (2.0 * dx)


def gradient_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, shifts=-1, dims=0) - torch.roll(field, shifts=1, dims=0)) / (2.0 * dx)


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(u, dx) + gradient_y(v, dx)


def make_initial_eta(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, size, device=device)
    x = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    main = 0.38 * torch.exp(-90.0 * ((xx + 0.35) ** 2 + (yy + 0.10) ** 2))
    trough = -0.24 * torch.exp(-110.0 * ((xx - 0.25) ** 2 + (yy - 0.15) ** 2))
    ridge = 0.10 * torch.exp(-35.0 * ((xx + 0.05) ** 2 + (yy - 0.45) ** 2))
    return main + trough + ridge


def simulate_uv(
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
    eta = make_initial_eta(size, device)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU h/u/v shallow-water surface simulation rendered in 3D.")
    parser.add_argument("--size", type=int, default=256, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=700, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=10, help="Save one render frame every N simulation steps.")
    parser.add_argument("--mean-depth", type=float, default=0.35, help="Mean water depth H.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravity coefficient g.")
    parser.add_argument("--dt", type=float, default=0.0025, help="Time step.")
    parser.add_argument("--damping", type=float, default=0.9995, help="Global damping per step.")
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

    frames = simulate_uv(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        mean_depth=args.mean_depth,
        gravity=args.gravity,
        dt=args.dt,
        damping=args.damping,
        device=device,
    )
    save_3d_outputs(
        frames,
        output_dir=args.output_dir,
        fps=args.fps,
        max_surface_points=args.max_surface_points,
        output_prefix="shallow_water_uv_surface_3d",
    )


if __name__ == "__main__":
    main()
