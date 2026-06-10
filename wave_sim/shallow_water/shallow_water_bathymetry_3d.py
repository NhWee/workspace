# Shallow water wave simulation with variable bathymetry in 2D using PyTorch for GPU acceleration, rendered in 3D.
# You can handle the parameters
# size, steps, frame_every, gravity, dt, damping, fps
import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch

from wave_sim.shallow_water.shallow_water_2d import make_edge_damping
from wave_sim.shallow_water.shallow_water_surface_3d import save_3d_outputs
from wave_sim.shallow_water.shallow_water_uv_3d import gradient_x, gradient_y


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


def divergence_flux(flux_x: torch.Tensor, flux_y: torch.Tensor, dx: float) -> torch.Tensor:
    return gradient_x(flux_x, dx) + gradient_y(flux_y, dx)


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
    cfl: float = 0.35,
    store_velocity: bool = False,
) -> tuple[list[torch.Tensor], torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    dx = 2.0 / (size - 1)
    depth, wet_mask = make_bathymetry(size, device)
    if dt is None:
        dt = compute_cfl_dt(depth, gravity, dx, cfl)
    eta = make_incoming_wave(size, device) * wet_mask
    u = torch.zeros_like(eta)
    v = torch.zeros_like(eta)
    edge_damping = make_edge_damping(size, width=max(8, size // 12), strength=0.10, device=device)
    frames = []
    u_frames = []
    v_frames = []

    for step in range(steps):
        surface_gradient_x = gradient_x(eta, dx)
        surface_gradient_y = gradient_y(eta, dx)
        u_next = damping * (u - gravity * dt * surface_gradient_x)
        v_next = damping * (v - gravity * dt * surface_gradient_y)

        u_next = u_next * wet_mask
        v_next = v_next * wet_mask

        flux_x = depth * u_next
        flux_y = depth * v_next
        eta_next = damping * (eta - dt * divergence_flux(flux_x, flux_y, dx))

        u_next = u_next * edge_damping
        v_next = v_next * edge_damping
        eta_next = eta_next * edge_damping * wet_mask

        eta, u, v = eta_next, u_next, v_next

        if step % frame_every == 0:
            frames.append(eta.detach().cpu())
            if store_velocity:
                u_frames.append(u.detach().cpu())
                v_frames.append(v.detach().cpu())

    if store_velocity:
        return frames, depth.detach().cpu(), u_frames, v_frames
    return frames, depth.detach().cpu()


def save_bathymetry_map(depth: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shallow_water_bathymetry_depth.png"
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    image = ax.imshow(depth, cmap="viridis", origin="lower", interpolation="bilinear")
    ax.set_title("Bathymetry depth map")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="depth")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved depth map: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU shallow-water surface simulation with variable bathymetry.")
    parser.add_argument("--size", type=int, default=256, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=900, help="Simulation steps.")
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
    save_3d_outputs(
        frames,
        output_dir=args.output_dir,
        fps=args.fps,
        max_surface_points=args.max_surface_points,
        output_prefix="shallow_water_bathymetry_surface_3d",
    )
    save_bathymetry_map(depth, args.output_dir)


if __name__ == "__main__":
    main()
