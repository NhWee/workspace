import argparse
from pathlib import Path

import torch

from shallow_water_bathymetry_3d import simulate_bathymetry
from wave_dataset import save_wave_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GPU bathymetry wave frames to a reusable NPZ dataset.")
    parser.add_argument("--size", type=int, default=256, help="Simulation grid size.")
    parser.add_argument("--steps", type=int, default=720, help="Simulation steps.")
    parser.add_argument("--frame-every", type=int, default=12, help="Save one frame every N simulation steps.")
    parser.add_argument("--gravity", type=float, default=1.0, help="Gravity coefficient g.")
    parser.add_argument("--dt", default="auto", help="Time step, or 'auto' to use a CFL-based value.")
    parser.add_argument("--cfl", type=float, default=0.35, help="CFL factor used when --dt auto.")
    parser.add_argument("--damping", type=float, default=0.9994, help="Global damping per step.")
    parser.add_argument("--store-velocity", action="store_true", help="Store u/v velocity frames in the NPZ dataset.")
    parser.add_argument("--output", type=Path, default=Path("outputs/wave_dataset.npz"), help="Output NPZ path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    dt = None if str(args.dt).lower() == "auto" else float(args.dt)

    result = simulate_bathymetry(
        size=args.size,
        steps=args.steps,
        frame_every=args.frame_every,
        gravity=args.gravity,
        dt=dt,
        damping=args.damping,
        device=device,
        cfl=args.cfl,
        store_velocity=args.store_velocity,
    )
    if args.store_velocity:
        frames, depth, u_frames, v_frames = result
    else:
        frames, depth = result
        u_frames = None
        v_frames = None
    metadata = {
        "solver": "bathymetry_shallow_water",
        "size": args.size,
        "steps": args.steps,
        "frame_every": args.frame_every,
        "gravity": args.gravity,
        "dt": args.dt,
        "cfl": args.cfl,
        "damping": args.damping,
        "device": str(device),
        "frame_count": len(frames),
        "stores_velocity": args.store_velocity,
    }
    save_wave_dataset(args.output, frames, depth, metadata, u_frames=u_frames, v_frames=v_frames)


if __name__ == "__main__":
    main()
