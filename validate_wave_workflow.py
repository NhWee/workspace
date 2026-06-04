import argparse
from pathlib import Path

import torch

from shallow_water_bathymetry_3d import compute_cfl_dt, make_bathymetry, simulate_bathymetry
from shallow_water_plotly_viewer import build_interactive_figure
from wave_dataset import load_wave_dataset, load_wave_dataset_with_velocity, save_wave_dataset


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_workflow(size: int, steps: int, frame_every: int, output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    depth_for_dt, _ = make_bathymetry(size, device)
    dx = 2.0 / (size - 1)
    dt = compute_cfl_dt(depth_for_dt, gravity=1.0, dx=dx, cfl=0.35)
    assert_condition(dt > 0.0, "CFL dt must be positive.")
    print(f"CFL dt: {dt:.6g}")

    frames, depth, u_frames, v_frames = simulate_bathymetry(
        size=size,
        steps=steps,
        frame_every=frame_every,
        gravity=1.0,
        dt=None,
        damping=0.9994,
        device=device,
        cfl=0.35,
        store_velocity=True,
    )
    expected_frames = (steps + frame_every - 1) // frame_every
    assert_condition(len(frames) == expected_frames, "Unexpected frame count.")
    assert_condition(frames[0].shape == (size, size), "Unexpected frame shape.")
    assert_condition(depth.shape == (size, size), "Unexpected depth shape.")
    assert_condition(torch.isfinite(frames[-1]).all().item(), "Frame contains non-finite values.")
    assert_condition(torch.isfinite(depth).all().item(), "Depth contains non-finite values.")
    assert_condition(len(u_frames) == len(frames), "Unexpected u velocity frame count.")
    assert_condition(len(v_frames) == len(frames), "Unexpected v velocity frame count.")
    assert_condition(torch.isfinite(u_frames[-1]).all().item(), "U velocity contains non-finite values.")
    assert_condition(torch.isfinite(v_frames[-1]).all().item(), "V velocity contains non-finite values.")
    print(f"Simulated frames: {len(frames)}")

    dataset_path = output_dir / "workflow_validation_dataset.npz"
    metadata = {
        "solver": "bathymetry_shallow_water_validation",
        "size": size,
        "steps": steps,
        "frame_every": frame_every,
        "device": str(device),
        "stores_velocity": True,
    }
    save_wave_dataset(dataset_path, frames, depth, metadata, u_frames=u_frames, v_frames=v_frames)

    loaded_frames, loaded_depth, loaded_metadata = load_wave_dataset(dataset_path)
    _, _, _, loaded_u_frames, loaded_v_frames = load_wave_dataset_with_velocity(dataset_path)
    assert_condition(len(loaded_frames) == len(frames), "Loaded frame count mismatch.")
    assert_condition(loaded_frames[0].shape == frames[0].shape, "Loaded frame shape mismatch.")
    assert_condition(loaded_depth.shape == depth.shape, "Loaded depth shape mismatch.")
    assert_condition(loaded_metadata["size"] == size, "Loaded metadata mismatch.")
    assert_condition(loaded_metadata["stores_velocity"] is True, "Loaded velocity metadata mismatch.")
    assert_condition(loaded_u_frames is not None and len(loaded_u_frames) == len(frames), "Loaded u velocity mismatch.")
    assert_condition(loaded_v_frames is not None and len(loaded_v_frames) == len(frames), "Loaded v velocity mismatch.")
    print(f"Reloaded dataset: {dataset_path}")

    fig = build_interactive_figure(loaded_frames, loaded_depth, max_surface_points=min(48, size))
    html_path = output_dir / "workflow_validation_viewer.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    html_text = html_path.read_text(encoding="utf-8")
    assert_condition("Plotly.newPlot" in html_text, "HTML is missing Plotly.newPlot.")
    assert_condition("Interactive bathymetry + GPU wave surface" in html_text, "HTML is missing viewer title.")
    print(f"Validated Plotly HTML: {html_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fast end-to-end validation of the wave workflow.")
    parser.add_argument("--size", type=int, default=96, help="Small validation grid size.")
    parser.add_argument("--steps", type=int, default=90, help="Validation simulation steps.")
    parser.add_argument("--frame-every", type=int, default=15, help="Frame interval for validation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_workflow(args.size, args.steps, args.frame_every, args.output_dir)
    print("Wave workflow validation passed.")


if __name__ == "__main__":
    main()
