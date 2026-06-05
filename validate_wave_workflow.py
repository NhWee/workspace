import argparse
from pathlib import Path

import torch

from compare_wave_datasets import (
    load_dataset_summary,
    make_markdown_table,
    save_final_frame_difference_heatmaps,
    save_frame_metric_chart,
    save_frame_metric_series,
    write_summary,
)
from shallow_water_bathymetry_3d import compute_cfl_dt, make_bathymetry, simulate_bathymetry
from shallow_water_particle_animation_viewer import build_particle_animation_figure
from shallow_water_particle_viewer import bilinear_sample, make_particle_seeds, make_wet_mask, trace_particles
from shallow_water_plotly_viewer import build_interactive_figure
from shallow_water_streamline_viewer import build_streamline_figure, trace_streamlines
from sweep_wave_experiments import parse_float_list, run_sweep
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
    assert_condition("created_at_utc" in loaded_metadata, "Loaded metadata is missing created_at_utc.")
    assert_condition("git_commit" in loaded_metadata, "Loaded metadata is missing git_commit.")
    assert_condition("git_is_dirty" in loaded_metadata, "Loaded metadata is missing git_is_dirty.")
    assert_condition(loaded_u_frames is not None and len(loaded_u_frames) == len(frames), "Loaded u velocity mismatch.")
    assert_condition(loaded_v_frames is not None and len(loaded_v_frames) == len(frames), "Loaded v velocity mismatch.")
    print(f"Reloaded dataset: {dataset_path}")

    dataset_summary = load_dataset_summary(dataset_path)
    assert_condition(dataset_summary["frame_count"] == len(frames), "Dataset summary frame count mismatch.")
    assert_condition(dataset_summary["stores_velocity"] is True, "Dataset summary velocity mismatch.")
    comparison_path = output_dir / "workflow_validation_dataset_comparison.md"
    duplicate_summary = load_dataset_summary(dataset_path)
    heatmap_dir = output_dir / "workflow_validation_diff_heatmaps"
    heatmap_paths = save_final_frame_difference_heatmaps([dataset_summary, duplicate_summary], heatmap_dir)
    assert_condition(len(heatmap_paths) == 1, "Expected one validation difference heatmap.")
    assert_condition(heatmap_paths[0].exists(), "Validation difference heatmap was not created.")
    frame_metrics_dir = output_dir / "workflow_validation_frame_metrics"
    frame_metric_paths = save_frame_metric_series([dataset_summary, duplicate_summary], frame_metrics_dir)
    assert_condition(len(frame_metric_paths) == 1, "Expected one validation frame metrics CSV.")
    assert_condition(frame_metric_paths[0].exists(), "Validation frame metrics CSV was not created.")
    frame_metric_text = frame_metric_paths[0].read_text(encoding="utf-8")
    assert_condition("frame_index,l2_vs_baseline,linf_vs_baseline" in frame_metric_text, "Frame metrics CSV header mismatch.")
    frame_metric_chart_path = output_dir / "workflow_validation_frame_metrics.html"
    save_frame_metric_chart([dataset_summary, duplicate_summary], frame_metric_chart_path)
    frame_metric_chart_text = frame_metric_chart_path.read_text(encoding="utf-8")
    assert_condition("Frame-wise wave dataset difference vs baseline" in frame_metric_chart_text, "Frame metrics chart title missing.")
    assert_condition("Plotly.newPlot" in frame_metric_chart_text, "Frame metrics chart is missing Plotly.newPlot.")
    assert_condition(duplicate_summary["final_l2_vs_baseline"] == 0.0, "Duplicate final L2 metric must be zero.")
    comparison_table = make_markdown_table([dataset_summary, duplicate_summary])
    assert_condition("frame_count" in comparison_table, "Dataset comparison table is missing frame_count.")
    assert_condition("final_l2_vs_baseline" in comparison_table, "Dataset comparison table is missing final L2 metric.")
    assert_condition("final_diff_heatmap" in comparison_table, "Dataset comparison table is missing heatmap column.")
    assert_condition("frames_l2_mean_vs_baseline" in comparison_table, "Dataset comparison table is missing frame L2 metric.")
    assert_condition("frame_metrics_csv" in comparison_table, "Dataset comparison table is missing frame metrics CSV column.")
    assert_condition(dataset_summary["final_l2_vs_baseline"] == 0.0, "Baseline final L2 metric must be zero.")
    assert_condition(dataset_summary["final_linf_vs_baseline"] == 0.0, "Baseline final Linf metric must be zero.")
    assert_condition(duplicate_summary["frames_l2_mean_vs_baseline"] == 0.0, "Duplicate frame L2 metric must be zero.")
    assert_condition(duplicate_summary["frames_linf_max_vs_baseline"] == 0.0, "Duplicate frame Linf metric must be zero.")
    write_summary(comparison_path, comparison_table)
    print(f"Validated dataset comparison summary: {comparison_path}")

    sweep_manifest = run_sweep(
        argparse.Namespace(
            experiment_name="workflow_validation_sweep",
            parameter="damping",
            values=parse_float_list("0.9993,0.9994"),
            size=max(32, size // 2),
            steps=max(24, steps // 3),
            frame_every=max(6, frame_every // 2),
            gravity=1.0,
            dt="auto",
            cfl=0.35,
            damping=0.9994,
            store_velocity=False,
            output_dir=output_dir / "workflow_validation_sweeps",
        )
    )
    sweep_outputs = sweep_manifest["outputs"]
    assert_condition(len(sweep_manifest["runs"]) == 2, "Sweep should create two validation runs.")
    assert_condition(Path(sweep_outputs["comparison"]).exists(), "Sweep comparison was not created.")
    assert_condition(Path(sweep_outputs["frame_metrics_chart"]).exists(), "Sweep frame metrics chart was not created.")
    assert_condition(len(sweep_outputs["diff_heatmaps"]) == 1, "Sweep should create one validation heatmap.")
    assert_condition(len(sweep_outputs["frame_metrics_csv"]) == 1, "Sweep should create one validation frame metrics CSV.")
    assert_condition(
        Path(sweep_manifest["runs"][0]["dataset_path"]).exists(),
        "Sweep baseline dataset was not created.",
    )
    print("Validated parameter sweep workflow.")

    wet_mask = make_wet_mask(loaded_depth, wet_depth_threshold=0.055)
    seed_x, seed_y = make_particle_seeds(3, 4)
    custom_seed_x, custom_seed_y = make_particle_seeds(2, 3, x_min=-0.75, x_max=-0.65, y_min=-0.30, y_max=0.30)
    assert_condition(custom_seed_x.shape == (6,), "Custom seed x shape mismatch.")
    assert_condition(custom_seed_y.shape == (6,), "Custom seed y shape mismatch.")
    assert_condition(abs(float(custom_seed_x.min()) + 0.75) < 1.0e-12, "Custom seed x min mismatch.")
    assert_condition(abs(float(custom_seed_x.max()) + 0.65) < 1.0e-12, "Custom seed x max mismatch.")
    assert_condition(abs(float(custom_seed_y.min()) + 0.30) < 1.0e-12, "Custom seed y min mismatch.")
    assert_condition(abs(float(custom_seed_y.max()) - 0.30) < 1.0e-12, "Custom seed y max mismatch.")
    print("Validated custom particle seed ranges.")
    for integrator in ("euler", "rk2"):
        paths_x, paths_y, _ = trace_particles(
            loaded_frames,
            loaded_u_frames,
            loaded_v_frames,
            seed_x,
            seed_y,
            step_scale=0.55,
            depth=loaded_depth,
            wet_depth_threshold=0.055,
            block_dry_cells=True,
            integrator=integrator,
        )
        assert_condition(paths_x.shape == (len(loaded_frames), len(seed_x)), f"{integrator} particle shape mismatch.")
        sampled_wet = bilinear_sample(wet_mask, paths_x.reshape(-1), paths_y.reshape(-1))
        assert_condition(float(sampled_wet.min()) >= 0.5, f"{integrator} particle path entered a dry cell.")
    print("Validated Euler/RK2 wet/dry particle clipping.")

    streamline_x, streamline_y, _ = trace_streamlines(
        loaded_frames[-1],
        loaded_depth,
        loaded_u_frames[-1],
        loaded_v_frames[-1],
        custom_seed_x,
        custom_seed_y,
        streamline_steps=8,
        step_scale=0.18,
        wet_depth_threshold=0.055,
        block_dry_cells=True,
        integrator="rk2",
    )
    sampled_streamline_wet = bilinear_sample(wet_mask, streamline_x.reshape(-1), streamline_y.reshape(-1))
    assert_condition(streamline_x.shape == (9, len(custom_seed_x)), "Streamline shape mismatch.")
    assert_condition(float(sampled_streamline_wet.min()) >= 0.5, "Streamline entered a dry cell.")
    print("Validated streamline tracing.")

    fig = build_interactive_figure(loaded_frames, loaded_depth, max_surface_points=min(48, size))
    html_path = output_dir / "workflow_validation_viewer.html"
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    html_text = html_path.read_text(encoding="utf-8")
    assert_condition("Plotly.newPlot" in html_text, "HTML is missing Plotly.newPlot.")
    assert_condition("Interactive bathymetry + GPU wave surface" in html_text, "HTML is missing viewer title.")
    print(f"Validated Plotly HTML: {html_path}")

    particle_html_path = output_dir / "workflow_validation_particle_animation_viewer.html"
    particle_fig = build_particle_animation_figure(
        loaded_frames,
        loaded_depth,
        loaded_u_frames,
        loaded_v_frames,
        max_surface_points=min(48, size),
        seed_count_x=3,
        seed_count_y=4,
        seed_x_min=-0.88,
        seed_x_max=-0.58,
        seed_y_min=-0.62,
        seed_y_max=0.62,
        particle_step_scale=0.55,
        wet_depth_threshold=0.055,
        block_dry_cells=True,
        particle_integrator="rk2",
        trail_length=4,
        frame_duration_ms=90,
    )
    particle_fig.write_html(particle_html_path, include_plotlyjs=True, full_html=True)
    particle_html_text = particle_html_path.read_text(encoding="utf-8")
    assert_condition("Animated particles over speed-colored 3D wave surface" in particle_html_text, "Particle HTML is missing viewer title.")
    assert_condition("particle trails" in particle_html_text, "Particle HTML is missing trails trace.")
    assert_condition("particles" in particle_html_text, "Particle HTML is missing marker trace.")
    print(f"Validated particle animation HTML: {particle_html_path}")

    streamline_html_path = output_dir / "workflow_validation_streamline_viewer.html"
    streamline_fig = build_streamline_figure(
        loaded_frames,
        loaded_depth,
        loaded_u_frames,
        loaded_v_frames,
        max_surface_points=min(48, size),
        frame_index=-1,
        seed_count_x=3,
        seed_count_y=4,
        seed_x_min=-0.75,
        seed_x_max=-0.65,
        seed_y_min=-0.30,
        seed_y_max=0.30,
        streamline_steps=8,
        streamline_step_scale=0.18,
        wet_depth_threshold=0.055,
        block_dry_cells=True,
        particle_integrator="rk2",
    )
    streamline_fig.write_html(streamline_html_path, include_plotlyjs=True, full_html=True)
    streamline_html_text = streamline_html_path.read_text(encoding="utf-8")
    assert_condition("Streamlines over speed-colored 3D wave surface" in streamline_html_text, "Streamline HTML is missing viewer title.")
    assert_condition("streamlines" in streamline_html_text, "Streamline HTML is missing streamlines trace.")
    print(f"Validated streamline HTML: {streamline_html_path}")


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
