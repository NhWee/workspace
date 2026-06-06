import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_spectral_wave import benchmark_size as benchmark_spectral_size, save_benchmark_chart
from compare_solver_benchmarks import run_comparison as run_solver_benchmark_comparison
from compare_wave_datasets import (
    load_dataset_summary,
    make_markdown_table,
    save_final_frame_difference_heatmaps,
    save_frame_metric_chart,
    save_frame_metric_series,
    write_summary,
)
from compare_spectral_choppy_asset_bundles import write_comparison as write_asset_bundle_comparison
from export_spectral_choppy_asset_bundle import write_asset_bundle
from evaluate_spectral_choppy_wave import evaluate_choppy_frames, write_metric_outputs
from export_spectral_choppy_mesh import (
    write_foam_ply,
    write_foam_sequence,
    write_metadata,
    write_obj_mesh,
    write_obj_sequence,
)
from export_spectral_choppy_gltf import (
    write_animated_glb_scene,
    write_animated_gltf_scene,
    write_glb_scene,
    write_glb_sequence,
    write_gltf_scene,
    write_gltf_sequence,
)
from report_spectral_choppy_asset_bundle import write_report as write_asset_bundle_report
from shallow_water_bathymetry_3d import compute_cfl_dt, make_bathymetry, simulate_bathymetry
from shallow_water_particle_animation_viewer import build_particle_animation_figure
from shallow_water_particle_viewer import bilinear_sample, make_particle_seeds, make_wet_mask, trace_particles
from shallow_water_plotly_viewer import build_interactive_figure
from shallow_water_streamline_viewer import build_streamline_figure, trace_streamlines
from shallow_water_velocity_viewer import build_velocity_figure
from spectral_choppy_wave_viewer import build_choppy_figure, simulate_choppy_frames
from spectral_wave_surface_3d import simulate_spectral_wave
from sweep_spectral_choppy_asset_bundles import run_asset_bundle_sweep
from sweep_wave_experiments import parse_float_list, run_sweep
from validate_spectral_choppy_asset_bundle import validate_bundle, write_validation_outputs
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
    assert_condition(sweep_manifest["runs"][0]["elapsed_sec"] > 0.0, "Sweep elapsed time must be positive.")
    assert_condition(sweep_manifest["runs"][0]["steps_per_sec"] > 0.0, "Sweep steps/sec must be positive.")
    assert_condition(
        sweep_manifest["runs"][0]["million_cell_steps_per_sec"] > 0.0,
        "Sweep throughput must be positive.",
    )
    assert_condition("peak_vram_gib" in sweep_manifest["runs"][0], "Sweep peak VRAM metric is missing.")
    assert_condition(Path(sweep_outputs["comparison"]).exists(), "Sweep comparison was not created.")
    assert_condition(Path(sweep_outputs["dashboard"]).exists(), "Sweep dashboard was not created.")
    assert_condition(Path(sweep_outputs["frame_metrics_chart"]).exists(), "Sweep frame metrics chart was not created.")
    assert_condition(Path(sweep_outputs["performance_chart"]).exists(), "Sweep performance chart was not created.")
    assert_condition(len(sweep_outputs["diff_heatmaps"]) == 1, "Sweep should create one validation heatmap.")
    assert_condition(len(sweep_outputs["frame_metrics_csv"]) == 1, "Sweep should create one validation frame metrics CSV.")
    sweep_dashboard_text = Path(sweep_outputs["dashboard"]).read_text(encoding="utf-8")
    assert_condition("Wave Sweep Dashboard" in sweep_dashboard_text, "Sweep dashboard title is missing.")
    assert_condition("Open frame metric chart" in sweep_dashboard_text, "Sweep dashboard chart link is missing.")
    assert_condition("Open performance chart" in sweep_dashboard_text, "Sweep dashboard performance link is missing.")
    assert_condition("million_cell_steps_per_sec" in sweep_dashboard_text, "Sweep dashboard throughput metric is missing.")
    sweep_performance_text = Path(sweep_outputs["performance_chart"]).read_text(encoding="utf-8")
    assert_condition("Wave sweep performance" in sweep_performance_text, "Sweep performance chart title is missing.")
    assert_condition("Plotly.newPlot" in sweep_performance_text, "Sweep performance chart is missing Plotly.newPlot.")
    sweep_comparison_text = Path(sweep_outputs["comparison"]).read_text(encoding="utf-8")
    assert_condition("elapsed_sec" in sweep_comparison_text, "Sweep comparison elapsed metric is missing.")
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

    spectral_result = simulate_spectral_wave(
        size=max(32, size // 2),
        steps=max(24, steps // 3),
        frame_every=max(6, frame_every // 2),
        domain_size=8.0,
        gravity=9.81,
        dt=0.04,
        wave_amplitude=0.06,
        peak_wavelength=1.2,
        bandwidth=0.32,
        wind_direction_degrees=25.0,
        directional_spread=6.0,
        damping=0.9995,
        seed=11,
        device=device,
        store_velocity=True,
    )
    spectral_frames, spectral_depth, spectral_u_frames, spectral_v_frames = spectral_result
    assert_condition(len(spectral_frames) > 0, "Spectral wave produced no frames.")
    assert_condition(torch.isfinite(spectral_frames[-1]).all().item(), "Spectral wave frame contains non-finite values.")
    assert_condition(spectral_depth.shape == spectral_frames[0].shape, "Spectral wave depth shape mismatch.")
    assert_condition(len(spectral_u_frames) == len(spectral_frames), "Spectral u velocity frame count mismatch.")
    assert_condition(len(spectral_v_frames) == len(spectral_frames), "Spectral v velocity frame count mismatch.")
    assert_condition(torch.isfinite(spectral_u_frames[-1]).all().item(), "Spectral u velocity contains non-finite values.")
    assert_condition(torch.isfinite(spectral_v_frames[-1]).all().item(), "Spectral v velocity contains non-finite values.")
    spectral_dataset_path = output_dir / "workflow_validation_spectral_dataset.npz"
    save_wave_dataset(
        spectral_dataset_path,
        spectral_frames,
        spectral_depth,
        {
            "solver": "spectral_wave_surface_validation",
            "size": spectral_frames[0].shape[0],
            "steps": max(24, steps // 3),
            "frame_every": max(6, frame_every // 2),
            "device": str(device),
            "stores_velocity": True,
        },
        u_frames=spectral_u_frames,
        v_frames=spectral_v_frames,
    )
    spectral_loaded_frames, spectral_loaded_depth, spectral_metadata, spectral_loaded_u, spectral_loaded_v = (
        load_wave_dataset_with_velocity(spectral_dataset_path)
    )
    assert_condition(spectral_metadata["solver"] == "spectral_wave_surface_validation", "Spectral metadata mismatch.")
    assert_condition(spectral_metadata["stores_velocity"] is True, "Spectral velocity metadata mismatch.")
    assert_condition(spectral_loaded_depth.shape == spectral_loaded_frames[0].shape, "Loaded spectral depth shape mismatch.")
    assert_condition(spectral_loaded_u is not None and len(spectral_loaded_u) == len(spectral_loaded_frames), "Loaded spectral u velocity mismatch.")
    assert_condition(spectral_loaded_v is not None and len(spectral_loaded_v) == len(spectral_loaded_frames), "Loaded spectral v velocity mismatch.")
    spectral_fig = build_interactive_figure(spectral_loaded_frames, spectral_loaded_depth, max_surface_points=min(48, size))
    spectral_fig.update_layout(title="Interactive GPU FFT spectral wave surface")
    spectral_html_path = output_dir / "workflow_validation_spectral_viewer.html"
    spectral_fig.write_html(spectral_html_path, include_plotlyjs=True, full_html=True)
    spectral_html_text = spectral_html_path.read_text(encoding="utf-8")
    assert_condition("Interactive GPU FFT spectral wave surface" in spectral_html_text, "Spectral HTML title missing.")
    assert_condition("Plotly.newPlot" in spectral_html_text, "Spectral HTML is missing Plotly.newPlot.")
    spectral_velocity_fig = build_velocity_figure(
        spectral_loaded_frames,
        spectral_loaded_depth,
        spectral_loaded_u,
        spectral_loaded_v,
        max_surface_points=min(48, size),
    )
    spectral_velocity_fig.update_layout(title="GPU FFT spectral wave surface colored by speed")
    spectral_velocity_html_path = output_dir / "workflow_validation_spectral_velocity_viewer.html"
    spectral_velocity_fig.write_html(spectral_velocity_html_path, include_plotlyjs=True, full_html=True)
    spectral_velocity_html_text = spectral_velocity_html_path.read_text(encoding="utf-8")
    assert_condition("GPU FFT spectral wave surface colored by speed" in spectral_velocity_html_text, "Spectral velocity HTML title missing.")
    assert_condition("Plotly.newPlot" in spectral_velocity_html_text, "Spectral velocity HTML is missing Plotly.newPlot.")
    print(f"Validated spectral wave dataset and viewer: {spectral_html_path}")

    choppy_frames = simulate_choppy_frames(
        size=max(32, size // 2),
        steps=max(24, steps // 3),
        frame_every=max(6, frame_every // 2),
        domain_size=8.0,
        gravity=9.81,
        dt=0.04,
        wave_amplitude=0.06,
        peak_wavelength=1.2,
        bandwidth=0.32,
        wind_direction_degrees=25.0,
        directional_spread=6.0,
        damping=0.9995,
        seed=19,
        choppiness=0.7,
        max_surface_points=min(48, size),
        device=device,
    )
    assert_condition(len(choppy_frames) > 0, "Choppy spectral viewer produced no frames.")
    choppy_x, choppy_y, choppy_z = choppy_frames[-1]
    assert_condition(choppy_x.shape == choppy_y.shape == choppy_z.shape, "Choppy frame shape mismatch.")
    assert_condition(bool(np.isfinite(choppy_z).all()), "Choppy frame contains non-finite eta values.")
    choppy_fig = build_choppy_figure(choppy_frames, domain_size=8.0)
    choppy_html_path = output_dir / "workflow_validation_spectral_choppy_viewer.html"
    choppy_fig.write_html(choppy_html_path, include_plotlyjs=True, full_html=True)
    choppy_html_text = choppy_html_path.read_text(encoding="utf-8")
    assert_condition("Interactive GPU FFT choppy wave surface" in choppy_html_text, "Choppy HTML title missing.")
    assert_condition("Plotly.newPlot" in choppy_html_text, "Choppy HTML is missing Plotly.newPlot.")
    assert_condition("foam highlights" in choppy_html_text, "Choppy HTML is missing foam highlights trace.")
    print(f"Validated spectral choppy wave viewer: {choppy_html_path}")

    choppy_metrics = evaluate_choppy_frames(choppy_frames, domain_size=8.0, foam_threshold=0.0)
    choppy_metric_outputs = write_metric_outputs(choppy_metrics, output_dir / "workflow_validation_spectral_choppy_metrics")
    choppy_metric_report = Path(choppy_metric_outputs["report"]).read_text(encoding="utf-8")
    assert_condition(choppy_metric_outputs["summary"]["frame_count"] == len(choppy_frames), "Choppy metrics frame count mismatch.")
    assert_condition(choppy_metric_outputs["summary"]["steepness_max_max"] > 0.0, "Choppy metrics steepness max should be positive.")
    assert_condition(choppy_metric_outputs["summary"]["folded_triangle_ratio_max"] <= 0.01, "Choppy metrics folded triangle ratio is unexpectedly high.")
    assert_condition("horizontal_displacement_p95_max" in choppy_metric_report, "Choppy metrics report displacement metric missing.")
    assert_condition("folded_triangle_ratio_max" in choppy_metric_report, "Choppy metrics report fold metric missing.")
    assert_condition(Path(choppy_metric_outputs["csv"]).exists(), "Choppy metrics CSV missing.")
    assert_condition(Path(choppy_metric_outputs["json"]).exists(), "Choppy metrics JSON missing.")

    choppy_mesh_path = output_dir / "workflow_validation_spectral_choppy_mesh.obj"
    choppy_mesh_summary = write_obj_mesh(choppy_mesh_path, choppy_x, choppy_y, choppy_z)
    choppy_sequence_dir = output_dir / "workflow_validation_spectral_choppy_mesh_sequence"
    choppy_sequence_summary = write_obj_sequence(choppy_sequence_dir, choppy_frames)
    choppy_foam_path = output_dir / "workflow_validation_spectral_choppy_foam.ply"
    choppy_foam_summary = write_foam_ply(
        choppy_foam_path,
        choppy_x,
        choppy_y,
        choppy_z,
        foam_threshold=0.0,
        max_foam_points=64,
        z_offset=0.01,
    )
    choppy_foam_sequence_dir = output_dir / "workflow_validation_spectral_choppy_foam_sequence"
    choppy_foam_sequence_summary = write_foam_sequence(
        choppy_foam_sequence_dir,
        choppy_frames,
        foam_threshold=0.0,
        max_foam_points=64,
        z_offset=0.01,
    )
    choppy_gltf_path = output_dir / "workflow_validation_spectral_choppy_scene.gltf"
    choppy_gltf_summary = write_gltf_scene(
        choppy_gltf_path,
        choppy_x,
        choppy_y,
        choppy_z,
        foam_threshold=0.0,
        max_foam_points=64,
        foam_z_offset=0.01,
    )
    choppy_glb_path = output_dir / "workflow_validation_spectral_choppy_scene.glb"
    choppy_glb_summary = write_glb_scene(
        choppy_glb_path,
        choppy_x,
        choppy_y,
        choppy_z,
        foam_threshold=0.0,
        max_foam_points=64,
        foam_z_offset=0.01,
    )
    choppy_gltf_sequence_dir = output_dir / "workflow_validation_spectral_choppy_gltf_sequence"
    choppy_gltf_sequence_summary = write_gltf_sequence(
        choppy_gltf_sequence_dir,
        choppy_frames,
        foam_threshold=0.0,
        max_foam_points=64,
        foam_z_offset=0.01,
    )
    choppy_glb_sequence_dir = output_dir / "workflow_validation_spectral_choppy_glb_sequence"
    choppy_glb_sequence_summary = write_glb_sequence(
        choppy_glb_sequence_dir,
        choppy_frames,
        foam_threshold=0.0,
        max_foam_points=64,
        foam_z_offset=0.01,
    )
    choppy_animated_gltf_path = output_dir / "workflow_validation_spectral_choppy_animated.gltf"
    choppy_animated_gltf_summary = write_animated_gltf_scene(
        choppy_animated_gltf_path,
        choppy_frames,
        frame_duration=0.48,
    )
    choppy_animated_glb_path = output_dir / "workflow_validation_spectral_choppy_animated.glb"
    choppy_animated_glb_summary = write_animated_glb_scene(
        choppy_animated_glb_path,
        choppy_frames,
        frame_duration=0.48,
    )
    choppy_metadata_path = output_dir / "workflow_validation_spectral_choppy_mesh.json"
    write_metadata(
        choppy_metadata_path,
        choppy_mesh_summary,
        {
            "size": max(32, size // 2),
            "steps": max(24, steps // 3),
            "frame_every": max(6, frame_every // 2),
            "domain_size": 8.0,
            "choppiness": 0.7,
        },
        device,
        choppy_sequence_summary,
        choppy_foam_summary,
        choppy_foam_sequence_summary,
    )
    choppy_mesh_text = choppy_mesh_path.read_text(encoding="utf-8")
    choppy_sequence_frame_path = choppy_sequence_dir / "frame_0000.obj"
    choppy_sequence_text = choppy_sequence_frame_path.read_text(encoding="utf-8")
    choppy_foam_text = choppy_foam_path.read_text(encoding="utf-8")
    choppy_foam_sequence_frame_path = choppy_foam_sequence_dir / "foam_0000.ply"
    choppy_foam_sequence_text = choppy_foam_sequence_frame_path.read_text(encoding="utf-8")
    choppy_gltf = json.loads(choppy_gltf_path.read_text(encoding="utf-8"))
    choppy_glb_header = choppy_glb_path.read_bytes()[:12]
    choppy_gltf_sequence_manifest = json.loads((choppy_gltf_sequence_dir / "sequence_manifest.json").read_text(encoding="utf-8"))
    choppy_glb_sequence_manifest = json.loads((choppy_glb_sequence_dir / "sequence_manifest.json").read_text(encoding="utf-8"))
    choppy_glb_sequence_header = (choppy_glb_sequence_dir / "frame_0000.glb").read_bytes()[:12]
    choppy_animated_gltf = json.loads(choppy_animated_gltf_path.read_text(encoding="utf-8"))
    choppy_animated_glb_header = choppy_animated_glb_path.read_bytes()[:12]
    choppy_metadata_text = choppy_metadata_path.read_text(encoding="utf-8")
    assert_condition("\nv " in choppy_mesh_text, "Choppy OBJ mesh is missing vertices.")
    assert_condition("\nvn " in choppy_mesh_text, "Choppy OBJ mesh is missing vertex normals.")
    assert_condition("\nf " in choppy_mesh_text, "Choppy OBJ mesh is missing faces.")
    assert_condition("\nv " in choppy_sequence_text, "Choppy OBJ sequence frame is missing vertices.")
    assert_condition("\nvn " in choppy_sequence_text, "Choppy OBJ sequence frame is missing vertex normals.")
    assert_condition("\nf " in choppy_sequence_text, "Choppy OBJ sequence frame is missing faces.")
    assert_condition("spectral_choppy_mesh" in choppy_metadata_text, "Choppy mesh metadata solver marker missing.")
    assert_condition("\"sequence\"" in choppy_metadata_text, "Choppy mesh metadata sequence marker missing.")
    assert_condition("\"foam\"" in choppy_metadata_text, "Choppy mesh metadata foam marker missing.")
    assert_condition("\"foam_sequence\"" in choppy_metadata_text, "Choppy mesh metadata foam sequence marker missing.")
    assert_condition("property float steepness" in choppy_foam_text, "Choppy foam PLY steepness property missing.")
    assert_condition("property float steepness" in choppy_foam_sequence_text, "Choppy foam sequence PLY steepness property missing.")
    assert_condition(choppy_mesh_summary["vertex_count"] > 0, "Choppy OBJ mesh has no vertices.")
    assert_condition(
        choppy_mesh_summary["normal_count"] == choppy_mesh_summary["vertex_count"],
        "Choppy OBJ mesh normal count mismatch.",
    )
    assert_condition(choppy_mesh_summary["quad_face_count"] > 0, "Choppy OBJ mesh has no faces.")
    assert_condition(
        choppy_sequence_summary["frame_count"] == len(choppy_frames),
        "Choppy OBJ sequence frame count mismatch.",
    )
    assert_condition(choppy_foam_summary["point_count"] > 0, "Choppy foam PLY has no points.")
    assert_condition(
        choppy_foam_sequence_summary["frame_count"] == len(choppy_frames),
        "Choppy foam sequence frame count mismatch.",
    )
    assert_condition(choppy_gltf["asset"]["version"] == "2.0", "Choppy glTF version mismatch.")
    assert_condition(choppy_gltf["buffers"][0]["uri"].startswith("data:application/octet-stream;base64,"), "Choppy glTF buffer is not embedded.")
    assert_condition(len(choppy_gltf["meshes"][0]["primitives"]) == 2, "Choppy glTF should contain water and foam primitives.")
    assert_condition(choppy_gltf_summary["triangle_count"] > 0, "Choppy glTF has no triangles.")
    assert_condition(choppy_gltf_summary["foam_point_count"] > 0, "Choppy glTF has no foam points.")
    assert_condition(choppy_glb_header[:4] == b"glTF", "Choppy GLB magic header mismatch.")
    assert_condition(int.from_bytes(choppy_glb_header[4:8], "little") == 2, "Choppy GLB version mismatch.")
    assert_condition(choppy_glb_summary["glb_length"] == choppy_glb_path.stat().st_size, "Choppy GLB length mismatch.")
    assert_condition(choppy_glb_summary["triangle_count"] > 0, "Choppy GLB has no triangles.")
    assert_condition(choppy_gltf_sequence_summary["frame_count"] == len(choppy_frames), "Choppy glTF sequence frame count mismatch.")
    assert_condition(choppy_gltf_sequence_manifest["format"] == "embedded_gltf_sequence", "Choppy glTF sequence manifest format mismatch.")
    assert_condition((choppy_gltf_sequence_dir / "frame_0000.gltf").exists(), "Choppy glTF sequence first frame missing.")
    assert_condition(choppy_glb_sequence_summary["frame_count"] == len(choppy_frames), "Choppy GLB sequence frame count mismatch.")
    assert_condition(choppy_glb_sequence_manifest["format"] == "glb_sequence", "Choppy GLB sequence manifest format mismatch.")
    assert_condition(choppy_glb_sequence_header[:4] == b"glTF", "Choppy GLB sequence first frame magic header mismatch.")
    assert_condition(int.from_bytes(choppy_glb_sequence_header[4:8], "little") == 2, "Choppy GLB sequence first frame version mismatch.")
    assert_condition(choppy_animated_gltf_summary["frame_count"] == len(choppy_frames), "Choppy animated glTF frame count mismatch.")
    assert_condition(choppy_animated_gltf_summary["morph_target_count"] == len(choppy_frames) - 1, "Choppy animated glTF morph target count mismatch.")
    assert_condition(choppy_animated_gltf_summary["foam_point_capacity"] == 900, "Choppy animated glTF foam capacity mismatch.")
    assert_condition(choppy_animated_gltf["animations"][0]["channels"][0]["target"]["path"] == "weights", "Choppy animated glTF should animate morph weights.")
    assert_condition(len(choppy_animated_gltf["meshes"][0]["primitives"]) == 2, "Choppy animated glTF should contain water and foam primitives.")
    assert_condition("targets" in choppy_animated_gltf["meshes"][0]["primitives"][0], "Choppy animated glTF morph targets missing.")
    assert_condition("targets" in choppy_animated_gltf["meshes"][0]["primitives"][1], "Choppy animated glTF foam morph targets missing.")
    assert_condition(choppy_animated_glb_header[:4] == b"glTF", "Choppy animated GLB magic header mismatch.")
    assert_condition(int.from_bytes(choppy_animated_glb_header[4:8], "little") == 2, "Choppy animated GLB version mismatch.")
    assert_condition(choppy_animated_glb_summary["glb_length"] == choppy_animated_glb_path.stat().st_size, "Choppy animated GLB length mismatch.")
    choppy_bundle_dir = output_dir / "workflow_validation_spectral_choppy_asset_bundle"
    choppy_bundle_manifest = write_asset_bundle(
        choppy_bundle_dir,
        choppy_frames,
        domain_size=8.0,
        simulation_parameters={
            "size": max(32, size // 2),
            "steps": max(24, steps // 3),
            "frame_every": max(6, frame_every // 2),
            "domain_size": 8.0,
            "dt": 0.04,
            "source": "workflow_validation",
        },
        device=device,
        foam_threshold=0.0,
        max_foam_points=64,
        foam_z_offset=0.01,
    )
    assert_condition(choppy_bundle_manifest["frame_count"] == len(choppy_frames), "Choppy asset bundle frame count mismatch.")
    assert_condition((choppy_bundle_dir / "bundle_manifest.json").exists(), "Choppy asset bundle manifest missing.")
    assert_condition((choppy_bundle_dir / "viewer.html").exists(), "Choppy asset bundle viewer missing.")
    assert_condition((choppy_bundle_dir / "final.glb").exists(), "Choppy asset bundle final GLB missing.")
    assert_condition((choppy_bundle_dir / "glb_sequence" / "frame_0000.glb").exists(), "Choppy asset bundle GLB sequence missing.")
    assert_condition((choppy_bundle_dir / "animated.glb").exists(), "Choppy asset bundle animated GLB missing.")
    assert_condition(choppy_bundle_manifest["animated_glb"]["morph_target_count"] == len(choppy_frames) - 1, "Choppy asset bundle animated GLB morph target mismatch.")
    assert_condition("metrics" in choppy_bundle_manifest, "Choppy asset bundle metrics missing.")
    assert_condition((choppy_bundle_dir / "choppy_wave_metrics.csv").exists(), "Choppy asset bundle metric CSV missing.")
    assert_condition(choppy_bundle_manifest["metrics"]["summary"]["frame_count"] == len(choppy_frames), "Choppy asset bundle metric frame count mismatch.")
    assert_condition("folded_triangle_ratio_max" in choppy_bundle_manifest["metrics"]["summary"], "Choppy asset bundle fold metric missing.")
    choppy_bundle_report_path = write_asset_bundle_report(choppy_bundle_dir)
    choppy_bundle_report_text = choppy_bundle_report_path.read_text(encoding="utf-8")
    assert_condition("Spectral Choppy Wave Asset Bundle Report" in choppy_bundle_report_text, "Choppy asset bundle report title missing.")
    assert_condition("## Metrics" in choppy_bundle_report_text, "Choppy asset bundle report metrics section missing.")
    assert_condition("final GLB" in choppy_bundle_report_text, "Choppy asset bundle report final GLB row missing.")
    assert_condition("GLB sequence" in choppy_bundle_report_text, "Choppy asset bundle report GLB sequence row missing.")
    assert_condition("animated GLB" in choppy_bundle_report_text, "Choppy asset bundle report animated GLB row missing.")
    assert_condition("missing_assets: `0`" in choppy_bundle_report_text, "Choppy asset bundle report found missing assets.")
    choppy_bundle_validation = validate_bundle(choppy_bundle_dir)
    choppy_bundle_validation_outputs = write_validation_outputs(choppy_bundle_validation, choppy_bundle_dir)
    choppy_bundle_validation_text = Path(choppy_bundle_validation_outputs["report"]).read_text(encoding="utf-8")
    assert_condition(choppy_bundle_validation["passed"], "Choppy asset bundle validation failed.")
    assert_condition("animated GLB weight animation" in choppy_bundle_validation_text, "Choppy asset bundle validation animated GLB check missing.")
    assert_condition("animated GLB water and foam primitives" in choppy_bundle_validation_text, "Choppy asset bundle validation animated foam check missing.")
    choppy_bundle_comparison_path = output_dir / "workflow_validation_spectral_choppy_asset_bundle_comparison.md"
    write_asset_bundle_comparison([choppy_bundle_dir], choppy_bundle_comparison_path)
    choppy_bundle_comparison_text = choppy_bundle_comparison_path.read_text(encoding="utf-8")
    assert_condition("Spectral Choppy Wave Asset Bundle Comparison" in choppy_bundle_comparison_text, "Choppy asset bundle comparison title missing.")
    assert_condition("GLB Seq Frames" in choppy_bundle_comparison_text, "Choppy asset bundle comparison GLB sequence column missing.")
    assert_condition("Animated GLB Size" in choppy_bundle_comparison_text, "Choppy asset bundle comparison animated GLB column missing.")
    assert_condition("Steepness P95 Max" in choppy_bundle_comparison_text, "Choppy asset bundle comparison metric column missing.")
    assert_condition("Fold Ratio Max" in choppy_bundle_comparison_text, "Choppy asset bundle comparison fold metric column missing.")
    assert_condition("workflow_validation_spectral_choppy_asset_bundle" in choppy_bundle_comparison_text, "Choppy asset bundle comparison row missing.")
    choppy_bundle_sweep = run_asset_bundle_sweep(
        argparse.Namespace(
            parameter="choppiness",
            values=[0.45, 0.75],
            size=32,
            steps=24,
            frame_every=6,
            domain_size=8.0,
            gravity=9.81,
            dt=0.04,
            wave_amplitude=0.06,
            peak_wavelength=1.2,
            bandwidth=0.32,
            wind_direction_degrees=25.0,
            directional_spread=6.0,
            damping=0.9995,
            seed=23,
            choppiness=0.7,
            max_surface_points=32,
            foam_threshold=0.0,
            max_foam_points=64,
            foam_z_offset=0.01,
            output_dir=output_dir / "workflow_validation_spectral_choppy_asset_bundle_sweep",
        ),
        device,
    )
    choppy_bundle_sweep_comparison = Path(choppy_bundle_sweep["outputs"]["comparison"])
    choppy_bundle_sweep_text = choppy_bundle_sweep_comparison.read_text(encoding="utf-8")
    assert_condition(len(choppy_bundle_sweep["runs"]) == 2, "Choppy asset bundle sweep should create two runs.")
    assert_condition("00_choppiness_0p45" in choppy_bundle_sweep_text, "Choppy asset bundle sweep first run missing.")
    assert_condition("01_choppiness_0p75" in choppy_bundle_sweep_text, "Choppy asset bundle sweep second run missing.")
    assert_condition("Animated GLB Size" in choppy_bundle_sweep_text, "Choppy asset bundle sweep comparison animated GLB column missing.")
    assert_condition("Steepness P95 Max" in choppy_bundle_sweep_text, "Choppy asset bundle sweep comparison metric column missing.")
    assert_condition("Fold Ratio Max" in choppy_bundle_sweep_text, "Choppy asset bundle sweep comparison fold metric column missing.")
    assert_condition("metrics" in choppy_bundle_sweep["runs"][0], "Choppy asset bundle sweep run metrics missing.")
    print(f"Validated spectral choppy OBJ mesh export: {choppy_mesh_path}")

    spectral_benchmark = benchmark_spectral_size(
        size=max(32, size // 2),
        steps=max(24, steps // 3),
        frame_every=max(6, frame_every // 2),
        domain_size=8.0,
        gravity=9.81,
        dt=0.04,
        wave_amplitude=0.06,
        peak_wavelength=1.2,
        bandwidth=0.32,
        wind_direction_degrees=25.0,
        directional_spread=6.0,
        damping=0.9995,
        seed=13,
        store_velocity=True,
        device=device,
    )
    assert_condition(spectral_benchmark["elapsed_sec"] > 0.0, "Spectral benchmark elapsed time must be positive.")
    assert_condition(
        spectral_benchmark["million_cell_steps_per_sec"] > 0.0,
        "Spectral benchmark throughput must be positive.",
    )
    assert_condition(spectral_benchmark["frame_count"] > 0, "Spectral benchmark frame count must be positive.")
    assert_condition(spectral_benchmark["max_speed"] > 0.0, "Spectral benchmark speed must be positive.")
    spectral_benchmark_chart_path = output_dir / "workflow_validation_spectral_benchmark.html"
    save_benchmark_chart([spectral_benchmark], spectral_benchmark_chart_path)
    spectral_benchmark_chart_text = spectral_benchmark_chart_path.read_text(encoding="utf-8")
    assert_condition("Spectral wave benchmark" in spectral_benchmark_chart_text, "Spectral benchmark chart title missing.")
    assert_condition("Plotly.newPlot" in spectral_benchmark_chart_text, "Spectral benchmark chart is missing Plotly.newPlot.")
    print("Validated spectral wave benchmark.")

    comparison_rows = run_solver_benchmark_comparison(
        argparse.Namespace(
            sizes=str(max(32, size // 2)),
            output_dir=output_dir / "workflow_validation_solver_benchmark",
            bathymetry_steps=max(24, steps // 3),
            bathymetry_gravity=1.0,
            bathymetry_damping=0.9994,
            bathymetry_cfl=0.35,
            spectral_steps=max(24, steps // 3),
            spectral_frame_every=max(6, frame_every // 2),
            spectral_domain_size=8.0,
            spectral_gravity=9.81,
            spectral_dt=0.04,
            spectral_wave_amplitude=0.06,
            spectral_peak_wavelength=1.2,
            spectral_bandwidth=0.32,
            spectral_wind_direction_degrees=25.0,
            spectral_directional_spread=6.0,
            spectral_damping=0.9995,
            spectral_seed=17,
            spectral_store_velocity=True,
        )
    )
    assert_condition(len(comparison_rows) == 2, "Solver benchmark comparison should contain two rows.")
    solver_benchmark_dir = output_dir / "workflow_validation_solver_benchmark"
    solver_benchmark_csv = solver_benchmark_dir / "wave_solver_benchmark_comparison.csv"
    solver_benchmark_html = solver_benchmark_dir / "wave_solver_benchmark_comparison.html"
    assert_condition(solver_benchmark_csv.exists(), "Solver benchmark comparison CSV missing.")
    assert_condition(solver_benchmark_html.exists(), "Solver benchmark comparison HTML missing.")
    solver_benchmark_html_text = solver_benchmark_html.read_text(encoding="utf-8")
    assert_condition("Wave solver benchmark comparison" in solver_benchmark_html_text, "Solver benchmark chart title missing.")
    assert_condition("Plotly.newPlot" in solver_benchmark_html_text, "Solver benchmark chart is missing Plotly.newPlot.")
    print("Validated solver benchmark comparison.")


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
