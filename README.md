# workspace

GPU 기반 3D 파도 시뮬레이션 workflow를 정리하는 작업 공간입니다.

현재 구조는 2D shallow-water prototype에서 시작해 3D surface, velocity/particle/streamline viewer, GPU FFT spectral wave, choppy surface, foam, glTF/GLB asset export, sweep/recommendation, production export plan까지 이어지는 실험 파이프라인입니다.

## Project Layout

```text
wave_sim/
  setup/             CUDA/PyTorch 동작 확인
  shallow_water/     2D/3D shallow-water solver와 viewer
  spectral_wave/     GPU FFT spectral wave와 choppy wave viewer
  choppy_assets/     metric 평가, OBJ/PLY/glTF/GLB asset export
  workflow_tools/    report, comparison, sweep, recommendation, validation
  benchmarks/        solver 성능 측정과 비교
  data/              dataset 저장, 로드, 비교
  three_d_fluid/     CUDA APIC-MPM 기반 오프라인 3D 자유수면 유체
  blender_fluid/     Blender Mantaflow FLIP 기반 고해상도 액체 장면 생성

docs/
  00_setup/
  01_shallow_water/
  02_spectral_wave/
  03_choppy_assets/
  04_workflow_tools/
  05_benchmarks/
  06_data/
```

Python 코드는 `wave_sim/` 패키지 안에 section별로 정리되어 있습니다. 실행은 repo root에서 `python -m ...` 형식을 사용합니다.

## Recommended Flow

```powershell
.\.venv\Scripts\python.exe -m wave_sim.setup.gpu_smoke_test
.\.venv\Scripts\python.exe -m wave_sim.shallow_water.shallow_water_combined --mode all
.\.venv\Scripts\python.exe -m wave_sim.spectral_wave.spectral_wave_surface_3d
.\.venv\Scripts\python.exe -m wave_sim.spectral_wave.spectral_choppy_wave_viewer
.\.venv\Scripts\python.exe -m wave_sim.choppy_assets.evaluate_spectral_choppy_wave
.\.venv\Scripts\python.exe -m wave_sim.choppy_assets.export_spectral_choppy_asset_bundle --output-dir outputs\spectral_choppy_asset_bundle
.\.venv\Scripts\python.exe -m wave_sim.workflow_tools.validate_spectral_choppy_asset_bundle --bundle-dir outputs\spectral_choppy_asset_bundle
.\.venv\Scripts\python.exe -m wave_sim.workflow_tools.sweep_spectral_choppy_asset_bundles --parameter choppiness --values "0.45,0.75"
.\.venv\Scripts\python.exe -m wave_sim.workflow_tools.recommend_spectral_choppy_asset_bundle --sweep-manifest outputs\spectral_choppy_asset_bundle_sweep\sweep_manifest.json
.\.venv\Scripts\python.exe -m wave_sim.workflow_tools.plan_spectral_choppy_production_export --recommendation-json outputs\spectral_choppy_asset_bundle_sweep\bundle_recommendation.json
.\.venv\Scripts\python.exe -m wave_sim.workflow_tools.validate_wave_workflow
.\.venv\Scripts\python.exe -m wave_sim.three_d_fluid.apic_wave_tank_3d --quality preview
.\.venv\Scripts\python.exe -m wave_sim.three_d_fluid.render_apic_particle_cache
.\tools\blender-5.1.2\blender-5.1.2-windows-x64\blender.exe --background --python wave_sim\blender_fluid\create_mantaflow_wave_tank.py -- --quality preview
```

## Shallow Water Quick Runs

```powershell
.\.venv\Scripts\python.exe -m wave_sim.shallow_water.shallow_water_2d
.\.venv\Scripts\python.exe -m wave_sim.shallow_water.shallow_water_surface_3d
.\.venv\Scripts\python.exe -m wave_sim.shallow_water.shallow_water_uv_3d
.\.venv\Scripts\python.exe -m wave_sim.shallow_water.shallow_water_bathymetry_3d
.\.venv\Scripts\python.exe -m wave_sim.shallow_water.shallow_water_bathymetry_scene_3d
```

## Outputs

Generated simulation/export results are written to `outputs/`.

`outputs/` is intentionally ignored by Git because these files are reproducible and can become large. Re-run the relevant command to recreate them.

## Current Verified State

Last verified on this workspace:

```powershell
.\.venv\Scripts\python.exe -m wave_sim.workflow_tools.validate_wave_workflow --size 64 --steps 48 --frame-every 12
.\.venv\Scripts\python.exe -m pip check
```

Both passed with CUDA on `NVIDIA GeForce RTX 4060 Ti`.
