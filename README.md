# workspace

GPU 기반 3D 파도 시뮬레이션 workflow를 정리한 작업 공간입니다.

현재 구조는 2D shallow-water prototype에서 시작해 GPU FFT spectral wave,
choppy surface, foam, glTF/GLB asset export, sweep/recommendation,
production export plan까지 이어지는 실험 파이프라인입니다.

## Sections

| Section | Purpose | Main files |
| --- | --- | --- |
| Setup | CUDA/PyTorch 동작 확인 | `gpu_smoke_test.py`, `requirements.txt` |
| 2D/3D Shallow Water | 기본 shallow-water solver, 3D surface, velocity/particle/streamline viewer | `shallow_water_*.py` |
| GPU Spectral Wave | FFT 기반 3D wave와 choppy wave, metric 평가 | `spectral_wave_surface_3d.py`, `spectral_choppy_wave_viewer.py`, `evaluate_spectral_choppy_wave.py` |
| Choppy Asset Export | OBJ/PLY/glTF/GLB/animated GLB/bundle export | `export_spectral_choppy_*.py` |
| Workflow Tools | report, comparison, sweep, recommendation, production plan, validation | `report_*`, `compare_*`, `sweep_*`, `recommend_*`, `plan_*`, `validate_*` |
| Benchmarks | solver 성능 측정과 비교 | `benchmark_*.py`, `compare_solver_benchmarks.py` |
| Data Utilities | dataset 저장/로드와 dataset 비교 | `wave_dataset.py`, `export_wave_dataset.py`, `compare_wave_datasets.py` |

Detailed guides are organized under `docs/`:

```text
docs/
  01_shallow_water/
  02_spectral_wave/
  03_choppy_assets/
  04_workflow_tools/
  05_benchmarks/
  06_data/
```

## Recommended Flow

```powershell
.\.venv\Scripts\python.exe gpu_smoke_test.py
.\.venv\Scripts\python.exe shallow_water_2d.py
.\.venv\Scripts\python.exe shallow_water_surface_3d.py
.\.venv\Scripts\python.exe spectral_wave_surface_3d.py
.\.venv\Scripts\python.exe spectral_choppy_wave_viewer.py
.\.venv\Scripts\python.exe evaluate_spectral_choppy_wave.py
.\.venv\Scripts\python.exe export_spectral_choppy_asset_bundle.py --output-dir outputs\spectral_choppy_asset_bundle
.\.venv\Scripts\python.exe validate_spectral_choppy_asset_bundle.py --bundle-dir outputs\spectral_choppy_asset_bundle
.\.venv\Scripts\python.exe sweep_spectral_choppy_asset_bundles.py --parameter choppiness --values "0.45,0.75"
.\.venv\Scripts\python.exe recommend_spectral_choppy_asset_bundle.py --sweep-manifest outputs\spectral_choppy_asset_bundle_sweep\sweep_manifest.json
.\.venv\Scripts\python.exe plan_spectral_choppy_production_export.py --recommendation-json outputs\spectral_choppy_asset_bundle_sweep\bundle_recommendation.json
.\.venv\Scripts\python.exe validate_wave_workflow.py
```

## Outputs

Generated simulation/export results are written to `outputs/`.

`outputs/` is intentionally ignored by Git because these files are reproducible and can become large. Re-run the relevant command to recreate them.

## Current Verified State

Last verified on this workspace:

```powershell
.\.venv\Scripts\python.exe validate_wave_workflow.py --size 64 --steps 48 --frame-every 12
.\.venv\Scripts\python.exe -m pip check
```

Both passed with CUDA on `NVIDIA GeForce RTX 4060 Ti`.

Latest pushed commit before this cleanup:

```text
20ec1e6 Add choppy production export plan
```
