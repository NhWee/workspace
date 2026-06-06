# workspace

GPU 실험과 고부하 계산 프로토타입을 기록하는 작업 공간입니다.

현재 목표는 PyTorch CUDA 환경에서 시작해 3차원 수면 파도 시뮬레이션으로
점진적으로 발전시키는 것입니다.

## Current Wave Simulation

- `gpu_smoke_test.py`: PyTorch CUDA 동작 확인
- `shallow_water_2d.py`: 2D height-field 파동 계산 및 2D GIF/PNG 저장
- `shallow_water_surface_3d.py`: 같은 높이장 계산을 3D surface `z = h(x, y, t)`로 렌더링
- `spectral_wave_surface_3d.py`: GPU FFT로 방향성 있는 spectral 3D wave surface dataset/viewer 생성
- `spectral_choppy_wave_viewer.py`: FFT wave에 수평 변위를 더한 choppy 3D surface viewer
- `export_spectral_choppy_mesh.py`: choppy wave final frame을 OBJ mesh로 export
- `shallow_water_uv_3d.py`: 수면 변위 `eta`와 속도장 `u, v`를 쓰는 GPU shallow-water surface 렌더링
- `shallow_water_bathymetry_3d.py`: 변수 수심 `H(x, y)`와 dry mask를 포함한 3D 수면 렌더링
- `shallow_water_bathymetry_scene_3d.py`: 바닥 지형 `z = -H(x, y)`와 수면 `z = eta(x, y, t)`를 함께 렌더링
- `shallow_water_plotly_viewer.py`: 브라우저에서 회전/확대/프레임 슬라이더를 쓸 수 있는 Plotly HTML 뷰어
- `shallow_water_velocity_viewer.py`: velocity dataset의 `u, v`로 수면 색상을 speed magnitude에 매핑하는 Plotly 뷰어
- `shallow_water_vector_viewer.py`: speed-colored surface 위에 `u, v` 방향 cone vector를 얹는 Plotly 뷰어
- `shallow_water_particle_viewer.py`: velocity field로 seed particle을 advect해 흐름 궤적을 그리는 Plotly 뷰어
- `shallow_water_particle_animation_viewer.py`: particle 위치와 trail을 시간 애니메이션으로 보여주는 Plotly 뷰어
- `shallow_water_streamline_viewer.py`: 한 frame의 velocity field를 따라 streamline을 그리는 Plotly 뷰어
- `export_wave_dataset.py`: GPU 계산 결과를 재사용 가능한 `.npz` 데이터셋으로 저장
- `wave_dataset.py`: wave frame/depth 데이터셋 저장/불러오기 유틸, 생성 시각과 Git metadata 기록
- `compare_wave_datasets.py`: 여러 `.npz` wave dataset의 metadata와 기본 수치 범위를 Markdown 표로 비교
- `sweep_wave_experiments.py`: 여러 damping/gravity/CFL 조건을 자동 실행하고 dashboard/비교 산출물을 생성
- `validate_wave_workflow.py`: solver, dataset, Plotly viewer를 빠르게 검증하는 end-to-end 체크
- `benchmark_bathymetry.py`: bathymetry solver의 GPU 성능과 안정성 벤치마크
- `benchmark_spectral_wave.py`: GPU FFT spectral wave solver의 성능 벤치마크
- `compare_solver_benchmarks.py`: bathymetry/spectral solver 성능을 한 chart에서 비교
- `shallow_water_2d_guide.txt`: 2D height-field 모델 설명
- `shallow_water_surface_3d_guide.txt`: 3D surface 렌더링 워크플로 설명
- `spectral_wave_surface_3d_guide.txt`: GPU FFT spectral wave surface 설명
- `spectral_choppy_wave_viewer_guide.txt`: spectral choppy wave surface viewer 설명
- `export_spectral_choppy_mesh_guide.txt`: choppy wave OBJ mesh export 설명
- `shallow_water_uv_3d_guide.txt`: `h/u/v` shallow-water solver 설명
- `shallow_water_bathymetry_3d_guide.txt`: bathymetry solver 설명
- `shallow_water_bathymetry_scene_3d_guide.txt`: 수면+바닥 통합 3D 장면 설명
- `shallow_water_plotly_viewer_guide.txt`: 인터랙티브 Plotly 뷰어 설명
- `shallow_water_velocity_viewer_guide.txt`: 속도 크기 색상 Plotly 뷰어 설명
- `shallow_water_vector_viewer_guide.txt`: 속도 방향 cone vector Plotly 뷰어 설명
- `shallow_water_particle_viewer_guide.txt`: velocity 기반 particle trace Plotly 뷰어 설명
- `shallow_water_particle_animation_viewer_guide.txt`: particle animation Plotly 뷰어 설명
- `shallow_water_streamline_viewer_guide.txt`: velocity frame 기반 streamline Plotly 뷰어 설명
- `wave_dataset_guide.txt`: 계산 결과 저장/재사용 워크플로 설명
- `compare_wave_datasets_guide.txt`: 여러 wave dataset 비교 CLI 설명
- `sweep_wave_experiments_guide.txt`: parameter sweep 실험 워크플로 설명
- `validate_wave_workflow_guide.txt`: 전체 워크플로 검증 스크립트 설명
- `benchmark_spectral_wave_guide.txt`: GPU FFT spectral wave benchmark 설명
- `compare_solver_benchmarks_guide.txt`: solver benchmark 통합 비교 리포트 설명

## Run

```powershell
.\.venv\Scripts\python.exe gpu_smoke_test.py
.\.venv\Scripts\python.exe shallow_water_2d.py
.\.venv\Scripts\python.exe shallow_water_surface_3d.py
.\.venv\Scripts\python.exe spectral_wave_surface_3d.py
.\.venv\Scripts\python.exe spectral_wave_surface_3d.py --store-velocity --output outputs\spectral_wave_velocity.npz --viewer-output outputs\spectral_wave_velocity.html
.\.venv\Scripts\python.exe spectral_choppy_wave_viewer.py
.\.venv\Scripts\python.exe export_spectral_choppy_mesh.py --output outputs\spectral_choppy_wave_final.obj
.\.venv\Scripts\python.exe export_spectral_choppy_mesh.py --sequence-output-dir outputs\spectral_choppy_mesh_sequence
.\.venv\Scripts\python.exe shallow_water_uv_3d.py
.\.venv\Scripts\python.exe shallow_water_bathymetry_3d.py
.\.venv\Scripts\python.exe shallow_water_bathymetry_scene_3d.py
.\.venv\Scripts\python.exe shallow_water_plotly_viewer.py
.\.venv\Scripts\python.exe export_wave_dataset.py
.\.venv\Scripts\python.exe export_wave_dataset.py --store-velocity --output outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe shallow_water_plotly_viewer.py --input-npz outputs\wave_dataset.npz
.\.venv\Scripts\python.exe shallow_water_velocity_viewer.py --input-npz outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe shallow_water_vector_viewer.py --input-npz outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe shallow_water_particle_viewer.py --input-npz outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe shallow_water_particle_animation_viewer.py --input-npz outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe shallow_water_streamline_viewer.py --input-npz outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe compare_wave_datasets.py outputs\wave_dataset_velocity.npz
.\.venv\Scripts\python.exe sweep_wave_experiments.py --size 192 --steps 360 --frame-every 18
.\.venv\Scripts\python.exe validate_wave_workflow.py
.\.venv\Scripts\python.exe benchmark_bathymetry.py --sizes 256,512,1024 --steps 600
.\.venv\Scripts\python.exe benchmark_spectral_wave.py --sizes 256,512,1024 --steps 360 --store-velocity
.\.venv\Scripts\python.exe compare_solver_benchmarks.py --sizes 256,512,1024 --spectral-store-velocity
```

생성 결과는 `outputs/`에 저장됩니다. 이 폴더는 재생성 가능한 산출물이므로
기본 Git 커밋에서는 제외합니다.

## Latest Benchmark

RTX 4060 Ti에서 bathymetry solver를 600 steps로 측정한 결과:

| Grid | Auto dt | Time | Throughput | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| 256 x 256 | 0.0036046 | 0.232s | 169.6M cell-steps/s | 0.004 GiB |
| 512 x 512 | 0.0017988 | 0.227s | 692.0M cell-steps/s | 0.018 GiB |
| 1024 x 1024 | 0.0008985 | 0.274s | 2299.6M cell-steps/s | 0.070 GiB |
