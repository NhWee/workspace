"""Offline 3D APIC-MPM free-surface wave tank running on CUDA.

This is deliberately an offline solver. It spends GPU work on a three-
dimensional particle/voxel fluid state instead of trying to sustain browser
frame rate. The model is weakly compressible water (equation of state), so it
can form a moving free surface, vortical flow, splashes, and detached droplets.
It is a substantial physical upgrade over a height-field ocean, but is still
not a production two-phase air-water CFD solver.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import taichi as ti


@dataclass(frozen=True)
class SimulationConfig:
    n_grid: int
    particles_per_cell: int
    frames: int
    substeps_per_frame: int
    dt: float
    export_every: int
    max_export_particles: int
    eos_stiffness: float
    gravity: float
    wave_force: float
    wave_frequency: float
    max_velocity: float


PRESETS = {
    "preview": SimulationConfig(
        n_grid=48,
        particles_per_cell=4,
        frames=48,
        substeps_per_frame=10,
        dt=2.0e-4,
        export_every=2,
        max_export_particles=80_000,
        eos_stiffness=55.0,
        gravity=9.8,
        wave_force=19.0,
        wave_frequency=2.0,
        max_velocity=5.0,
    ),
    "production": SimulationConfig(
        n_grid=128,
        particles_per_cell=6,
        frames=240,
        substeps_per_frame=28,
        dt=8.0e-5,
        export_every=8,
        max_export_particles=180_000,
        eos_stiffness=120.0,
        gravity=9.8,
        wave_force=24.0,
        wave_frequency=1.7,
        max_velocity=7.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline CUDA 3D APIC-MPM wave-tank simulation.")
    parser.add_argument("--quality", choices=sorted(PRESETS), default="production")
    parser.add_argument("--n-grid", type=int, help="Override cubic voxel resolution.")
    parser.add_argument("--particles-per-cell", type=int, help="Override particle density in the initial water volume.")
    parser.add_argument("--frames", type=int, help="Override exported simulation frames.")
    parser.add_argument("--substeps-per-frame", type=int, help="Override physics substeps between exported frames.")
    parser.add_argument("--dt", type=float, help="Override simulation timestep.")
    parser.add_argument("--export-every", type=int, help="Export every Nth frame.")
    parser.add_argument("--max-export-particles", type=int, help="Particle cap per saved frame; solver still uses every particle.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/apic_wave_tank_3d.npz"))
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    base = PRESETS[args.quality]
    return SimulationConfig(
        n_grid=args.n_grid or base.n_grid,
        particles_per_cell=args.particles_per_cell or base.particles_per_cell,
        frames=args.frames or base.frames,
        substeps_per_frame=args.substeps_per_frame or base.substeps_per_frame,
        dt=args.dt or base.dt,
        export_every=args.export_every or base.export_every,
        max_export_particles=args.max_export_particles or base.max_export_particles,
        eos_stiffness=base.eos_stiffness,
        gravity=base.gravity,
        wave_force=base.wave_force,
        wave_frequency=base.wave_frequency,
        max_velocity=base.max_velocity,
    )


@ti.data_oriented
class APICWaveTank:
    def __init__(self, config: SimulationConfig, seed: int) -> None:
        self.config = config
        self.n_grid = config.n_grid
        self.dx = 1.0 / config.n_grid
        self.inv_dx = float(config.n_grid)
        self.bound = 3
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol
        self.gamma = 4.0

        fluid_volume = 0.76 * 0.25 * 0.64
        self.n_particles = max(1, int(fluid_volume * config.n_grid**3 * config.particles_per_cell))

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)
        self.J = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))

        rng = np.random.default_rng(seed)
        positions = rng.uniform(
            low=np.array([0.12, 0.16, 0.18], dtype=np.float32),
            high=np.array([0.88, 0.41, 0.82], dtype=np.float32),
            size=(self.n_particles, 3),
        ).astype(np.float32)
        velocities = np.zeros_like(positions)
        velocities[:, 0] = 0.32 * np.sin((positions[:, 2] - 0.5) * np.pi * 3.0)
        velocities[:, 1] = 0.05 * np.cos((positions[:, 0] - 0.5) * np.pi * 4.0)
        velocities[:, 2] = 0.12 * np.sin((positions[:, 0] - 0.5) * np.pi * 2.0)
        self.x.from_numpy(positions)
        self.v.from_numpy(velocities)
        self.J.fill(1.0)
        self.C.fill(0.0)

    @ti.kernel
    def p2g(self):
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            self.J[p] = ti.min(1.06, ti.max(0.94, self.J[p] * (1.0 + self.config.dt * self.C[p].trace())))
            pressure = self.config.eos_stiffness * (self.J[p] ** (-self.gamma) - 1.0)
            stress = -self.config.dt * self.p_vol * 4.0 * self.inv_dx * self.inv_dx * pressure
            affine = stress * ti.Matrix.identity(ti.f32, 3) + self.p_mass * self.C[p]
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def grid_operations(self, sim_time: ti.f32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0.0:
                velocity = self.grid_v[I] / self.grid_m[I]
                velocity[1] -= self.config.dt * self.config.gravity
                position = ti.cast(I, ti.f32) * self.dx
                left_driver = ti.exp(-((position[0] - 0.16) ** 2) / 0.010 - ((position[1] - 0.31) ** 2) / 0.060)
                cross_driver = ti.exp(-((position[2] - 0.50) ** 2) / 0.080)
                velocity[1] += self.config.dt * self.config.wave_force * ti.sin(sim_time * self.config.wave_frequency * 6.2831853) * left_driver * cross_driver
                velocity[2] += self.config.dt * 0.38 * self.config.wave_force * ti.cos(sim_time * self.config.wave_frequency * 4.1) * left_driver
                for d in ti.static(range(3)):
                    if I[d] < self.bound and velocity[d] < 0.0:
                        velocity[d] = 0.0
                    if I[d] > self.n_grid - self.bound and velocity[d] > 0.0:
                        velocity[d] = 0.0
                    velocity[d] = ti.min(self.config.max_velocity, ti.max(-self.config.max_velocity, velocity[d]))
                self.grid_v[I] = velocity

    @ti.kernel
    def g2p(self):
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                g_v = self.grid_v[base + offset]
                weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                new_v += weight * g_v
                new_C += 4.0 * self.inv_dx * weight * g_v.outer_product(dpos)
            for d in ti.static(range(3)):
                new_v[d] = ti.min(self.config.max_velocity, ti.max(-self.config.max_velocity, new_v[d]))
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.config.dt * self.v[p]
            for d in ti.static(range(3)):
                self.x[p][d] = ti.min(1.0 - self.bound * self.dx, ti.max(self.bound * self.dx, self.x[p][d]))

    def substep(self, sim_time: float) -> None:
        self.grid_v.fill(0.0)
        self.grid_m.fill(0.0)
        self.p2g()
        self.grid_operations(sim_time)
        self.g2p()

    def make_export_indices(self, max_particles: int, rng: np.random.Generator) -> np.ndarray | None:
        if self.n_particles <= max_particles:
            return None
        return rng.choice(self.n_particles, size=max_particles, replace=False)

    def export_particles(self, indices: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        positions = self.x.to_numpy()
        velocities = self.v.to_numpy()
        if indices is not None:
            positions = positions[indices]
            velocities = velocities[indices]
        return positions.astype(np.float32), velocities.astype(np.float32)


def run_simulation(config: SimulationConfig, seed: int, output: Path) -> None:
    ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=0.78, offline_cache=True)
    simulator = APICWaveTank(config, seed)
    rng = np.random.default_rng(seed + 1)
    frames: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    sim_time = 0.0
    start = time.perf_counter()
    export_indices = simulator.make_export_indices(config.max_export_particles, rng)

    print(
        f"CUDA APIC-MPM: grid={config.n_grid}^3, particles={simulator.n_particles:,}, "
        f"frames={config.frames}, substeps/frame={config.substeps_per_frame}"
    )
    for frame in range(config.frames):
        for _ in range(config.substeps_per_frame):
            simulator.substep(sim_time)
            sim_time += config.dt
        if frame % config.export_every == 0 or frame == config.frames - 1:
            positions, velocity = simulator.export_particles(export_indices)
            frames.append(positions)
            velocities.append(velocity)
        if (frame + 1) % max(1, config.frames // 12) == 0 or frame == config.frames - 1:
            elapsed = time.perf_counter() - start
            print(f"frame {frame + 1}/{config.frames}  elapsed={elapsed:.1f}s")

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "solver": "CUDA APIC-MPM weakly-compressible free-surface fluid",
        "n_grid": config.n_grid,
        "particles": simulator.n_particles,
        "frames": config.frames,
        "substeps_per_frame": config.substeps_per_frame,
        "dt": config.dt,
        "export_every": config.export_every,
        "max_export_particles": config.max_export_particles,
        "eos_stiffness": config.eos_stiffness,
        "gravity": config.gravity,
        "wave_force": config.wave_force,
        "wave_frequency": config.wave_frequency,
        "max_velocity": config.max_velocity,
    }
    np.savez_compressed(
        output,
        positions=np.stack(frames),
        velocities=np.stack(velocities),
        metadata=json.dumps(metadata),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved particle cache: {output}")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if config.n_grid < 16:
        raise ValueError("--n-grid must be at least 16.")
    if config.particles_per_cell < 1:
        raise ValueError("--particles-per-cell must be positive.")
    run_simulation(config, args.seed, args.output)


if __name__ == "__main__":
    main()
