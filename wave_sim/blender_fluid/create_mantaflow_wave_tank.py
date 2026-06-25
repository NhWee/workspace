"""Create an offline Blender Mantaflow FLIP wave-tank scene.

Run through Blender, not the regular Python interpreter:

  blender --background --python wave_sim/blender_fluid/create_mantaflow_wave_tank.py -- --quality preview

The scene uses a liquid domain, an initial water body, an animated paddle, and
collision rocks. It is intentionally configured for an offline bake with mesh
and secondary foam/bubble/spray particles enabled.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy


PRESETS = {
    "preview": {"resolution": 112, "frame_end": 120, "timesteps_max": 6, "mesh_scale": 3, "fps": 24},
    "long_preview": {"resolution": 96, "frame_end": 360, "timesteps_max": 5, "mesh_scale": 3, "fps": 48},
    "production": {"resolution": 256, "frame_end": 220, "timesteps_max": 8, "mesh_scale": 4, "fps": 24},
}


def script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Blender Mantaflow FLIP wave tank.")
    parser.add_argument("--quality", choices=sorted(PRESETS), default="production")
    parser.add_argument("--output", type=Path, default=Path("outputs/mantaflow_wave_tank.blend"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/mantaflow_wave_tank_cache"))
    parser.add_argument("--frame-end", type=int, default=None, help="Override the preset timeline length.")
    parser.add_argument("--fps", type=int, default=None, help="Override the scene playback frame rate.")
    parser.add_argument("--bake", action="store_true", help="Bake data, mesh, and secondary particles after scene creation.")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def set_if_present(settings: object, name: str, value: object) -> None:
    if hasattr(settings, name):
        setattr(settings, name, value)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def cube(name: str, location: tuple[float, float, float], dimensions: tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def liquid_flow(name: str, location: tuple[float, float, float], dimensions: tuple[float, float, float]) -> bpy.types.Object:
    obj = cube(name, location, dimensions)
    modifier = obj.modifiers.new("Liquid Flow", type="FLUID")
    modifier.fluid_type = "FLOW"
    settings = modifier.flow_settings
    settings.flow_type = "LIQUID"
    settings.flow_behavior = "GEOMETRY"
    # Emit the initial water block on frame 1 only. A permanently enabled flow
    # recreates its vertical faces every frame; disabling it from the start
    # removes the liquid entirely.
    set_if_present(settings, "use_inflow", True)
    if hasattr(settings, "use_inflow"):
        settings.keyframe_insert(data_path="use_inflow", frame=1)
        settings.use_inflow = False
        settings.keyframe_insert(data_path="use_inflow", frame=2)
    set_if_present(settings, "surface_distance", 1.5)
    set_if_present(settings, "subframes", 2)
    obj.hide_render = True
    obj.display_type = "WIRE"
    return obj


def effector(name: str, location: tuple[float, float, float], dimensions: tuple[float, float, float]) -> bpy.types.Object:
    obj = cube(name, location, dimensions)
    modifier = obj.modifiers.new("Fluid Collision", type="FLUID")
    modifier.fluid_type = "EFFECTOR"
    settings = modifier.effector_settings
    settings.effector_type = "COLLISION"
    set_if_present(settings, "surface_distance", 0.001)
    return obj


def make_water_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Water")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.015, 0.12, 0.18, 1.0)
    principled.inputs["Roughness"].default_value = 0.14
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["IOR"].default_value = 1.333
    principled.inputs["Transmission Weight"].default_value = 0.12
    return material


def setup_domain(preset: dict[str, int], cache_dir: Path) -> bpy.types.Object:
    domain = cube("Liquid Domain", (0.0, 0.0, 2.5), (12.0, 5.0, 5.0))
    modifier = domain.modifiers.new("Liquid Domain", type="FLUID")
    modifier.fluid_type = "DOMAIN"
    settings = modifier.domain_settings
    settings.domain_type = "LIQUID"
    settings.resolution_max = preset["resolution"]
    settings.cache_type = "MODULAR"
    settings.cache_frame_start = 1
    settings.cache_frame_end = preset["frame_end"]
    settings.cache_directory = str(cache_dir.resolve())
    settings.cache_data_format = "OPENVDB"
    set_if_present(settings, "cache_mesh_format", "BOBJECT")
    settings.timesteps_min = 2
    settings.timesteps_max = preset["timesteps_max"]
    settings.cfl_condition = 2.0
    settings.gravity = (0.0, 0.0, -9.81)
    settings.use_mesh = True
    settings.mesh_scale = preset["mesh_scale"]
    set_if_present(settings, "mesh_particle_radius", 1.55)
    set_if_present(settings, "mesh_smoothen_pos", 4)
    set_if_present(settings, "mesh_smoothen_neg", 4)
    set_if_present(settings, "use_foam_particles", True)
    set_if_present(settings, "use_bubble_particles", True)
    set_if_present(settings, "use_spray_particles", True)
    set_if_present(settings, "use_tracer_particles", True)
    set_if_present(settings, "particle_radius", 1.45)
    set_if_present(settings, "particle_band_width", 3)
    set_if_present(settings, "flip_ratio", 0.97)
    set_if_present(settings, "vorticity", 1.0)
    set_if_present(settings, "surface_tension", 0.6)
    # Mantaflow attaches the generated liquid mesh to the domain object, so
    # hiding the domain would also hide the baked water surface.
    domain.hide_render = False
    domain.display_type = "WIRE"
    return domain


def animate_paddle(paddle: bpy.types.Object, frame_end: int) -> None:
    # The paddle begins in air, then periodically grazes the leading water edge.
    # Keeping the forcing active makes long bakes continue like a driven wave
    # tank instead of calming down after the first impact.
    keyframes = [(1, -4.55), (20, -4.55)]
    cycle_start = 52
    cycle = 56
    frame = cycle_start
    while frame <= frame_end + cycle:
        keyframes.append((frame, -3.55))
        keyframes.append((frame + 24, -4.52))
        frame += cycle
    for frame, x in keyframes:
        paddle.location.x = x
        paddle.keyframe_insert(data_path="location", index=0, frame=frame)


def setup_camera_and_lights() -> None:
    bpy.ops.object.camera_add(location=(15.0, -17.0, 10.5), rotation=(math.radians(65), 0.0, math.radians(38)))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="SUN", location=(2.0, -4.0, 11.0))
    sun = bpy.context.active_object
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(30), math.radians(-20), math.radians(35))
    bpy.ops.object.light_add(type="AREA", location=(-3.0, -2.0, 9.0))
    bpy.context.active_object.data.energy = 1600.0
    bpy.context.active_object.data.shape = "DISK"
    bpy.context.active_object.data.size = 8.0


def create_scene(args: argparse.Namespace) -> bpy.types.Object:
    preset = dict(PRESETS[args.quality])
    if args.frame_end is not None:
        preset["frame_end"] = args.frame_end
    if args.fps is not None:
        preset["fps"] = args.fps
    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = preset["frame_end"]
    scene.render.fps = preset["fps"]
    scene.sync_mode = "FRAME_DROP"
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.world.color = (0.008, 0.02, 0.035)

    domain = setup_domain(preset, args.cache_dir)
    liquid_flow("Initial Water", (-0.25, 0.0, 1.15), (6.8, 4.5, 2.0))
    paddle = effector("Wave Paddle", (-4.55, 0.0, 1.65), (0.20, 4.7, 2.8))
    animate_paddle(paddle, preset["frame_end"])

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0, location=(1.2, 0.0, 1.10))
    rock = bpy.context.active_object
    rock.name = "Breakwater Rock"
    rock.scale = (1.6, 1.15, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    modifier = rock.modifiers.new("Fluid Collision", type="FLUID")
    modifier.fluid_type = "EFFECTOR"
    modifier.effector_settings.effector_type = "COLLISION"

    water = make_water_material()
    domain.data.materials.append(water)
    setup_camera_and_lights()
    return domain


def bake(domain: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = domain
    domain.select_set(True)
    bpy.ops.fluid.bake_data()
    bpy.ops.fluid.bake_mesh()
    try:
        bpy.ops.fluid.bake_particles()
    except RuntimeError:
        pass


def main() -> None:
    args = script_args()
    domain = create_scene(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    if args.bake:
        bake(domain)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"Saved Mantaflow wave tank: {args.output}")


if __name__ == "__main__":
    main()
