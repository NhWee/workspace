"""Create an offline Blender Mantaflow FLIP wave-tank scene.

Run through Blender, not the regular Python interpreter:

  blender --background --python wave_sim/blender_fluid/create_mantaflow_wave_tank.py -- --quality preview

The scene uses a liquid domain, an initial water body, an animated paddle,
sloped absorbing beach geometry, and collision rocks. It is intentionally
configured for an offline bake with mesh and secondary foam/bubble/spray
particles enabled.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy

PRESETS = {
    "preview": {"resolution": 112, "frame_end": 120, "timesteps_max": 6, "mesh_scale": 3, "fps": 24},
    "long_preview": {"resolution": 96, "frame_end": 360, "timesteps_max": 5, "mesh_scale": 3, "fps": 48},
    "ocean_preview": {"resolution": 144, "frame_end": 300, "timesteps_max": 8, "mesh_scale": 4, "fps": 30},
    "production": {"resolution": 256, "frame_end": 360, "timesteps_max": 10, "mesh_scale": 4, "fps": 30},
}

DOMAIN_SIZE = (18.0, 6.0, 5.5)
DOMAIN_CENTER = (1.5, 0.0, 2.75)


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


def shade_smooth(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    finally:
        obj.select_set(False)


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


def mesh_effector(name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    modifier = obj.modifiers.new("Fluid Collision", type="FLUID")
    modifier.fluid_type = "EFFECTOR"
    modifier.effector_settings.effector_type = "COLLISION"
    set_if_present(modifier.effector_settings, "surface_distance", 0.002)
    return obj


def make_rock_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Wet Rock")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.09, 0.08, 0.07, 1.0)
    principled.inputs["Roughness"].default_value = 0.62
    principled.inputs["Metallic"].default_value = 0.0
    return material


def make_beach_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Wet Absorbing Beach")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.16, 0.15, 0.13, 1.0)
    principled.inputs["Roughness"].default_value = 0.78
    principled.inputs["Metallic"].default_value = 0.0
    return material


def make_foam_material() -> bpy.types.Material:
    material = bpy.data.materials.new("White Foam")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.94, 0.97, 0.96, 1.0)
    principled.inputs["Roughness"].default_value = 0.82
    principled.inputs["Metallic"].default_value = 0.0
    return material


def make_spray_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Mist Spray")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.86, 0.95, 1.0, 1.0)
    principled.inputs["Alpha"].default_value = 0.68
    principled.inputs["Roughness"].default_value = 0.22
    principled.inputs["Metallic"].default_value = 0.0
    material.blend_method = "BLEND"
    return material


def make_water_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Water")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.015, 0.12, 0.18, 1.0)
    principled.inputs["Roughness"].default_value = 0.14
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["IOR"].default_value = 1.333
    principled.inputs["Transmission Weight"].default_value = 0.12
    principled.inputs["Alpha"].default_value = 0.72
    material.blend_method = "BLEND"
    return material


def particle_instance(name: str, radius: float, material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=(0.0, 0.0, -100.0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    shade_smooth(obj)
    obj.hide_viewport = True
    return obj


def configure_secondary_particle_render(domain: bpy.types.Object) -> None:
    foam_object = bpy.data.objects.get("Foam Render Droplet") or particle_instance(
        "Foam Render Droplet", 1.0, bpy.data.materials.get("White Foam") or make_foam_material()
    )
    spray_object = bpy.data.objects.get("Spray Render Droplet") or particle_instance(
        "Spray Render Droplet", 1.0, bpy.data.materials.get("Mist Spray") or make_spray_material()
    )
    settings_by_name = {
        "Foam": {"object": foam_object, "size": 0.060, "random": 0.72, "display": 100},
        "Spray": {"object": spray_object, "size": 0.030, "random": 0.85, "display": 100},
        "Bubbles": {"object": spray_object, "size": 0.018, "random": 0.65, "display": 45},
        "Tracers": {"object": spray_object, "size": 0.014, "random": 0.80, "display": 25},
        "Spray + Foam + Bubbles": {"object": foam_object, "size": 0.050, "random": 0.80, "display": 100},
    }
    for particle_system in domain.particle_systems:
        config = settings_by_name.get(particle_system.name)
        if not config:
            continue
        settings = particle_system.settings
        settings.render_type = "OBJECT"
        settings.instance_object = config["object"]
        settings.particle_size = config["size"]
        settings.size_random = config["random"]
        settings.display_percentage = config["display"]
        set_if_present(settings, "use_rotations", True)
        set_if_present(settings, "use_modifier_stack", True)


def foam_proxy_emitter(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    frame_end: int,
    count: int,
    lifetime: int,
    particle_size: float,
    instance_object: bpy.types.Object,
    upward: float,
    sideways: float,
) -> bpy.types.Object:
    emitter = cube(name, location, dimensions)
    bpy.context.view_layer.objects.active = emitter
    emitter.select_set(True)
    bpy.ops.object.particle_system_add()
    particle_system = emitter.particle_systems[-1]
    settings = particle_system.settings
    settings.name = f"{name} Particles"
    settings.count = count
    settings.frame_start = 12
    settings.frame_end = frame_end
    settings.lifetime = lifetime
    settings.lifetime_random = 0.65
    settings.emit_from = "FACE"
    settings.physics_type = "NEWTON"
    settings.render_type = "OBJECT"
    settings.instance_object = instance_object
    settings.particle_size = particle_size
    settings.size_random = 0.8
    settings.normal_factor = upward
    settings.tangent_factor = sideways
    settings.object_align_factor[0] = 0.20
    settings.object_align_factor[1] = 0.05
    settings.object_align_factor[2] = upward * 0.35
    settings.brownian_factor = 0.45
    settings.drag_factor = 0.24
    if hasattr(settings, "effector_weights"):
        settings.effector_weights.gravity = 0.08
    settings.display_percentage = 60
    set_if_present(settings, "use_rotations", True)
    emitter.hide_viewport = True
    emitter.hide_render = False
    emitter.show_instancer_for_render = False
    emitter.show_instancer_for_viewport = False
    emitter.select_set(False)
    return emitter


def setup_foam_proxy_emitters(frame_end: int) -> None:
    foam_object = bpy.data.objects.get("Foam Render Droplet") or particle_instance(
        "Foam Render Droplet", 1.0, bpy.data.materials.get("White Foam") or make_foam_material()
    )
    spray_object = bpy.data.objects.get("Spray Render Droplet") or particle_instance(
        "Spray Render Droplet", 1.0, bpy.data.materials.get("Mist Spray") or make_spray_material()
    )
    foam_proxy_emitter(
        "Rock Foam Proxy Emitter",
        (2.35, 0.0, 1.55),
        (3.4, 2.4, 0.06),
        frame_end,
        2600,
        30,
        0.040,
        foam_object,
        0.035,
        0.18,
    )
    foam_proxy_emitter(
        "Beach Foam Proxy Emitter",
        (5.75, 0.0, 1.30),
        (2.6, 4.8, 0.06),
        frame_end,
        1800,
        36,
        0.035,
        foam_object,
        0.025,
        0.10,
    )
    foam_proxy_emitter(
        "Breaking Spray Proxy Emitter",
        (1.55, -0.15, 1.85),
        (1.6, 1.4, 0.10),
        frame_end,
        950,
        24,
        0.022,
        spray_object,
        0.42,
        0.28,
    )


def setup_domain(preset: dict[str, int], cache_dir: Path) -> bpy.types.Object:
    domain = cube("Liquid Domain", DOMAIN_CENTER, DOMAIN_SIZE)
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
    set_if_present(settings, "cache_particle_format", "UNI")
    set_if_present(settings, "cache_mesh_format", "BOBJECT")
    settings.timesteps_min = 2
    settings.timesteps_max = preset["timesteps_max"]
    settings.cfl_condition = 1.5
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
    set_if_present(settings, "particle_band_width", 5)
    set_if_present(settings, "particle_number", 3)
    set_if_present(settings, "particle_min", 12)
    set_if_present(settings, "particle_max", 128)
    set_if_present(settings, "particle_scale", 1)
    set_if_present(settings, "sndparticle_combined_export", "SPRAY_FOAM_BUBBLES")
    set_if_present(settings, "sndparticle_boundary", "PUSHOUT")
    set_if_present(settings, "sndparticle_life_min", 28.0)
    set_if_present(settings, "sndparticle_life_max", 110.0)
    set_if_present(settings, "sndparticle_potential_min_wavecrest", 0.05)
    set_if_present(settings, "sndparticle_potential_max_wavecrest", 30.0)
    set_if_present(settings, "sndparticle_potential_min_trappedair", 0.05)
    set_if_present(settings, "sndparticle_potential_max_trappedair", 35.0)
    set_if_present(settings, "sndparticle_potential_min_energy", 0.02)
    set_if_present(settings, "sndparticle_potential_max_energy", 25.0)
    set_if_present(settings, "sndparticle_sampling_wavecrest", 600)
    set_if_present(settings, "sndparticle_sampling_trappedair", 160)
    set_if_present(settings, "sndparticle_potential_radius", 3)
    set_if_present(settings, "sndparticle_update_radius", 3)
    set_if_present(settings, "sndparticle_bubble_buoyancy", 0.72)
    set_if_present(settings, "sndparticle_bubble_drag", 0.48)
    set_if_present(settings, "sys_particle_maximum", 2500000)
    set_if_present(settings, "flip_ratio", 0.97)
    set_if_present(settings, "vorticity", 1.8)
    set_if_present(settings, "surface_tension", 0.25)
    # Mantaflow attaches the generated liquid mesh to the domain object, so
    # hiding the domain would also hide the baked water surface.
    domain.hide_render = False
    domain.display_type = "WIRE"
    return domain


def animate_paddle(paddle: bpy.types.Object, frame_end: int) -> None:
    # Combine two stroke periods so the tank receives uneven wave groups
    # instead of a single mechanical back-and-forth pulse.
    for frame in range(1, frame_end + 25, 12):
        t = frame / 30.0
        stroke = 0.72 * math.sin(2.0 * math.pi * t / 2.6)
        stroke += 0.28 * math.sin(2.0 * math.pi * t / 1.35 + 0.7)
        paddle.location.x = -6.95 + stroke
        paddle.rotation_euler[1] = math.radians(4.0 * math.sin(2.0 * math.pi * t / 1.8))
        paddle.keyframe_insert(data_path="location", index=0, frame=frame)
        paddle.keyframe_insert(data_path="rotation_euler", index=1, frame=frame)

def setup_absorbing_beach_and_tank() -> None:
    # The sloped beach dissipates wave energy before it reaches the right wall,
    # reducing the tall reflected sheet that appeared in earlier previews.
    beach = mesh_effector(
        "Sloped Absorbing Beach",
        [
            (4.8, -2.95, 0.00),
            (8.95, -2.95, 0.00),
            (8.95, -2.95, 2.15),
            (4.8, 2.95, 0.00),
            (8.95, 2.95, 0.00),
            (8.95, 2.95, 2.15),
        ],
        [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (0, 2, 5, 3)],
    )
    beach.data.materials.append(make_beach_material())

    floor = effector("Tank Floor", (1.5, 0.0, -0.08), (17.6, 5.8, 0.16))
    left_wall = effector("Left Wave Wall", (-7.35, 0.0, 2.0), (0.18, 5.8, 4.0))
    back_wall = effector("Back Side Wall", (1.5, 3.0, 2.0), (17.6, 0.12, 4.0))
    front_wall = effector("Front Side Wall", (1.5, -3.0, 2.0), (17.6, 0.12, 4.0))
    for obj in (floor, left_wall, back_wall, front_wall):
        obj.hide_render = True
        obj.display_type = "WIRE"


def setup_breakwater_rocks() -> None:
    rock_material = make_rock_material()
    rocks = [
        ((1.7, -0.5, 0.85), (1.55, 1.05, 0.85), 4),
        ((2.65, 0.8, 0.72), (1.25, 0.85, 0.68), 3),
        ((3.45, -1.05, 0.58), (1.05, 0.75, 0.55), 3),
    ]
    for index, (location, scale, subdivisions) in enumerate(rocks, start=1):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, radius=1.0, location=location)
        rock = bpy.context.active_object
        rock.name = f"Breakwater Rock {index}"
        rock.scale = scale
        rock.rotation_euler = (math.radians(8 * index), math.radians(-13 * index), math.radians(17 * index))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        modifier = rock.modifiers.new("Fluid Collision", type="FLUID")
        modifier.fluid_type = "EFFECTOR"
        modifier.effector_settings.effector_type = "COLLISION"
        rock.data.materials.append(rock_material)
        shade_smooth(rock)


def setup_camera_and_lights() -> None:
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(1.8, 0.0, 1.0))
    target = bpy.context.active_object
    target.name = "Camera Target"
    bpy.ops.object.camera_add(location=(12.0, -13.0, 7.6))
    camera = bpy.context.active_object
    camera.data.lens = 32
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = target
    camera.data.dof.aperture_fstop = 8.0
    direction = target.location - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="SUN", location=(2.0, -4.0, 11.0))
    sun = bpy.context.active_object
    sun.data.energy = 3.2
    sun.rotation_euler = (math.radians(33), math.radians(-18), math.radians(31))
    bpy.ops.object.light_add(type="AREA", location=(-3.5, -3.5, 8.0))
    bpy.context.active_object.data.energy = 1600.0
    bpy.context.active_object.data.shape = "DISK"
    bpy.context.active_object.data.size = 7.0


def setup_rendering(scene: bpy.types.Scene) -> None:
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 96
        scene.cycles.use_denoising = True
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
        scene.render.use_motion_blur = False
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"


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
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.world.color = (0.008, 0.02, 0.035)
    setup_rendering(scene)

    domain = setup_domain(preset, args.cache_dir)
    liquid_flow("Initial Water", (-1.35, 0.0, 1.05), (9.2, 5.25, 1.9))
    setup_absorbing_beach_and_tank()
    paddle = effector("Wave Paddle", (-6.95, 0.0, 1.45), (0.24, 5.55, 2.5))
    paddle.hide_render = True
    paddle.display_type = "WIRE"
    animate_paddle(paddle, preset["frame_end"])
    setup_breakwater_rocks()
    setup_foam_proxy_emitters(preset["frame_end"])

    water = make_water_material()
    domain.data.materials.append(water)
    configure_secondary_particle_render(domain)
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
    configure_secondary_particle_render(domain)


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
