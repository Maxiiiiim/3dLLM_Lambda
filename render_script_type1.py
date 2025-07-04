# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 04, 2024
#
# This code is licensed under the MIT License.
# ==============================================================================

# ./blender-3.4.1-linux-x64/blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
# C:\Users\mminecz\PycharmProjects\main\3D_LLM\DiffuRank\blender -b -P render_script_type1.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
import bpy
import sys
import mathutils
from mathutils import Vector, Matrix, Euler
import argparse
import numpy as np
import math
import os
import time
import pickle
from PIL import Image
import random
import json

### solve the division problem
from decimal import Decimal, getcontext

getcontext().prec = 28  # Set the precision for the decimal calculations.

parser = argparse.ArgumentParser()
parser.add_argument('--object_path_pkl', type = str, default = './example_material/example_object_path.pkl')
parser.add_argument('--parent_dir', type = str, default = './example_material')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

uid_paths = pickle.load(open(args.object_path_pkl, 'rb'))
# uid_paths = [#'./example_material/glbs/e2ec7e244a1b4e328c92f57b4709257e.glb',
#              './example_material/glbs/00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj',
#              './example_material/glbs/00000003_1ffb81a71e5b402e966b9341_trimesh_002.obj',
#              './example_material/glbs/00000058_666139e3bff64d4e8a6ce183_trimesh_007.obj']
# random.shuffle(uids)

def compute_bounding_box(mesh_objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in mesh_objects:
        matrix_world = obj.matrix_world
        mesh = obj.data

        for vert in mesh.vertices:
            global_coord = matrix_world @ vert.co

            min_coords = Vector((min(min_coords[i], global_coord[i]) for i in range(3)))
            max_coords = Vector((max(max_coords[i], global_coord[i]) for i in range(3)))

    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    return bbox_center, bbox_size

def get_3x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def clear_socket_input(tree, socket):
    for link in list(tree.links):
        if link.to_socket == socket:
            tree.links.remove(link)

FORMAT_VERSION = 6
MAX_DEPTH = 5.0

UNIFORM_LIGHT_DIRECTION = None
BASIC_AMBIENT_COLOR = None
BASIC_DIFFUSE_COLOR = None

def setup_nodes(output_path, capturing_material_alpha: bool = False, basic_lighting: bool = False):
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants.
    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    def node_add(x, y, **kwargs):
        return node_op("ADD", x, y, **kwargs)

    def node_abs(x, **kwargs):
        return node_op("ABSOLUTE", x, **kwargs)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]
        if basic_lighting:
            # Compute diffuse lighting
            normal_xyz = tree.nodes.new(type="CompositorNodeSeparateXYZ")
            tree.links.new(input_sockets["Normal"], normal_xyz.inputs[0])
            normal_x, normal_y, normal_z = [normal_xyz.outputs[i] for i in range(3)]
            dot = node_add(
                node_mul(UNIFORM_LIGHT_DIRECTION[0], normal_x),
                node_add(
                    node_mul(UNIFORM_LIGHT_DIRECTION[1], normal_y),
                    node_mul(UNIFORM_LIGHT_DIRECTION[2], normal_z),
                ),
            )
            diffuse = node_abs(dot)
            # Compute ambient + diffuse lighting
            brightness = node_add(BASIC_AMBIENT_COLOR, node_mul(BASIC_DIFFUSE_COLOR, diffuse))
            # Modulate the RGB channels using the total brightness.
            rgba_node = tree.nodes.new(type="CompositorNodeSepRGBA")
            tree.links.new(raw_color_socket, rgba_node.inputs[0])
            combine_node = tree.nodes.new(type="CompositorNodeCombRGBA")
            for i in range(3):
                tree.links.new(node_mul(rgba_node.outputs[i], brightness), combine_node.inputs[i])
            tree.links.new(rgba_node.outputs[3], combine_node.inputs[3])
            raw_color_socket = combine_node.outputs[0]

        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear Rec.2020"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]
    split_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    # split_node.base_path =f"{output_path}_rgba"
    tree.links.new(color_socket, split_node.inputs[0])
    # Create separate file output nodes for every channel we care about.
    # The process calling this script must decide how to recombine these
    # channels, possibly into a single image.
    for i, channel in enumerate("rgba") if not capturing_material_alpha else [(0, "MatAlpha")]:
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_{channel}"
        links.new(split_node.outputs[i], output_node.inputs[0])

    if capturing_material_alpha:
        # No need to re-write depth here.
        return

    depth_out = node_clamp(node_mul(input_sockets["Depth"], 1 / MAX_DEPTH))
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    output_node.base_path = f"{output_path}_depth"
    links.new(depth_out, output_node.inputs[0])

def get_socket_value(tree, socket):
    default = socket.default_value
    if not isinstance(default, float):
        default = list(default)
    for link in tree.links:
        if link.to_socket == socket:
            return (link.from_socket, default)
    return (None, default)

def set_socket_value(tree, socket, socket_and_default):
    clear_socket_input(tree, socket)
    old_source_socket, default = socket_and_default
    if isinstance(default, float) and not isinstance(socket.default_value, float):
        # Codepath for setting Emission to a previous alpha value.
        socket.default_value = [default] * 3 + [1.0]
    else:
        socket.default_value = default
    if old_source_socket is not None:
        tree.links.new(old_source_socket, socket)

def setup_material_extraction_shader_for_material(mat, capturing_material_alpha: bool):
    mat.use_nodes = True

    # By default, most imported models should use the regular
    # "Principled BSDF" material, so we should always find this.
    # If not, this shader manipulation logic won't work.
    bsdf_node = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf_node = node
    assert bsdf_node is not None, "material has no Principled BSDF node to modify"

    socket_map = {}
    for input in bsdf_node.inputs:
        socket_map[input.name] = input
    for name in ["Base Color", "Emission", "Emission Strength", "Alpha", "Specular"]:
        assert name in socket_map.keys(), f"{name} not in {list(socket_map.keys())}"

    old_base_color = get_socket_value(mat.node_tree, socket_map["Base Color"])
    old_alpha = get_socket_value(mat.node_tree, socket_map["Alpha"])
    old_emission = get_socket_value(mat.node_tree, socket_map["Emission"])
    old_emission_strength = get_socket_value(mat.node_tree, socket_map["Emission Strength"])
    old_specular = get_socket_value(mat.node_tree, socket_map["Specular"])

    # Make sure the base color of all objects is black and the opacity
    # is 1, so that we are effectively just telling the shader what color
    # to make the pixels.
    clear_socket_input(mat.node_tree, socket_map["Base Color"])
    socket_map["Base Color"].default_value = [0, 0, 0, 1]
    clear_socket_input(mat.node_tree, socket_map["Alpha"])
    socket_map["Alpha"].default_value = 1
    clear_socket_input(mat.node_tree, socket_map["Specular"])
    socket_map["Specular"].default_value = 0.0

    old_blend_method = mat.blend_method
    mat.blend_method = "OPAQUE"

    if capturing_material_alpha:
        set_socket_value(mat.node_tree, socket_map["Emission"], old_alpha)
    else:
        set_socket_value(mat.node_tree, socket_map["Emission"], old_base_color)
    clear_socket_input(mat.node_tree, socket_map["Emission Strength"])
    socket_map["Emission Strength"].default_value = 1.0

    def undo_fn():
        mat.blend_method = old_blend_method
        set_socket_value(mat.node_tree, socket_map["Base Color"], old_base_color)
        set_socket_value(mat.node_tree, socket_map["Alpha"], old_alpha)
        set_socket_value(mat.node_tree, socket_map["Emission"], old_emission)
        set_socket_value(mat.node_tree, socket_map["Emission Strength"], old_emission_strength)
        set_socket_value(mat.node_tree, socket_map["Specular"], old_specular)

    return undo_fn

def find_materials():
    all_materials = set()
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        for mat in obj.data.materials:
            all_materials.add(mat)
    return all_materials

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def setup_material_extraction_shaders(capturing_material_alpha: bool):
    """
    Change every material to emit texture colors (or alpha) rather than having
    an actual reflective color. Returns a function to undo the changes to the
    materials.
    """
    # Objects can share materials, so we first find all of the
    # materials in the project, and then modify them each once.
    undo_fns = []
    for mat in find_materials():
        undo_fn = setup_material_extraction_shader_for_material(mat, capturing_material_alpha)
        if undo_fn is not None:
            undo_fns.append(undo_fn)
    return lambda: [undo_fn() for undo_fn in undo_fns]

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

def create_camera():
    # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object

def set_camera(direction, camera_dist=2.0):
    camera_pos = -camera_dist * direction
    bpy.context.scene.camera.location = camera_pos

    # https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    rot_quat = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

    bpy.context.view_layer.update()


from mathutils.noise import random_unit_vector
def randomize_camera(camera_dist=2.0):
    direction = random_unit_vector()
    set_camera(direction, camera_dist=camera_dist)


def pan_camera(time, axis="Z", camera_dist=2.0, elevation=0.1):
    angle = time * math.pi * 2
    direction = [-math.cos(angle), -math.sin(angle), elevation]
    assert axis in ["X", "Y", "Z"]
    if axis == "X":
        direction = [direction[2], *direction[:2]]
    elif axis == "Y":
        direction = [direction[0], elevation, direction[1]]
    direction = Vector(direction).normalized()
    set_camera(direction, camera_dist=camera_dist)


def place_camera(time, camera_pose_mode="random", camera_dist_min=2.0, camera_dist_max=2.0):
    camera_dist = random.uniform(camera_dist_min, camera_dist_max)
    if camera_pose_mode == "random":
        randomize_camera(camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular":
        pan_camera(time, axis="Z", camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular-elevated":
        pan_camera(time, axis="Z", camera_dist=camera_dist, elevation=-0.2617993878)
    else:
        raise ValueError(f"Unknown camera pose mode: {camera_pose_mode}")

def create_vertex_color_shaders():
    # By default, Blender will ignore vertex colors in both the
    # Eevee and Cycles backends, since these colors aren't
    # associated with a material.
    #
    # What we do here is create a simple material shader and link
    # the vertex color to the material color.
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            continue

        if len(obj.data.materials):
            # We don't want to override any existing materials.
            continue

        color_keys = (obj.data.vertex_colors or {}).keys()
        if not len(color_keys):
            # Many objects will have no materials *or* vertex colors.
            continue

        mat = bpy.data.materials.new(name="VertexColored")
        mat.use_nodes = True

        # There should be a Principled BSDF by default.
        bsdf_node = None
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf_node = node
        assert bsdf_node is not None, "material has no Principled BSDF node to modify"

        socket_map = {}
        for input in bsdf_node.inputs:
            socket_map[input.name] = input

        # Make sure nothing lights the object except for the diffuse color.
        socket_map["Specular"].default_value = 0.0
        socket_map["Roughness"].default_value = 1.0

        v_color = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        v_color.layer_name = color_keys[0]

        mat.node_tree.links.new(v_color.outputs[0], socket_map["Base Color"])

        obj.data.materials.append(mat)

def scene_fov():
    x_fov = bpy.context.scene.camera.data.angle_x
    y_fov = bpy.context.scene.camera.data.angle_y
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    if bpy.context.scene.camera.data.angle == x_fov:
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * height / width)
    else:
        x_fov = 2 * math.atan(math.tan(y_fov / 2) * width / height)
    return x_fov, y_fov

def write_camera_metadata(path):
    x_fov, y_fov = scene_fov()
    bbox_min, bbox_max = scene_bbox()
    matrix = bpy.context.scene.camera.matrix_world
    with open(path, "w") as f:
        json.dump(
            dict(
                format_version=FORMAT_VERSION,
                max_depth=MAX_DEPTH,
                bbox=[list(bbox_min), list(bbox_max)],
                origin=list(matrix.col[3])[:3],
                x_fov=x_fov,
                y_fov=y_fov,
                x=list(matrix.col[0])[:3],
                y=list(-matrix.col[1])[:3],
                z=list(-matrix.col[2])[:3],
            ),
            f,
        )

def render_scene(output_path, fast_mode: bool, extract_material: bool, basic_lighting: bool):
    use_workbench=True
    bpy.context.scene.render.engine == "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.view_layer.update()
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    if basic_lighting:
        bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_settings.view_transform = "Raw"  # sRGB done in graph nodes
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "BW"
    bpy.context.scene.render.image_settings.color_depth = "16"
    bpy.context.scene.render.filepath = output_path
    if extract_material:
        for do_alpha in [False, True]:
            undo_fn = setup_material_extraction_shaders(capturing_material_alpha=do_alpha)
            setup_nodes(output_path, capturing_material_alpha=do_alpha)
            bpy.ops.render.render(write_still=True)
            undo_fn()
    else:
        setup_nodes(output_path, basic_lighting=basic_lighting)
        bpy.ops.render.render(write_still=True)

    # The output images must be moved from their own sub-directories, or
    # discarded if we are using workbench for the color.
    for channel_name in ["r", "g", "b", "a", "depth", *(["MatAlpha"] if extract_material else [])]:
        sub_dir = f"{output_path}_{channel_name}"
        image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])
        name, ext = os.path.splitext(output_path)
        if channel_name == "depth" or not use_workbench:
            os.rename(image_path, f"{name}_{channel_name}{ext}")
        else:
            os.remove(image_path)
        os.removedirs(sub_dir)

    if use_workbench:
        # Re-render RGBA using workbench with texture mode, since this seems
        # to show the most reasonable colors when lighting is broken.
        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.engine == "CYCLES"
        bpy.context.scene.cycles.samples = 16
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.color_depth = "16"
        ### change background to grey
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.81, 0.81, 0.81, 0.81)
        bpy.context.scene.render.film_transparent = False
        os.remove(output_path)
        bpy.ops.render.render(write_still=True)

start_time = time.time()
load_time = []
render_time = []
file_size = []
dis = []
for uid in uid_paths:
    print(uid)

    if not os.path.exists(uid):
        print('object not exist, check the file path')
        continue
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    t_load = time.time()
    path = uid
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
        # bpy.ops.import_scene.obj(filepath=path)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError(f"unexpected extension: {ext}")
    load_time.append(time.time() - t_load)

    t_render = time.time()
    print('begin*************')


    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    normalize_scene()
    bbox_center, bbox_size = compute_bounding_box(mesh_objects)
    distance = max(bbox_size.x, bbox_size.y, bbox_size.z)
    dis.append(distance)
    create_vertex_color_shaders()
    num_images = 8
    camera_pose = "z-circular-elevated"
    cur_path = os.path.join(args.parent_dir, 'rendered_imgs/%s'%(uid.split('/')[-1].split('.')[0]))
    os.makedirs(cur_path, exist_ok=True)
    for i in range(num_images):
        t = i / max(num_images - 1, 1)  # same as np.linspace(0, 1, num_images)
        place_camera(
           t,
           camera_pose_mode=camera_pose,
           camera_dist_min=2.0,
           camera_dist_max=2.0,
        )
        camera = bpy.context.scene.camera
        transform_matrix = bpy.context.scene.camera.matrix_world
        rotation = transform_matrix.to_euler()[2]
        transform_matrix_list = []
        for row in transform_matrix:
            transform_matrix_list.append(list(row))
        try:
            render_scene(
               os.path.join(cur_path, f"{i+20:05}.png"),
               fast_mode=True,
               extract_material=True,
               basic_lighting=True,
            )
        except:
            render_scene(
               os.path.join(cur_path, f"{i+20:05}.png"),
               fast_mode=True,
               extract_material=False,
               basic_lighting=False,
            )
            print('render_scene with material failed')
        write_camera_metadata(os.path.join(cur_path, f"{i+20:05}.json"))
    continue
