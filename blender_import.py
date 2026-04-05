"""
blender_import.py

Imports a scene.json produced by the pipeline into Blender.

Usage (inside Blender Script Editor):
    1. Open Blender
    2. Go to the Scripting workspace
    3. Open this file
    4. Set SCENE_JSON to the path of your scene.json
    5. Press Run Script

Usage (command line, headless):
    blender --python blender_import.py -- --scene output/dining/scene.json

Coordinate conversion:
    Pipeline camera space  →  Blender world space
    X (right)              →  X (right)
    Y (down, scaled)       →  Z (up, negated)
    Z (depth)              →  Y (forward)
"""

import bpy
import bmesh
import mathutils
import math
import json
import os
import sys

# ─────────────────────────────────────────────
#  CONFIGURE THIS when running from Script Editor
# ─────────────────────────────────────────────
SCENE_JSON  = r"output/dining/scene.json"
CLEAR_SCENE = True
# ─────────────────────────────────────────────

PLANE_COLORS = {
    "floor":    (0.40, 0.30, 0.20, 1),
    "ground":   (0.35, 0.28, 0.18, 1),
    "carpet":   (0.55, 0.35, 0.30, 1),
    "rug":      (0.60, 0.40, 0.30, 1),
    "ceiling":  (0.90, 0.90, 0.88, 1),
    "roof":     (0.70, 0.65, 0.60, 1),
    "wall":     (0.80, 0.78, 0.72, 1),
    "room":     (0.80, 0.78, 0.72, 1),
    "road":     (0.30, 0.30, 0.30, 1),
    "pavement": (0.45, 0.45, 0.45, 1),
    "sidewalk": (0.55, 0.55, 0.52, 1),
    "pathway":  (0.50, 0.45, 0.35, 1),
    "grass":    (0.20, 0.50, 0.18, 1),
    "lawn":     (0.22, 0.52, 0.20, 1),
    "sky":      (0.50, 0.70, 1.00, 1),
    "water":    (0.20, 0.40, 0.80, 1),
    "river":    (0.20, 0.40, 0.80, 1),
    "pool":     (0.25, 0.55, 0.85, 1),
}


def dominant_color_from_crop(crop_path):
    """Read a crop PNG and return the most common non-background color as (R,G,B,1) in 0-1 range."""
    try:
        from PIL import Image
        img = Image.open(crop_path).convert("RGB")
        pixels = list(img.getdata())
        # Filter out the gray background (RGB ~128,128,128 from the 0.5 composite)
        fg_pixels = [p for p in pixels if not (120 < p[0] < 136 and 120 < p[1] < 136 and 120 < p[2] < 136)]
        if not fg_pixels:
            fg_pixels = pixels
        # Quantize to reduce noise: round each channel to nearest 16
        quantized = [(r // 16 * 16, g // 16 * 16, b // 16 * 16) for r, g, b in fg_pixels]
        # Find most common quantized color
        from collections import Counter
        most_common = Counter(quantized).most_common(1)[0][0]
        return (most_common[0] / 255.0, most_common[1] / 255.0, most_common[2] / 255.0, 1.0)
    except Exception as e:
        print(f"    Could not read crop for color: {e}")
        return (0.6, 0.6, 0.6, 1.0)


def cam_to_blender(pos):
    """Camera space (X right, Y down, Z depth) → Blender (X right, Y depth, Z up)."""
    return (pos["x"], pos["z"], -pos["y"])


def _deselect_all():
    bpy.ops.object.select_all(action="DESELECT")


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def make_material(name, rgba):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.8
    return mat


def import_mesh_object(entry, base_dir):
    mesh_path = entry.get("mesh_path")
    if not mesh_path:
        print(f"    No mesh path for '{entry['class_name']}', skipping")
        return None

    abs_path = os.path.normpath(os.path.join(base_dir, mesh_path))
    if not os.path.exists(abs_path):
        print(f"    Mesh not found: {abs_path}")
        return None

    _deselect_all()
    before = set(o.name for o in bpy.data.objects)

    ext = os.path.splitext(abs_path)[1].lower()
    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=abs_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=abs_path)
    else:
        print(f"    Unsupported format: {ext}")
        return None

    new_objs = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
    if not new_objs:
        print(f"    Import produced no mesh objects")
        return None

    # Join all imported pieces into one object
    _deselect_all()
    for o in new_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    if len(new_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = f"{entry['id']:03d}_{entry['class_name']}"

    # If the mesh has no textured materials, extract dominant color from the crop
    has_texture = False
    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    has_texture = True
                    break
        if has_texture:
            break

    if not has_texture:
        crop_path = entry.get("crop_path")
        if crop_path:
            abs_crop = os.path.normpath(os.path.join(base_dir, crop_path))
            color = dominant_color_from_crop(abs_crop)
        else:
            color = (0.6, 0.6, 0.6, 1.0)
        mat = make_material(f"{entry['class_name']}_color", color)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        print(f"    Assigned color from crop: ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

    return obj


def place_mesh_object(obj, entry):
    # Use perspective-corrected scale from scene.json
    scale = entry.get("scale", {})
    tgt_w = scale.get("width",  1.0)
    tgt_h = scale.get("height", 1.0)
    tgt_d = scale.get("depth",  min(tgt_w, tgt_h))

    # Reset transforms so bounding box is accurate
    _deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bb = obj.bound_box  # 8 local corners
    xs = [v[0] for v in bb]
    ys = [v[1] for v in bb]
    zs = [v[2] for v in bb]
    cur_w = max(xs) - min(xs)
    cur_d = max(ys) - min(ys)
    cur_h = max(zs) - min(zs)

    sx = tgt_w / cur_w if cur_w > 1e-6 else 1.0
    sy = tgt_d / cur_d if cur_d > 1e-6 else 1.0
    sz = tgt_h / cur_h if cur_h > 1e-6 else 1.0
    obj.scale = (sx, sy, sz)

    # 180° X fix (TripoSR/Hunyuan3D orientation) + scene Y-axis rotation
    rot_y_rad = math.radians(entry.get("rotation_y_deg", 0.0))
    obj.rotation_euler = (math.pi, 0.0, rot_y_rad)

    # Apply scale+rotation so bounding box is in world space
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Position from scene.json (camera space → Blender)
    pos = cam_to_blender(entry["position"])
    obj.location = pos

    # Ground-snap: shift object so its bottom sits at the correct height above ground
    bpy.context.view_layer.update()
    bb_world = [obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box]
    bb_min_z = min(v.z for v in bb_world)
    ground_offset = entry.get("height_from_ground_m", 0.0)
    obj.location.z += ground_offset - bb_min_z


def create_plane_object(entry):
    dims   = entry.get("dimensions", {})
    width  = max(dims.get("width",  5.0), 0.1)
    height = max(dims.get("height", 5.0), 0.1)
    length = max(dims.get("length", 5.0), 0.1)

    pos    = cam_to_blender(entry["position"])
    normal = entry.get("plane_normal", "up")

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=pos)
    obj = bpy.context.active_object
    obj.name = f"{entry['id']:03d}_{entry['class_name']}"

    if normal == "up":
        obj.rotation_euler = (0, 0, 0)
        obj.scale = (width, length, 1.0)
    elif normal == "down":
        obj.rotation_euler = (math.pi, 0, 0)
        obj.scale = (width, length, 1.0)
    elif normal == "forward":
        obj.rotation_euler = (math.pi / 2, 0, 0)
        obj.scale = (width, height, 1.0)

    color = PLANE_COLORS.get(entry["class_name"].lower(), (0.65, 0.65, 0.65, 1))
    mat = make_material(entry["class_name"], color)
    obj.data.materials.append(mat)

    return obj


def add_camera_light():
    # Simple sun lamp for visibility
    bpy.ops.object.light_add(type="SUN", location=(0, -5, 5))
    sun = bpy.context.active_object
    sun.name = "SceneSun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))


def main():
    # Allow CLI override: blender --python blender_import.py -- --scene path/to/scene.json
    json_path = SCENE_JSON
    argv = sys.argv
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        for i, a in enumerate(rest):
            if a == "--scene" and i + 1 < len(rest):
                json_path = rest[i + 1]

    if not os.path.isabs(json_path):
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
        json_path = os.path.join(blend_dir, json_path)

    json_path = os.path.normpath(json_path)
    base_dir  = os.path.dirname(json_path)

    print(f"\nLoading scene JSON: {json_path}")
    with open(json_path) as f:
        scene_data = json.load(f)

    print(f"Scene: {scene_data['scene']}  |  Objects: {scene_data['object_count']}")

    if CLEAR_SCENE:
        clear_scene()

    placed = 0
    for entry in scene_data["objects"]:
        print(f"  [{entry['id']:03d}] {entry['type']:<6}  {entry['class_name']}")
        try:
            if entry["type"] == "plane":
                create_plane_object(entry)
                placed += 1
            else:
                obj = import_mesh_object(entry, base_dir)
                if obj:
                    place_mesh_object(obj, entry)
                    placed += 1
        except Exception as e:
            print(f"    ERROR: {e}")

    add_camera_light()

    print(f"\nDone — {placed}/{scene_data['object_count']} objects placed.")


main()
