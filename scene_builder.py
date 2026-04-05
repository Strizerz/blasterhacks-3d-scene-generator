import os
import sys
import json
import subprocess
import numpy as np
import cv2
from PIL import Image as PILImage

TRIPOSR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TripoSR")
HUNYUAN3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")

# Lazy-loaded Hunyuan3D pipelines (loaded once, reused for all objects)
_hunyuan_shapegen = None
_hunyuan_texgen = None
_hunyuan_rembg = None

PLANE_CLASSES = {
    "floor", "ground", "carpet", "rug", "mat",
    "ceiling", "roof",
    "wall", "room", "background",
    "road", "pavement", "sidewalk", "pathway", "path", "dirt path", "gravel",
    "sky", "water", "river", "pond", "pool", "lake",
    "grass", "lawn",
}

PLANE_NORMALS = {
    "floor": "up", "ground": "up", "carpet": "up", "rug": "up", "mat": "up",
    "grass": "up", "lawn": "up", "road": "up", "pavement": "up",
    "sidewalk": "up", "pathway": "up", "path": "up", "dirt path": "up", "gravel": "up",
    "water": "up", "river": "up", "pond": "up", "pool": "up", "lake": "up",
    "ceiling": "down", "roof": "down",
    "wall": "forward", "room": "forward", "background": "forward", "sky": "forward",
}


def estimate_world_scale(bbox_2d, depth_value, image_wh, fov_deg=60.0):
    """
    Estimate the real-world size of an object from its 2D bbox and depth.

    Larger bbox at same depth = bigger object.
    Same bbox at farther depth = bigger object (perspective).

    Returns (width_m, height_m) in metres.
    """
    img_w, img_h = image_wh
    x1, y1, x2, y2 = bbox_2d
    bbox_w_px = x2 - x1
    bbox_h_px = y2 - y1

    distance_m = 1.0 + (1.0 - depth_value) * 9.0

    fov_rad = np.radians(fov_deg)
    sensor_half_w = 2.0 * distance_m * np.tan(fov_rad / 2.0)
    px_to_m = sensor_half_w / img_w

    width_m  = bbox_w_px * px_to_m
    height_m = bbox_h_px * px_to_m

    return round(float(width_m), 4), round(float(height_m), 4)


def _find_mesh(search_dir):
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith((".glb", ".obj", ".ply")):
                return os.path.join(root, f)
    return None


def run_triposr(crop_path, mesh_output_dir):
    abs_out = os.path.abspath(mesh_output_dir)
    cached = _find_mesh(abs_out)
    if cached:
        print(f"    Cached: {cached}")
        return cached
    os.makedirs(abs_out, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "run.py",
            os.path.abspath(crop_path),
            "--output-dir", abs_out,
            "--no-remove-bg",
        ],
        cwd=TRIPOSR_DIR,
        check=True,
    )
    result = _find_mesh(abs_out)
    if result is None:
        raise FileNotFoundError(f"TripoSR produced no mesh in {abs_out}")
    return result


def _load_hunyuan3d(device="cuda"):
    global _hunyuan_shapegen, _hunyuan_texgen, _hunyuan_rembg

    if _hunyuan_shapegen is not None:
        return _hunyuan_shapegen, _hunyuan_texgen, _hunyuan_rembg

    # Add Hunyuan3D-2 to path so hy3dgen can be imported
    if HUNYUAN3D_DIR not in sys.path:
        sys.path.insert(0, HUNYUAN3D_DIR)

    # Ensure CUDA DLLs from torch are findable (Windows needs this for custom extensions)
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if hasattr(os, "add_dll_directory") and os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)

    # Ensure custom_rasterizer egg is importable (kernel DLL needs torch DLLs above)
    cr_egg = os.path.join(
        os.path.dirname(torch.__file__), "..",
        "custom_rasterizer-0.1-py3.12-win-amd64.egg"
    )
    cr_egg = os.path.normpath(cr_egg)
    if os.path.isdir(cr_egg) and cr_egg not in sys.path:
        sys.path.insert(0, cr_egg)
    try:
        import custom_rasterizer_kernel  # noqa: F401
        import custom_rasterizer  # noqa: F401
        print("  custom_rasterizer loaded OK")
    except Exception as e:
        print(f"  custom_rasterizer pre-import failed: {e}")

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    print("  Loading Hunyuan3D shape generator...")
    _hunyuan_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    _hunyuan_rembg = BackgroundRemover()

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        print("  Loading Hunyuan3D texture painter...")
        _hunyuan_texgen = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
        print("  Texture painter loaded.")
    except Exception as e:
        print(f"  Texture painter unavailable ({e}), will generate untextured meshes.")
        _hunyuan_texgen = None

    return _hunyuan_shapegen, _hunyuan_texgen, _hunyuan_rembg


def run_hunyuan3d(crop_path, mesh_output_dir, device="cuda"):
    abs_out = os.path.abspath(mesh_output_dir)
    cached = _find_mesh(abs_out)
    if cached:
        print(f"    Cached: {cached}")
        return cached
    os.makedirs(abs_out, exist_ok=True)

    shapegen, texgen, rembg = _load_hunyuan3d(device)

    image = PILImage.open(crop_path)
    if image.mode == "RGB":
        image = rembg(image.convert("RGBA"))
    else:
        image = image.convert("RGBA")

    mesh = shapegen(image=image)[0]

    if texgen is not None:
        try:
            mesh = texgen(mesh, image=image)
        except Exception as e:
            print(f"    Texture painting failed ({e}), exporting untextured mesh.")

    glb_path = os.path.join(abs_out, "mesh.glb")
    mesh.export(glb_path)

    result = _find_mesh(abs_out)
    if result is None:
        raise FileNotFoundError(f"Hunyuan3D produced no mesh in {abs_out}")
    return result


def segment_and_extract(image_path, boxes_3d, crops_dir, device="cuda"):
    import torch
    from ultralytics import SAM

    os.makedirs(crops_dir, exist_ok=True)

    print("  Loading SAM...")
    sam = SAM("sam2_b.pt")

    frame_bgr = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    crop_paths = []
    seg_vis = frame_bgr.copy()
    rng = np.random.default_rng(42)
    colors = [tuple(int(c) for c in rng.integers(80, 255, 3)) for _ in boxes_3d]

    for idx, box_3d in enumerate(boxes_3d):
        class_name = box_3d["class_name"]
        x1, y1, x2, y2 = [int(c) for c in box_3d["bbox_2d"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = colors[idx]

        if x2 - x1 < 10 or y2 - y1 < 10:
            print(f"  [{idx}] {class_name} — bbox too small, skipping")
            crop_paths.append(None)
            continue

        mask = None
        try:
            results = sam(image_path, bboxes=[[x1, y1, x2, y2]], device=device, verbose=False)
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    mask = masks[0].astype(bool)
        except Exception as e:
            print(f"  [{idx}] {class_name} — SAM error: {e}")

        if mask is not None:
            overlay = seg_vis.copy()
            overlay[mask] = (overlay[mask].astype(np.float32) * 0.5 +
                             np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
            seg_vis = overlay
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_vis, contours, -1, color, 2)

        cv2.rectangle(seg_vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(seg_vis, class_name[:14], (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = frame_rgb.astype(np.float32) / 255.0
        if mask is not None:
            rgba[:, :, 3] = mask.astype(np.float32)
        else:
            rgba[y1:y2, x1:x2, 3] = 1.0

        rgb_comp = rgba[:, :, :3] * rgba[:, :, 3:4] + (1 - rgba[:, :, 3:4]) * 0.5
        composite = (rgb_comp * 255).astype(np.uint8)

        pad = 10
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = composite[cy1:cy2, cx1:cx2]

        safe_name = class_name.replace(" ", "_").replace("/", "_")
        crop_path = os.path.join(crops_dir, f"{idx:03d}_{safe_name}.png")
        PILImage.fromarray(crop).save(crop_path)
        crop_paths.append(crop_path)
        print(f"  [{idx}] {class_name} → {os.path.basename(crop_path)}")

    del sam
    torch.cuda.empty_cache()

    return crop_paths, seg_vis


def _find_existing_crops(boxes_3d, crops_dir):
    """Find existing crop images on disk, return list matching boxes_3d order."""
    crop_paths = []
    for idx, box_3d in enumerate(boxes_3d):
        safe_name = box_3d["class_name"].replace(" ", "_").replace("/", "_")
        path = os.path.join(crops_dir, f"{idx:03d}_{safe_name}.png")
        crop_paths.append(path if os.path.exists(path) else None)
    return crop_paths


def build_scene(image_path, boxes_3d, output_dir, device="cuda",
                skip_mesh_gen=False, reuse_existing=False, generator="hunyuan3d"):
    print("\n=== Building Scene ===")

    crops_dir  = os.path.join(output_dir, "crops")
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    frame = cv2.imread(image_path)
    img_h, img_w = frame.shape[:2]

    if reuse_existing and os.path.isdir(crops_dir):
        print(f"Reusing existing crops from {crops_dir}")
        crop_paths = _find_existing_crops(boxes_3d, crops_dir)
        found = sum(1 for p in crop_paths if p is not None)
        print(f"  Found {found}/{len(boxes_3d)} existing crops")
        seg_vis = None
    else:
        print(f"Segmenting {len(boxes_3d)} objects with SAM...")
        crop_paths, seg_vis = segment_and_extract(image_path, boxes_3d, crops_dir, device)
        cv2.imwrite(os.path.join(output_dir, "viz_sam.png"), seg_vis)
        print(f"  SAM visualization → viz_sam.png")

    use_hunyuan = generator == "hunyuan3d" and os.path.isdir(HUNYUAN3D_DIR)
    if generator == "hunyuan3d" and not os.path.isdir(HUNYUAN3D_DIR):
        print(f"  Hunyuan3D-2 not found at {HUNYUAN3D_DIR}, falling back to TripoSR")
        use_hunyuan = False

    gen_name = "Hunyuan3D" if use_hunyuan else "TripoSR"
    print(f"  3D generator: {gen_name}")

    scene_objects = []

    for idx, (box_3d, crop_path) in enumerate(zip(boxes_3d, crop_paths)):
        class_name = box_3d["class_name"]
        loc  = box_3d["location"]
        dims = box_3d["dimensions"]
        safe_name = class_name.replace(" ", "_").replace("/", "_")
        is_plane = class_name.lower().strip() in PLANE_CLASSES
        plane_normal = PLANE_NORMALS.get(class_name.lower().strip(), "up") if is_plane else None
        obj_type = "plane" if is_plane else "mesh"

        world_w, world_h = estimate_world_scale(
            box_3d["bbox_2d"], box_3d["depth_value"], (img_w, img_h)
        )

        mesh_rel = None
        if not is_plane:
            mesh_out = os.path.join(meshes_dir, f"{idx:03d}_{safe_name}")
            # Check for already-generated mesh on disk
            existing = _find_mesh(mesh_out) if os.path.isdir(mesh_out) else None
            if existing:
                mesh_rel = os.path.relpath(existing, output_dir).replace("\\", "/")
                print(f"  [{idx}] {class_name} → cached {mesh_rel}")
            elif crop_path is not None and not skip_mesh_gen:
                print(f"  {gen_name} [{idx}] {class_name}...")
                try:
                    if use_hunyuan:
                        mesh_abs = run_hunyuan3d(crop_path, mesh_out, device)
                    else:
                        mesh_abs = run_triposr(crop_path, mesh_out)
                    mesh_rel = os.path.relpath(mesh_abs, output_dir).replace("\\", "/")
                    print(f"    → {mesh_rel}")
                except Exception as e:
                    print(f"    {gen_name} failed: {e}")
                    if use_hunyuan:
                        print(f"    Falling back to TripoSR...")
                        try:
                            mesh_abs = run_triposr(crop_path, mesh_out)
                            mesh_rel = os.path.relpath(mesh_abs, output_dir).replace("\\", "/")
                            print(f"    → {mesh_rel}")
                        except Exception as e2:
                            print(f"    TripoSR also failed: {e2}")
            else:
                print(f"  [{idx}] {class_name} → no mesh (skipped)")
        else:
            print(f"  [{idx}] {class_name} → plane")

        entry = {
            "id":           idx,
            "type":         obj_type,
            "class_name":   class_name,
            "score":        round(float(box_3d["score"]), 3),
            "mesh_path":    mesh_rel,
            "crop_path":    os.path.relpath(crop_path, output_dir).replace("\\", "/") if crop_path else None,
            "position": {
                "x": round(float(loc[0]), 4),
                "y": round(float(loc[1]), 4),
                "z": round(float(loc[2]), 4),
            },
            "rotation_y_deg":       round(float(np.degrees(box_3d["orientation"])), 2),
            "scale": {
                "width":  world_w,
                "height": world_h,
                "depth":  round(min(world_w, world_h), 4),
            },
            "dimensions": {
                "height": round(float(dims[0]), 4),
                "width":  round(float(dims[1]), 4),
                "length": round(float(dims[2]), 4),
            },
            "height_from_ground_m": round(float(box_3d.get("height_from_gnd", 0.0)), 4),
            "depth_value":          round(float(box_3d["depth_value"]), 4),
            "bbox_2d":              [int(c) for c in box_3d["bbox_2d"]],
        }
        if is_plane:
            entry["plane_normal"] = plane_normal
        scene_objects.append(entry)

    json_path = os.path.join(output_dir, "scene.json")
    with open(json_path, "w") as f:
        json.dump({
            "scene":        os.path.basename(output_dir),
            "source_image": os.path.abspath(image_path).replace("\\", "/"),
            "image_size":   [img_w, img_h],
            "generator":    gen_name,
            "object_count": len(scene_objects),
            "objects":      scene_objects,
        }, f, indent=2)

    # Free Hunyuan3D VRAM if it was loaded
    global _hunyuan_shapegen, _hunyuan_texgen, _hunyuan_rembg
    if _hunyuan_shapegen is not None:
        import torch
        del _hunyuan_shapegen, _hunyuan_texgen, _hunyuan_rembg
        _hunyuan_shapegen = _hunyuan_texgen = _hunyuan_rembg = None
        torch.cuda.empty_cache()

    print(f"\nScene JSON → {json_path}")
    print(f"Objects with meshes: {sum(1 for o in scene_objects if o['mesh_path'])}/{len(scene_objects)}")

    return scene_objects
