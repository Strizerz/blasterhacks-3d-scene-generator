"""
scene_builder.py

Reads SAM output (blobs_metadata.json + blob PNGs), runs TripoSR on each
detected object, then assembles a single GLB scene with objects placed
according to their image position and depth.

Usage:
    python scene_builder.py --metadata SAM/output/blobs_metadata.json
                            --blobs-dir SAM/output/blobs
                            --output-dir output/scene
                            --scene-out output/scene/scene.glb

Depth map convention (Depth-Anything): higher value = closer to camera.
Objects are placed on a Z axis where closer objects have smaller Z.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image

# --- TripoSR imports ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TripoSR"))
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
from tsr.bake_texture import bake_texture

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

IMG_W = 1200
IMG_H = 1200
SCENE_SCALE = 10.0      # world units across the full image width
DEPTH_SCALE = 8.0       # world units from nearest to farthest object


def depth_to_z(depth_median: float) -> float:
    """Convert 0-255 depth value to world Z. Higher depth = closer = smaller Z."""
    # depth 255 → z = 0 (closest), depth 0 → z = DEPTH_SCALE (farthest)
    return (1.0 - depth_median / 255.0) * DEPTH_SCALE


def bbox_to_xy(bbox: list) -> tuple:
    """Convert pixel bbox [x1,y1,x2,y2] to centred world XY coordinates."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x_world = (cx / IMG_W - 0.5) * SCENE_SCALE
    y_world = -(cy / IMG_H - 0.5) * SCENE_SCALE   # flip Y for 3D
    return x_world, y_world


def bbox_size_scale(bbox: list) -> float:
    """Estimate a rough scale factor based on object size in image."""
    x1, y1, x2, y2 = bbox
    diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    img_diag = (IMG_W ** 2 + IMG_H ** 2) ** 0.5
    return (diag / img_diag) * SCENE_SCALE * 1.5


def run_triposr(model, image_path: str, output_dir: str, texture_resolution: int = 1024) -> str:
    """Run TripoSR on a single image and return path to output GLB."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "mesh.glb")

    if os.path.exists(out_path):
        log.info(f"  Cached: {out_path}")
        return out_path

    image = Image.open(image_path)

    # Remove background if image has no alpha
    if image.mode != "RGBA":
        image = remove_background(image, rembg_session)
    image = resize_foreground(image, 0.85)

    with torch.no_grad():
        scene_codes = model([image], device=device)

    meshes = model.extract_mesh(scene_codes, resolution=256)

    bake_out = bake_texture(meshes[0], model, scene_codes[0], texture_resolution)
    texture_img = Image.fromarray(
        (bake_out["colors"] * 255.0).astype(np.uint8)
    ).transpose(Image.FLIP_TOP_BOTTOM)

    vertices = meshes[0].vertices[bake_out["vmapping"]]
    faces = bake_out["indices"]
    uvs = bake_out["uvs"]

    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture_img)
    visuals = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, process=False)
    mesh.export(out_path)

    log.info(f"  Exported: {out_path}")
    return out_path


def build_scene(metadata: list, blobs_dir: str, meshes_dir: str) -> trimesh.Scene:
    scene = trimesh.Scene()

    for obj in metadata:
        label = obj["label"]
        blob_file = obj["file"]
        bbox = obj["bbox"]
        depth_median = obj["depth"]["median"]

        log.info(f"Processing: {label} ({blob_file})")

        blob_path = os.path.join(blobs_dir, blob_file)
        obj_output_dir = os.path.join(meshes_dir, os.path.splitext(blob_file)[0])

        try:
            glb_path = run_triposr(triposr_model, blob_path, obj_output_dir)
        except Exception as e:
            log.warning(f"  TripoSR failed for {blob_file}: {e}")
            continue

        mesh = trimesh.load(glb_path, force="scene")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Normalise mesh to unit size then scale by bbox size
        extents = mesh.bounding_box.extents
        max_extent = max(extents) if max(extents) > 0 else 1.0
        target_scale = bbox_size_scale(bbox)
        scale = target_scale / max_extent

        x, y = bbox_to_xy(bbox)
        z = depth_to_z(depth_median)

        transform = trimesh.transformations.scale_and_translate(
            scale=[scale, scale, scale],
            translate=[x, y, z]
        )
        scene.add_geometry(mesh, transform=transform, node_name=f"{label}_{blob_file}")
        log.info(f"  Placed at ({x:.2f}, {y:.2f}, {z:.2f}), scale={scale:.3f}")

    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="SAM/output/blobs_metadata.json")
    parser.add_argument("--blobs-dir", default="SAM/output/blobs")
    parser.add_argument("--output-dir", default="output/scene/meshes")
    parser.add_argument("--scene-out", default="output/scene/scene.glb")
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load rembg session for background removal
    import rembg
    rembg_session = rembg.new_session()

    # Load TripoSR model
    log.info("Loading TripoSR model...")
    triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    triposr_model.renderer.set_chunk_size(8192)
    triposr_model.to(device)
    log.info("Model loaded.")

    with open(args.metadata) as f:
        metadata = json.load(f)

    os.makedirs(os.path.dirname(args.scene_out), exist_ok=True)

    scene = build_scene(metadata, args.blobs_dir, args.output_dir)
    scene.export(args.scene_out)
    log.info(f"\nScene saved to: {args.scene_out} ({len(scene.geometry)} objects)")
