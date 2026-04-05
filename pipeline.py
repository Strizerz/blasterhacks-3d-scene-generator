import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DA_DIR      = os.path.join(BASE_DIR, "Depth-Anything-V2")
TRIPOSR_DIR = os.path.join(BASE_DIR, "TripoSR")
SAM_DIR     = os.path.join(BASE_DIR, "SAM")

sys.path.insert(0, DA_DIR)
sys.path.insert(0, TRIPOSR_DIR)

SCENE_SCALE = 10.0
DEPTH_SCALE = 8.0


def run_depth(image_path: str, output_dir: str, encoder: str = "vitl") -> str:
    from depth_anything_v2.dpt import DepthAnythingV2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    ckpt = os.path.join(DA_DIR, "checkpoints", f"depth_anything_v2_{encoder}.pth")
    log.info(f"[1/5] Loading Depth-Anything-V2 ({encoder}) from {ckpt}")

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    raw = cv2.imread(image_path)
    depth = model.infer_image(raw, 518)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    depth_path = os.path.join(output_dir, "depth.png")
    cv2.imwrite(depth_path, depth)

    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(output_dir, "viz_depth.png"), depth_color)
    log.info(f"  Depth map saved: {depth_path}")
    return depth_path, depth


def run_3d_detection(image_path: str, depth_map: np.ndarray, output_dir: str) -> str:
    from ultralytics import YOLO

    log.info("[1.5/5] Running 3D object detection with YOLO")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO("yolov8x.pt")
    results = yolo(image_path, conf=0.3, device=device)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections_3d = []
    depth_normalized = depth_map.astype(np.float32) / 255.0

    if results[0].boxes:
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = results[0].names[int(cls)]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bbox_h, bbox_w = y2 - y1, x2 - x1

            depth_crop = depth_normalized[y1:y2, x1:x2]
            depth_val = np.median(depth_crop) if depth_crop.size > 0 else 0.5

            z_world = (1.0 - depth_val) * DEPTH_SCALE
            x_world = (cx / w - 0.5) * SCENE_SCALE
            y_world = -(cy / h - 0.5) * SCENE_SCALE

            size_norm = (bbox_w + bbox_h) / (w + h)
            size_3d = size_norm * SCENE_SCALE * 2

            detections_3d.append({
                "label": label,
                "conf": float(conf),
                "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                "pos_3d": [float(x_world), float(y_world), float(z_world)],
                "size_3d": float(size_3d),
            })

    metadata_path = os.path.join(output_dir, "detections_3d.json")
    with open(metadata_path, "w") as f:
        json.dump(detections_3d, f, indent=2)
    log.info(f"  3D detections saved: {len(detections_3d)} objects")
    return metadata_path, detections_3d


def generate_bev(img: np.ndarray, detections_3d: list, output_dir: str):
    log.info("[1.75/5] Generating bird's eye view")

    bev_size = 800
    bev_range = 15.0
    bev = np.ones((bev_size, bev_size, 3), dtype=np.uint8) * 255

    for det in detections_3d:
        x, y, z = det["pos_3d"]
        size = det["size_3d"]
        label = det["label"]

        px = int((x / bev_range + 1) * bev_size / 2)
        py = int((y / bev_range + 1) * bev_size / 2)
        ps = max(int(size / bev_range * bev_size / 2), 10)

        color = (int(det["conf"] * 255), int(det["conf"] * 255), 200)
        cv2.rectangle(bev, (px - ps, py - ps), (px + ps, py + ps), color, 2)
        cv2.circle(bev, (px, py), 3, (0, 0, 255), -1)

        cv2.putText(bev, label[:8], (px - 30, py - ps - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.line(bev, (bev_size // 2 - 50, bev_size // 2), (bev_size // 2 + 50, bev_size // 2), (0, 0, 0), 1)
    cv2.line(bev, (bev_size // 2, bev_size // 2 - 50), (bev_size // 2, bev_size // 2 + 50), (0, 0, 0), 1)
    cv2.putText(bev, f"Range: {bev_range}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    bev_path = os.path.join(output_dir, "bev.png")
    cv2.imwrite(bev_path, bev)
    log.info(f"  Bird's eye view saved: {bev_path}")


def run_segmentation(image_path: str, output_dir: str) -> str:
    from ultralytics import SAM

    blobs_dir = os.path.join(output_dir, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("[2/5] Running SAM automatic segmentation with grid prompts")
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    sam = SAM(os.path.join(SAM_DIR, "sam2.1_b.pt"))

    grid_step = 60
    points = []
    for y in range(grid_step//2, h, grid_step):
        for x in range(grid_step//2, w, grid_step):
            points.append([x, y])

    log.info(f"  Running SAM with {len(points)} point prompts (grid {grid_step}px, batched)...")

    all_sam_results = []
    batch_size = 50
    for batch_idx in range(0, len(points), batch_size):
        batch_points = np.array(points[batch_idx:batch_idx + batch_size])
        sam_results = sam(image_path, points=batch_points, device=device)
        if sam_results[0].masks is not None:
            all_sam_results.append(sam_results[0])

    if not all_sam_results:
        log.warning("  No objects detected by SAM.")
        metadata_path = os.path.join(output_dir, "blobs_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump([], f, indent=2)
        return metadata_path

    all_metadata = []
    name_counts = {}
    PALETTE = [
        (54, 162, 235), (255, 99, 132), (75, 192, 192),
        (255, 206, 86), (153, 102, 255), (255, 159, 64),
    ]

    viz_sam = img_bgr.copy()
    collected_masks = []
    processed_masks = []

    for sam_result in all_sam_results:
        if sam_result.masks is None:
            continue

        for mask_tensor in sam_result.masks.data:
            seg = (mask_tensor.cpu().numpy().astype(np.uint8) * 255)
            ys, xs = np.where(seg > 0)
            if len(xs) < 100:
                continue

            mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()

            is_duplicate = False
            for prev_seg, *_ in processed_masks:
                intersection = np.sum((seg > 0) & (prev_seg > 0))
                union = np.sum((seg > 0) | (prev_seg > 0))
                iou = intersection / (union + 1e-6)
                if iou > 0.7:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            processed_masks.append((seg, mx1, my1, mx2, my2))
            color = PALETTE[len(processed_masks) % len(PALETTE)]
            label = "object"
            name_counts[label] = name_counts.get(label, 0) + 1
            file_name = f"{label}_{name_counts[label]}.png"

            r, g, b = cv2.split(img_rgb)
            rgba = cv2.merge([r, g, b, seg])
            cropped = rgba[my1:my2, mx1:mx2]
            cv2.imwrite(os.path.join(blobs_dir, file_name), cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))

            all_metadata.append({
                "label": label,
                "file": file_name,
                "yolo_conf": 0.9,
                "bbox": [int(mx1), int(my1), int(mx2), int(my2)],
                "mask_bbox": [int(mx1), int(my1), int(mx2), int(my2)],
            })
            log.info(f"  Saved blob: {file_name}")
            collected_masks.append((seg, color, label, 0.9, mx1, my1, mx2, my2))

    for seg, color, label, conf, x1, y1, x2, y2 in collected_masks:
        overlay = viz_sam.copy()
        overlay[seg > 0] = color
        cv2.addWeighted(overlay, 0.45, viz_sam, 0.55, 0, viz_sam)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_sam, contours, -1, color, 2)

    cv2.imwrite(os.path.join(output_dir, "viz_sam.png"), viz_sam)
    log.info(f"  Saved viz_sam.png")

    metadata_path = os.path.join(output_dir, "blobs_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    log.info(f"  Metadata saved: {metadata_path} ({len(all_metadata)} objects)")
    return metadata_path


def add_depth(metadata_path: str, depth_path: str):
    log.info("[3/5] Adding depth values to metadata")
    with open(metadata_path) as f:
        metadata = json.load(f)

    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    blobs_dir = os.path.join(os.path.dirname(metadata_path), "blobs")

    for obj in metadata:
        blob = cv2.imread(os.path.join(blobs_dir, obj["file"]), cv2.IMREAD_UNCHANGED)
        mx1, my1, mx2, my2 = obj["mask_bbox"]
        depth_crop = depth_img[my1:my2, mx1:mx2]
        mask_crop  = blob[:, :, 3] if blob is not None and blob.shape[2] == 4 else np.ones(depth_crop.shape, np.uint8) * 255
        if mask_crop.shape != depth_crop.shape:
            mask_crop = cv2.resize(mask_crop, (depth_crop.shape[1], depth_crop.shape[0]))
        masked = depth_crop[mask_crop > 0]
        if len(masked) == 0:
            masked = depth_crop.flatten()
        obj["depth"] = {
            "mean":   float(np.mean(masked)),
            "median": float(np.median(masked)),
            "min":    float(np.min(masked)),
            "max":    float(np.max(masked)),
        }
        log.info(f"  {obj['file']}: depth median={obj['depth']['median']:.1f}")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_img2img_pipeline(device: str):
    from diffusers import StableDiffusionImg2ImgPipeline
    log.info("Loading Stable Diffusion img2img pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def clean_blob_with_sd(pipe, blob_path: str, label: str, out_path: str) -> str:
    if os.path.exists(out_path):
        return out_path

    rgba = Image.open(blob_path).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    rgb = bg.convert("RGB").resize((512, 512))

    prompt = (
        f"a {label}, isometric 3/4 angle view, white background, centered, "
        "clean product photography, studio lighting, sharp focus, isolated object, "
        "full object visible, three-dimensional"
    )
    negative = (
        "blurry, cluttered, shadows, multiple objects, text, watermark, "
        "flat, top-down, overhead view, 2d, painting, sketch"
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=rgb,
        strength=0.65,
        guidance_scale=9.0,
        num_inference_steps=40,
    ).images[0]

    result.save(out_path)
    log.info(f"  SD cleaned: {out_path}")
    return out_path


def run_triposr_on_blob(model, rembg_session, blob_path: str, out_dir: str,
                        device: str, texture_res: int = 1024,
                        sd_pipe=None, label: str = "object",
                        img_w: int = 512, img_h: int = 512,
                        obj_x1: int = 0, obj_y1: int = 0) -> str:
    from tsr.utils import remove_background, resize_foreground

    os.makedirs(out_dir, exist_ok=True)
    glb_path = os.path.join(out_dir, "mesh.glb")
    if os.path.exists(glb_path):
        log.info(f"  Cached: {glb_path}")
        return glb_path

    from tsr.utils import resize_foreground

    if sd_pipe is not None:
        cleaned_path = os.path.join(out_dir, "cleaned.png")
        clean_blob_with_sd(sd_pipe, blob_path, label, cleaned_path)
        sd_img = Image.open(cleaned_path).convert("RGB")
        rgba = remove_background(sd_img, rembg_session)
        rgba = resize_foreground(rgba, 0.85)
    else:
        blob_rgba = Image.open(blob_path).convert("RGBA")
        canvas = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))
        canvas.paste(blob_rgba, (obj_x1, obj_y1), mask=blob_rgba.split()[3])
        raw = canvas.convert("RGB")
        rgba = remove_background(raw, rembg_session)
        rgba = resize_foreground(rgba, 0.85)

    rgba_np = np.array(rgba).astype(np.float32) / 255.0
    rgb_np = rgba_np[:, :, :3] * rgba_np[:, :, 3:4] + (1 - rgba_np[:, :, 3:4]) * 0.5
    image = Image.fromarray((rgb_np * 255.0).astype(np.uint8))

    input_path = os.path.abspath(os.path.join(out_dir, "triposr_input.png"))
    image.save(input_path)

    import subprocess
    cmd = [
        sys.executable,
        os.path.join(TRIPOSR_DIR, "run.py"),
        input_path,
        "--output-dir", os.path.abspath(out_dir),
        "--bake-texture",
        "--model-save-format", "glb",
        "--no-remove-bg",
    ]
    log.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=TRIPOSR_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"TripoSR run.py failed with code {result.returncode}")

    nested = os.path.join(out_dir, "0", "mesh.glb")
    if os.path.exists(nested):
        import shutil
        shutil.move(nested, glb_path)

    return glb_path


def assemble_scene(metadata: list, blobs_dir: str, meshes_dir: str,
                   triposr_model, rembg_session, device: str,
                   img_w: int, img_h: int, texture_res: int,
                   sd_pipe=None):
    log.info("[5/5] Assembling scene")
    all_meshes = []

    for obj in metadata:
        log.info(f"  [{obj['label']}] {obj['file']}")
        blob_path = os.path.join(blobs_dir, obj["file"])
        obj_dir   = os.path.join(meshes_dir, os.path.splitext(obj["file"])[0])

        try:
            x1, y1 = obj["mask_bbox"][0], obj["mask_bbox"][1]
            glb_path = run_triposr_on_blob(triposr_model, rembg_session,
                                           blob_path, obj_dir, device, texture_res,
                                           sd_pipe=sd_pipe, label=obj["label"],
                                           img_w=img_w, img_h=img_h,
                                           obj_x1=x1, obj_y1=y1)
        except Exception as e:
            log.warning(f"  TripoSR failed: {e}")
            continue

        mesh = trimesh.load(glb_path, force="scene")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        mesh.apply_translation(-mesh.bounding_box.centroid)

        rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        mesh.apply_transform(rot)

        x1, y1, x2, y2 = obj["bbox"]
        bbox_diag  = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        img_diag   = (img_w ** 2 + img_h ** 2) ** 0.5
        target_sz  = (bbox_diag / img_diag) * SCENE_SCALE * 1.5
        max_extent = max(mesh.bounding_box.extents)
        scale      = target_sz / max_extent if max_extent > 0 else 1.0

        cx      = (x1 + x2) / 2.0
        cy      = (y1 + y2) / 2.0
        x_world = (cx / img_w - 0.5) * SCENE_SCALE
        y_world = -(cy / img_h - 0.5) * SCENE_SCALE
        z_world = (1.0 - obj["depth"]["median"] / 255.0) * DEPTH_SCALE

        mesh.apply_scale(scale)
        mesh.apply_translation([x_world, y_world, z_world])
        all_meshes.append(mesh)
        log.info(f"    pos=({x_world:.2f}, {y_world:.2f}, {z_world:.2f})  scale={scale:.3f}")

    if not all_meshes:
        return None
    return trimesh.util.concatenate(all_meshes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image → 3D scene pipeline")
    parser.add_argument("--image",       required=True,  help="Input scene image")
    parser.add_argument("--output-dir",  default="output/pipeline")
    parser.add_argument("--encoder",     default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--texture-res", type=int, default=1024)
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--no-sd",       action="store_true", help="Skip SD img2img cleanup step")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.image)
    img_h, img_w = img.shape[:2]

    depth_path, depth_map = run_depth(args.image, args.output_dir, args.encoder)

    det_3d_path, detections_3d = run_3d_detection(args.image, depth_map, args.output_dir)

    generate_bev(img, detections_3d, args.output_dir)

    metadata_path = run_segmentation(args.image, args.output_dir)

    add_depth(metadata_path, depth_path)

    log.info("[4/5] TripoSR will be invoked per object via run.py")
    triposr_model = None

    import rembg
    rembg_session = rembg.new_session()

    sd_pipe = None
    if not args.no_sd:
        sd_pipe = load_img2img_pipeline(device)

    with open(metadata_path) as f:
        metadata = json.load(f)

    blobs_dir  = os.path.join(args.output_dir, "blobs")
    meshes_dir = os.path.join(args.output_dir, "meshes")
    mesh = assemble_scene(metadata, blobs_dir, meshes_dir,
                          triposr_model, rembg_session, device,
                          img_w, img_h, args.texture_res,
                          sd_pipe=sd_pipe)

    scene_path = os.path.join(args.output_dir, "scene.glb")
    if mesh is None:
        log.error("No objects were successfully processed — scene is empty.")
        sys.exit(1)
    mesh.export(scene_path)
    log.info(f"\nDone. Scene saved to: {scene_path}  ({len(metadata)} objects)")
