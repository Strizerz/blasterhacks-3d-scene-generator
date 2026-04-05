import argparse
import sys
import os
import cv2
import numpy as np
import torch
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "YOLO-3D"))

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

PALETTE = [
    (255,  60,  60),
    ( 60, 180,  60),
    ( 60,  60, 255),
    (255, 180,   0),
    (180,   0, 255),
    (  0, 220, 220),
    (255, 100, 180),
    (100, 255, 140),
    (255, 160,  60),
    (140,  60, 255),
    ( 60, 220, 180),
    (220, 220,  60),
    (255,  60, 180),
    ( 60, 140, 255),
    (180, 255,  60),
    (255, 120,  60),
    ( 60, 255, 220),
    (220,  60, 255),
    (120, 200, 255),
    (255, 200, 120),
]

FALLBACK_CLASSES = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "tree", "bush", "shrub", "grass", "flower", "plant", "palm tree",
    "rock", "stone", "boulder",
    "pathway", "road", "sidewalk", "pavement", "dirt path", "gravel",
    "building", "house", "wall", "fence", "gate", "door", "window",
    "bench", "chair", "table", "trash can", "fire hydrant", "street light",
    "sign", "traffic sign", "mailbox", "pole",
    "dog", "cat", "bird",
    "sky", "cloud", "water", "river", "pond", "pool",
    "stairs", "ramp", "bridge", "pillar", "column",
    "bag", "backpack", "umbrella", "bicycle rack",
]

_class_color_registry = {}


def get_class_color(class_name):
    key = class_name.lower().strip()
    if key not in _class_color_registry:
        idx = len(_class_color_registry) % len(PALETTE)
        _class_color_registry[key] = PALETTE[idx]
    return _class_color_registry[key]


def describe_scene(image_path, device="cuda"):
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    from PIL import Image as PILImage

    print("Loading InstructBLIP scene describer...")
    dtype = torch.float16 if "cuda" in device else torch.float32

    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-7b",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    image = PILImage.open(image_path).convert("RGB")

    def vqa(question, max_tokens=256):
        inputs = processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=5,
                temperature=1.0,
                repetition_penalty=1.5,
            )
        new_tokens = ids[0][input_len:]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("  Generating scene description...")
    description = vqa("Describe this image in extensive detail, mentioning every object you can see including items on walls, floors, surfaces, background, lighting, and decorations.")

    print("  Querying scene details...")
    questions = [
        "What objects are on the walls such as paintings, mirrors, clocks, shelves or decorations?",
        "What furniture is visible in this scene?",
        "What objects are on the floor?",
        "What items are on any tables, counters or surfaces?",
        "What is visible in the background or far away?",
        "What lighting fixtures or windows are visible?",
        "What plants, artwork or decorative items are present?",
        "What are all the distinct objects in this image? List them separated by commas.",
    ]

    all_answers = [description]
    for q in questions:
        answer = vqa(q)
        print(f"    Q: {q[:60]}...")
        print(f"    A: {answer}")
        all_answers.append(answer)

    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    combined_text = " ".join(all_answers)
    noun_classes = _parse_nouns(combined_text)

    last_answer = all_answers[-1]
    comma_items = [o.strip().lower() for o in last_answer.split(",")]
    comma_list = [item for item in comma_items if 2 < len(item) <= 40 and " " in item or len(item.split()) <= 3]

    combined = list(dict.fromkeys(noun_classes + comma_list))
    combined = _filter_classes(combined)

    return description, combined


# Words that are not physical objects — skip these as YOLO-World classes
_ABSTRACT_WORDS = {
    "image", "scene", "photo", "picture", "view", "foreground",
    "area", "space", "side", "part", "section", "piece", "thing", "object",
    "item", "element", "detail", "feature", "style", "type", "kind", "way",
    "number", "amount", "variety", "collection", "group", "set", "pair",
    "color", "colour", "shade", "tone", "pattern", "texture", "design",
    "addition", "presence", "absence", "use", "example", "instance",
    "room", "center", "centre", "corner", "edge", "surface", "top", "bottom",
    "left", "right", "front", "back", "middle", "end",
    "atmosphere", "formation", "touch", "elegance", "beauty", "feel",
    "illumination", "spending", "time", "meal", "friend", "friends", "family",
    "light", "pieces", "furniture", "objects", "items", "surfaces",
    "fixture", "utensil", "utensils", "decoration", "decorations",
    "counter", "counters", "vas", "background", "overall design",
    "natural light", "additional illumination", "distinct objects",
    "dining room", "dining area", "living room", "bedroom", "kitchen",
    "bathroom", "hallway", "entrance",
}


def _filter_classes(classes):
    """Filter out non-object classes: abstract words, sentences, duplicates."""
    filtered = []
    seen = set()
    seen_base = set()

    for cls in classes:
        cls = cls.strip().rstrip(".,;:!?")
        # Skip if too long (likely a sentence fragment)
        if len(cls) > 30 or len(cls.split()) > 3:
            continue
        # Skip single-char or very short
        if len(cls) < 3:
            continue
        # Skip if contains sentence markers, filler, or quantity words
        skip_phrases = ["there is", "there are", "which ", "include",
                        "in this", "no visible", "this image",
                        "this scene", "can be", "identified", "semi-circle",
                        "formation", "overall", "additional", "distinct",
                        "two ", "three ", "four ", "five ", "six ", "seven ",
                        "eight ", "nine ", "ten "]
        skip_exact = {"that", "these", "those", "which", "it", "they", "them",
                      "what", "where", "how", "who", "its", "their"}
        if cls in skip_exact or any(p in cls for p in skip_phrases):
            continue
        # Skip numbering patterns like "1.", "2."
        if cls[0].isdigit():
            cls = cls.lstrip("0123456789.)- ").strip()
            if len(cls) < 3:
                continue
        # Strip leading articles/quantifiers
        for art in ["a ", "an ", "the ", "some ", "several ", "many ", "few ", "various ", "other "]:
            if cls.startswith(art):
                cls = cls[len(art):]
        cls = cls.strip()
        if len(cls) < 3:
            continue
        # Skip abstract/non-physical words
        if cls in _ABSTRACT_WORDS:
            continue
        # Deduplicate singular/plural: "chair" and "chairs" → keep first seen
        # Generate all candidate base forms
        candidates = {cls}
        if cls.endswith("ves") and len(cls) > 5:
            candidates.add(cls[:-3] + "f")    # shelves → shelf
        if cls.endswith("ies") and len(cls) > 5:
            candidates.add(cls[:-3] + "y")    # berries → berry
        if cls.endswith("es") and len(cls) > 4:
            candidates.add(cls[:-2])           # glasses → glass
        if cls.endswith("s") and len(cls) > 3 and not cls.endswith("ss"):
            candidates.add(cls[:-1])           # chairs → chair, vases → vase
        # Also check if this is "adj + noun" where the base noun is already seen
        words = cls.split()
        if len(words) >= 2 and (words[-1] in seen or words[-1].rstrip("s") in seen_base):
            continue
        if candidates & seen_base:
            continue
        seen.add(cls)
        seen_base.update(candidates)
        filtered.append(cls)
    return filtered


def _parse_nouns(text):
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True, capture_output=True,
            )
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        seen = set()
        nouns = []
        for chunk in doc.noun_chunks:
            # Only take the root noun (not the full "several distinct objects" phrase)
            noun = chunk.root.lemma_.lower().strip()
            if len(noun) > 2 and noun not in seen:
                seen.add(noun)
                nouns.append(noun)
            # For short phrases (2-3 words), also keep the cleaned phrase
            # Strip leading determiners/adjectives, keep "dining table" but not "several distinct objects"
            words = chunk.text.lower().strip().split()
            # Remove leading determiners
            det = {"a", "an", "the", "some", "several", "many", "few", "this", "that", "these", "those", "its", "their"}
            while words and words[0] in det:
                words.pop(0)
            phrase = " ".join(words)
            if 1 < len(words) <= 3 and phrase not in seen and len(phrase) > 2:
                seen.add(phrase)
                nouns.append(phrase)
        return nouns
    except Exception:
        words = text.lower().split()
        stop = {"a", "an", "the", "and", "or", "in", "on", "at", "with",
                "of", "is", "are", "there", "some", "many", "several"}
        return list(dict.fromkeys(w.strip(".,;:") for w in words
                                  if len(w) > 3 and w not in stop))


class LabelledBirdEyeView(BirdEyeView):
    def reset(self):
        super().reset()
        ox, oy = self.origin_x, self.origin_y
        pts = np.array([[ox, oy - 14], [ox - 10, oy + 4], [ox + 10, oy + 4]], np.int32)
        cv2.fillPoly(self.bev_image, [pts], (0, 200, 255))
        cv2.putText(self.bev_image, "CAM", (ox - 18, oy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    def draw_box(self, box_3d, color=None):
        try:
            class_name = box_3d["class_name"]
            depth_value = box_3d.get("depth_value", 0.5)

            scene_depth_m = getattr(self, "scene_depth_m", 10.0)
            available_px = self.origin_y - 20
            raw_depth = 1.0 - depth_value
            min_raw = getattr(self, "_min_raw_depth", 0.0)
            max_raw = getattr(self, "_max_raw_depth", 1.0)
            raw_range = max(max_raw - min_raw, 1e-6)
            normalized = (raw_depth - min_raw) / raw_range
            depth_m = 1.5 + normalized * (scene_depth_m - 1.5)

            x1, y1, x2, y2 = box_3d["bbox_2d"]
            center_x_2d = (x1 + x2) / 2
            rel_x = (center_x_2d / self.bev_image.shape[1]) - 0.5
            bev_x = int(self.origin_x + rel_x * self.width * 0.6)
            bev_y = int(self.origin_y - (depth_m / scene_depth_m) * available_px)

            bev_x = max(20, min(bev_x, self.width - 20))
            bev_y = max(20, min(bev_y, self.origin_y - 10))

            if color is None:
                color = get_class_color(class_name)

            dims = box_3d.get("dimensions")
            size = max(4, int(dims[1] * self.scale * 0.25)) if dims is not None else 8

            cv2.rectangle(self.bev_image,
                          (bev_x - size, bev_y - size),
                          (bev_x + size, bev_y + size),
                          color, -1)
            cv2.line(self.bev_image, (self.origin_x, self.origin_y),
                     (bev_x, bev_y), (70, 70, 70), 1)
            cv2.putText(self.bev_image, class_name[:10], (bev_x - 20, bev_y - size - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        except Exception as e:
            print(f"BEV draw error: {e}")


def make_legend_panel(registry, height):
    panel_w = 160
    panel = np.zeros((height, panel_w, 3), dtype=np.uint8)
    cv2.putText(panel, "LEGEND", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y = 38
    for class_name, color in sorted(registry.items()):
        cv2.rectangle(panel, (8, y), (22, y + 14), color, -1)
        cv2.putText(panel, class_name[:14], (28, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        y += 20
        if y + 20 > height:
            break
    return panel


def estimate_ground_plane_y(depth_map, bbox3d_estimator):
    h, w = depth_map.shape[:2]
    ground_strip = depth_map[int(h * 0.8):, :]
    ground_depth_val = float(np.median(ground_strip))
    ground_dist = 1.0 + (1.0 - ground_depth_val) * 9.0
    ground_pt = bbox3d_estimator._backproject_point(w // 2, int(h * 0.95), ground_dist)
    return float(ground_pt[1])


def compute_object_heights(bbox, depth_map, ground_y, bbox3d_estimator):
    x1, y1, x2, y2 = [int(c) for c in bbox]
    h, w = depth_map.shape[:2]
    cx = (x1 + x2) // 2
    cy_top = max(0, min(y1, h - 1))
    cy_bot = max(0, min(y2, h - 1))

    depth_top = float(depth_map[cy_top, cx])
    depth_bot = float(depth_map[cy_bot, cx])
    dist_top = 1.0 + (1.0 - depth_top) * 9.0
    dist_bot = 1.0 + (1.0 - depth_bot) * 9.0

    pt_top = bbox3d_estimator._backproject_point(cx, cy_top, dist_top)
    pt_bot = bbox3d_estimator._backproject_point(cx, cy_bot, dist_bot)

    obj_height_m   = abs(float(pt_top[1] - pt_bot[1]))
    height_from_gnd = max(0.0, abs(float(pt_bot[1] - ground_y)))
    return obj_height_m, height_from_gnd


def run(image_path, output_dir, conf=0.25, device="cuda", use_describer=True):
    _class_color_registry.clear()
    os.makedirs(output_dir, exist_ok=True)

    frame = cv2.imread(image_path)
    desc_file = os.path.join(output_dir, "scene_description.txt")

    if use_describer and os.path.exists(desc_file):
        print(f"Found cached scene description at {desc_file}, skipping InstructBLIP.")
        with open(desc_file) as f:
            content = f.read()
        parts = content.split("\n\nDiscovered classes:\n")
        description = parts[0]
        world_classes = _filter_classes([c.strip() for c in parts[1].split("\n") if c.strip()]) if len(parts) > 1 else FALLBACK_CLASSES
        print(f"\nScene description:\n  {description}\n")
        print(f"Classes to detect ({len(world_classes)}): {world_classes}\n")
    elif use_describer:
        description, discovered = describe_scene(image_path, device)
        print(f"\nScene description:\n  {description}\n")
        world_classes = discovered if discovered else FALLBACK_CLASSES
        print(f"Classes to detect ({len(world_classes)}): {world_classes}\n")
        with open(desc_file, "w") as f:
            f.write(description + "\n\nDiscovered classes:\n")
            f.write("\n".join(world_classes))
    else:
        world_classes = FALLBACK_CLASSES

    detector = ObjectDetector(model_size="world", conf_thres=conf, device=device)
    detector.set_world_classes(world_classes)
    depth_estimator = DepthEstimator(model_size="small", device=device)

    # Build camera intrinsics from actual image dimensions (not KITTI defaults)
    img_h, img_w = frame.shape[:2]
    fov_h_deg = 55.0
    fx = (img_w / 2.0) / np.tan(np.radians(fov_h_deg / 2.0))
    fy = fx  # square pixels
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    bbox3d_estimator = BBox3DEstimator(camera_matrix=K)
    bev = LabelledBirdEyeView(scale=60, size=(600, 600))

    detection_frame = frame.copy()
    detection_frame, detections = detector.detect(detection_frame, track=False)

    depth_map = depth_estimator.estimate_depth(frame)
    depth_colored = depth_estimator.colorize_depth(depth_map)

    ground_y = estimate_ground_plane_y(depth_map, bbox3d_estimator)

    result_frame = frame.copy()
    boxes_3d = []

    for detection in detections:
        bbox, score, class_id, obj_id = detection
        class_name = detector.get_class_names()[class_id]

        if class_name.lower() in ["person", "cat", "dog"]:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            depth_value = depth_estimator.get_depth_at_point(depth_map, cx, cy)
        else:
            depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method="median")

        estimated = bbox3d_estimator.estimate_3d_box(bbox, depth_value, class_name, obj_id)
        obj_height_m, height_from_gnd = compute_object_heights(bbox, depth_map, ground_y, bbox3d_estimator)

        box_3d = {
            "bbox_2d":          bbox,
            "depth_value":      depth_value,
            "depth_method":     "center" if class_name.lower() in ["person", "cat", "dog"] else "median",
            "class_name":       class_name,
            "object_id":        obj_id,
            "score":            score,
            "location":         estimated["location"],
            "dimensions":       estimated["dimensions"],
            "orientation":      estimated["orientation"],
            "obj_height_m":     obj_height_m,
            "height_from_gnd":  height_from_gnd,
        }
        boxes_3d.append(box_3d)

    viz_2d = frame.copy()
    for box_3d in boxes_3d:
        color = get_class_color(box_3d["class_name"])
        result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
        x1, y1, x2, y2 = [int(c) for c in box_3d["bbox_2d"]]
        cv2.rectangle(viz_2d, (x1, y1), (x2, y2), color, 2)
        cv2.putText(viz_2d, box_3d["class_name"][:16], (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if boxes_3d:
        raw_depths = [1.0 - b["depth_value"] for b in boxes_3d]
        min_raw = min(raw_depths)
        max_raw = max(raw_depths)
        bev._min_raw_depth = min_raw
        bev._max_raw_depth = max_raw
        bev.scene_depth_m = max(10.0, 1.0 + max_raw * 15.0)
    else:
        bev._min_raw_depth = 0.0
        bev._max_raw_depth = 1.0
        bev.scene_depth_m = 10.0

    bev.reset()
    bev_order = sorted(boxes_3d,
                       key=lambda b: b["dimensions"][1] if b.get("dimensions") is not None else 0,
                       reverse=True)
    for box_3d in bev_order:
        bev.draw_box(box_3d)
    bev_image = bev.get_image()

    legend_panel = make_legend_panel(_class_color_registry, bev_image.shape[0])
    bev_with_legend = np.hstack([bev_image, legend_panel])

    cv2.imwrite(os.path.join(output_dir, "viz_2d.png"), viz_2d)
    cv2.imwrite(os.path.join(output_dir, "viz_3d.png"), result_frame)
    cv2.imwrite(os.path.join(output_dir, "viz_bev.png"), bev_with_legend)
    cv2.imwrite(os.path.join(output_dir, "viz_depth_colored.png"), depth_colored)

    print(f"\nDetected {len(boxes_3d)} objects")
    print(f"  viz_2d.png            — 2D bounding boxes")
    print(f"  viz_3d.png            — 3D bounding boxes")
    print(f"  viz_bev.png           — Bird's eye view")
    print(f"  viz_depth_colored.png — Depth map")
    if use_describer:
        print(f"  scene_description.txt — Caption + class list")

    for b in boxes_3d:
        loc  = b["location"]
        dims = b["dimensions"]
        rot  = float(np.degrees(b["orientation"]))
        print(f"  {b['class_name']:<20} score={b['score']:.2f}  "
              f"pos=({loc[0]:.2f},{loc[1]:.2f},{loc[2]:.2f})  "
              f"rot={rot:.1f}°  gnd={b.get('height_from_gnd',0):.2f}m")

    return boxes_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D bounding box + BEV visualization")
    parser.add_argument("--image",         required=True)
    parser.add_argument("--output",        default=".")
    parser.add_argument("--conf",          type=float, default=0.25)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--no-describe",   action="store_true",
                        help="Skip InstructBLIP scene description, use fallback class list")
    parser.add_argument("--build-scene",   action="store_true",
                        help="Run SAM segmentation + 3D generation on each object and save scene.json")
    parser.add_argument("--skip-mesh-gen", action="store_true",
                        help="With --build-scene: segment and save JSON but skip 3D generation")
    parser.add_argument("--reuse-all",     action="store_true",
                        help="With --build-scene: reuse existing crops and meshes, only recompute positions")
    parser.add_argument("--generator",     default="hunyuan3d",
                        choices=["hunyuan3d", "triposr"],
                        help="3D mesh generator (default: hunyuan3d, falls back to triposr)")
    args = parser.parse_args()

    boxes = run(args.image, args.output, args.conf, args.device,
                use_describer=not args.no_describe)

    if args.build_scene and boxes:
        from scene_builder import build_scene
        build_scene(
            image_path=args.image,
            boxes_3d=boxes,
            output_dir=args.output,
            device=args.device,
            skip_mesh_gen=args.skip_mesh_gen,
            reuse_existing=args.reuse_all,
            generator=args.generator,
        )
