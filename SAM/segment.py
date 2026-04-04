import os
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO, SAM

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "input", "test.png")
output_root = os.path.join(script_dir, "output")
blobs_dir = os.path.join(output_root, "blobs")
os.makedirs(blobs_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO("yolo11x.pt")
sam_model = SAM("sam2.1_b.pt")

yolo_results = yolo_model(image_path, conf=0.25, device=device)

yolo_boxes = []
yolo_labels = []
yolo_confs = []
if yolo_results[0].boxes:
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    yolo_labels = [yolo_results[0].names[int(c)] for c in yolo_results[0].boxes.cls.cpu().numpy()]
    yolo_confs = yolo_results[0].boxes.conf.cpu().numpy()

img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

all_metadata = []
name_counts = {}

if len(yolo_boxes) == 0:
    print("No YOLO detections found.")
else:
    sam_results = sam_model(image_path, bboxes=yolo_boxes, device=device)

    for i, (box, label, conf) in enumerate(zip(yolo_boxes, yolo_labels, yolo_confs)):
        x1, y1, x2, y2 = map(int, box)

        mask = None
        if sam_results[0].masks is not None and i < len(sam_results[0].masks.data):
            mask_tensor = sam_results[0].masks.data[i]
            mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255

        if mask is None or mask.sum() == 0:
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            continue

        mx1, my1 = x_indices.min(), y_indices.min()
        mx2, my2 = x_indices.max(), y_indices.max()

        box_mask = np.zeros_like(mask)
        box_mask[y1:y2, x1:x2] = 255
        intersection = np.logical_and(mask > 0, box_mask > 0).sum()
        mask_area = (mask > 0).sum()
        box_area = (box_mask > 0).sum()
        union = mask_area + box_area - intersection
        iou = float(intersection / union) if union > 0 else 0.0

        name_counts[label] = name_counts.get(label, 0) + 1
        file_name = f"{label}_{name_counts[label]}.png"
        file_path = os.path.join(blobs_dir, file_name)

        r, g, b = cv2.split(img_rgb)
        rgba_img = cv2.merge([r, g, b, mask])
        cropped = rgba_img[my1:my2, mx1:mx2]
        cv2.imwrite(file_path, cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))

        all_metadata.append({
            "label": label,
            "file": file_name,
            "yolo_conf": float(conf),
            "mask_iou": iou,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "mask_bbox": [int(mx1), int(my1), int(mx2), int(my2)],
        })
        print(f"Saved: {file_name} (conf={conf:.2f}, iou={iou:.2f})")

metadata_path = os.path.join(output_root, "blobs_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(all_metadata, f, indent=4)

print(f"\nDone. {len(all_metadata)} objects saved to {blobs_dir}")