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

yolo_results = yolo_model(image_path, conf=0.15, device=device)
sam_results = sam_model(image_path, device=device)

all_metadata = []
name_counts = {}

yolo_boxes = []
yolo_labels = []
if yolo_results[0].boxes:
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    yolo_labels = [yolo_results[0].names[int(c)] for c in yolo_results[0].boxes.cls.cpu().numpy()]

for r_idx, sam_result in enumerate(sam_results):
    img = sam_result.orig_img
    if sam_result.masks is not None:
        for i, mask_tensor in enumerate(sam_result.masks.data):
            mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255
            y_indices, x_indices = np.where(mask > 0)
            
            if len(x_indices) == 0: continue
            
            x1, y1, x2, y2 = x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()
            
            final_label = "unidentified"
            max_iou = 0
            
            for b_idx, y_box in enumerate(yolo_boxes):
                inter_x1 = max(x1, y_box[0])
                inter_y1 = max(y1, y_box[1])
                inter_x2 = min(x2, y_box[2])
                inter_y2 = min(y2, y_box[3])
                
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter_area > 0:
                    box_area = (y_box[2] - y_box[0]) * (y_box[3] - y_box[1])
                    mask_bbox_area = (x2 - x1) * (y2 - y1)
                    union_area = box_area + mask_bbox_area - inter_area
                    iou = inter_area / union_area
                    
                    if iou > max_iou:
                        max_iou = iou
                        final_label = yolo_labels[b_idx]

            name_counts[final_label] = name_counts.get(final_label, 0) + 1
            file_name = f"{final_label}_{name_counts[final_label]}.png"
            file_path = os.path.join(blobs_dir, file_name)

            b, g, r = cv2.split(img)
            rgba_img = cv2.merge([r, g, b, mask])
            cropped_blob = rgba_img[y1:y2, x1:x2]
            cv2.imwrite(file_path, cv2.cvtColor(cropped_blob, cv2.COLOR_RGBA2BGRA))

            all_metadata.append({"label": final_label, "file": file_name, "iou_score": float(max_iou)})

metadata_path = os.path.join(output_root, "blobs_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(all_metadata, f, indent=4)