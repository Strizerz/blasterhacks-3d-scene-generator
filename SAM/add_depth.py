import os
import cv2
import json
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
depth_path = os.path.join(script_dir, "output", "test.png")
metadata_path = os.path.join(script_dir, "output", "blobs_metadata.json")
blobs_dir = os.path.join(script_dir, "output", "blobs")

with open(metadata_path, "r") as f:
    all_metadata = json.load(f)

depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

for obj in all_metadata:
    blob_path = os.path.join(blobs_dir, obj["file"])
    blob = cv2.imread(blob_path, cv2.IMREAD_UNCHANGED)

    mx1, my1, mx2, my2 = obj["mask_bbox"]

    if depth_img.shape != blob.shape[:2]:
        depth_resized = cv2.resize(depth_img, (depth_img.shape[1], depth_img.shape[0]))
    else:
        depth_resized = depth_img

    depth_crop = depth_resized[my1:my2, mx1:mx2]
    mask_crop = blob[:, :, 3] if blob.shape[2] == 4 else np.ones(blob.shape[:2], dtype=np.uint8) * 255
    masked_depth = depth_crop[mask_crop > 0]

    obj["depth"] = {
        "mean": float(np.mean(masked_depth)),
        "median": float(np.median(masked_depth)),
        "min": float(np.min(masked_depth)),
        "max": float(np.max(masked_depth)),
        "std": float(np.std(masked_depth)),
        "crop": depth_crop.tolist(),
        "mask_crop": mask_crop.tolist(),
    }
    print(f"Depth added: {obj['file']} (median={obj['depth']['median']:.1f})")

with open(metadata_path, "w") as f:
    json.dump(all_metadata, f, indent=4)

print(f"\nDone. Depth added to {len(all_metadata)} objects.")