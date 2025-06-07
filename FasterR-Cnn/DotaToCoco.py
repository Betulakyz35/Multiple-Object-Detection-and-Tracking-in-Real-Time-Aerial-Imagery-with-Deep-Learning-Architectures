import os
import json
import cv2
from tqdm import tqdm

def convert_dota_to_coco(image_dir, label_dir, output_json_path):
    images = []
    annotations = []
    categories = [
        {"id": 11, "name": "plane"},
        {"id": 12, "name": "ship"},
        {"id": 13, "name": "storage-tank"},
        {"id": 14, "name": "baseball-diamond"},
        {"id": 15, "name": "tennis-court"},
        {"id": 16, "name": "basketball-court"},
        {"id": 17, "name": "ground-track-field"},
        {"id": 18, "name": "harbor"},
        {"id": 19, "name": "bridge"},
        {"id": 20, "name": "large-vehicle"},
        {"id": 21, "name": "small-vehicle"},
        {"id": 22, "name": "helicopter"},
        {"id": 23, "name": "roundabout"},
        {"id": 24, "name": "soccer-ball-field"},
        {"id": 25, "name": "swimming-pool"}
    ]

    category_mapping = {c["name"]: c["id"] for c in categories}

    image_id = 0
    annotation_id = 0

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        images.append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        if os.path.exists(label_file):
            with open(label_file) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                label_name = parts[8]
                # Get axis-aligned bbox
                xmin = min(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                xmax = max(x1, x2, x3, x4)
                ymax = max(y1, y2, y3, y4)
                width_box = xmax - xmin
                height_box = ymax - ymin
                if width_box <=0 or height_box <=0:
                    continue
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_mapping.get(label_name, 1), # default to 1 if missing
                    "bbox": [xmin, ymin, width_box, height_box],
                    "area": width_box * height_box,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

convert_dota_to_coco(
    image_dir= r'C:\Users\marsh\Desktop\DotaVal\images',
    label_dir= r'C:\Users\marsh\Desktop\DotaVal\annotations',
    output_json_path='dota_val_coco.json'
)
