import os
import json
import cv2
from tqdm import tqdm

def visdrone_to_coco(image_dir, annot_dir, output_json_path):
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"}
    ]

    valid_category_ids = set(range(1, 11))  # 1-10 arası geçerli ID’ler
    image_id = 0
    annotation_id = 0

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for img_file in tqdm(image_files, desc="Dönüştürülüyor"):
        img_path = os.path.join(image_dir, img_file)
        ann_file = os.path.join(annot_dir, img_file.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Görsel okunamadı: {img_path}")
            continue

        height, width = img.shape[:2]

        images.append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        if os.path.exists(ann_file):
            with open(ann_file) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 8 or '' in parts:
                    continue
                try:
                    x, y, w, h = map(int, parts[:4])
                    category_id = int(parts[5])
                    if category_id == 0 or category_id == 11 or category_id not in valid_category_ids:
                        continue
                    if w <= 0 or h <= 0:
                        continue

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                except ValueError as e:
                    print(f"Hatalı satır: {line.strip()} → {e}")
                    continue

        image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"\n✅ COCO JSON başarıyla oluşturuldu: {output_json_path}")


# Örnek kullanım:
visdrone_to_coco(
    image_dir=r'C:\Users\marsh\Desktop\VisDroneDataSet\val\images',
    annot_dir=r'C:\Users\marsh\Desktop\VisDroneDataSet\val\annotations',
    output_json_path='visdrone_val_coco.json'
)
