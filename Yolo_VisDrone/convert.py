import os
import cv2

# Konfigürasyonlar
input_train_dir = 'C:/Users/marsh/Desktop/VisDroneDataSet/train/annotations'
input_val_dir = 'C:/Users/marsh/Desktop/VisDroneDataSet/val/annotations'
output_train_dir = 'C:/Users/marsh/Desktop/VisDroneDataSet/train/labels'
output_val_dir = 'C:/Users/marsh/Desktop/VisDroneDataSet/val/labels'

# Class isimleri - index 1'den başlar (çünkü 0 = ignored, 11 = others dahil edilmeyecek)
class_names = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor"
]

# YOLO formatına dönüştürme fonksiyonu
def convert_to_yolo_format(annotation_line, img_width, img_height):
    parts = annotation_line.strip().split(',')

    if len(parts) < 6:
        print(f"Geçersiz format: {annotation_line}")
        return None

    try:
        class_id_original = int(parts[5])

        # 0 = ignored region, 11 = others → atla!
        if class_id_original == 0 or class_id_original == 11:
            return None

        # class_id'yi YOLO eğitim dizinine göre yeniden indexle (çünkü 1 → 0, 2 → 1, ... 10 → 9)
        class_id = class_id_original - 1

        x_min = float(parts[0])
        y_min = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])

        x_center = x_min + width / 2
        y_center = y_min + height / 2

        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        return f"{class_id} {x_center} {y_center} {width} {height}"
    except ValueError as e:
        print(f"ValueError occurred: {e} in annotation {annotation_line}")
        return None

def process_annotations(input_dir, output_dir, image_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            annotation_file = os.path.join(input_dir, filename)
            image_file = os.path.join(image_dir, filename.replace('.txt', '.jpg'))

            if not os.path.exists(image_file):
                print(f"Görsel bulunamadı: {image_file}")
                continue

            img = cv2.imread(image_file)
            if img is None:
                print(f"Görsel okunamadı: {image_file}")
                continue

            img_height, img_width, _ = img.shape

            with open(annotation_file, 'r') as file:
                lines = file.readlines()

            yolo_annotations = []
            for line in lines:
                yolo_annotation = convert_to_yolo_format(line, img_width, img_height)
                if yolo_annotation:
                    yolo_annotations.append(yolo_annotation)

            output_annotation_file = os.path.join(output_dir, filename)
            with open(output_annotation_file, 'w') as output_file:
                output_file.write("\n".join(yolo_annotations))

    print(f"{input_dir} için dönüştürme tamamlandı!")

process_annotations(input_train_dir, output_train_dir, 'C:/Users/marsh/Desktop/VisDroneDataSet/train/images')
process_annotations(input_val_dir, output_val_dir, 'C:/Users/marsh/Desktop/VisDroneDataSet/val/images')

print("Tüm dönüştürme işlemleri tamamlandı")
