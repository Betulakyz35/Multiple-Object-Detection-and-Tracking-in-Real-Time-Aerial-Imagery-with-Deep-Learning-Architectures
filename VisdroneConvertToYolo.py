import os
import shutil
from tqdm import tqdm

visdrone_root = r"C:\Users\Betul\VisDrone_MOT"
yolo_root = r"C:\Users\Betul\VisDrone_MOT\yoloFormat"

def convert_visdrone_to_yolo(visdrone_root, yolo_root):
    sets = ['train', 'val', 'test']
    for dataset in sets:
        image_dir = os.path.join(yolo_root, 'images', dataset)
        label_dir = os.path.join(yolo_root, 'labels', dataset)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        sequences_dir = os.path.join(visdrone_root, dataset, 'sequences')
        sequence_list = os.listdir(sequences_dir)
        total_files = 0
        for sequence in sequence_list:
            sequence_path = os.path.join(sequences_dir, sequence)
            total_files += len(os.listdir(sequence_path))

        pbar = tqdm(total=total_files, desc=f"Converting {dataset}")
        for sequence in sequence_list:
            sequence_path = os.path.join(sequences_dir, sequence)
            annotation_path = os.path.join(visdrone_root, dataset, 'annotations', f"{sequence}.txt")

            if os.path.exists(annotation_path):
                annotation_dict = {}
                with open(annotation_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        frame_id = int(parts[0])
                        x, y, w, h = map(int, parts[2:6])
                        class_id = int(parts[7])
                        if frame_id not in annotation_dict:
                            annotation_dict[frame_id] = []
                        annotation_dict[frame_id].append((x, y, w, h, class_id))

                image_list = sorted(os.listdir(sequence_path))
                for image_file in image_list:
                    image_num = int(image_file.split('.')[0])
                    image_path = os.path.join(sequence_path, image_file)
                    new_image_name = f"{sequence}_{image_file}"
                    new_image_path = os.path.join(image_dir, new_image_name)
                    shutil.copy(image_path, new_image_path)

                    if image_num in annotation_dict:
                        img = plt.imread(new_image_path)
                        img_height, img_width, _ = img.shape
                        yolo_annotations = []
                        for x, y, w, h, class_id in annotation_dict[image_num]:
                            center_x = (x + w / 2) / img_width
                            center_y = (y + h / 2) / img_height
                            yolo_w = w / img_width
                            yolo_h = h / img_height
                            yolo_annotations.append(f"{class_id} {center_x} {center_y} {yolo_w} {yolo_h}")

                        label_file = os.path.splitext(new_image_name)[0] + '.txt'
                        label_path = os.path.join(label_dir, label_file)
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_annotations))
                    pbar.update(1)
            else:
                for image_file in os.listdir(sequence_path):
                    image_path = os.path.join(sequence_path, image_file)
                    new_image_name = f"{sequence}_{image_file}"
                    new_image_path = os.path.join(image_dir, new_image_name)
                    shutil.copy(image_path, new_image_path)
                    pbar.update(1)
        pbar.close()

import matplotlib.pyplot as plt
convert_visdrone_to_yolo(visdrone_root, yolo_root)