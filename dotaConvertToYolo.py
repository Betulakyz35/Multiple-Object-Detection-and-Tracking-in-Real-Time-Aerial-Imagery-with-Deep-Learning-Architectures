import os
import cv2

# Define class mapping starting from index 12
class_mapping = {
    "plane": 12,
    "ship": 13,
    "storage-tank": 14,
    "baseball-diamond": 15,
    "tennis-court": 16,
    "basketball-court": 17,
    "ground-track-field": 18,
    "harbor": 19,
    "bridge": 20,
    "large-vehicle": 21,
    "small-vehicle": 22,
    "helicopter": 23,
    "roundabout": 24,
    "soccer-ball-field": 25,
    "swimming-pool": 26,
    "container-crane": 27
}

# Function to get image size
def get_image_size(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found!")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Couldn't read image {image_path}")
        return None
    return img.shape[1], img.shape[0]  # (width, height)


# Function to convert a single DOTA annotation file
def convert_dota_to_yolo(dota_file, image_folder, output_folder):
    file_name = os.path.splitext(os.path.basename(dota_file))[0]

    # Find corresponding image
    image_extensions = [".jpg", ".png", ".jpeg"]
    image_path = None
    for ext in image_extensions:
        possible_image_path = os.path.join(image_folder, file_name + ext)
        if os.path.exists(possible_image_path):
            image_path = possible_image_path
            break

    if not image_path:
        print(f"Skipping {dota_file}: No matching image found.")
        return

    # Get image dimensions
    image_size = get_image_size(image_path)
    if not image_size:
        return
    image_width, image_height = image_size

    with open(dota_file, 'r') as f:
        lines = f.readlines()

    yolo_labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue  # Skip invalid lines

        # Extract class and difficulty level
        class_name = parts[-2]
        difficulty = parts[-1]  # Ignoring difficulty level

        # Convert class name to YOLO class ID
        if class_name not in class_mapping:
            continue  # Skip unknown classes

        class_id = class_mapping[class_name]

        # Extract bounding box coordinates
        coords = list(map(float, parts[:-2]))
        x_coords = coords[0::2]
        y_coords = coords[1::2]

        # Compute YOLO bounding box
        x_center = sum(x_coords) / 4 / image_width
        y_center = sum(y_coords) / 4 / image_height
        bbox_width = (max(x_coords) - min(x_coords)) / image_width
        bbox_height = (max(y_coords) - min(y_coords)) / image_height

        # Format YOLO annotation
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Save converted annotation
    os.makedirs(output_folder, exist_ok=True)
    yolo_file = os.path.join(output_folder, file_name + ".txt")

    with open(yolo_file, 'w') as f:
        f.write("\n".join(yolo_labels))

    print(f"Converted: {dota_file} -> {yolo_file}")


# Process all annotation files in a folder
def batch_convert_dota_to_yolo(dota_folder, image_folder, output_folder):
    dota_files = [f for f in os.listdir(dota_folder) if f.endswith(".txt")]

    if not dota_files:
        print("No DOTA annotation files found!")
        return

    print(f"Found {len(dota_files)} annotation files. Processing...")

    for dota_file in dota_files:
        dota_path = os.path.join(dota_folder, dota_file)
        convert_dota_to_yolo(dota_path, image_folder, output_folder)


# Example usage
dota_folder = r"C:\Users\Betul\Desktop\aaa"  # Folder containing .txt annotation files
image_folder = r"C:\Users\Betul\Desktop\aaa"  # Folder containing images
output_folder = r"C:\Users\Betul\Desktop\aaa\nnn"  # Folder to save YOLO annotations

batch_convert_dota_to_yolo(dota_folder, image_folder, output_folder)
print("Conversion completed!")
