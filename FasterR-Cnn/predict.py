import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches


model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=26)
model.load_state_dict(torch.load("fasterrcnn_final.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


id2label = {
    0: "background",
    # VisDrone categories (1-10)
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    # DOTA categories (11-25)
    11: "plane",
    12: "ship",
    13: "storage-tank",
    14: "baseball-diamond",
    15: "tennis-court",
    16: "basketball-court",
    17: "ground-track-field",
    18: "harbor",
    19: "bridge",
    20: "large-vehicle",
    21: "small-vehicle",
    22: "helicopter",
    23: "roundabout",
    24: "soccer-ball-field",
    25: "swimming-pool"
}



def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    return transform(image).unsqueeze(0), image


def predict_and_plot(image_path, threshold=0.5):
    image_tensor, original_image = load_image(image_path)
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)[0]

    plt.figure(figsize=(12, 8))
    plt.imshow(original_image)
    ax = plt.gca()

    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score >= threshold:
            x1, y1, x2, y2 = box.tolist()
            category = id2label.get(label.item(), str(label.item()))
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1, f"{category} ({score:.2f})", fontsize=10,
                    bbox=dict(facecolor="yellow", alpha=0.5))

    plt.axis("off")
    plt.show()


predict_and_plot("deneme.png", threshold=0.01)
