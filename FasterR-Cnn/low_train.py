import os
import torch
import gc
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MAX_BOXES = 30

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, image_root, transforms=None):
        self.coco = COCO(json_path)
        self.image_root = image_root
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_path = os.path.join(self.image_root, coco.loadImgs(img_id)[0]['file_name'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in anns[:MAX_BOXES]:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [1]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

def save_predictions(model, data_loader, device, writer, epoch):
    model.eval()
    images, _ = next(iter(data_loader))
    images = [img.to(device) for img in images]

    with torch.no_grad():
        preds = model(images)

    for i, pred in enumerate(preds[:4]):
        img = images[i].cpu()
        img = TF.to_pil_image(img)
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for box in pred['boxes']:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        writer.add_figure(f'Predictions/Image_{i}', fig, epoch)
        plt.close(fig)

def main():
    torch.cuda.empty_cache()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    writer = SummaryWriter(log_dir='runs/fasterrcnn_exp1')

    # Paths
    train_vis_json = 'visdrone_train_coco.json'
    train_vis_img = r'C:\Users\marsh\Desktop\VisDrone\images\train'
    train_dota_json = 'dota_train_coco.json'
    train_dota_img = r'C:\Users\marsh\Desktop\DotaTrain\images'

    val_vis_json = 'visdrone_val_coco.json'
    val_vis_img = r'C:\Users\marsh\Desktop\VisDroneDataSet\val\images'
    val_dota_json = 'dota_val_coco.json'
    val_dota_img = r'C:\Users\marsh\Desktop\DotaVal\images'

    # Datasets
    train_ds1 = CocoDataset(train_vis_json, train_vis_img, get_transform())
    train_ds2 = CocoDataset(train_dota_json, train_dota_img, get_transform())
    train_dataset = ConcatDataset([train_ds1, train_ds2])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    val_ds1 = CocoDataset(val_vis_json, val_vis_img, get_transform())
    val_ds2 = CocoDataset(val_dota_json, val_dota_img, get_transform())
    val_dataset = ConcatDataset([val_ds1, val_ds2])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=26)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler('cuda')

    epoch_losses = []
    val_losses = []
    mAPs = []
    recalls = []
    precisions = []
    num_epochs = 50

    print("🚀 Hızlı Eğitim Başladı...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print("❌ NaN veya inf bulundu, batch atlandı.")
                continue

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()
            gc.collect()

        epoch_losses.append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        print(f"🎯 Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), f"fasterrcnn_epoch_{epoch + 1}.pth")

        # Validation
        model.eval()
        val_loss = 0
        metric = MeanAveragePrecision()
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.amp.autocast('cuda'):
                    model.train()
                    loss_dict = model(images, targets)
                    model.eval()

                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()

                    preds = model(images)
                    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
                    target_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
                    metric.update(preds, target_cpu)

        val_losses.append(val_loss)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"📊 Validation Loss (Epoch {epoch + 1}): {val_loss:.4f}")

        result = metric.compute()
        mAPs.append(result['map'].item())
        recalls.append(result['mar_100'].item())  # Mean Average Recall @100
        precisions.append(result['map_50'].item())  # mAP @ IoU=0.50 (Pascal VOC style)

        writer.add_scalar('Metrics/mAP', result['map'].item(), epoch)
        writer.add_scalar('Metrics/Recall_mar100', result['mar_100'].item(), epoch)
        writer.add_scalar('Metrics/Precision_map50', result['map_50'].item(), epoch)

        print(f"📈 mAP: {result['map']:.4f} | Recall@100: {result['mar_100']:.4f} | Precision@50: {result['map_50']:.4f}")

        save_predictions(model, val_loader, device, writer, epoch)

        # Plot all metrics
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(range(1, epoch + 2), epoch_losses, label="Train Loss")
        plt.plot(range(1, epoch + 2), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(range(1, epoch + 2), mAPs, label="mAP")
        plt.plot(range(1, epoch + 2), precisions, label="Precision")
        plt.plot(range(1, epoch + 2), recalls, label="Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Metrics over Epochs")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig(f"metrics_epoch_{epoch + 1}.png")
        plt.close()

    torch.save(model.state_dict(), "fasterrcnn_final.pth")
    writer.close()
    print("✅ Eğitim tamamlandı ve model kaydedildi.")


if __name__ == "__main__":
    main()
