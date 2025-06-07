from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLOv11
model = YOLO('yolo11s.pt') 

# Eğitim parametreleri
data_config = 'visdrone.yaml'  
epochs = 25  
batch_size = 10 
img_size = 480  
augment = True
flipud = 0.0
fliplr = 0.5
degrees = 10
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
scale = 0.8
translate = 0.1
optimizer = 'Adam'
lrf = 0.001


# Eğitim
model.train(data=data_config,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='yolov11-visdrone',
            device='cpu',
            augment = augment,
            flipud = flipud,
            fliplr = fliplr,
            degrees = degrees,
            hsv_h = hsv_h,
            hsv_s = hsv_s,
            scale = scale,
            translate = translate,
            optimizer = optimizer,
            lrf = lrf,
            )  
