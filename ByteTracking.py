import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Video ve model
video_path = r"C:\Users\Betul\Downloads\tokat.mp4"
cap = cv2.VideoCapture(video_path)
model_name = r"C:\Users\Betul\Downloads\runs\runs\detect\train11\weights\best.pt"
model = YOLO(model_name)

# Senin sınıf isimlerin (custom)
custom_class_names = [
    'ignored', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
    'motor', 'others','plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
    'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout',
    'soccer ball field', 'swimming pool', 'container crane'
]

# 'ignored' sadece gösterilecek, takip edilmeyecek
ignored_class_id = 0  # 'ignored' indexi

# Takip geçmişi
track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=1280)

    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")

    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Sınıf adını güvenli şekilde al
        class_name = custom_class_names[class_id] if class_id < len(custom_class_names) else f"class_{class_id}"
        color = (255, 0, 0) if class_id == ignored_class_id else (0, 255, 0)

        # Kutu çizimi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Sınıf adı (takipli ya da takipsiz)
        label = f"ID:{track_id} {class_name.upper()}" if class_id != ignored_class_id else class_name.upper()
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Eğer 'ignored' değilse takip et
        if class_id != ignored_class_id:
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 15:
                track.pop(0)
            points = np.hstack(track).astype("int32").reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
