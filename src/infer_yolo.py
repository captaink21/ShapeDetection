import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json

MODEL_PATH = "runs/yolo_exp1/weights/weights/best.pt"
TEST_DIR = "yolo_dataset/images/test"
OUT_DIR = "python_results"
NUM_IMAGES = 5

CLASSES = ["triangle", "rhombus", "circle", "hexagon"]
COLORS = {
    0: (0, 255, 0),      # triangle - зелёный
    1: (255, 0, 0),      # rhombus - синий
    2: (0, 0, 255),      # circle - красный
    3: (255, 255, 0)     # hexagon - голубой
}

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

images = sorted(Path(TEST_DIR).glob("*.jpg"))

stats = {
    "total_detections": 0,
    "per_image": [],
    "per_class": {cls: [] for cls in CLASSES}
}

for i, img_path in enumerate(images[:NUM_IMAGES]):
    
    image = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
        
    results = model.predict(
        str(img_path),
        conf=0.25,
        iou=0.45,
        verbose=False,
        imgsz=256
    )
    result = results[0]

    detections = []

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for j, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls_id)
        class_name = CLASSES[class_id]
        
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        detections.append({
            "box": [x1, y1, x2, y2],
            "class": class_name,
            "class_id": class_id,
            "confidence": float(score),
            "area": area
        })
        
        stats["total_detections"] += 1
        stats["per_class"][class_name].append(float(score))

    result_image = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cls_id = detection["class_id"]
        cls_name = detection["class"]
        conf = detection["confidence"]
        
        color = COLORS[cls_id]
        
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        y_label = max(y1 - 5, label_size[1] + 5)
        cv2.rectangle(
            result_image,
            (x1, y_label - label_size[1] - 5),
            (x1 + label_size[0] + 5, y_label + baseline),
            color,
            -1
        )
        cv2.putText(
            result_image,
            label,
            (x1 + 2, y_label - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    out_path = os.path.join(OUT_DIR, f"result_{img_path.name}")
    cv2.imwrite(out_path, result_image)

    json_path = os.path.join(OUT_DIR, f"result_{img_path.stem}.json")
    with open(json_path, "w") as f:
        json.dump({
            "image": img_path.name,
            "size": [orig_w, orig_h],
            "detections": detections,
            "total": len(detections)
        }, f, indent=2)

print(f"\nОбщее количество обнаружений: {stats['total_detections']}")
print("\nПо классам:")

for cls in CLASSES:
    scores = stats["per_class"][cls]
    if scores:
        avg_conf = np.mean(scores)
        min_conf = np.min(scores)
        max_conf = np.max(scores)
        print(f"  {cls:12} | count={len(scores):3d} | "
              f"avg_conf={avg_conf:.3f} | min={min_conf:.3f} | max={max_conf:.3f}")
    else:
        print(f"  {cls:12} | count=  0 | не обнаружено")

print(f"Результаты в: {OUT_DIR}/\n")
