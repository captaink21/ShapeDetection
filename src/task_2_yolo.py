import os
import json
import random
import shutil
from pathlib import Path
import pandas as pd

import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ultralytics import YOLO

from task_1 import ShapeGeneratorCV


SEED = 42
IMG_SIZE = 256
NUM_CLASSES = 4
TRAIN_DIR = "dataset_train"
TEST_DIR = "dataset_test"
OUT_DIR = "runs/yolo_exp1"
YOLO_DATASET_DIR = "yolo_dataset"

DATA_MODE_TRAIN = "on_the_fly"
DATA_MODE_TEST = "on_the_fly"
EPOCHS = 12
BATCH_SIZE = 4
NUM_WORKERS = 0
LR = 1e-3
SCORE_THR = 0.5
IOU_THR = 0.5
MID_EPOCH = 5

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASS2ID = {"triangle": 0, "rhombus": 1, "circle": 2, "hexagon": 3}
ID2CLASS = {v: k for k, v in CLASS2ID.items()}


def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]


def xywh_to_yolo_normalized(x, y, w, h, img_size):
    x_center = (x + w / 2) / img_size
    y_center = (y + h / 2) / img_size
    w_norm = w / img_size
    h_norm = h / img_size
    return x_center, y_center, w_norm, h_norm


class ShapesOnTheFly(Dataset):

    def __init__(self, n=5000, img_size=256, augment=False,
                 allowed_shapes=None, must_include=None):
        self.n = n
        self.generator = ShapeGeneratorCV(
            img_size=img_size,
            out_dir=None,
            save_to_disk=False,
            allowed_shapes=allowed_shapes,
            must_include=must_include
        )
        self.augment = augment
        self.class_counts = {k: 0 for k in CLASS2ID.keys()}

        for i in range(self.n):
            _, anns = self.generator.generate_image(i, return_data=True)
            for a in anns:
                if a["name"] in CLASS2ID:
                    self.class_counts[a["name"]] += 1

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GaussianBlur(p=0.2)
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3))
        else:
            self.transform = A.Compose([A.NoOp()],
                                       bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, anns = self.generator.generate_image(idx, return_data=True)
        boxes, labels = [], []

        for a in anns:
            if a["name"] not in CLASS2ID:
                continue
            xmin, ymin, xmax, ymax = xywh_to_xyxy(
                a["x"], a["y"], a["w"], a["h"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS2ID[a["name"]])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        transformed = self.transform(
            image=img, bboxes=boxes, labels=labels.tolist())
        img = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t,
                  "image_id": torch.tensor([idx])}
        return img_t, target


class ShapesFolderAlb(Dataset):

    def __init__(self, root: str, augment=False):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.paths = sorted([f for f in os.listdir(
            self.img_dir) if f.lower().endswith(".png")])
        self.class_counts = {k: 0 for k in CLASS2ID.keys()}

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GaussianBlur(p=0.2)
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3))
        else:
            self.transform = A.Compose([A.NoOp()],
                                       bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(
            self.ann_dir, img_name.replace(".png", ".json"))

        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with open(ann_path, "r", encoding="utf-8") as f:
            anns = json.load(f)

        boxes, labels = [], []
        for a in anns:
            name = a.get("name", "")
            if name not in CLASS2ID:
                continue
            xmin, ymin, xmax, ymax = xywh_to_xyxy(
                a["x"], a["y"], a["w"], a["h"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS2ID[name])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros(
            (0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros(
            (0,), dtype=np.int64)

        transformed = self.transform(
            image=img, bboxes=boxes, labels=labels.tolist())
        img = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t,
                  "image_id": torch.tensor([idx])}
        return img_t, target


class BalancedOnTheFly(ShapesOnTheFly):

    def __init__(self, total_n=12000, img_size=256, cycle_classes=None,
                 augment=False, force_include_every_image=False, must_include=None):
        super().__init__(n=total_n, img_size=img_size, augment=augment,
                         allowed_shapes=cycle_classes, must_include=must_include)
        self.cycle_classes = cycle_classes or list(CLASS2ID.keys())
        self.force_include = force_include_every_image
        self.must_include = must_include

    def __getitem__(self, idx):
        if self.force_include:
            shape = self.cycle_classes[idx % len(self.cycle_classes)]
            self.generator.allowed_shapes = [shape]
        if self.must_include:
            self.generator.must_include = self.must_include
        return super().__getitem__(idx)


def create_yolo_dataset(dataset_obj, split_name, yolo_dir=YOLO_DATASET_DIR, img_size=IMG_SIZE):

    split_img_dir = os.path.join(yolo_dir, "images", split_name)
    split_label_dir = os.path.join(yolo_dir, "labels", split_name)
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_label_dir, exist_ok=True)

    for idx in range(len(dataset_obj)):
        img_tensor, target = dataset_obj[idx]

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_path = os.path.join(split_img_dir, f"{idx:06d}.jpg")
        cv2.imwrite(img_path, img_bgr)

        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        yolo_labels = []
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            x = x1 + w / 2
            y = y1 + h / 2

            x_norm = x / img_size
            y_norm = y / img_size
            w_norm = w / img_size
            h_norm = h / img_size

            yolo_labels.append(
                f"{int(label)} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = os.path.join(split_label_dir, f"{idx:06d}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))


def get_yolo_model(model_name="yolov8n", num_classes=NUM_CLASSES, pretrained=True):
    if pretrained:
        model = YOLO(f'{model_name}.pt')
    else:
        model = YOLO(f'{model_name}.yaml')

    return model


def create_yolo_yaml(dataset_path, yaml_path, nc=NUM_CLASSES):
    names_list = [ID2CLASS[i] for i in range(nc)]

    yaml_content = f"""path: {os.path.abspath(dataset_path)}
train: images/train
val: images/val
test: images/test

nc: {nc}
names: {names_list}
"""

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    areaB = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)


def greedy_match_tp_fp_fn(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                          iou_thr=0.5, score_thr=0.5):
    order = np.argsort(-pred_scores)
    used_gt = set()
    TP = 0
    FP = 0
    FN = len(gt_boxes)
    ious = []

    for idx in order:
        if pred_scores[idx] < score_thr:
            continue

        pb = pred_boxes[idx].tolist() if isinstance(
            pred_boxes[idx], np.ndarray) else pred_boxes[idx]
        pl = int(pred_labels[idx])

        best_iou, best_j = 0.0, -1
        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if j in used_gt or int(gl) != pl:
                continue
            gb = gb.tolist() if isinstance(gb, np.ndarray) else gb
            i = iou_xyxy(pb, gb)
            if i > best_iou:
                best_iou, best_j = i, j

        if best_j >= 0 and best_iou >= iou_thr:
            TP += 1
            FN -= 1
            used_gt.add(best_j)
            ious.append(best_iou)
        else:
            FP += 1

    return TP, FP, FN, ious


def draw_boxes_rgb(img_rgb, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                   out_path, title=""):
    img = img_rgb.copy()

    for b, l in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"GT:{ID2CLASS.get(int(l), '?')}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"P:{ID2CLASS.get(int(l), '?')} {s:.2f}", (x1, min(img.shape[0]-3, y1+15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    if title:
        cv2.putText(img, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 200, 50), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def evaluate_yolo_model(model, test_loader, device, out_tag="final", score_thr=SCORE_THR, iou_thr=IOU_THR):

    model.eval()
    TP_sum = FP_sum = FN_sum = 0
    all_ious = []
    per_class_gt = {c: 0 for c in CLASS2ID.values()}
    per_class_det = {c: 0 for c in CLASS2ID.values()}
    per_image_mean_iou = []
    per_image_cache = []

    with torch.no_grad():
        for batch_idx, (imgs, gt) in enumerate(test_loader):
            imgs_np = [(img.permute(1, 2, 0).cpu().numpy() *
                        255).astype(np.uint8) for img in imgs]

            results = model(imgs_np, verbose=False)

            for i, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()

                gt_cpu = {
                    "boxes": gt[i]["boxes"].cpu().numpy(),
                    "labels": gt[i]["labels"].cpu().numpy()
                }

                TP, FP, FN, ious = greedy_match_tp_fp_fn(
                    boxes, labels, scores,
                    gt_cpu["boxes"], gt_cpu["labels"],
                    iou_thr=iou_thr, score_thr=score_thr
                )

                TP_sum += TP
                FP_sum += FP
                FN_sum += FN
                all_ious.extend(ious)

                for l in gt_cpu["labels"]:
                    per_class_gt[l] += 1

                for pl, sc in zip(labels, scores):
                    if sc >= score_thr:
                        per_class_det[pl] += 1

                img_mean_iou = float(np.mean(ious)) if ious else 0.0
                per_image_mean_iou.append(img_mean_iou)

                img_rgb = imgs_np[i]
                save_name = f"gen_{batch_idx:06d}_{i:02d}"

                keep = scores >= score_thr
                pred_boxes = boxes[keep]
                pred_labels = labels[keep]
                pred_scores = scores[keep]

                per_image_cache.append({
                    "img_rgb": img_rgb,
                    "gt_boxes": gt_cpu["boxes"],
                    "gt_labels": gt_cpu["labels"],
                    "pred_boxes": pred_boxes,
                    "pred_labels": pred_labels,
                    "pred_scores": pred_scores,
                    "img_name": save_name
                })

    precision = TP_sum / (TP_sum + FP_sum + 1e-9)
    recall = TP_sum / (TP_sum + FN_sum + 1e-9)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    if per_image_mean_iou:
        i_max = int(np.argmax(per_image_mean_iou))
        i_mid = int(np.argsort(per_image_mean_iou)[len(per_image_mean_iou)//2])
        i_min = int(np.argmin(per_image_mean_iou))

        viz_dir = os.path.join(OUT_DIR, f"viz_{out_tag}")
        os.makedirs(viz_dir, exist_ok=True)

        for tag, idx in [("best", i_max), ("mid", i_mid), ("worst", i_min)]:
            ex = per_image_cache[idx]
            draw_boxes_rgb(
                ex["img_rgb"], ex["gt_boxes"], ex["gt_labels"],
                ex["pred_boxes"], ex["pred_labels"], ex["pred_scores"],
                out_path=os.path.join(viz_dir, f"{tag}_{ex['img_name']}.png"),
                title=f"{tag.upper()} {out_tag}: mean IoU={per_image_mean_iou[idx]:.3f}"
            )

    metrics = {
        "mean_iou": mean_iou,
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "tp": int(TP_sum),
        "fp": int(FP_sum),
        "fn": int(FN_sum),
        "per_class_gt": per_class_gt,
        "per_class_det": per_class_det,
        "per_image_mean_iou_min": float(np.min(per_image_mean_iou)) if per_image_mean_iou else 0.0,
        "per_image_mean_iou_max": float(np.max(per_image_mean_iou)) if per_image_mean_iou else 0.0,
    }

    return metrics


def build_loaders():
    if DATA_MODE_TRAIN == "on_the_fly":
        train_ds = ShapesOnTheFly(n=5000, img_size=IMG_SIZE, augment=True)
    elif DATA_MODE_TRAIN == "from_disk":
        train_ds = ShapesFolderAlb(TRAIN_DIR, augment=True)
    else:
        raise ValueError("Unknown DATA_MODE_TRAIN")

    if DATA_MODE_TEST == "on_the_fly":
        test_ds = ShapesOnTheFly(n=1000, img_size=IMG_SIZE, augment=False)
    elif DATA_MODE_TEST == "from_disk":
        test_ds = ShapesFolderAlb(TEST_DIR, augment=False)
    else:
        raise ValueError("Unknown DATA_MODE_TEST")

    def collate_fn(batch):
        imgs, tgts = list(zip(*batch))
        return list(imgs), list(tgts)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

    return train_ds, test_ds, train_loader, test_loader


def run_task2_yolo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется device: {device}")

    print("Создание datasets...")
    train_ds, test_ds, train_loader, test_loader = build_loaders()

    print("Создание YOLO dataset...")
    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)

    create_yolo_dataset(train_ds, "train", YOLO_DATASET_DIR, IMG_SIZE)
    create_yolo_dataset(test_ds, "val", YOLO_DATASET_DIR, IMG_SIZE)
    create_yolo_dataset(test_ds, "test", YOLO_DATASET_DIR, IMG_SIZE)



    yaml_path = create_yolo_yaml(
        YOLO_DATASET_DIR, os.path.join(OUT_DIR, "data.yaml"), NUM_CLASSES)
    print(f"YAML конфиг создан: {yaml_path}")

    print("Загрузка YOLO модели...")
    model = YOLO("yolov8n.pt")

    print(f"Обучение YOLO на {EPOCHS} эпохах...")
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        patience=None,
        val=False,
        save=True,
        project=OUT_DIR,
        name="weights",
        lr0=LR,
        verbose=True,
        seed=SEED,
        freeze=10,
        save_period=1
    )


    csv_file = os.path.join(OUT_DIR, "weights", "results.csv")

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        losses = df['train/box_loss'].tolist()


    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, markersize=6, color='#1f77b4')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training loss", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    mid_model_path = os.path.join(OUT_DIR, "weights", "weights", f"epoch{MID_EPOCH}.pt")
    mid_model = YOLO(mid_model_path)

    metrics_mid = evaluate_yolo_model(
        mid_model, test_loader, device, out_tag=f"mid_epoch{MID_EPOCH}",
        score_thr=SCORE_THR, iou_thr=IOU_THR
    )

    with open(os.path.join(OUT_DIR, "metrics_mid.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_mid, f, ensure_ascii=False, indent=2)

    final_model_path = os.path.join(OUT_DIR, "model_final.pt")
    model.save(final_model_path)
    print(f"Модель сохранена: {final_model_path}")

    model.export(format="onnx", simplify=False, dynamic=False, half=False)

    if torch.cuda.is_available():
        try:
            model.export(format="engine", half=True, workspace=4)
        except Exception as e:
            print(f"TensorRT экспорт пропущен: {e}")

    try:
        model.export(format="openvino", half=False, imgsz=IMG_SIZE)
    except Exception as e:
        print(f"⚠️ OpenVINO экспорт пропущен: {e}")

    print("Оценка на тестовом наборе...")
    metrics_final = evaluate_yolo_model(model, test_loader, device, out_tag="final",
                                        score_thr=SCORE_THR, iou_thr=IOU_THR)

    with open(os.path.join(OUT_DIR, "metrics_final.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_final, f, ensure_ascii=False, indent=2)

    print("Финальные метрики:", metrics_final)




    with open(os.path.join(OUT_DIR, "train_class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(getattr(train_ds, "class_counts", {}),
                  f, ensure_ascii=False, indent=2)

    total_images = len(train_ds)
    with open(os.path.join(OUT_DIR, "train_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_images": total_images,
            "per_class_images": train_ds.class_counts,
            "data_mode": DATA_MODE_TRAIN
        }, f, ensure_ascii=False, indent=2)

    print("Обучение завершено!")
    return model, metrics_final


if __name__ == "__main__":
    run_task2_yolo()
