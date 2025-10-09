import os
import json
import random

import cv2
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torchvision.models.detection import retinanet_resnet50_fpn
from task_1 import ShapeGeneratorCV

SEED = 42
IMG_SIZE = 256
NUM_CLASSES = 5 
TRAIN_DIR = "dataset_train"
TEST_DIR  = "dataset_test"
OUT_DIR   = "runs/exp1"

DATA_MODE_TRAIN = "on_the_fly"
DATA_MODE_TEST  = "on_the_fly"

MID_EPOCH    = 5
EPOCHS       = 12
BATCH_SIZE   = 4
NUM_WORKERS  = 0
LR           = 1e-3
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
SCORE_THR    = 0.5
IOU_THR      = 0.5

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


CLASS2ID = {"triangle": 1, "rhombus": 2, "circle": 3, "hexagon": 4}
ID2CLASS = {v: k for k, v in CLASS2ID.items()}


def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]



class ShapesOnTheFly(Dataset):
    def __init__(self, n=5000, img_size=256, augment=False,
                 allowed_shapes= None, must_include = None):
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
            xmin, ymin, xmax, ymax = xywh_to_xyxy(a["x"], a["y"], a["w"], a["h"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS2ID[a["name"]])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        transformed = self.transform(image=img, bboxes=boxes, labels=labels.tolist())
        img = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}
        return img_t, target


class ShapesFolderAlb(Dataset):
    def __init__(self, root: str, augment=False):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.paths = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(".png")])
        self.class_counts = {k: 0 for k in CLASS2ID.keys()}

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
        ann_path = os.path.join(self.ann_dir, img_name.replace(".png", ".json"))

        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with open(ann_path, "r", encoding="utf-8") as f:
            anns = json.load(f)

        boxes, labels = [], []
        for a in anns:
            name = a.get("name", "")
            if name not in CLASS2ID:
                continue
            xmin, ymin, xmax, ymax = xywh_to_xyxy(a["x"], a["y"], a["w"], a["h"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS2ID[name])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        transformed = self.transform(image=img, bboxes=boxes, labels=labels.tolist())
        img = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([idx])}
        return img_t, target


#для 3 задания
class BalancedOnTheFly(ShapesOnTheFly):
    def __init__(self, total_n=12000, img_size=256, cycle_classes=None,
                 augment=False, force_include_every_image=False,
             must_include=None):
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


def collate_fn(batch):
    imgs, tgts = list(zip(*batch))
    return list(imgs), list(tgts)


def iou_xyxy(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    areaB = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)


def greedy_match_tp_fp_fn(pred, gt, iou_thr=0.5, score_thr=0.5):
    order = torch.argsort(pred["scores"], descending=True)
    used_gt = set()
    TP = 0; FP = 0; FN = len(gt["boxes"])
    ious = []
    for idx in order.tolist():
        if pred["scores"][idx].item() < score_thr:
            continue
        pb = pred["boxes"][idx].tolist()
        pl = int(pred["labels"][idx].item())

        best_iou, best_j = 0.0, -1
        for j, (gb, gl) in enumerate(zip(gt["boxes"].tolist(), gt["labels"].tolist())):
            if j in used_gt or int(gl) != pl:
                continue
            i = iou_xyxy(pb, gb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0 and best_iou >= iou_thr:
            TP += 1; FN -= 1; used_gt.add(best_j); ious.append(best_iou)
        else:
            FP += 1
    return TP, FP, FN, ious


def draw_boxes_rgb(img_rgb,
                   gt_boxes, gt_labels,
                   pred_boxes, pred_labels, pred_scores,
                   out_path,
                   title=""):
    img = img_rgb.copy()
    for b, l in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"GT:{ID2CLASS.get(int(l),'?')}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img, f"P:{ID2CLASS.get(int(l),'?')} {s:.2f}", (x1, min(img.shape[0]-3, y1+15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,50), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_model(num_classes=NUM_CLASSES, pretrained=True):
    if pretrained:
        model = retinanet_resnet50_fpn(
            weights=None, weights_backbone="ResNet50_Weights.IMAGENET1K_V1", num_classes=num_classes)
    else:
        model = retinanet_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes)

    model.transform.min_size = (IMG_SIZE,)
    model.transform.max_size = IMG_SIZE
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for imgs, tgts in loader:
        imgs = [im.to(device) for im in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))



@torch.no_grad()
def evaluate_and_visualize(model,
                           loader,
                           device,
                           out_tag="final",
                           score_thr=SCORE_THR,
                           iou_thr=IOU_THR):
    model.eval()
    TP_sum = FP_sum = FN_sum = 0
    all_ious = []

    per_class_gt  = {c: 0 for c in CLASS2ID.values()}
    per_class_det = {c: 0 for c in CLASS2ID.values()}

    per_image_mean_iou = []
    per_image_cache    = []

    for imgs, gt in loader:
        imgs = [im.to(device) for im in imgs]
        outs = model(imgs)
        out  = outs[0]
        pred = {"boxes": out["boxes"].cpu(), "labels": out["labels"].cpu(), "scores": out["scores"].cpu()}
        gt_cpu = {"boxes": gt[0]["boxes"].cpu(), "labels": gt[0]["labels"].cpu()}

        TP, FP, FN, ious = greedy_match_tp_fp_fn(pred, gt_cpu, iou_thr=iou_thr, score_thr=score_thr)
        TP_sum += TP; FP_sum += FP; FN_sum += FN
        all_ious.extend(ious)

        for l in gt_cpu["labels"].tolist():
            per_class_gt[l] += 1
        for pl, sc in zip(pred["labels"].tolist(), pred["scores"].tolist()):
            if sc >= score_thr:
                per_class_det[pl] += 1

        img_mean_iou = float(np.mean(ious)) if ious else 0.0
        per_image_mean_iou.append(img_mean_iou)

        ds = loader.dataset
        img_id = int(gt[0]["image_id"].item())
        if hasattr(ds, "paths"):
            img_name = ds.paths[img_id]
            img_path = os.path.join(ds.root, "images", img_name)
            img_bgr  = cv2.imread(img_path)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
            if img_rgb is None:
                img_rgb = (imgs[0].detach().cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
            save_name = img_name.replace(".png", "")
        else:
            img_rgb  = (imgs[0].detach().cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
            save_name = f"gen_{img_id:06d}"

        keep = pred["scores"] >= score_thr
        pred_boxes  = pred["boxes"][keep].tolist()
        pred_labels = pred["labels"][keep].tolist()
        pred_scores = pred["scores"][keep].tolist()

        per_image_cache.append({
            "img_rgb": img_rgb, "gt_boxes": gt_cpu["boxes"].tolist(), "gt_labels": gt_cpu["labels"].tolist(),
            "pred_boxes": pred_boxes, "pred_labels": pred_labels, "pred_scores": pred_scores, "img_name": save_name
        })

    precision = TP_sum / (TP_sum + FP_sum + 1e-9)
    recall    = TP_sum / (TP_sum + FN_sum + 1e-9)
    mean_iou  = float(np.mean(all_ious)) if all_ious else 0.0

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
            with open(os.path.join(viz_dir, f"{tag}_{ex['img_name']}_pred.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "pred_boxes": ex["pred_boxes"],
                    "pred_labels": ex["pred_labels"],
                    "pred_scores": ex["pred_scores"]
                }, f, ensure_ascii=False, indent=2)

    metrics = {
        "mean_iou": mean_iou,
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "tp": int(TP_sum), "fp": int(FP_sum), "fn": int(FN_sum),
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    return train_ds, test_ds, train_loader, test_loader



def run_task2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds, train_loader, test_loader = build_loaders()
    model = get_model(NUM_CLASSES, pretrained=True).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

    losses = []
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, opt, device)
        losses.append(avg_loss)
        sched.step()
        print(f"[Epoch {epoch}/{EPOCHS}] loss={avg_loss:.4f}")

        if epoch == MID_EPOCH:
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_mid_epoch{epoch}.pth"))

    final_path = os.path.join(OUT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved checkpoint: {final_path}")

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training loss")
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png")); plt.close()

    mid_path = os.path.join(OUT_DIR, f"model_mid_epoch{MID_EPOCH}.pth")
    if os.path.isfile(mid_path):
        model.load_state_dict(torch.load(mid_path, map_location=device))
        metrics_mid = evaluate_and_visualize(model, test_loader, device, out_tag="mid")
        with open(os.path.join(OUT_DIR, "metrics_mid.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_mid, f, ensure_ascii=False, indent=2)
        print("MID metrics:", metrics_mid)

    model.load_state_dict(torch.load(final_path, map_location=device))
    metrics_final = evaluate_and_visualize(model, test_loader, device, out_tag="final")
    with open(os.path.join(OUT_DIR, "metrics_final.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_final, f, ensure_ascii=False, indent=2)
    print("FINAL metrics:", metrics_final)

    with open(os.path.join(OUT_DIR, "train_class_counts.json"), "w", encoding="utf-8") as f:
        json.dump(getattr(train_ds, "class_counts", {}), f, ensure_ascii=False, indent=2)

    total_images = len(train_ds)
    with open(os.path.join(OUT_DIR, "train_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_images": total_images,
            "per_class_images": train_ds.class_counts,
            "data_mode": DATA_MODE_TRAIN
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_task2()
