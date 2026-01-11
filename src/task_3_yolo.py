import os
import json
import torch
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

from task_2_yolo import (
    BalancedOnTheFly,
    evaluate_yolo_model,
    create_yolo_yaml,
    NUM_CLASSES, OUT_DIR, SCORE_THR, IOU_THR, IMG_SIZE,
    CLASS2ID, ID2CLASS, YOLO_DATASET_DIR
)

ITERATIONS = 20
EPOCHS_PER_ITER = 5
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = os.path.join(OUT_DIR, "yolo_result")
os.makedirs(RESULTS_DIR, exist_ok=True)


def collate_fn_cf(batch):
    imgs, tgts = list(zip(*batch))
    return list(imgs), list(tgts)


def save_batch_to_disk(dataset, start_idx, img_dir, lbl_dir):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(len(dataset)):
        img_tensor, target = dataset[i]

        img_np = img_tensor.permute(1, 2, 0).numpy() * 255
        img_np = img_np.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        filename = f"gen_{start_idx + i:06d}"

        img_path = os.path.join(img_dir, f"{filename}.jpg")
        cv2.imwrite(img_path, img_bgr)

        h, w = img_np.shape[:2]
        boxes = target["boxes"]  # [x1, y1, x2, y2]
        labels = target["labels"]  # [1, 2, 3, 4]

        lbl_path = os.path.join(lbl_dir, f"{filename}.txt")
        with open(lbl_path, "w") as f:
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                cls = int(label)

                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + bw / 2) / w
                cy = (y1 + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")


def count_class_distribution(labels_dir):
    class_counts = Counter()
    if not os.path.exists(labels_dir):
        return {0: 0, 1: 0, 2: 0, 3: 0}
    
    for txt_file in Path(labels_dir).glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                except (ValueError, IndexError):
                    continue
    return dict(class_counts)


def save_distribution_json(class_counts, output_path, iteration=None):
    CLASS_NAMES = {0: "triangle", 1: "rhombus", 2: "circle", 3: "hexagon"}
    distribution = {CLASS_NAMES.get(cid, f"id_{cid}"): count for cid, count in class_counts.items()}
    distribution["total"] = sum(class_counts.values())
    if iteration is not None:
        distribution["iteration"] = iteration
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)


def plot_history_graphs(history, save_dir):
    iterations = history["iterations"]
    
    #ГРАФИК 1: Метрики (Precision, Recall, IoU)
    plt.figure(figsize=(12, 6))
    
    # Test1 (Без гексагона)
    t1_prec = [m["precision@0.5"] for m in history["test1_metrics"]]
    t1_rec = [m["recall@0.5"] for m in history["test1_metrics"]]
    t1_iou = [m["mean_iou"] for m in history["test1_metrics"]]
    
    plt.plot(iterations, t1_prec, 'o-', label='Precision Test1 (no hex)', color='#1f77b4') # Blue
    plt.plot(iterations, t1_rec, 'o-', label='Recall Test1 (no hex)', color='#ff7f0e')    # Orange
    plt.plot(iterations, t1_iou, 'o-', label='IoU Test1 (no hex)', color='#2ca02c')       # Green
    
    # Test2 (С гексагоном)
    t2_prec = [m["precision@0.5"] for m in history["test2_metrics"]]
    t2_rec = [m["recall@0.5"] for m in history["test2_metrics"]]
    t2_iou = [m["mean_iou"] for m in history["test2_metrics"]]
    
    plt.plot(iterations, t2_prec, 's--', label='Precision Test2 (w/ hex)', color='#d62728') # Red
    plt.plot(iterations, t2_rec, 's--', label='Recall Test2 (w/ hex)', color='#9467bd')     # Purple
    plt.plot(iterations, t2_iou, 's--', label='IoU Test2 (w/ hex)', color='#8c564b')        # Brown
    
    plt.title("Task3: Precision / Recall / IoU by iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Metric value")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "metrics_history.png"))
    plt.close()
    
    # --- ГРАФИК 2: Покрытие классов (Det/GT) ---
    plt.figure(figsize=(12, 6))
    
    # Имена классов (согласно твоему маппингу)
    class_names = {0: "Triangle", 1: "Rhombus", 2: "Circle", 3: "Hexagon"}
    
    for cls_id, cls_name in class_names.items():
        ratios = []
        for m in history["test2_metrics"]:
            gt = m["per_class_gt"].get(str(cls_id), 0) + 1e-9 
            det = m["per_class_det"].get(cls_id, 0) 
            
            # Фикс для разных форматов ключей в JSON (строка vs число)
            if det == 0 and str(cls_id) in m["per_class_det"]:
                det = m["per_class_det"][str(cls_id)]
                
            ratios.append(det / gt)
            
        plt.plot(iterations, ratios, '--', label=f'Test2 {cls_name}')

    plt.title("Class Coverage (Det/GT) by iteration [Test2]")
    plt.xlabel("Iteration")
    plt.ylabel("Det / GT Ratio")
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "coverage_history.png"))
    plt.close()
    
    print(f"Графики сохранены в {save_dir}")

def run_task3():
    print("\n")
    print(f"Total Iterations: {ITERATIONS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Workers: 0")

    yaml_path = os.path.join(OUT_DIR, "data_cf_analysis.yaml")
    yaml_path = create_yolo_yaml(YOLO_DATASET_DIR, yaml_path, NUM_CLASSES)
    print(f"\nYAML Config: {yaml_path}")

    print("\nИнициализация датасетов")

    print("Создание начального датасета (12,000 без гексагона)")

    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)

    initial_train_ds = BalancedOnTheFly(
        total_n=12000,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle"],
        augment=True,
        force_include_every_image=False,
        must_include=None
    )

    img_dir_train = os.path.join(YOLO_DATASET_DIR, "images", "train")
    lbl_dir_train = os.path.join(YOLO_DATASET_DIR, "labels", "train")

    save_batch_to_disk(initial_train_ds, 0, img_dir_train, lbl_dir_train)

    print(f"\nНачальный датасет создан: {len(initial_train_ds)} изображений")

    dummy_ds = BalancedOnTheFly(
        total_n=100,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle"],
        augment=False
    )

    save_batch_to_disk(dummy_ds, 0,
                       os.path.join(YOLO_DATASET_DIR, "images", "val"),
                       os.path.join(YOLO_DATASET_DIR, "labels", "val")
                       )
    save_batch_to_disk(dummy_ds, 0,
                       os.path.join(YOLO_DATASET_DIR, "images", "test"),
                       os.path.join(YOLO_DATASET_DIR, "labels", "test")
                       )
    # БЕЗ гексагона (3000 изображений)
    test_ds1 = BalancedOnTheFly(
        total_n=3000,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle"],
        augment=False,
        force_include_every_image=False,
        must_include=None
    )
    test_loader1 = DataLoader(
        test_ds1, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn_cf
    )
    print(f"Test Dataset 1 (No Hexagon): {len(test_ds1)} images")

    # С гексагоном (3000 изображений, обязательно содержит гексагон)
    test_ds2 = BalancedOnTheFly(
        total_n=3000,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle", "hexagon"],
        augment=False,
        force_include_every_image=False,
        must_include=["hexagon"]
    )
    test_loader2 = DataLoader(
        test_ds2, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn_cf
    )
    print(f"Test Dataset 2 (With Hexagon): {len(test_ds2)} images")

    print("ПРЕДВАРИТЕЛЬНОЕ ОБУЧЕНИЕ")

    model = YOLO("yolov8n.yaml")

    start_result = model.train(
        data=yaml_path,
        epochs=EPOCHS_PER_ITER,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        pretrained=False,
        amp=False,
        patience=None,
        val=False,
        save=True,
        project=RESULTS_DIR,
        name="00_start_learning",
        verbose=False,
        seed=42,
        exist_ok=True,
        workers=0
    )

    start_weights_path = os.path.join(
        RESULTS_DIR, "00_start_learning", "weights", "best.pt")
    print(f"✓ Start Learning завершено!")
    print(f"  Веса сохранены: {start_weights_path}")

    history = {
        "iterations": [],
        "test1_metrics": [],
        "test2_metrics": [],
        "train_sizes": []
    }

    print("Запуск цикла исследования (20 итераций)")

    for it in range(1, ITERATIONS + 1):
        print(f"\nИТЕРАЦИЯ {it}/{ITERATIONS}")

        if it == 1:
            eval_weights = start_weights_path
        else:
            eval_weights = os.path.join(
                RESULTS_DIR, f"{it-1:02d}_retrain", "weights", "best.pt"
            )

        print(f"\nТестирование")
        print(f"  Модель: {os.path.basename(os.path.dirname(eval_weights))}")

        del model
        torch.cuda.empty_cache()
        eval_model = YOLO(eval_weights)

        metrics1 = evaluate_yolo_model(
            eval_model, test_loader1, DEVICE,
            out_tag=f"iter{it:02d}_test1",
            score_thr=SCORE_THR, iou_thr=IOU_THR
        )

        metrics2 = evaluate_yolo_model(
            eval_model, test_loader2, DEVICE,
            out_tag=f"iter{it:02d}_test2",
            score_thr=SCORE_THR, iou_thr=IOU_THR
        )

        print(f"\n  Test1 (No Hexagon):")
        print(f"    Precision@0.5: {metrics1['precision@0.5']:.4f}")
        print(f"    Recall@0.5:    {metrics1['recall@0.5']:.4f}")
        print(f"    Mean IoU:      {metrics1['mean_iou']:.4f}")

        print(f"\n  Test2 (With Hexagon):")
        print(f"    Precision@0.5: {metrics2['precision@0.5']:.4f}")
        print(f"    Recall@0.5:    {metrics2['recall@0.5']:.4f}")
        print(f"    Mean IoU:      {metrics2['mean_iou']:.4f}")

        new_train_size = 12000 + it * 400
        print(f"\n Генерация новых данных")
        print(f"  Генерируем 400 изображений с гексагоном...")
        print(f"  Размер тренировочного датасета: 12000 -> {new_train_size}")

        temp_ds = BalancedOnTheFly(
            total_n=400,
            img_size=IMG_SIZE,
            cycle_classes=["triangle", "rhombus", "circle", "hexagon"],
            augment=True,
            must_include=["hexagon"],
            force_include_every_image=False
        )

        img_dir = os.path.join(YOLO_DATASET_DIR, "images", "train")
        lbl_dir = os.path.join(YOLO_DATASET_DIR, "labels", "train")
        start_idx = 12000 + (it - 1) * 400

        print(
            f"  Сохраняем на диск (индексы {start_idx}-{start_idx + 400})...")
        save_batch_to_disk(temp_ds, start_idx, img_dir, lbl_dir)
        print(f"  Файлы сохранены в {img_dir} и {lbl_dir}")


        train_distribution = count_class_distribution(lbl_dir)
        save_distribution_json(
        train_distribution,
        os.path.join(RESULTS_DIR, f"train_distribution_iter_{it:02d}.json"),
        iteration=it
        )
        print(f"\n Очистка кэша")
        cache_files = [
            os.path.join(YOLO_DATASET_DIR, "images", "train.cache"),
            os.path.join(YOLO_DATASET_DIR, "images", "val.cache"),
            os.path.join(YOLO_DATASET_DIR, "labels", "train.cache"),
            os.path.join(YOLO_DATASET_DIR, "labels", "val.cache"),
        ]
        for cf in cache_files:
            if os.path.exists(cf):
                os.remove(cf)

        if it == 1:
            train_weights = start_weights_path
        else:
            train_weights = os.path.join(
                RESULTS_DIR, f"{it-1:02d}_retrain", "weights", "best.pt"
            )
            print(f"  Продолжаем обучение с итерации {it-1}...")

        model = YOLO(train_weights)

        retrain_result = model.train(
            data=yaml_path,
            epochs=EPOCHS_PER_ITER,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            patience=None,
            val=False,
            save=True,
            project=RESULTS_DIR,
            name=f"{it:02d}_retrain",
            verbose=False,
            seed=42,
            exist_ok=True,
            workers=0
        )

        history["iterations"].append(it)
        history["test1_metrics"].append(metrics1)
        history["test2_metrics"].append(metrics2)
        history["train_sizes"].append(new_train_size)

        with open(os.path.join(RESULTS_DIR, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"\n Итерация {it} завершена, результаты сохранены")

    print("ФИНАЛЬНЫЙ АНАЛИЗ")

    test1_prec = [m["precision@0.5"] for m in history["test1_metrics"]]
    test1_rec = [m["recall@0.5"] for m in history["test1_metrics"]]
    test1_iou = [m["mean_iou"] for m in history["test1_metrics"]]

    test2_prec = [m["precision@0.5"] for m in history["test2_metrics"]]
    test2_rec = [m["recall@0.5"] for m in history["test2_metrics"]]
    test2_iou = [m["mean_iou"] for m in history["test2_metrics"]]

    prec_degradation = (
        (test1_prec[0] - test1_prec[-1]) / (test1_prec[0] + 1e-9)) * 100
    rec_degradation = (
        (test1_rec[0] - test1_rec[-1]) / (test1_rec[0] + 1e-9)) * 100

    print(f"\n ДЕГРАДАЦИЯ НА БАЗОВЫХ КЛАССАХ (Test1, БЕЗ гексагона):")
    print(
        f"  Precision: {test1_prec[0]:.4f} → {test1_prec[-1]:.4f} ({prec_degradation:+.2f}%)")
    print(
        f"  Recall:    {test1_rec[0]:.4f} → {test1_rec[-1]:.4f} ({rec_degradation:+.2f}%)")
    print(f"  Mean IoU:  {test1_iou[0]:.4f} → {test1_iou[-1]:.4f}")

    print(f"\n ПРОИЗВОДИТЕЛЬНОСТЬ НА НОВОМ КЛАССЕ (Test2, С гексагоном):")
    print(f"  Avg Precision: {np.mean(test2_prec):.4f}")
    print(f"  Avg Recall:    {np.mean(test2_rec):.4f}")
    print(f"  Avg Mean IoU:  {np.mean(test2_iou):.4f}")
    print(f"  Min Precision: {np.min(test2_prec):.4f}")
    print(f"  Max Precision: {np.max(test2_prec):.4f}")

    final_report = {
        "task": "task_3_yolo.py",
        "iterations": ITERATIONS,
        "epochs_per_iter": EPOCHS_PER_ITER,
        "initial_train_size": 12000,
        "increment_per_iter": 400,
        "test1_dataset_size": 3000,
        "test2_dataset_size": 3000,
        "device": str(DEVICE),
        "model": "YOLOv8n",

        "yolo_result": {
            "test1_initial_precision": float(test1_prec[0]),
            "test1_final_precision": float(test1_prec[-1]),
            "precision_degradation_percent": float(prec_degradation),
            "test1_initial_recall": float(test1_rec[0]),
            "test1_final_recall": float(test1_rec[-1]),
            "recall_degradation_percent": float(rec_degradation),
        },

        "new_class_performance": {
            "avg_precision": float(np.mean(test2_prec)),
            "avg_recall": float(np.mean(test2_rec)),
            "avg_mean_iou": float(np.mean(test2_iou)),
            "min_precision": float(np.min(test2_prec)),
            "max_precision": float(np.max(test2_prec)),
        },

        "history": history
    }

    with open(os.path.join(RESULTS_DIR, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    plot_history_graphs(history, RESULTS_DIR)
    print(f"\n АНАЛИЗ ЗАВЕРШЕН!")
    print(f" Результаты: {RESULTS_DIR}")
    print(
        f" Финальный отчет: {os.path.join(RESULTS_DIR, 'final_report.json')}")

    return final_report


if __name__ == "__main__":
    try:
        report = run_task3()
    except Exception as e:
        print(f"\n ОШИБКА: {e}")
