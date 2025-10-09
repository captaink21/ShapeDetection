import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader

from task_2 import (
    BalancedOnTheFly, get_model, train_one_epoch, evaluate_and_visualize,
    collate_fn, NUM_CLASSES, OUT_DIR, SCORE_THR, IOU_THR
)


IMG_SIZE = 256
EPOCHS_PER_ITER = 5
BATCH_SIZE = 4
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


def run_task3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = BalancedOnTheFly(
        total_n=12000,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle"],
        augment=True,
        force_include_every_image=True
    )

    test1_ds = BalancedOnTheFly(
        total_n=3000,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle"],
        augment=False,
        force_include_every_image=True
    )

    test2_ds = BalancedOnTheFly(
        total_n=3000,
        img_size=IMG_SIZE,
        cycle_classes=["triangle", "rhombus", "circle", "hexagon"],
        must_include=["hexagon"],
        augment=False
    )

    test1_loader = DataLoader(test1_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test2_loader = DataLoader(test2_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = get_model(NUM_CLASSES, pretrained=False).to(device)
    start_ckpt = os.path.join(OUT_DIR, "start_learning.pth")
    torch.save(model.state_dict(), start_ckpt)
    print(f"Saved checkpoint: {start_ckpt}")

    results = []

    for it in range(1, 21):
        print(f"\n итерация {it}/20")

        opt = torch.optim.SGD(model.parameters(), lr=LR,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0, pin_memory=True)

        for epoch in range(EPOCHS_PER_ITER):
            avg_loss = train_one_epoch(model, train_loader, opt, device)
            print(
                f"[Iter {it}] Epoch {epoch+1}/{EPOCHS_PER_ITER}: loss={avg_loss:.4f}")

        metrics1 = evaluate_and_visualize(model, test1_loader, device,
                                          out_tag=f"iter{it}_test1",
                                          score_thr=SCORE_THR, iou_thr=IOU_THR)
        metrics2 = evaluate_and_visualize(model, test2_loader, device,
                                          out_tag=f"iter{it}_test2",
                                          score_thr=SCORE_THR, iou_thr=IOU_THR)

        results.append({
            "iteration": it,
            "train_size": len(train_ds),
            "metrics_test1": metrics1,
            "metrics_test2": metrics2
        })

        print(f"Iter {it}- Test1: {metrics1}")
        print(f"Iter {it}- Test2: {metrics2}")

        train_ds.n += 400
        train_ds.must_include = ["hexagon"]

    results_path = os.path.join(OUT_DIR, "task3_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {results_path}")

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    data = []
    for r in results:
        it = r["iteration"]
        for tag, metrics in [("Test1", r["metrics_test1"]), ("Test2", r["metrics_test2"])]:
            data.append({
                "Iteration": it,
                "Dataset": tag,
                "Precision": metrics["precision@0.5"],
                "Recall": metrics["recall@0.5"],
                "IoU": metrics["mean_iou"]
            })
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Iteration", y="Precision",
                 hue="Dataset", marker="o")
    sns.lineplot(data=df, x="Iteration", y="Recall",
                 hue="Dataset", marker="s", linestyle="--")
    sns.lineplot(data=df, x="Iteration", y="IoU",
                 hue="Dataset", marker="^", linestyle=":")
    plt.title("Динамика Precision / Recall / IoU по итерациям (Seaborn)")
    plt.legend(title="Датасет")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task3_metrics_seaborn.png"))
    plt.close()

    corr = df[["Precision", "Recall", "IoU"]].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляция метрик между собой")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task3_metrics_correlation.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Iteration", y="IoU", hue="Dataset", palette="Set2")
    plt.title("Распределение IoU по итерациям")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task3_iou_boxplot.png"))
    plt.close()

    print(f"Seaborn-графики сохранены в {OUT_DIR}")

    iters = [r["iteration"] for r in results]
    prec1 = [r["metrics_test1"]["precision@0.5"] for r in results]
    rec1 = [r["metrics_test1"]["recall@0.5"] for r in results]
    iou1 = [r["metrics_test1"]["mean_iou"] for r in results]

    prec2 = [r["metrics_test2"]["precision@0.5"] for r in results]
    rec2 = [r["metrics_test2"]["recall@0.5"] for r in results]
    iou2 = [r["metrics_test2"]["mean_iou"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(iters, prec1, "-o", label="Precision Test1 (no hexagon)")
    plt.plot(iters, rec1, "-o", label="Recall Test1 (no hexagon)")
    plt.plot(iters, iou1, "-o", label="IoU Test1 (no hexagon)")
    plt.plot(iters, prec2, "-s", label="Precision Test2 (with hexagon)")
    plt.plot(iters, rec2, "-s", label="Recall Test2 (with hexagon)")
    plt.plot(iters, iou2, "-s", label="IoU Test2 (with hexagon)")
    plt.xlabel("Iteration")
    plt.ylabel("Metric value")
    plt.title("Task3: Precision / Recall / IoU by iteration")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "task3_metrics_curve.png"))
    plt.close()

    print(f"Графики сохранены в {OUT_DIR}/task3_metrics_curve.png")

    with open(os.path.join(OUT_DIR, "train_class_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(train_ds.class_counts, f, ensure_ascii=False, indent=2)

    print(
        f"Статистика по классам сохранена в {OUT_DIR}/train_class_distribution.json")


if __name__ == "__main__":
    run_task3()
