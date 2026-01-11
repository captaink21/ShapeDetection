# 🎯 Детекция геометрических фигур | YOLOv8 + PyTorch

**Система детекции геометрических фигур с поэтапным расширением модели на новый класс без потери качества на базовых классах.**

---

## 📋 Описание проекта

Этот проект демонстрирует эффективный подход к **инкрементальному обучению** (incremental learning) в контексте детекции объектов:

- **Начальное обучение**: модель YOLOv8n обучается на 3 классах (треугольник, ромб, круг) с использованием синтетически сгенерированных данных.
- **Поэтапное расширение**: на протяжении 20 итераций добавляется новый класс (шестиугольник).
- **На каждой итерации**:
    - Генерируются дополнительные 400 изображений с новым классом
    - Модель переобучается на расширенном датасете
    - Считаются метрики качества на двух тестовых наборах:
        - **Test1** (без гексагонов) — контроль деградации старых классов
        - **Test2** (с гексагонами) — оценка нового класса
- **Результат**: все 4 класса достигают близкого к идеальному качества (Precision/Recall ≈ 1.0) **без деградации на базовых классах**.

---

## 🛠️ Стек технологий

| **Компонент** | **Технология** |
| --- | --- |
| **Модель детекции** | YOLOv8n (Ultralytics) |
| **DL-фреймворк** | PyTorch 2.0+ |
| **Генерация синтетических данных** | OpenCV, NumPy |
| **Аугментация** | Albumentations |
| **Анализ и визуализация** | Pandas, Matplotlib, Seaborn |
| **Экспорт модели** | ONNX Runtime |
| **Production-инференс (C++)** | ONNX Runtime C++ API |

---

## 📁 Структура проекта

`textShapeDetection/
├── task_1.py                    # Генерация и визуализация синтетических фигур
├── task_2_yolo.py               # Вспомогательные функции для YOLO
│                                #   - BalancedOnTheFly (датасет)
│                                #   - evaluate_yolo_model (оценка)
│                                #   - create_yolo_yaml (конфигурация)
├── task_3_yolo.py               # Основной эксперимент (инкрементальное обучение)
├── infer_yolo.py                # Инференс PyTorch-модели YOLOv8
├── infer_onnx.cpp               # Инференс ONNX-модели в C++
├── spark_generate.py            # Утилита для параллельной генерации данных
├── Makefile                     # Сборка C++ компонентов
└── runs/yolo_exp1/yolo_result/  # Результаты эксперимента (генерируются автоматически)
    ├── history.json             # История метрик по итерациям
    └── final_report.json        # Финальный отчет с результатами`

---

## 🚀 Установка

## Требования

- **Python**: 3.10 или выше
- **CUDA** (опционально, для ускорения на GPU)

---

## 💻 Использование

## 1️⃣ Основной эксперимент (Task 3) — Инкрементальное обучение

Это основной скрипт, который выполняет весь цикл экспериментов:

`python3 task_3_yolo.py`

**Что он делает:**

1. Создает начальный датасет (12,000 изображений без гексагонов)
2. Обучает модель YOLOv8n на базовых 3 классах
3. Выполняет 20 итераций:
    - Генерирует +400 изображений с новым классом (гексагон)
    - Переобучает модель на расширенном датасете
    - Оценивает метрики на Test1 и Test2
4. Сохраняет результаты:
    - **`history.json`** — история метрик
    - **`final_report.json`** — финальный отчет

**Результаты сохраняются в**: **`runs/yolo_exp1/yolo_result/`**

---

## 2️⃣ Генерация синтетических данных (Task 1)

Просмотр и проверка синтетически сгенерированных фигур:

`python3 task_1.py`

Создает примеры изображений с геометрическими фигурами.

---

## 3️⃣ Инференс на PyTorch

Использование обученной модели для детекции фигур на новых изображениях:

`python3 infer_yolo.py --weights <path_to_weights.pt> \
                     --source <image_or_folder> \
                     --output <output_folder>`

**Пример:**

`python3 infer_yolo.py --weights runs/yolo_exp1/yolo_result/20_retrain/weights/best.pt \
                     --source test_images/ \
                     --output results/`

---

## 4️⃣ Инференс на C++ с ONNX (Production)

Для интеграции в production-системы на C++:

## Сборка C++ примера

`На Linux:
make`

---

## 🔍 Ключевые файлы и их роли

## **`task_3_yolo.py`**

**Основной скрипт эксперимента**

- Генерирует синтетические датасеты (train/val/test)
- Управляет циклом из 20 итераций
- Вызывает обучение YOLOv8 через **`ultralytics.YOLO`**
- Использует **`evaluate_yolo_model()`** для подсчета метрик:
    - Precision@0.5, Recall@0.5, Mean IoU
    - Per-class статистика (GT/Det)
- Сохраняет историю в JSON

**Основные параметры (можно менять):**

- **`ITERATIONS = 20`** — количество итераций
- **`EPOCHS_PER_ITER = 5`** — эпох обучения на итерацию
- **`BATCH_SIZE = 64`** — размер батча

---

## **`task_2_yolo.py`**

**Вспомогательные компоненты**

- **`BalancedOnTheFly`** — класс для синтетической генерации батчей
    - Создает изображения с геометрическими фигурами в реальном времени
    - Поддерживает аугментацию (Albumentations)
    - Параметры: количество фигур, размер изображения, список классов
- **`evaluate_yolo_model()`** — оценка модели на DataLoader
    - Считает Precision, Recall, IoU
    - Сохраняет per-class статистику
    - Возвращает словарь метрик
- **`create_yolo_yaml()`** — генерирует YAML конфигурацию
    - Форматированный под требования Ultralytics YOLO
    - Указывает пути к датасетам и количество классов

**Константы:**

- **`NUM_CLASSES = 4`**
- **`CLASS2ID = {0: "triangle", 1: "rhombus", 2: "circle", 3: "hexagon"}`**
- **`IMG_SIZE = 256`**

---

## **`infer_yolo.py`**

**Инференс на PyTorch**

Загружает обученную YOLOv8-модель и выполняет детекцию:

`python# Пример программного использования
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="image.jpg", conf=0.5)`

---

## **`infer_onnx.cpp`**

**Инференс на C++ через ONNX Runtime**

Демонстрирует, как использовать экспортированную ONNX-модель в production:

**Преимущества ONNX:**

- Отсутствие зависимости на Python
- Быстрый инференс
- Кросс-платформенность
- Интеграция в C++

---

## 📈 Интерпретация результатов

**coverage_history.png:**

- Если все линии близко к 1.0 → модель видит все объекты
- Если линии < 1.0 → есть пропуски объектов
- Если линии > 1.0 → есть ложные срабатывания (дубли)

---

## 🎨 Примеры использования в коде

## Загрузка истории и построение собственных графиков

`import json
import matplotlib.pyplot as plt

# Загрузить историю
with open("runs/yolo_exp1/yolo_result/history.json", "r") as f:
    history = json.load(f)

# Вытащить метрики
iterations = history["iterations"]
precision = [m["precision@0.5"] for m in history["test1_metrics"]]

# Построить график
plt.plot(iterations, precision, 'o-', label='Precision Test1')
plt.xlabel('Iteration')
plt.ylabel('Precision')
plt.legend()
plt.savefig('my_custom_plot.png')
plt.show()`

## Использование обученной модели для детекции

`python
from ultralytics import YOLO
import cv2

# Загрузить модель
model = YOLO("runs/yolo_exp1/yolo_result/20_retrain/weights/best.pt")

# Выполнить детекцию
results = model.predict(source="test_image.jpg", conf=0.5, iou=0.45)

# Вывести результаты
for result in results:
    print(f"Обнаружено объектов: {len(result.boxes)}")
    for box in result.boxes:
        print(f"  Класс: {box.cls}, Confidence: {box.conf:.2f}")`

---

## 📄 Результаты в JSON

После завершения эксперимента доступны два важных файла:

## **`history.json`**

Содержит полную историю метрик по всем 20 итерациям:

`json{
  "iterations": [1, 2, 3, ..., 20],
  "test1_metrics": [
    {
      "precision@0.5": 0.9999,
      "recall@0.5": 0.9999,
      "mean_iou": 0.9998,
      "per_class_gt": {"0": 750, "1": 750, "2": 750},
      "per_class_det": {"0": 750, "1": 750, "2": 750}
    },
    ...
  ],
  "test2_metrics": [...],
  "train_sizes": [12000, 12400, 12800, ...]
}`

## **`final_report.json`**

Агрегированный отчет с итоговыми результатами:

`json{
  "task": "task_3_yolo.py",
  "iterations": 20,
  "epochs_per_iter": 5,
  "initial_train_size": 12000,
  "increment_per_iter": 400,
  "yolo_result": {
    "test1_initial_precision": 0.9999,
    "test1_final_precision": 0.9999,
    "precision_degradation_percent": 0.0,
    ...
  },
  "new_class_performance": {
    "avg_precision": 0.9998,
    "avg_recall": 0.9996,
    "avg_mean_iou": 0.9992,
    ...
  }
}`

---