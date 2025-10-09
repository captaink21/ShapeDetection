# 🧠 NLP Text Classification — от TF-IDF до Fine-Tuned BERT

### 🎯 Цель проекта

Создать систему классификации текстов и сравнить качество различных подходов — от классических ML-моделей с TF-IDF до современных нейросетевых архитектур (Sentence-BERT и Fine-Tuned BERT).

Проект демонстрирует **путь развития NLP-моделей**:

> "От простых статистических методов — к контекстным трансформерам."
> 

---

## ⚙️ Структура проекта

| Раздел | Описание |
| --- | --- |
| **1. Импорт и установка библиотек** | Подключение необходимых пакетов: `sklearn`, `xgboost`, `lightgbm`, `catboost`, `sentence-transformers`, `transformers` |
| **2. Загрузка данных** | Используются `train_nlp.csv` и `test_nlp.csv`. Train содержит `text` и `label`. |
| **3. EDA (анализ данных)** | Проверка структуры, баланса классов, распределения длины текстов и визуализация WordCloud по классам. |
| **4. TF-IDF векторизация** | Преобразование текстов в числовое представление для обучения классических моделей. |
| **5. Базовые модели** | Logistic Regression, Linear SVM — как сильные baseline-решения. |
| **6. Градиентные бустинги** | CatBoost, XGBoost, LightGBM — проверка, как они справляются с разреженными признаками. |
| **7. Sentence-BERT + Logistic Regression** | Использование эмбеддингов с `all-MiniLM-L6-v2` для контекстного представления текста. |
| **8. Fine-Tuning BERT** | Дообучение модели `bert-base-multilingual-cased` под задачу классификации. |
| **9. Сравнение моделей** | Визуализация F1-score и выбор лучшего решения. |
| **10. Финальные предсказания** | Использование лучшей модели для генерации `submission_final.csv`. |
| **11. Выводы** | Итоговое сравнение и направления развития проекта. |

---

## 📊 Используемые данные

| Файл | Назначение | Пример |
| --- | --- | --- |
| `train_nlp.csv` | Обучение и валидация | `{"text": "This show seemed to be kinda good. ", "label": 1}` |
| `test_nlp.csv` | Финальное тестирование | `{"text": "This show seemed to be kinda good. ", "label": 1}` |

---

## 🧩 Модели и подходы

| Подход |
| --- |
| **TF-IDF + Logistic Regression** |
| **TF-IDF + LinearSVM** |
| **CatBoost / XGBoost / LightGBM** |
| **Sentence-BERT + LogReg** |
| **Fine-tuned BERT** |

---

## 🚀 Запуск проекта

```bash
# 1. Клонировать репозиторий
git clone https://github.com/<your-username>/nlp-text-classification.git
cd nlp-text-classification

# 2. Запустить ноутбук
jupyter notebook NLP_Text_Classification.ipynb

```

---

---

## 📚 Технологии

- Python 3.10
- scikit-learn
- CatBoost / XGBoost / LightGBM
- Sentence-Transformers
- HuggingFace Transformers
- Matplotlib / Seaborn / WordCloud

---

## 🏁 Выводы

- TF-IDF остаётся мощным baseline для коротких текстов.
- Контекстные модели (SBERT, BERT) выигрывают на длинных и сложных примерах.
- Fine-tuning даёт максимальную гибкость, но требует больше ресурсов.

---

## 🧑‍💻 Автор

**Фархад Миннеханов**

📧 minfarkhad@gmail.com

🔗 Telegram @miehao