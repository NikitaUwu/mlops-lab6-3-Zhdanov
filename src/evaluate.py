import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import json
import os


def evaluate_model():
    """Оценка качества модели на тестовых данных"""
    # Загрузка данных
    test_path = "data/processed/test.csv"
    df = pd.read_csv(test_path)

    # Разделение на признаки и целевую переменную
    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Загрузка модели из последнего эксперимента
    # В реальности здесь будет выбор лучшей модели из Model Registry
    model_uri = "models:/churn-prediction/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    # Предсказание
    y_pred = model.predict(X)

    # Расчет метрик
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred)
    }

    # Логирование метрик в MLflow
    with mlflow.start_run():
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

    # Сохранение метрик в файл
    os.makedirs("metrics", exist_ok=True)

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model evaluation completed:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")

    return metrics


if __name__ == "__main__":
     evaluate_model()
