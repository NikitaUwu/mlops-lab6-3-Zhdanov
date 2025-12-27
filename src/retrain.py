import os
import subprocess
import datetime
from git import Repo
import mlflow


def retrain_model():
    """Запускает процесс переобучения модели"""
    print("Starting model retraining...")

    # 1. Обновление данных
    update_data()

    # 2. Валидация данных
    if not validate_data():
        print("Data validation failed. Aborting retraining.")
        return False

    # 3. Обучение новой модели
    train_new_model()

    # 4. Оценка качества
    metrics = evaluate_model()

    # 5. Сравнение с production-моделью
    if not compare_with_production(metrics):
        print("New model is not better than production model. Aborting deployment.")
        return False

    # 6. Регистрация модели в Model Registry
    register_model()

    # 7. Развертывание в staging
    deploy_to_staging()

    # 8. A/B тестирование
    if not run_ab_test():
        print("A/B test failed. Aborting production deployment.")
        return False

    # 9. Развертывание в production
    deploy_to_production()

    print("Model retraining and deployment completed successfully.")
    return True


def update_data():
    """Обновляет данные для обучения"""
    print("Updating data...")

    # В реальности здесь будет загрузка новых данных
    # Например, из базы данных или API

    # Для примера, обновим данные через DVC
    subprocess.run(["dvc", "pull"], check=True)
    subprocess.run(["dvc", "repro", "data/processed/train.csv.dvc"], check=True)


def validate_data():
    """Валидирует данные перед обучением"""
    print("Validating data...")

    # Запуск скрипта валидации
    result = subprocess.run(["python", "src/validate_data.py"],
    capture_output=True, text=True)

    if result.returncode != 0:
        rint("Data validation failed:")
        print(result.stderr)
        return False
    return True


def train_new_model():
    """Обучает новую модель"""
    print("Training new model...")

    # Запуск скрипта обучения
    subprocess.run(["python", "src/train.py"], check=True)


def evaluate_model():
    """Оценивает качество модели"""
    print("Evaluating model...")

    # Запуск скрипта оценки
    subprocess.run(["python", "src/evaluate.py"], check=True)

    # Загрузка метрик
    with open("metrics/metrics.json") as f:
        metrics = json.load(f)

    return metrics


def compare_with_production(new_metrics):
    """Сравнивает новую модель с production-моделью"""
    print("Comparing with production model...")

    # Загрузка метрик production-модели
    client = mlflow.tracking.MlflowClient()
    production_model = client.get_latest_versions("churn-prediction", stages=["Production"])[0]
    production_run = client.get_run(production_model.run_id)

    production_metrics = {
        "f1": production_run.data.metrics["f1_score"],
        "roc_auc": production_run.data.metrics["roc_auc"]
    }

    # Сравнение метрик
    improvement = new_metrics["f1"] - production_metrics["f1"]
    print(f"F1 improvement: {improvement:.4f}")

    # Требуем улучшения как минимум на 0.01
    return improvement >= 0.01


def register_model():
    """Регистрирует модель в Model Registry"""
    print("Registering model in Model Registry...")

    # Получение последнего запуска
    client = mlflow.tracking.MlflowClient()
    last_run = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("customer-churnprediction").experiment_id],
        max_results=1
    )[0]

    # Регистрация модели
    model_uri = f"runs:/{last_run.info.run_id}/model"
    model_details = client.create_registered_model("churn-prediction")
    version = client.create_model_version(
        name="churn-prediction",
        source=model_uri,
        run_id=last_run.info.run_id
    )

    # Переход в staging
    client.transition_model_version_stage(
        name="churn-prediction",
        version=version.version,
        stage="Staging"
    )


def deploy_to_staging():
    """Развертывает модель в staging"""
    print("Deploying to staging environment...")

    # В реальности здесь будет вызов CI/CD пайплайна
    # Например, через GitHub Actions API
    subprocess.run(["kubectl", "apply", "-f", "k8s/staging.yaml"], check=True)


def run_ab_test():
    """Проводит A/B тестирование"""
    print("Running A/B test...")

    # Имитация A/B теста
    time.sleep(60) # Ждем, пока система стабилизируется

    # Проверка метрик
    # В реальности здесь будут реальные метрики из мониторинга
    new_model_metrics = {
        "latency_p95": 150, # ms
        "error_rate": 0.005,
        "f1": 0.87
    }

    production_metrics = {
        "latency_p95": 140, # ms
        "error_rate": 0.006,
        "f1": 0.86
    }

    # Проверка, что новые метрики не хуже
    if (new_model_metrics["latency_p95"] < production_metrics["latency_p95"] * 1.1 and new_model_metrics["error_rate"] < production_metrics["error_rate"] * 1.1):
        print("A/B test passed!")
        return True
    else:
        print("A/B test failed!")
        return False


def deploy_to_production():
    """Развертывает модель в production"""
    print("Deploying to production...")

    # Получение последней версии в staging
    client = mlflow.tracking.MlflowClient()
    staging_versions = client.get_latest_versions("churn-prediction",
    stages=["Staging"])

    if not staging_versions:
        print("No model in staging. Aborting deployment.")
    return

    # Переход в production
    client.transition_model_version_stage(
        name="churn-prediction",
        version=staging_versions[0].version,
        stage="Production"
    )

    # Развертывание через CI/CD
    subprocess.run(["kubectl", "apply", "-f", "k8s/production.yaml"], check=True)
    print("Model deployed to production successfully!")


if __name__ == "__main__":
    retrain_model()