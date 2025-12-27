import time
import numpy as np
import pandas as pd
from scipy import stats
import mlflow
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    RegressionQualityMetric,
    ClassificationQualityMetric
)
from prometheus_client import Gauge, start_http_server
import requests
import json
import logging
from datetime import datetime, timedelta


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model-monitor")


# Запуск сервера метрик Prometheus
start_http_server(8002)

# Метрики для мониторинга
DATA_DRIFT_PSI = Gauge(
    'model_data_drift_psi',
    'Population Stability Index for data drift'
)
MODEL_QUALITY_F1 = Gauge(
    'model_quality_f1',
    'F1 score of the model'
)
LATENCY_P95 = Gauge(
    'model_latency_p95',
    '95th percentile of model latency'
)


def calculate_psi(expected, actual, buckets=10):
    """
    Рассчитывает Population Stability Index между двумя распределениями

    Args:
    expected: Ожидаемое распределение (обучающие данные)
    actual: Фактическое распределение (production данные)
    buckets: Количество бакетов для группировки

    Returns:
    PSI: Population Stability Index
    """
    # Создание бакетов
    breakpoints = np.linspace(0, 1, buckets + 1)

    # Расчет процентов в каждом бакете
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Добавление небольшого числа для избежания деления 
    epsilon = 1e-10
    expected_percents = np.where(expected_percents == 0, epsilon,
    expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon,
    actual_percents)

    # Расчет PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi


def check_data_drift():
    """Проверяет дрейф данных и генерирует алерты при необходимости"""
    logger.info("Checking for data drift...")

    # Загрузка последних production данных
    # В реальности здесь будут данные из вашей системы
    production_data = pd.read_csv("data/production/requests.csv")

    # Загрузка обучающих данных
    train_data = pd.read_csv("data/processed/train.csv")

    # Выбор признаков для проверки дрейфа
    features = [col for col in train_data.columns if col != "churn"]

    # Расчет PSI для каждого признака
    psi_values = {}
    for feature in features:
        if train_data[feature].dtype in ['int64', 'float64']:
            # Для числовых признаков
            psi = calculate_psi(
                train_data[feature].fillna(train_data[feature].median()),
                production_data[feature].fillna(production_data[feature].median())
            )
            psi_values[feature] = psi

    # Расчет среднего PSI
    avg_psi = np.mean(list(psi_values.values()))
    logger.info(f"Average PSI: {avg_psi:.4f}")

    # Логирование в Prometheus
    DATA_DRIFT_PSI.set(avg_psi)

    # Проверка на превышение порога
    drift_threshold = 0.2 # Можно настроить в зависимости от варианта
    if avg_psi > drift_threshold:
        logger.warning(f"Data drift detected! Average PSI={avg_psi:.4f} > {drift_threshold}")
        send_alert(
            "Data Drift Detected",
            f"Average PSI={avg_psi:.4f} exceeds threshold of {drift_threshold}",
            "warning"
        )
        return True
    return False


def check_model_quality():
    """Проверяет качество модели в production"""
    logger.info("Checking model quality...")

    # Загрузка данных с ground truth (если доступно)
    # В реальности это могут быть данные из A/B теста или implicit feedback
    try:
        production_data = pd.read_csv("data/production/requests_with_labels.csv")

        # Загрузка модели
        model_uri = "models:/churn-prediction/Production"
        model = mlflow.pyfunc.load_model(model_uri)

        # Предсказание
        X = production_data.drop(["churn", "prediction"], axis=1)
        y_true = production_data["churn"]
        y_pred = model.predict(X)

        # Расчет F1
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred)
        logger.info(f"Current F1 score: {f1:.4f}")

        # Логирование в Prometheus
        MODEL_QUALITY_F1.set(f1)

        # Сравнение с baseline
        baseline_f1 = 0.85 # Можно взять из Model Registry
        if f1 < baseline_f1 * 0.95: # 5% падение
            logger.warning(f"Model quality drop detected! F1={f1:.4f} < {baseline_f1*0.95:.4f}")
            send_alert(
                "Model Quality Drop",
                f"F1 score dropped to {f1:.4f} (baseline: {baseline_f1:.4f})",
                "warning"
            )
            return True

        return False

    except Exception as e:
        logger.warning(f"Cannot evaluate model quality: {str(e)}")
    
    return False


def send_alert(title, message, severity):
    """Отправляет алерт в Slack"""
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not slack_webhook:
        logger.error("SLACK_WEBHOOK_URL not set")
    return
    
    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*:{get_severity_emoji(severity)}{severity.upper()}: {title}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Сообщение:*\n{message}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Время:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Запустить переобучение"
                        },
                        "style": "primary",
                        "value": "retrain"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Посмотреть отчет"
                        },
                        "url": "http://monitoring.example.com/report"
                    }
                ]
            }
        ]
    }

    response = requests.post(
        slack_webhook,
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        logger.error(f"Failed to send alert to Slack: {response.text}")


def get_severity_emoji(severity):
    """Возвращает эмодзи в зависимости от уровня серьезности"""
    if severity == "critical":
        return "rotating_light"
    elif severity == "warning":
        return "warning"
    else:
        return "information_source"


def should_retrain(data_drift, quality_drop, last_retrain):
    """
    Определяет, нужно ли переобучать модель

    Args:
    data_drift: Обнаружен ли дрейф данных
    quality_drop: Упала ли точность модели
    last_retrain: Дата последнего переобучения

    Returns:
    bool: Нужно ли переобучать модель
    str: Причина переобучения
    """
    # Проверка максимального интервала
    max_interval = timedelta(weeks=2)
    time_alert = (datetime.now() - last_retrain) > max_interval

    if data_drift or quality_drop or time_alert:
        reason = "Data drift" if data_drift else \
            "Quality drop" if quality_drop else \
            "Time interval"
        return True, reason

    return False, None


def main():
    """Основной цикл мониторинга"""
    last_retrain = datetime.now() - timedelta(days=7) # Предположим, последнее переобучение неделю назад

    while True:
        try:
            # Проверка дрейфа данных
            data_drift = check_data_drift()

            # Проверка качества модели
            quality_drop = check_model_quality()

            # Проверка необходимости переобучения
            retrain, reason = should_retrain(data_drift, quality_drop,
            last_retrain)

            if retrain:
                logger.info(f"Triggering retraining because of {reason}")
                trigger_retraining(reason)
                last_retrain = datetime.now()

            # Ожидание перед следующей проверкой
            time.sleep(3600) # Проверять каждые час

        except Exception as e:
            logger.exception(f"Error in monitoring loop: {str(e)}")
            time.sleep(600) # Подождать 10 минут при ошибке


def trigger_retraining(reason):
    """Запускает процесс переобучения"""
    logger.info(f"Triggering retraining because: {reason}")

    # В реальности здесь будет вызов CI/CD пайплайна
    # Например, через GitHub Actions API или запуск скрипта
    print("Retraining triggered!")


if __name__ == "__main__":
     main()