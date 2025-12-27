import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import logging


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("churn-prediction")
app = FastAPI()

# Загрузка модели
model_uri = "models:/churn-prediction/Production"
model = mlflow.pyfunc.load_model(model_uri)


class PredictionRequest(BaseModel):
    customer_id: str
    tenure: int
    monthly_charges: float
    total_charges: float
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    # Добавьте остальные признаки в зависимости от вашей задачи


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    model_version: str
    latency_ms: float


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_version": model.metadata.run_id}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Предсказание оттока клиента"""
    start_time = datetime.now()

    try:
        # Преобразование запроса в DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Предсказание
        churn_probability = model.predict(input_data)[0]
        churn_prediction = 1 if churn_probability > 0.5 else 0
        
        # Расчет времени обработки
        latency = (datetime.now() - start_time).total_seconds() * 1000

        # Логирование
        logger.info(
            f"Prediction for {request.customer_id}: "
            f"probability={churn_probability:.4f}, "
            f"prediction={churn_prediction}, "
            f"latency={latency:.2f}ms"
        )

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=float(churn_probability),
            churn_prediction=int(churn_prediction),
            model_version=model.metadata.run_id,
            latency_ms=float(latency)
        )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Метрики для мониторинга (будут использоваться Prometheus)
from prometheus_client import Counter, Histogram, start_http_server


REQUESTS_TOTAL = Counter(
    'churn_prediction_requests_total',
    'Total number of prediction requests'
)
REQUEST_LATENCY = Histogram(
    'churn_prediction_request_latency_seconds',
    'Prediction request latency'
)
ERRORS_TOTAL = Counter(
    'churn_prediction_errors_total',
    'Total number of prediction errors'
)


@app.middleware("http")
async def monitor_requests(request, call_next):
    """Middleware для мониторинга запросов"""
    start_time = datetime.now()
    try:
        response = await call_next(request)
        REQUESTS_TOTAL.inc()
        return response
    except Exception as e:
        ERRORS_TOTAL.inc()
        raise
    finally:
        latency = (datetime.now() - start_time).total_seconds()
        REQUEST_LATENCY.observe(latency)

# Запуск сервера метрик Prometheus
start_http_server(8001)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)