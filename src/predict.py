import os
import logging
from datetime import datetime
from typing import Optional, Any

import pandas as pd
import mlflow
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, start_http_server


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("churn-prediction")

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Config
DEFAULT_MODEL_URI = "models:/churn-prediction/Production"
MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)

# If set, prefer local model file (Docker-friendly)
MODEL_PATH = os.getenv("MODEL_PATH")  # e.g. "models/model.joblib"

ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS", "1")  # "1" or "0"
PROM_PORT = int(os.getenv("PROMETHEUS_METRICS_PORT", "8001"))

# Metrics
REQUESTS_TOTAL = Counter(
    "churn_prediction_requests_total",
    "Total number of prediction requests"
)
REQUEST_LATENCY = Histogram(
    "churn_prediction_request_latency_seconds",
    "Prediction request latency (seconds)"
)
ERRORS_TOTAL = Counter(
    "churn_prediction_errors_total",
    "Total number of prediction errors"
)

# Model holder
_model: Optional[Any] = None
_model_version: Optional[str] = None
_model_source: Optional[str] = None


def _predict_probability(model: Any, df: pd.DataFrame) -> float:
    """
    Returns P(class=1) if available; otherwise best-effort.
    """
    # Prefer predict_proba if present (sklearn pipelines often expose it)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        # proba shape: (n, 2) for binary
        try:
            return float(proba[0][1])
        except Exception:
            pass

    # Fallback: model.predict()
    pred = model.predict(df)
    try:
        # If it already is probability-like
        p = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            # likely class label
            p = 1.0 if p >= 1.0 else p
        return p
    except Exception:
        return 0.0


def _load_model() -> None:
    """
    Load model either from local MODEL_PATH (preferred if set) or from MLflow MODEL_URI.
    """
    global _model, _model_version, _model_source

    # 1) Local model file
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        logger.info(f"Loading local model from: {MODEL_PATH}")
        m = joblib.load(MODEL_PATH)
        _model = m
        _model_source = f"file:{MODEL_PATH}"
        # Use file mtime as a simple version marker
        _model_version = str(int(os.path.getmtime(MODEL_PATH)))
        logger.info(f"Local model loaded. version={_model_version}")
        return

    # 2) MLflow model URI
    logger.info(f"Loading MLflow model from: {MODEL_URI}")
    m = mlflow.pyfunc.load_model(MODEL_URI)
    _model = m
    _model_source = f"mlflow:{MODEL_URI}"

    try:
        _model_version = getattr(getattr(m, "metadata", None), "run_id", None)
    except Exception:
        _model_version = None

    logger.info(f"MLflow model loaded. version={_model_version}")


@app.on_event("startup")
def on_startup():
    # Start Prometheus metrics server (best-effort)
    if ENABLE_PROMETHEUS.strip() == "1":
        try:
            start_http_server(PROM_PORT, addr="0.0.0.0")
            logger.info(f"Prometheus metrics server started on :{PROM_PORT}")
        except OSError as e:
            logger.warning(f"Could not start Prometheus metrics server on :{PROM_PORT}: {e}")

    _load_model()


class PredictionRequest(BaseModel):
    customer_id: str
    tenure: int
    monthly_charges: float
    total_charges: float
    gender: str
    senior_citizen: int
    partner: str
    dependents: str


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    model_version: str
    latency_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "ok" if _model is not None else "not_ready",
        "model_source": _model_source,
        "model_version": _model_version
    }


@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = datetime.now()
    try:
        response = await call_next(request)
        REQUESTS_TOTAL.inc()
        return response
    except Exception:
        ERRORS_TOTAL.inc()
        raise
    finally:
        latency = (datetime.now() - start_time).total_seconds()
        REQUEST_LATENCY.observe(latency)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = datetime.now()

    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        input_data = pd.DataFrame([request.dict()])

        churn_probability = _predict_probability(_model, input_data)
        churn_prediction = 1 if churn_probability > 0.5 else 0

        latency_ms = (datetime.now() - start_time).total_seconds() * 1000.0

        logger.info(
            f"Prediction for {request.customer_id}: "
            f"probability={churn_probability:.4f}, "
            f"prediction={churn_prediction}, "
            f"latency={latency_ms:.2f}ms"
        )

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=float(churn_probability),
            churn_prediction=int(churn_prediction),
            model_version=str(_model_version),
            latency_ms=float(latency_ms)
        )

    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e))
