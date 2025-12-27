import os
import importlib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    # Чтобы тесты не пытались поднять Prometheus-сервер на порту (в CI/локально это часто конфликтует)
    os.environ["ENABLE_PROMETHEUS"] = "0"

    # Тесты требуют MLflow URI (в CI он задаётся через workflow)
    if not (os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_REGISTRY_URI")):
        pytest.skip("MLFLOW_TRACKING_URI/MLFLOW_REGISTRY_URI not set; skipping API tests")

    is_ci = os.getenv("CI", "").lower() == "true"

    mod = importlib.import_module("src.predict")

    try:
        # Контекст-менеджер гарантирует запуск lifespan/startup => модель должна загрузиться
        with TestClient(mod.app) as c:
            yield c
    except Exception as e:
        # Локально удобнее “skip”, в CI — пусть падает
        if is_ci:
            raise
        pytest.skip(f"API startup failed (likely MLflow/Model Registry not ready): {e}")


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body.get("model_version") is not None


def test_predict_contract(client: TestClient):
    payload = {
        "customer_id": "C999999",
        "tenure": 12,
        "monthly_charges": 75.5,
        "total_charges": 850.0,
        "gender": "M",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["customer_id"] == payload["customer_id"]
    assert 0.0 <= float(body["churn_probability"]) <= 1.0
    assert int(body["churn_prediction"]) in (0, 1)
    assert body.get("model_version") is not None
    assert float(body["latency_ms"]) >= 0.0
