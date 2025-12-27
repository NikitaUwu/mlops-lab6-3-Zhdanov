FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=models/model.joblib \
    ENABLE_PROMETHEUS=1 \
    PROMETHEUS_METRICS_PORT=8001

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 8000 8001

CMD ["python", "-m", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
