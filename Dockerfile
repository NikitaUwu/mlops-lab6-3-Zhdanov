FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && if [ -s requirements.txt ]; then python -m pip install -r requirements.txt; fi \
    && python -m pip install "uvicorn[standard]" fastapi

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
