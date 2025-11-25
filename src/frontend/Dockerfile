# Dockerfile para la UI (Streamlit) del proyecto Stress Level Prediction
FROM python:3.13-slim

LABEL maintainer="monica.ibarra@iteso.mx"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ="America/Mexico_City" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS="0.0.0.0"

WORKDIR /app

# Instalar dependencias del sistema necesarias (libgomp para xgboost, procps para debug, tzdata para zona horaria)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libgomp1 \
        procps \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copiar requirements e instalar (asumimos que tienes requirements.txt en la raíz)
COPY --chown=appuser:appuser ./requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copiar la app Streamlit. Si tu archivo se llama ui.py, renómbralo a main.py o cambia aquí.
COPY --chown=appuser:appuser ./main.py /app/main.py

# Crear carpeta para artefactos que será montada en tiempo de ejecución (dv.b, scaler.b, features.pkl)
RUN mkdir -p /app/preprocessor && chown -R appuser:appuser /app/preprocessor

EXPOSE 8501

# Variables de entorno por defecto para la API
ENV API_BASE_URL="http://host.docker.internal:8000" \
    PREDICT_ENDPOINT="/predict"

# Command to run streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]