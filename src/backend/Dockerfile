# =====================================================================
# Dockerfile para FastAPI + MLflow Model Registry + Preprocesadores
# =====================================================================

FROM python:3.13-slim

WORKDIR /code

# Copiar requirements si existe
COPY ./requirements.txt /code/requirements.txt

# Dependencias del sistema
RUN apt-get update && apt-get install -y procps && \
    pip install -U pip && \
    rm /etc/localtime && \
    ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime && \
    pip install -r ./requirements.txt


# Copiar la app
COPY ./api.py /code/

# Exponer puerto FastAPI
EXPOSE 8000

# Comando de ejecución
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
