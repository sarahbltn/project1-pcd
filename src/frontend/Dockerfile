FROM python:3.13-slim

WORKDIR /code

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y procps && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r ./requirements.txt

# Copiar app
COPY ./main.py /code/

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
