# =====================================================================
# api.py - FastAPI para predicción de nivel de estrés usando MLflow Registry
# =====================================================================

import pickle
import os
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# =========================
# Configuración API
# =========================
app = FastAPI(
    title="Stress Level Prediction API",
    description="Servicio para predecir nivel de estrés usando modelo MLFlow (Registry)",
    version="1.0.0"
)

# =========================
# Cargar preprocesadores y modelo desde MLflow Registry
# =========================
load_dotenv(override=True)

# Preprocesadores locales
try:
    with open("preprocessor/dv.b", "rb") as f:
        dv = pickle.load(f)
    with open("preprocessor/scaler.b", "rb") as f:
        scaler = pickle.load(f)
    with open("preprocessor/features.pkl", "rb") as f:
        features = pickle.load(f)

except Exception as e:
    print("Error cargando preprocesadores:", e)
    dv = None
    scaler = None
    features = None


# Modelo MLflow Registry
try:
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    
    MODEL_REGISTRY_NAME = "workspace.default.equipo1-proyecto"
    ALIAS = "Champion"
    model_uri = f"models:/{MODEL_REGISTRY_NAME}@{ALIAS}"
    
    champion_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    model_loaded = True
except Exception as e:
    print("Error cargando modelo desde MLflow Registry:", e)
    champion_model = None
    model_loaded = False

# =========================
# Mapeos auxiliares
# =========================
def _map_continent(country):
    mapping = {
        "Canada": "America", "USA": "America", "Mexico": "America", "Brazil": "America",
        "Norway": "Europe", "Sweden": "Europe", "UK": "Europe", "Finland": "Europe",
        "Italy": "Europe", "Belgium": "Europe", "Germany": "Europe", "France": "Europe",
        "Switzerland": "Europe", "Netherlands": "Europe", "Spain": "Europe",
        "India": "Asia", "China": "Asia", "South Korea": "Asia", "Japan": "Asia",
        "Australia": "Oceania"
    }
    return mapping.get(country, np.nan)

sleep_map = {"Poor": "Bad", "Fair": "Bad", "Good": "Good", "Excellent": "Good"}

# =========================
# Pydantic input
# =========================
class InputData(BaseModel):
    sleep_quality: str 
    occupation: str 
    coffee_intake: float 
    physical_activity_hours: float 
    country: str 
    bmi: float 
    alcohol_consumption: str 
    age: int 
    gender: str 
    heart_rate: int 
    smoking: str

# =========================
# Preprocesamiento
# =========================
def preprocess(input_data: InputData):
    if not dv or not scaler or not features:
        raise HTTPException(status_code=500, detail="Preprocesadores no cargados")

    df = pd.DataFrame([input_data.dict()])

    # Sleep quality mapping
    df["Sleep_Group"] = df["sleep_quality"].map(sleep_map)
    df = df.drop(columns=["sleep_quality"], errors="ignore")

    # Continent
    df["Continent"] = df["country"].map(_map_continent)
    df = df.drop(columns=["country"], errors="ignore")

    # Gender encoding
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    # Bool -> int
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(int)

    # DictVectorizer
    X_dicts = df.to_dict(orient="records")
    X_encoded = dv.transform(X_dicts).astype(float)
    X_encoded[~np.isfinite(X_encoded)] = 0.0
    X_df = pd.DataFrame(X_encoded, columns=dv.get_feature_names_out())

    # Asegurar columnas
    for f in features:
        if f not in X_df.columns:
            X_df[f] = 0.0
    X_df = X_df[features]

    # Escalado
    X_df_scaled = X_df.copy()
    numeric_cols = X_df_scaled.select_dtypes(include=np.number).columns
    X_df_scaled[numeric_cols] = scaler.transform(X_df_scaled[numeric_cols])

    return X_df_scaled

# =========================
# Predicción
# =========================
def predict(input_data: InputData):
    if champion_model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    X = preprocess(input_data)  # ahora devuelve DataFrame con nombres
    y_pred = champion_model.predict(X)
    return float(y_pred[0])


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    status = "ok" if model_loaded else "error"
    return {"status": status, "model_loaded": model_loaded}

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    try:
        pred = predict(input_data)
        return JSONResponse(
            status_code=200,
            content={
                "prediction": pred,
                "features_used": features
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






