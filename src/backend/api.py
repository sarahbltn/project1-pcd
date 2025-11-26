import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

load_dotenv(override=True)

# =========================================
# CONFIG MLflow
# =========================================
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/priscila.cervantes@iteso.mx/project1-experiment"
model_name = "workspace.default.equipo1-proyecto"   # <- nombre correcto del pipeline
alias = "Champion"

# =========================================
# Cargar preprocesadores guardados
# =========================================

# Descarga artifacts del champion
champ_version = client.get_model_version_by_alias(model_name, alias)
run_id = champ_version.run_id

client.download_artifacts(run_id, "preprocessor", ".")

with open("preprocessor/dv.b", "rb") as f:
    dv = pickle.load(f)

with open("preprocessor/scaler.b", "rb") as f:
    scaler = pickle.load(f)

with open("preprocessor/features.pkl", "rb") as f:
    features = pickle.load(f)

# =========================================
# Cargar modelo champion desde MLflow
# =========================================
model_uri = f"models:/{model_name}@{alias}"
champion_model = mlflow.pyfunc.load_model(model_uri)

# =========================================
# Preprocesamiento (versión API)
# Igual que preprocessing_eval del pipeline
# =========================================

def _map_continent(country):
    pais_a_continente = {
        "Canada": "America", "USA": "America", "Mexico": "America", "Brazil": "America",
        "Norway": "Europe", "Sweden": "Europe", "UK": "Europe", "Finland": "Europe",
        "Italy": "Europe", "Belgium": "Europe", "Germany": "Europe", "France": "Europe",
        "Switzerland": "Europe", "Netherlands": "Europe", "Spain": "Europe",
        "India": "Asia", "China": "Asia", "South Korea": "Asia", "Japan": "Asia",
        "Australia": "Oceania"
    }
    return pais_a_continente.get(country, np.nan)

sleep_map = {"Poor": "Bad", "Fair": "Bad", "Good": "Good", "Excellent": "Good"}


def preprocess(input_data):

    df = pd.DataFrame([input_data.dict()])

    if "Sleep_Quality" in df.columns:
        df["Sleep_Group"] = df["Sleep_Quality"].map(sleep_map)
        df = df.drop(columns=["Sleep_Quality"], errors="ignore")

    df = df.drop(columns=['Health_Issues', 'Caffeine_mg', 'Sleep_Hours'], errors="ignore")

    if "Country" in df.columns:
        df["Continent"] = df["Country"].map(_map_continent)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    df = df.drop(columns=["Country", "ID"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # DictVectorizer
    X_dicts = df.to_dict(orient="records")
    X_encoded = dv.transform(X_dicts).astype(float)
    X_encoded[~np.isfinite(X_encoded)] = 0.0

    X_df = pd.DataFrame(X_encoded, columns=dv.get_feature_names_out())

    # asegurar order de columnas features.pkl
    for f in features:
        if f not in X_df.columns:
            X_df[f] = 0.0

    X_df = X_df[features]

    X_scaled = scaler.transform(X_df.values)

    return X_scaled


# =========================================
# Predicción
# =========================================
def predict(input_data):
    X = preprocess(input_data)
    return champion_model.predict(X)
    


# =========================================
# FastAPI
# =========================================

app = FastAPI()

class InputData(BaseModel):
    Sleep_Quality: str
    Occupation: str 
    Coffee_Intake: float 
    Physical_Activity_Hours: float 
    Country: str 
    BMI: float 
    Alcohol_Consumption: str 
    Age: int
    Gender: str 
    Heart_Rate: int 
    Smoking: bool  


@app.post("/api/v1/predict")
def predict_endpoint(payload: InputData):
    pred = predict(payload)[0]
    return {"prediction": float(pred)}



