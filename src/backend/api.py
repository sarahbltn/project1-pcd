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

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/sarahbeltrang@gmail.com.mx/project1-experiment"

# Buscar mejor run
run_ = mlflow.search_runs(
    order_by=['metrics.f1 DESC'], #cambiar porque no se como guardamos si f1 o f1_score o como
    output_format="list",
    experiment_names=[EXPERIMENT_NAME]
)[0]

run_id = run_.info.run_id

# Descargar preprocesadores del run
client.download_artifacts(run_id, "preprocessor", ".")

with open("preprocessor/dv.b", "rb") as f:
    dv = pickle.load(f)

with open("preprocessor/scaler.b", "rb") as f:
    scaler = pickle.load(f)

with open("preprocessor/top_features.pkl", "rb") as f:
    top_features = pickle.load(f)


# ================================
# Cargar modelo champion
# ================================

model_name = "workspace.default.project1-model" #cambiar segun el nombre del modelo no s eocmo le pusimos
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(model_uri)


# ================================
# Funciones de Preprocesamiento  
# (versión para API = eval)
# ================================

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

    # 1. Sleep_Quality -> Sleep_Group
    if "Sleep_Quality" in df.columns:
        df["Sleep_Group"] = df["Sleep_Quality"].map(sleep_map)
        df = df.drop(columns=["Sleep_Quality"], errors="ignore")

    # 2. Eliminar columnas que no van
    df = df.drop(columns=["Health_Issues", "Caffeine_mg", "Sleep_Hours"], errors="ignore")

    # 3. Country -> Continent
    if "Country" in df.columns:
        df["Continent"] = df["Country"].map(_map_continent)

    # 4. Gender map
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # 5. Drop columnas irrelevantes
    df = df.drop(columns=["Country", "ID"], errors="ignore")

    # 6. bool → int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # 7. Transformación con DictVectorizer
    X_dicts = df.to_dict(orient="records")
    X_encoded = dv.transform(X_dicts).astype(float)

    # 8. Reemplazar NaN e infinitos
    X_encoded[~np.isfinite(X_encoded)] = 0.0

    X_df = pd.DataFrame(X_encoded, columns=dv.get_feature_names_out())

    # 9. Asegurar que todas las top_features existan
    for f in top_features:
        if f not in X_df.columns:
            X_df[f] = 0.0

    # 10. Ordenar columnas
    X_df = X_df[top_features]

    # 11. Escalar
    X_scaled = scaler.transform(X_df.values)

    return X_scaled



# ================================
# Predicción
# ================================

def predict(input_data):
    X = preprocess(input_data)
    pred = champion_model.predict(X)
    return pred


# ================================
# FastAPI
# ================================

app = FastAPI()

class InputData(BaseModel):
    Sleep_Quality: str | None = None
    Sleep_Group: str | None = None  # opcional si ya la mandan
    Occupation: str | None = None
    Coffee_Intake: float | None = None
    Physical_Activity_Hours: float | None = None
    Continent: str | None = None
    Country: str | None = None
    BMI: float | None = None
    Alcohol_Consumption: float | None = None
    Age: float | None = None
    Gender: str | None = None
    Heart_Rate: float | None = None
    Smoking: int | None = None
    ID: int | None = None
    Health_Issues: int | None = None
    Caffeine_mg: float | None = None
    Sleep_Hours: float | None = None


@app.post("/api/v1/predict")
def predict_endpoint(payload: InputData):
    pred = predict(payload)[0]
    return {"prediction": float(pred)}
