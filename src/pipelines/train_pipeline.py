# train_pipeline_prefect.py
import os
import pathlib
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import optuna
from optuna.samplers import TPESampler

# -------------------------
# Config
# -------------------------
load_dotenv(override=True)

EXPERIMENT_NAME = "/Users/sarahbeltrang@gmail.com.mx/project1-experiment"
MODEL_REGISTRY_NAME = "workspace.default.equipo1-proyecto"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)

# Desactivar autolog por defecto
mlflow.sklearn.autolog(log_models=False)
mlflow.xgboost.autolog(log_models=False)

# -------------------------
# Helpers / Preprocessing
# -------------------------
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

def preprocessing_train(df: pd.DataFrame, n_top_features: int = 20):
    df = df.copy()

    # Agrupar Sleep_Quality
    sleep_map = {"Poor": "Bad", "Fair": "Bad", "Good": "Good", "Excellent": "Good"}
    if "Sleep_Quality" in df.columns:
        df["Sleep_Group"] = df["Sleep_Quality"].map(sleep_map)
        df = df.drop(columns=["Sleep_Quality"], errors='ignore')

    df = df.drop(columns=['Health_Issues', 'Caffeine_mg', 'Sleep_Hours'], errors='ignore')
    if "Gender" in df.columns:
        df = df[df["Gender"] != "Other"]
    if "Country" in df.columns:
        df["Continent"] = df["Country"].map(_map_continent)

    # target mapping
    df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    if "Gender" in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df = df.drop(columns=["Country","ID"], errors='ignore')

    # bool -> int
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    y = df["Stress_Level"].values
    X_df = df.drop(columns=["Stress_Level"])

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(X_df.to_dict(orient="records"))
    X_df_encoded = pd.DataFrame(X, columns=dv.get_feature_names_out())

    # eliminar dummy redundante
    feature_names = dv.get_feature_names_out().tolist()
    groups = {}
    for f in feature_names:
        if '=' in f:
            pref = f.split('=')[0]
            groups.setdefault(pref, []).append(f)
    to_drop_dummy = []
    for feats in groups.values():
        if len(feats) > 1:
            to_drop_dummy.append(sorted(feats)[0])
    X_df_encoded = X_df_encoded.drop(columns=[c for c in to_drop_dummy if c in X_df_encoded.columns], errors='ignore')

    # eliminar columnas constantes
    constant_cols = X_df_encoded.nunique()[X_df_encoded.nunique() <= 1].index.tolist()
    X_df_encoded = X_df_encoded.drop(columns=constant_cols, errors='ignore')

    # eliminar altamente correlacionadas
    corr_matrix = X_df_encoded.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.9)]
    X_filtered = X_df_encoded.drop(columns=to_drop_corr, errors='ignore')

    # mutual info
    mi = mutual_info_classif(X_filtered.values, y, random_state=42)
    mi_series = pd.Series(mi, index=X_filtered.columns)
    top_features = mi_series.nlargest(n_top_features).index.tolist()
    top_features = [f for f in top_features if f in X_filtered.columns]
    X_selected = X_filtered[top_features]

    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected.values)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)

    dropped = {"dropped_dummy": to_drop_dummy, "dropped_corr": to_drop_corr}
    return X_bal, y_bal, dv, top_features, dropped, scaler

def preprocessing_eval(df: pd.DataFrame, dv: DictVectorizer, features, scaler: RobustScaler):
    df = df.copy()
    sleep_map = {"Poor": "Bad", "Fair": "Bad", "Good": "Good", "Excellent": "Good"}
    if "Sleep_Quality" in df.columns:
        df["Sleep_Group"] = df["Sleep_Quality"].map(sleep_map)
        df = df.drop(columns=["Sleep_Quality"], errors='ignore')

    df = df.drop(columns=['Health_Issues', 'Caffeine_mg', 'Sleep_Hours'], errors='ignore')
    if "Gender" in df.columns:
        df = df[df["Gender"] != "Other"]
    if "Country" in df.columns:
        df["Continent"] = df["Country"].map(_map_continent)
    df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    if "Gender" in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df = df.drop(columns=["Country","ID"], errors='ignore')

    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    dicts = df.drop(columns=["Stress_Level"]).to_dict(orient="records")
    X_encoded = dv.transform(dicts).astype(float)
    X_encoded[~np.isfinite(X_encoded)] = 0.0
    X_df_encoded = pd.DataFrame(X_encoded, columns=dv.get_feature_names_out())

    # asegurar columnas faltantes
    for f in features:
        if f not in X_df_encoded.columns:
            X_df_encoded[f] = 0.0

    X_scaled = scaler.transform(X_df_encoded[features].values)
    y = df["Stress_Level"].values
    return X_scaled, y

# -------------------------
# Prefect Tasks
# -------------------------
@task(name="Read CSV")
def read_csv_task(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV no encontrado en: {file_path}")
    return pd.read_csv(file_path)

@task(name="Split data")
def split_task(df: pd.DataFrame, test_size: float = 0.4, seed: int = 42):
    target = "Stress_Level"
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df[target])
    return train_df, val_df, test_df

@task(name="Preprocess train")
def preprocess_train_task(train_df: pd.DataFrame, n_top_features: int = 20):
    return preprocessing_train(train_df, n_top_features)

@task(name="Preprocess eval")
def preprocess_eval_task(df_eval: pd.DataFrame, dv, features, scaler):
    return preprocessing_eval(df_eval, dv, features, scaler)

# -------------------------
# Tasks por modelo
# -------------------------
def train_model_with_optuna(model_class, model_name, X_train, y_train, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials=10):
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    with mlflow.start_run(run_name=f"{model_name} - Parent", nested=False) as parent_run:
        mlflow.set_tag("model_family", model_name)
        mlflow.log_param("optuna_n_trials", n_trials)
        parent_run_id = parent_run.info.run_id

        def objective(trial):
            if model_name == "logistic_regression":
                penalty = trial.suggest_categorical("penalty", ["l1","l2","elasticnet"])
                params = {
                    "penalty": penalty,
                    "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                    "class_weight": trial.suggest_categorical("class_weight", [None,"balanced"]),
                    "solver":"saga",
                    "multi_class":"multinomial",
                    "max_iter":2000,
                    "random_state":42,
                    "n_jobs":-1
                }
                if penalty=="elasticnet":
                    params["l1_ratio"]=trial.suggest_float("l1_ratio",0,1)
                model = LogisticRegression(**{k:v for k,v in params.items() if v is not None})

            elif model_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators",50,1000),
                    "max_depth": trial.suggest_int("max_depth",3,30),
                    "min_samples_split": trial.suggest_int("min_samples_split",2,20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf",1,20),
                    "max_features": trial.suggest_categorical("max_features",["sqrt","log2",None]),
                    "class_weight":None,
                    "random_state":42,
                    "n_jobs":-1
                }
                model = RandomForestClassifier(**params)

            elif model_name=="xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth",3,16),
                    "n_estimators": trial.suggest_int("n_estimators",50,800),
                    "learning_rate": trial.suggest_float("learning_rate",1e-3,0.5,log=True),
                    "gamma": trial.suggest_float("gamma",0.0,5.0),
                    "min_child_weight": trial.suggest_float("min_child_weight",1e-1,50.0,log=True),
                    "subsample": trial.suggest_float("subsample",0.5,1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree",0.4,1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha",1e-8,10.0,log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda",1e-8,10.0,log=True),
                    "use_label_encoder":False,
                    "verbosity":0,
                    "random_state":16
                }
                model = xgb.XGBClassifier(**params)

            with mlflow.start_run(run_name=f"Trial-{trial.number}", nested=True):
                mlflow.log_params({k:(v if v is not None else "None") for k,v in params.items()})
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_pred, average="macro")
                mlflow.log_metric("f1", val_f1)
                return val_f1

        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_val = study.best_value
        mlflow.log_metric("best_f1", best_val)
        mlflow.log_params({f"best_{k}":v for k,v in best_params.items()})

        # Modelo final
        with mlflow.start_run(run_name="Final", nested=True) as final_run:
            if model_name=="logistic_regression":
                model = LogisticRegression(**{k:v for k,v in best_params.items() if v is not None},
                                           n_jobs=-1, random_state=42)
            elif model_name=="random_forest":
                model = RandomForestClassifier(**{k:v for k,v in best_params.items() if v is not None},
                                               n_jobs=-1, random_state=42)
            elif model_name=="xgboost":
                model = xgb.XGBClassifier(**{k:v for k,v in best_params.items() if v is not None},
                                          n_jobs=-1, random_state=16,use_label_encoder=False,verbosity=0)

            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)
            test_f1 = f1_score(y_test, y_test_pred, average="macro")
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
            mlflow.log_metric("test_log_loss", log_loss(y_test, y_test_proba))

            # Guardar preprocessor
            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/dv.b","wb") as f: pickle.dump(dv,f)
            with open("preprocessor/scaler.b","wb") as f: pickle.dump(scaler,f)
            with open("preprocessor/features.pkl","wb") as f: pickle.dump(features,f)
            mlflow.log_artifact("preprocessor/dv.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/scaler.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/features.pkl", artifact_path="preprocessor")

            input_example = pd.DataFrame(X_test[:5], columns=features)
            signature = infer_signature(input_example, y_test[:5])
            if model_name=="xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model", input_example=input_example, signature=signature)
            else:
                mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

            final_run_id = final_run.info.run_id

    return {"parent_run_id": parent_run_id, "final_run_id": final_run_id, "best_params": best_params, "best_f1": best_val, "test_f1": test_f1}

# -------------------------
# Prefect tasks wrappers
# -------------------------
@task
def train_lr_task(*args, **kwargs):
    return train_model_with_optuna(LogisticRegression, "logistic_regression", *args, **kwargs)

@task
def train_rf_task(*args, **kwargs):
    return train_model_with_optuna(RandomForestClassifier, "random_forest", *args, **kwargs)

@task
def train_xgb_task(*args, **kwargs):
    return train_model_with_optuna(xgb.XGBClassifier, "xgboost", *args, **kwargs)

# -------------------------
# Register Champion / Challenger
# -------------------------
@task
def register_models_task(experiment_name: str, model_registry_name: str):
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.best_f1 DESC"], output_format="list")
    if not runs: return
    champion_run = runs[0]
    client.set_registered_model_alias(
        name=model_registry_name,
        alias="Champion",
        version=mlflow.register_model(f"runs:/{champion_run.info.run_id}/model", model_registry_name).version
    )
    if len(runs) > 1:
        challenger_run = runs[1]
        client.set_registered_model_alias(
            name=model_registry_name,
            alias="Challenger",
            version=mlflow.register_model(f"runs:/{challenger_run.info.run_id}/model", model_registry_name).version
        )

# -------------------------
# Flow principal
# -------------------------
@flow(name="Stress Level - Training pipeline V2")
def main_flow_v2(csv_path: str = "data/raw/synthetic_coffee_health_10000.csv",
                 n_top_features: int = 20,
                 n_trials: int = 10):
    mlflow.set_tracking_uri("databricks")
    EXPERIMENT_NAME = "/Users/monica.ibarra@iteso.mx/project1-experiment"
    MODEL_REGISTRY_NAME = "workspace.default.equipo1-proyecto"
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = read_csv_task(csv_path)
    train_df, val_df, test_df = split_task(df)

    X_train_bal, y_train_bal, dv, features, dropped, scaler = preprocess_train_task(train_df, n_top_features)
    X_val, y_val = preprocess_eval_task(val_df, dv, features, scaler)
    X_test, y_test = preprocess_eval_task(test_df, dv, features, scaler)

    res_lr = train_lr_task(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials)
    res_rf = train_rf_task(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials, wait_for=[res_lr])
    res_xgb = train_xgb_task(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials, wait_for=[res_lr, res_rf])

    register_models_task(EXPERIMENT_NAME, MODEL_REGISTRY_NAME, wait_for=[res_lr, res_rf, res_xgb])

    return {"lr": res_lr, "rf": res_rf, "xgb": res_xgb, "dropped": dropped}

if __name__=="__main__":
    CSV_PATH = os.getenv("DATA_CSV_PATH","data/raw/synthetic_coffee_health_10000.csv")
    main_flow_v2(CSV_PATH)