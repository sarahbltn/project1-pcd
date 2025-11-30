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

import matplotlib
matplotlib.use("Agg")

# Cargar credenciales
load_dotenv(override=True)

EXPERIMENT_NAME = "/Users/sarahbeltrang@gmail.com/project1-experiment"
MODEL_REGISTRY_NAME = "workspace.default.equipo1-proyecto"

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.sklearn.autolog(log_models=False)
mlflow.xgboost.autolog(log_models=False)

# Preprocesamiento
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


    df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    if "Gender" in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df = df.drop(columns=["Country","ID"], errors='ignore')

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

# Prefect Tasks
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

# Tasks por modelo
@task(name="Tune RF (Optuna) - single-process")
def tune_rf_task(X_train, y_train, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials: int = 10):
    def objective_rf(trial: optuna.trial.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1
        }

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "random_forest")
            mlflow.log_params(params)

            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)

            y_proba = clf.predict_proba(X_val)
            y_pred = clf.predict(X_val)

            val_logloss = log_loss(y_val, y_proba)
            val_acc = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, average="macro")

            mlflow.log_metric("log_loss", float(val_logloss))
            mlflow.log_metric("accuracy", float(val_acc))
            mlflow.log_metric("f1_macro", float(val_f1))

            signature = infer_signature(X_val, y_pred[:5])
            mlflow.sklearn.log_model(clf, "model", input_example=X_val[:5], signature=signature)

        return float(val_f1)

    # Flujo de búsqueda
    study_rf = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    with mlflow.start_run(run_name="RandomForest Hyperparameter Optimization (Optuna)", nested=False):
        study_rf.optimize(objective_rf, n_trials=10)
        best_rf = study_rf.best_params
        mlflow.log_params(best_rf)
        mlflow.set_tags({"project": "Stress Level Predicition",
                            "optimizer_engine": "optuna",
                            "model_family": "random_forest",
                            "feature_set_version": 1,
                            })
        mlflow.sklearn.autolog(log_models=False)

        # Run Final
        with mlflow.start_run(run_name="Final", nested=True):
            mlflow.log_params({f"final_{k}": v for k, v in best_rf.items()})
            final_model = RandomForestClassifier(**best_rf, n_jobs=-1, random_state=42)
            final_model.fit(X_train, y_train)

            # Evaluar en test
            y_test_pred = final_model.predict(X_test)
            y_test_proba = final_model.predict_proba(X_test)
            test_f1 = f1_score(y_test, y_test_pred, average="macro")
            mlflow.log_metric("test_f1", float(test_f1))
            mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, y_test_pred)))
            mlflow.log_metric("test_log_loss", float(log_loss(y_test, y_test_proba)))

            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/dv.b","wb") as f: pickle.dump(dv,f)
            with open("preprocessor/scaler.b","wb") as f: pickle.dump(scaler,f)
            with open("preprocessor/features.pkl","wb") as f: pickle.dump(features,f)
            mlflow.log_artifact("preprocessor/dv.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/scaler.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/features.pkl", artifact_path="preprocessor")

            input_example = pd.DataFrame(X_test[:5], columns=features)
            signature = infer_signature(input_example, y_test[:5])
            mlflow.sklearn.log_model(final_model, artifact_path="model", input_example=input_example, signature=signature)

    return {"model": "random_forest", "best_params": best_rf, "test_f1": float(test_f1)}

@task(name="Tune XGB (Optuna) - single-process")
def tune_xgb_task(X_train, y_train, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials: int = 10):

    def objective_xgb(trial: optuna.trial.Trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "seed":42
        }
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "xgboost")
            mlflow.log_params(params)

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_proba = clf.predict_proba(X_val)
            y_pred = clf.predict(X_val)

            val_logloss = log_loss(y_val, y_proba)
            val_acc = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, average="macro")

            mlflow.log_metric("log_loss", float(val_logloss))
            mlflow.log_metric("f1_macro", float(val_f1))
            mlflow.log_metric("accuracy", float(val_acc))

            signature = infer_signature(X_val, y_val[:5])
            mlflow.xgboost.log_model(clf, "model", input_example=X_val[:5], signature=signature)
        return float(val_f1)

    study_xgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    with mlflow.start_run(run_name="XGBoost Optimization (Optuna)", nested=False):
        study_xgb.optimize(objective_xgb, n_trials=10)
        best_xgb = study_xgb.best_params
        mlflow.log_params(best_xgb)

        mlflow.set_tags({
                    "project": "Stress Level Predicition",
                    "optimizer_engine": "optuna",
                    "model_family": "random_forest",
                    "feature_set_version": 1,
                })
        mlflow.sklearn.autolog(log_models=False)

        with mlflow.start_run(run_name="Final", nested=True):
            mlflow.log_params({f"final_{k}": v for k, v in best_xgb.items()})
            final_model = xgb.XGBClassifier(**best_xgb, use_label_encoder=False, verbosity=0)
            final_model.fit(X_train, y_train)

            y_test_pred = final_model.predict(X_test)
            y_test_proba = final_model.predict_proba(X_test)
            test_f1 = f1_score(y_test, y_test_pred, average="macro")
            mlflow.log_metric("test_f1", float(test_f1))
            mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, y_test_pred)))
            mlflow.log_metric("test_log_loss", float(log_loss(y_test, y_test_proba)))

            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/dv.b","wb") as f: pickle.dump(dv,f)
            with open("preprocessor/scaler.b","wb") as f: pickle.dump(scaler,f)
            with open("preprocessor/features.pkl","wb") as f: pickle.dump(features,f)
            mlflow.log_artifact("preprocessor/dv.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/scaler.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/features.pkl", artifact_path="preprocessor")

            input_example = pd.DataFrame(X_test[:5], columns=features)
            signature = infer_signature(input_example, y_test[:5])
            mlflow.xgboost.log_model(final_model, artifact_path="model", input_example=input_example, signature=signature)

    return {"model": "xgboost", "best_params": best_xgb, "test_f1": float(test_f1)}

@task(name="Tune LR (Optuna) - single-process")
def tune_lr_task(X_train, y_train, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials: int = 10):
    def objective_lr(trial: optuna.trial.Trial):
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
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "logistic_regression")
            mlflow.log_params(params)

            lr = LogisticRegression(**{k:v for k,v in params.items() if v is not None})
            lr.fit(X_train, y_train)
            y_proba = lr.predict_proba(X_val)
            y_pred = lr.predict(X_val)

            val_logloss = log_loss(y_val, y_proba)
            val_acc = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, average="macro")

            mlflow.log_metric("log_loss", float(val_logloss))
            mlflow.log_metric("f1_macro", float(val_f1))
            mlflow.log_metric("accuracy", float(val_acc))

            signature = infer_signature(X_val, y_val[:5])
            mlflow.sklearn.log_model(lr, "model", input_example=X_val[:5], signature=signature)
        return float(val_f1)

    mlflow.sklearn.autolog(log_models=False)
    study_lr = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    with mlflow.start_run(run_name="Logistic Regression Optimization (Optuna)", nested=False):
        study_lr.optimize(objective_lr, n_trials=10)
        best_lr = study_lr.best_params
        mlflow.log_params(best_lr)
        mlflow.set_tags({
            "project": "Stress Level Prediction",
            "optimizer_engine": "optuna",
            "model_family": "logistic_regression",
            "feature_set_version": 1,
            })
        mlflow.sklearn.autolog(log_models=False)

        with mlflow.start_run(run_name="Final", nested=True):
            mlflow.log_params({f"final_{k}": v for k, v in best_lr.items()})
            final_model = LogisticRegression(**{k:v for k,v in best_lr.items() if v is not None}, n_jobs=-1, random_state=42)
            final_model.fit(X_train, y_train)

            y_test_pred = final_model.predict(X_test)
            y_test_proba = final_model.predict_proba(X_test)
            test_f1 = f1_score(y_test, y_test_pred, average="macro")
            mlflow.log_metric("test_f1", float(test_f1))
            mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, y_test_pred)))
            mlflow.log_metric("test_log_loss", float(log_loss(y_test, y_test_proba)))

            pathlib.Path("preprocessor").mkdir(exist_ok=True)
            with open("preprocessor/dv.b","wb") as f: pickle.dump(dv,f)
            with open("preprocessor/scaler.b","wb") as f: pickle.dump(scaler,f)
            with open("preprocessor/features.pkl","wb") as f: pickle.dump(features,f)
            mlflow.log_artifact("preprocessor/dv.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/scaler.b", artifact_path="preprocessor")
            mlflow.log_artifact("preprocessor/features.pkl", artifact_path="preprocessor")

            input_example = pd.DataFrame(X_test[:5], columns=features)
            signature = infer_signature(input_example, y_test[:5])
            mlflow.sklearn.log_model(final_model, artifact_path="model", input_example=input_example, signature=signature)

    return {"model": "logistic_regression", "best_params": best_lr, "test_f1": float(test_f1)}

# Registrar Champion / Challenger
@task(name="Register champion/challenger")
def register_models_task(experiment_name: str, model_registry_name: str = MODEL_REGISTRY_NAME):
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.f1_macro DESC"], output_format="list")

    champion_run = runs[0]
    champion_run_id = champion_run.info.run_id
    print("Champion run:", champion_run_id)

    champ = mlflow.register_model(model_uri=f"runs:/{champion_run_id}/model", name=model_registry_name)
    client.set_registered_model_alias(name=model_registry_name, alias="Champion", version=champ.version)

    if len(runs) > 1:
        challenger_run = runs[1]
        challenger_run_id = challenger_run.info.run_id
        chal = mlflow.register_model(model_uri=f"runs:/{challenger_run_id}/model", name=model_registry_name)
        client.set_registered_model_alias(name=model_registry_name, alias="Challenger", version=chal.version)

# Main Flow
@flow(name="Stress Level - Training pipeline V2")
def main_flow_v2(csv_path: str = "data/raw/synthetic_coffee_health_10000.csv",
                 n_top_features: int = 20,
                 n_trials: int = 10):
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1) Cargar
    df = read_csv_task(csv_path)

    # 2) Split
    train_df, val_df, test_df = split_task(df)

    # 3) Preprocesamiento
    X_train_bal, y_train_bal, dv, features, dropped, scaler = preprocess_train_task(train_df, n_top_features)
    X_val, y_val = preprocess_eval_task(val_df, dv, features, scaler)
    X_test, y_test = preprocess_eval_task(test_df, dv, features, scaler)

    # 4) Tunear cada modelo
    rf_res = tune_rf_task(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials)
    xgb_res = tune_xgb_task(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials, wait_for=[rf_res])
    lr_res = tune_lr_task(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test, dv, features, scaler, n_trials, wait_for=[rf_res, xgb_res])

    # 5) Registrar champion / challenger
    register_models_task(EXPERIMENT_NAME, MODEL_REGISTRY_NAME, wait_for=[rf_res, xgb_res, lr_res])

    return {"rf": rf_res, "xgb": xgb_res, "lr": lr_res, "dropped": dropped}

if __name__ == "__main__":
    CSV_PATH = os.getenv("DATA_CSV_PATH","data/raw/synthetic_coffee_health_10000.csv")
    out = main_flow_v2(CSV_PATH, n_top_features=20, n_trials=10)
    print("Pipeline finished. Results:", out)