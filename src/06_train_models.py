
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

GOLD_DIR = PROJECT_DIR / "data" / "gold"
REPORTS_DIR = PROJECT_DIR / "reports"
MLFLOW_DB = PROJECT_DIR / "mlflow.db"

FEATURES_PATH = GOLD_DIR / "daily_sales_features.parquet"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Fichier Gold introuvable : {FEATURES_PATH}"
    )

# Utiliser la même base SQLite que l'interface MLflow.
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
mlflow.set_experiment("sales-forecasting-m5")


# ==================================================
# 2. CHARGEMENT DES DONNÉES GOLD
# ==================================================

print("Chargement de la couche Gold...")

con = duckdb.connect()

try:
    df = con.execute(
        "SELECT * FROM read_parquet(?)",
        [str(FEATURES_PATH)]
    ).fetchdf()
finally:
    con.close()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "store_id", "item_id"]
).reset_index(drop=True)

print("Dimensions :", df.shape)
print("Première date :", df["date"].min())
print("Dernière date :", df["date"].max())


# ==================================================
# 3. DÉFINITION DE LA CIBLE ET DES VARIABLES
# ==================================================

TARGET = "sales"

CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

NUMERIC_FEATURES = [
    "sell_price",
    "wm_yr_wk",
    "month",
    "year",
    "day_of_month",
    "quarter",
    "day_of_week",
    "wday",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sales_lag_7",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES
)

required_columns = FEATURE_COLUMNS + [TARGET, "date"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colonnes manquantes : {missing_columns}"
    )

# Écarter les lignes sans historique de 28 jours.
df = df.dropna(
    subset=["sales_lag_28"]
).copy()

if df.empty:
    raise ValueError(
        "Aucune ligne disponible après le filtrage."
    )


# ==================================================
# 4. SÉPARATION CHRONOLOGIQUE
# ==================================================

unique_dates = np.sort(df["date"].unique())

if len(unique_dates) <= 28:
    raise ValueError(
        "Il faut plus de 28 dates pour entraîner et tester."
    )

# Les 28 dernières dates sont réservées au test.
test_dates = unique_dates[-28:]
test_start = pd.Timestamp(test_dates[0])

train_df = df[df["date"] < test_start].copy()
test_df = df[df["date"].isin(test_dates)].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "L'ensemble d'entraînement ou de test est vide."
    )

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df[TARGET].astype(float)

X_test = test_df[FEATURE_COLUMNS].copy()
y_test = test_df[TARGET].astype(float)

print("\nDébut de la période de test :", test_start.date())
print("Lignes d'entraînement :", len(train_df))
print("Lignes de test :", len(test_df))
print("Nombre de dates de test :", test_df["date"].nunique())


# ==================================================
# 5. FONCTIONS D'ÉVALUATION
# ==================================================

def evaluate_model(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Les quantités vendues ne peuvent pas être négatives.
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    total_sales = np.sum(np.abs(y_true))

    wape = (
        np.sum(np.abs(y_true - y_pred))
        / total_sales * 100
        if total_sales > 0
        else np.nan
    )

    return {
        "model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE_percent": float(wape),
    }


# ==================================================
# 6. MODÈLE DE RÉFÉRENCE : LAG 7
# ==================================================

print("\nÉvaluation de la baseline...")

baseline_predictions = (
    test_df["sales_lag_7"].fillna(0).to_numpy()
)

baseline_metrics = evaluate_model(
    "Baseline_lag_7",
    y_test,
    baseline_predictions
)


# ==================================================
# 7. PRÉTRAITEMENT
# ==================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore")
    ),
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        NUMERIC_FEATURES
    ),
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    ),
])


# ==================================================
# 8. MODÈLE XGBOOST
# ==================================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])


# ==================================================
# 9. ENTRAÎNEMENT ET SUIVI MLFLOW
# ==================================================

print("\nEntraînement de XGBoost...")

with mlflow.start_run(
    run_name="xgboost_first_experiment"
) as run:

    mlflow.log_params({
        "model": "XGBRegressor",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_days": 28,
        "training_start": str(
            train_df["date"].min().date()
        ),
        "training_end": str(
            train_df["date"].max().date()
        ),
        "test_start": str(
            test_df["date"].min().date()
        ),
        "test_end": str(
            test_df["date"].max().date()
        ),
    })

    pipeline.fit(X_train, y_train)

    predictions = np.clip(
        pipeline.predict(X_test),
        0,
        None
    )

    xgb_metrics = evaluate_model(
        "XGBoost",
        y_test,
        predictions
    )

    # Enregistrer les métriques XGBoost.
    mlflow.log_metrics({
        "MAE": xgb_metrics["MAE"],
        "RMSE": xgb_metrics["RMSE"],
        "WAPE_percent": xgb_metrics["WAPE_percent"],
        "Baseline_MAE": baseline_metrics["MAE"],
        "Baseline_RMSE": baseline_metrics["RMSE"],
        "Baseline_WAPE_percent": baseline_metrics["WAPE_percent"],
    })

    # Enregistrer le modèle et ses artefacts.
    mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="xgboost-pipeline",
    skops_trusted_types=[
        "numpy.dtype",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBRegressor",
        ],
    )

    # Comparaison des modèles.
    results_df = pd.DataFrame([
        baseline_metrics,
        xgb_metrics,
    ]).sort_values("MAE").reset_index(drop=True)

    results_path = (
        REPORTS_DIR / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # Sauvegarder les prédictions et les valeurs réelles.
    predictions_df = test_df[
        ["date", "item_id", "store_id", "sales"]
    ].copy()

    predictions_df["prediction_baseline"] = (
        baseline_predictions
    )

    predictions_df["prediction_xgboost"] = predictions

    predictions_path = (
        REPORTS_DIR / "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # Joindre les rapports à cette expérience MLflow.
    mlflow.log_artifact(str(results_path))
    mlflow.log_artifact(str(predictions_path))

    print("\nRun MLflow :", run.info.run_id)


# ==================================================
# 10. AFFICHAGE FINAL
# ==================================================

print("\n========== COMPARAISON DES MODÈLES ==========")
print(results_df.round(4).to_string(index=False))

print("\nFichiers générés :")
print(f"- {results_path}")
print(f"- {predictions_path}")

print("\nEntraînement terminé avec succès.")
