
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SILVER_PATH = PROJECT_DIR / "data" / "silver" / "sales_silver.parquet"
GOLD_PATH = PROJECT_DIR / "data" / "gold" / "daily_sales_features.parquet"
CALENDAR_PATH = PROJECT_DIR / "data" / "bronze" / "calendar.parquet"
PRICES_PATH = PROJECT_DIR / "data" / "bronze" / "sell_prices.parquet"

REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_DAYS = 28

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

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ==================================================
# 2. CHARGEMENT DES DONNEES
# ==================================================

print("Chargement des données...")

con = duckdb.connect()

try:
    silver = con.execute(
        "SELECT * FROM read_parquet(?)",
        [str(SILVER_PATH)],
    ).fetchdf()

    gold = con.execute(
        "SELECT * FROM read_parquet(?)",
        [str(GOLD_PATH)],
    ).fetchdf()

    calendar = con.execute(
        """
        SELECT *
        FROM read_parquet(?)
        WHERE CAST(date AS DATE) > DATE '2016-04-24'
          AND CAST(date AS DATE) <= DATE '2016-04-24' + INTERVAL 28 DAYS
        ORDER BY date
        """,
        [str(CALENDAR_PATH)],
    ).fetchdf()

    prices = con.execute(
        "SELECT * FROM read_parquet(?)",
        [str(PRICES_PATH)],
    ).fetchdf()

finally:
    con.close()

silver["date"] = pd.to_datetime(silver["date"])
gold["date"] = pd.to_datetime(gold["date"])
calendar["date"] = pd.to_datetime(calendar["date"])

if silver.empty or gold.empty or calendar.empty:
    raise ValueError("Silver, Gold ou calendrier ne contient pas de données.")

last_history_date = silver["date"].max()
future_dates = sorted(calendar["date"].unique())

if len(future_dates) != FORECAST_DAYS:
    raise ValueError(
        f"28 dates futures attendues, {len(future_dates)} trouvées."
    )

print("Dernière date historique :", last_history_date.date())
print("Première date à prévoir :", pd.Timestamp(future_dates[0]).date())
print("Dernière date à prévoir :", pd.Timestamp(future_dates[-1]).date())


# ==================================================
# 3. ENTRAINEMENT SUR TOUT L'HISTORIQUE
# ==================================================

print("\nPréparation du modèle...")

gold = gold.dropna(subset=["sales_lag_28"]).copy()

missing_features = [
    col for col in FEATURES + ["sales", "date"]
    if col not in gold.columns
]

if missing_features:
    raise ValueError(
        f"Colonnes absentes de Gold : {missing_features}"
    )

train_df = gold[gold["date"] <= last_history_date].copy()

X_train = train_df[FEATURES]
y_train = train_df["sales"].astype(float)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])

model = XGBRegressor(
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
    ("model", model),
])

pipeline.fit(X_train, y_train)

print("Modèle entraîné sur", len(train_df), "lignes.")


# ==================================================
# 4. PREPARATION DE L'HISTORIQUE PAR PRODUIT
# ==================================================

products = (
    silver.sort_values("date")
    .groupby(["item_id", "store_id"], as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

metadata_columns = [
    "item_id", "store_id", "state_id", "dept_id", "cat_id"
]

products = products[metadata_columns].drop_duplicates(
    ["item_id", "store_id"]
)

history = {}

for (item_id, store_id), group in silver.groupby(
    ["item_id", "store_id"]
):
    group = group.sort_values("date")
    history[(item_id, store_id)] = group["sales"].astype(float).tolist()


# ==================================================
# 5. PREVISION RECURSIVE SUR 28 JOURS
# ==================================================

print("\nGénération des prévisions...")

forecast_rows = []

for current_date in future_dates:
    current_date = pd.Timestamp(current_date)

    day_calendar = calendar[
        calendar["date"] == current_date
    ].copy()

    if day_calendar.empty:
        raise ValueError(f"Date absente du calendrier : {current_date}")

    # Chaque produit reçoit les variables du jour courant.
    day_data = products.merge(day_calendar, how="cross")

    day_data = day_data.merge(
        prices,
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    if day_data["sell_price"].isna().any():
        missing = int(day_data["sell_price"].isna().sum())
        raise ValueError(
            f"{missing} prix manquants pour le {current_date.date()}."
        )

    day_data["date"] = current_date
    day_data["day_of_month"] = current_date.day
    day_data["quarter"] = current_date.quarter
    day_data["day_of_week"] = current_date.dayofweek

    # Les historiques ne contiennent que les ventes passées
    # et les prédictions déjà produites.
    lag7 = []
    lag28 = []
    rolling7 = []
    rolling28 = []

    for _, product in day_data.iterrows():
        key = (product["item_id"], product["store_id"])
        values = history[key]

        if len(values) < 28:
            raise ValueError(f"Historique insuffisant pour {key}")

        lag7.append(values[-7])
        lag28.append(values[-28])
        rolling7.append(float(np.mean(values[-7:])))
        rolling28.append(float(np.mean(values[-28:])))

    day_data["sales_lag_7"] = lag7
    day_data["sales_lag_28"] = lag28
    day_data["sales_rolling_mean_7"] = rolling7
    day_data["sales_rolling_mean_28"] = rolling28

    # Même ordre de variables que pendant l'entraînement.
    predictions = np.clip(
        pipeline.predict(day_data[FEATURES]),
        0,
        None,
    )

    day_data["predicted_sales"] = predictions

    for _, row in day_data.iterrows():
        key = (row["item_id"], row["store_id"])

        forecast_rows.append({
            "date": current_date,
            "item_id": row["item_id"],
            "store_id": row["store_id"],
            "predicted_sales": float(row["predicted_sales"]),
            "sell_price": float(row["sell_price"]),
        })

        # Les prédictions deviennent l'historique du jour suivant.
        history[key].append(float(row["predicted_sales"]))


# ==================================================
# 6. EXPORT DES RESULTATS
# ==================================================

forecast_df = pd.DataFrame(forecast_rows)

forecast_path = REPORTS_DIR / "forecast_28_days.csv"

forecast_df.to_csv(forecast_path, index=False)

daily_forecast = (
    forecast_df.groupby("date", as_index=False)
    .agg(
        predicted_sales=("predicted_sales", "sum"),
        products_forecasted=("item_id", "nunique"),
    )
)

daily_path = REPORTS_DIR / "forecast_daily_summary.csv"
daily_forecast.to_csv(daily_path, index=False)

print("\n===== APERCU DES PREVISIONS =====")
print(daily_forecast.head(10).round(2).to_string(index=False))

print("\nNombre de lignes prévues :", len(forecast_df))
print("Total prédit sur 28 jours :", round(
    forecast_df["predicted_sales"].sum(), 2
))

print("\nFichiers générés :")
print(forecast_path)
print(daily_path)
print("\nPrévision terminée.")
