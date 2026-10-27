
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"

PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {PREDICTIONS_PATH}\n"
        "Exécute d'abord src/06_train_models.py."
    )

df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

required = [
    "date",
    "item_id",
    "store_id",
    "sales",
    "prediction_baseline",
    "prediction_xgboost",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")

# ==========================================
# 2. METRIQUES GLOBALES
# ==========================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(
        np.asarray(predicted, dtype=float), 0, None
    )

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    total = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / total * 100
        if total > 0 else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }


results = []

for model_name, prediction_col in [
    ("Baseline_lag_7", "prediction_baseline"),
    ("XGBoost", "prediction_xgboost"),
]:
    metrics = calculate_metrics(df["sales"], df[prediction_col])
    metrics["model"] = model_name
    results.append(metrics)

comparison = pd.DataFrame(results)[
    ["model", "MAE", "RMSE", "WAPE_percent"]
]

comparison.to_csv(
    REPORTS_DIR / "evaluation_metrics.csv", index=False
)

print("\n===== COMPARAISON GLOBALE =====")
print(comparison.round(3).to_string(index=False))

# ==========================================
# 3. ANALYSE PAR PRODUIT
# ==========================================

df["error_baseline"] = (
    df["sales"] - df["prediction_baseline"]
).abs()

df["error_xgboost"] = (
    df["sales"] - df["prediction_xgboost"]
).abs()

product_metrics = df.groupby(
    ["item_id", "store_id"], as_index=False
).agg(
    total_sales=("sales", "sum"),
    MAE_baseline=("error_baseline", "mean"),
    MAE_xgboost=("error_xgboost", "mean"),
)

product_metrics["MAE_improvement"] = (
    product_metrics["MAE_baseline"]
    - product_metrics["MAE_xgboost"]
)

product_metrics = product_metrics.sort_values(
    "MAE_improvement", ascending=False
)

product_metrics.to_csv(
    REPORTS_DIR / "evaluation_by_product.csv", index=False
)

print("\n===== 10 PRODUITS AVEC LA PLUS GRANDE AMELIORATION =====")
print(product_metrics.head(10).round(3).to_string(index=False))

# ==========================================
# 4. AGREGATION JOURNALIERE
# ==========================================

daily = df.groupby("date", as_index=False).agg(
    actual_sales=("sales", "sum"),
    baseline_sales=("prediction_baseline", "sum"),
    xgboost_sales=("prediction_xgboost", "sum"),
)

daily.to_csv(
    REPORTS_DIR / "daily_forecast_comparison.csv", index=False
)

# ==========================================
# 5. GRAPHIQUE DES VENTES JOURNALIERES
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily["date"], daily["actual_sales"],
    label="Ventes réelles"
)
plt.plot(
    daily["date"], daily["baseline_sales"],
    label="Baseline lag 7", alpha=0.8
)
plt.plot(
    daily["date"], daily["xgboost_sales"],
    label="XGBoost", alpha=0.8
)

plt.title("Ventes réelles vs prévisions - période de test")
plt.xlabel("Date")
plt.ylabel("Quantité vendue")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "actual_vs_predicted.png",
    dpi=150
)
plt.close()

# ==========================================
# 6. GRAPHIQUE DES ERREURS PAR MODELE
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["model"],
    comparison["MAE"]
)

plt.title("Comparaison du MAE")
plt.ylabel("MAE - plus bas est meilleur")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "model_mae_comparison.png",
    dpi=150
)
plt.close()

print("\nEvaluation terminee.")
print("Fichiers générés dans :", REPORTS_DIR)
