
from pathlib import Path
import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_DIR / "data" / "gold"

features_path = (
    GOLD_DIR / "daily_sales_features.parquet"
).resolve()

kpi_path = (
    GOLD_DIR / "sales_kpi_daily.parquet"
).resolve()

con = duckdb.connect()

def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")

for path in [features_path, kpi_path]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier absent : {path}")

features = sql_path(features_path)
kpis = sql_path(kpi_path)

print("=== Variables ML ===")
print(con.execute(f"""
    SELECT
        COUNT(*) AS lignes,
        COUNT(DISTINCT item_id) AS produits,
        COUNT(DISTINCT store_id) AS magasins,
        MIN(date) AS date_debut,
        MAX(date) AS date_fin,
        SUM(CASE WHEN sales_lag_7 IS NULL THEN 1 ELSE 0 END)
            AS lag_7_manquant,
        SUM(CASE WHEN sales_rolling_mean_28 IS NULL THEN 1 ELSE 0 END)
            AS moyenne_28_manquante
    FROM read_parquet('{features}')
""").fetchdf().to_string(index=False))

print("\n=== KPI quotidiens ===")
print(con.execute(f"""
    SELECT *
    FROM read_parquet('{kpis}')
    ORDER BY date
    LIMIT 10
""").fetchdf().to_string(index=False))

print("\n=== Aperçu des variables de prévision ===")
print(con.execute(f"""
    SELECT
        item_id, store_id, date, sales,
        sales_lag_7, sales_lag_28,
        sales_rolling_mean_7,
        sales_rolling_mean_28
    FROM read_parquet('{features}')
    WHERE sales_lag_28 IS NOT NULL
    ORDER BY date, item_id
    LIMIT 10
""").fetchdf().to_string(index=False))

con.close()
