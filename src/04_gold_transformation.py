
from pathlib import Path
import duckdb

# --------------------------------------------------
# 1. Chemins
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = PROJECT_DIR / "data" / "silver"
GOLD_DIR = PROJECT_DIR / "data" / "gold"

GOLD_DIR.mkdir(parents=True, exist_ok=True)

silver_path = (
    SILVER_DIR / "sales_silver.parquet"
).resolve()

features_path = (
    GOLD_DIR / "daily_sales_features.parquet"
).resolve()

kpi_path = (
    GOLD_DIR / "sales_kpi_daily.parquet"
).resolve()

if not silver_path.exists():
    raise FileNotFoundError(
        f"Couche Silver introuvable : {silver_path}"
    )

def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")

silver_sql = sql_path(silver_path)
features_sql = sql_path(features_path)
kpi_sql = sql_path(kpi_path)

con = duckdb.connect()
con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 4")

# --------------------------------------------------
# 2. Gold : données détaillées pour le ML
# --------------------------------------------------
features_query = f"""
COPY (
    WITH base AS (
        SELECT *
        FROM read_parquet('{silver_sql}')
    ),

    features AS (
        SELECT
            *,

            LAG(sales, 7) OVER product_window
                AS sales_lag_7,

            LAG(sales, 28) OVER product_window
                AS sales_lag_28,

            AVG(sales) OVER (
                PARTITION BY item_id, store_id
                ORDER BY date
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
            ) AS sales_rolling_mean_7,

            AVG(sales) OVER (
                PARTITION BY item_id, store_id
                ORDER BY date
                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
            ) AS sales_rolling_mean_28

        FROM base

        WINDOW product_window AS (
            PARTITION BY item_id, store_id
            ORDER BY date
        )
    )

    SELECT
        *,
        EXTRACT(DAY FROM date) AS day_of_month,
        EXTRACT(QUARTER FROM date) AS quarter,
        EXTRACT(DOW FROM date) AS day_of_week
    FROM features
)
TO '{features_sql}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

# --------------------------------------------------
# 3. Gold : indicateurs quotidiens par magasin
# --------------------------------------------------
kpi_query = f"""
COPY (
    SELECT
        store_id,
        date,
        SUM(sales) AS total_units_sold,
        COUNT(DISTINCT item_id) AS number_of_products,
        AVG(sell_price) AS average_available_price,
        SUM(
            CASE WHEN sales > 0 THEN 1 ELSE 0 END
        ) AS product_store_pairs_with_sales

    FROM read_parquet('{silver_sql}')
    GROUP BY store_id, date
)
TO '{kpi_sql}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

try:
    print("Création des variables ML...")
    con.execute(features_query)
    print(f"Table créée : {features_path}")

    print("\nCréation des KPI quotidiens...")
    con.execute(kpi_query)
    print(f"Table créée : {kpi_path}")

    print("\nVérification des volumes :")
    for path in [features_path, kpi_path]:
        path_sql_value = sql_path(path)
        result = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path_sql_value}')"
        ).fetchone()[0]
        print(f"{path.name} : {result:,} lignes")

finally:
    con.close()
