
from pathlib import Path
import duckdb

# --------------------------------------------------
# 1. Configuration des chemins
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = PROJECT_DIR / "data" / "bronze"
SILVER_DIR = PROJECT_DIR / "data" / "silver"

SILVER_DIR.mkdir(parents=True, exist_ok=True)

sales_path = (
    BRONZE_DIR / "sales_train_validation.parquet"
).resolve()

calendar_path = (
    BRONZE_DIR / "calendar.parquet"
).resolve()

prices_path = (
    BRONZE_DIR / "sell_prices.parquet"
).resolve()

output_path = (
    SILVER_DIR / "sales_silver.parquet"
).resolve()

for path in [sales_path, calendar_path, prices_path]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

# Échapper les apostrophes dans les chemins SQL
def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")

sales_sql = sql_path(sales_path)
calendar_sql = sql_path(calendar_path)
prices_sql = sql_path(prices_path)
output_sql = sql_path(output_path)

con = duckdb.connect()

# Limiter la consommation mémoire
con.execute("SET memory_limit = '4GB'")
con.execute("SET threads = 4")

# --------------------------------------------------
# 2. Identifier automatiquement les colonnes d_1...
# --------------------------------------------------
description = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{sales_sql}')"
).fetchall()

day_columns = [
    row[0]
    for row in description
    if row[0].startswith("d_")
    and row[0][2:].isdigit()
]

if not day_columns:
    raise ValueError("Aucune colonne de ventes d_1, d_2... trouvée.")

print(f"Nombre de jours détectés : {len(day_columns)}")

# Les colonnes proviennent du schéma du fichier, pas d'une saisie utilisateur.
day_columns_sql = ", ".join(f'"{col}"' for col in day_columns)

# --------------------------------------------------
# 3. Dépivoter les ventes et joindre le calendrier
# --------------------------------------------------
query = f"""
COPY (
    WITH selected_sales AS (
        SELECT *
        FROM read_parquet('{sales_sql}')
        WHERE store_id = 'CA_1'
        ORDER BY item_id
        LIMIT 50
    ),

    sales_long AS (
        UNPIVOT selected_sales
        ON {day_columns_sql}
        INTO
            NAME day_id
            VALUE sales
    ),

    enriched_sales AS (
        SELECT
            s.id,
            s.item_id,
            s.dept_id,
            s.cat_id,
            s.store_id,
            s.state_id,

            c.d AS day_id,
            CAST(c.date AS DATE) AS date,
            CAST(s.sales AS INTEGER) AS sales,

            c.wm_yr_wk,
            c.weekday,
            c.wday,
            c.month,
            c.year,
            c.event_name_1,
            c.event_type_1,
            c.event_name_2,
            c.event_type_2,
            c.snap_CA,
            c.snap_TX,
            c.snap_WI

        FROM sales_long s
        INNER JOIN read_parquet('{calendar_sql}') c
            ON s.day_id = c.d
    )

    SELECT
        e.*,
        p.sell_price
    FROM enriched_sales e
    LEFT JOIN read_parquet('{prices_sql}') p
        ON e.store_id = p.store_id
        AND e.item_id = p.item_id
        AND e.wm_yr_wk = p.wm_yr_wk

)
TO '{output_sql}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

try:
    con.execute(query)
    print(f"Couche Silver créée : {output_path}")
finally:
    con.close()
