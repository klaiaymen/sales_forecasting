
from pathlib import Path
import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
PRICES_PATH = PROJECT_DIR / "data" / "bronze" / "sell_prices.parquet"

con = duckdb.connect()

if not PRICES_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {PRICES_PATH}")

print("===== STRUCTURE DES PRIX =====")
print(
    con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(PRICES_PATH)]
    ).fetchdf().to_string(index=False)
)

print("\n===== COUVERTURE DES PRIX FUTURS =====")

query = """
WITH products AS (
    SELECT DISTINCT item_id, store_id
    FROM read_parquet(?)
),
future_calendar AS (
    SELECT DISTINCT date, wm_yr_wk
    FROM read_parquet(?)
    WHERE CAST(date AS DATE)
          BETWEEN DATE '2016-04-25' AND DATE '2016-05-22'
),
expected AS (
    SELECT COUNT(*) AS expected_rows
    FROM products CROSS JOIN future_calendar
),
available AS (
    SELECT COUNT(*) AS available_rows
    FROM products p
    CROSS JOIN future_calendar c
    INNER JOIN read_parquet(?) prices
        ON prices.item_id = p.item_id
        AND prices.store_id = p.store_id
        AND prices.wm_yr_wk = c.wm_yr_wk
)
SELECT
    expected_rows,
    available_rows,
    expected_rows - available_rows AS missing_rows
FROM expected CROSS JOIN available
"""

result = con.execute(
    query,
    [
        str(PROJECT_DIR / "data" / "silver" / "sales_silver.parquet"),
        str(PROJECT_DIR / "data" / "bronze" / "calendar.parquet"),
        str(PRICES_PATH),
    ],
).fetchdf()

print(result.to_string(index=False))
con.close()
