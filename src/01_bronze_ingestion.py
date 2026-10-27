
from pathlib import Path
import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
BRONZE_DIR = PROJECT_DIR / "data" / "bronze"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv",
]

con = duckdb.connect()

for filename in files:
    source = (RAW_DIR / filename).resolve()
    target = (BRONZE_DIR / filename.replace(".csv", ".parquet")).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Fichier introuvable : {source}")

    source_sql = str(source).replace("'", "''")
    target_sql = str(target).replace("'", "''")

    con.execute(f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{source_sql}')
        )
        TO '{target_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print(f"Bronze créé : {target.name}")

con.close()
