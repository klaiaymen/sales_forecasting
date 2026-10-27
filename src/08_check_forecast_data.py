
from pathlib import Path
import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent

silver_path = PROJECT_DIR / "data" / "silver" / "sales_silver.parquet"
calendar_path = PROJECT_DIR / "data" / "bronze" / "calendar.parquet"

con = duckdb.connect()

for label, path in [
    ("SILVER", silver_path),
    ("CALENDRIER BRONZE", calendar_path),
]:
    print(f"\n===== {label} =====")

    if not path.exists():
        print(f"Fichier introuvable : {path}")
        continue

    # Afficher les colonnes disponibles
    schema = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(path)]
    ).fetchdf()

    print("Colonnes :", schema["column_name"].tolist())

    # Vérifier les dates si la colonne date existe
    if "date" in schema["column_name"].tolist():
        result = con.execute(
            """
            SELECT
                MIN(CAST(date AS DATE)) AS premiere_date,
                MAX(CAST(date AS DATE)) AS derniere_date,
                COUNT(*) AS nombre_lignes
            FROM read_parquet(?)
            """,
            [str(path)]
        ).fetchdf()

        print(result.to_string(index=False))
    else:
        print("Pas de colonne 'date' : vérifier le schéma ci-dessus.")

con.close()
