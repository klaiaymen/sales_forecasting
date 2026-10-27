
from pathlib import Path
import duckdb

PROJECT_DIR = Path(__file__).resolve().parent.parent
silver_path = (
    PROJECT_DIR / "data" / "silver" / "sales_silver.parquet"
).resolve()

if not silver_path.exists():
    raise FileNotFoundError(f"Fichier introuvable : {silver_path}")

path_sql = str(silver_path).replace("\\", "/").replace("'", "''")
con = duckdb.connect()

# Vérifier le volume et les dates
print("\n--- Statistiques générales ---")
print(con.execute(f"""
    SELECT
        COUNT(*) AS total_lignes,
        COUNT(DISTINCT item_id) AS nombre_produits,
        COUNT(DISTINCT date) AS nombre_jours,
        MIN(date) AS premiere_date,
        MAX(date) AS derniere_date,
        SUM(CASE WHEN sales IS NULL THEN 1 ELSE 0 END)
            AS ventes_manquantes,
        SUM(CASE WHEN sell_price IS NULL THEN 1 ELSE 0 END)
            AS prix_manquants
    FROM read_parquet('{path_sql}')
""").fetchdf().to_string(index=False))

# Vérifier les doublons sur la clé métier
print("\n--- Contrôle des doublons ---")
print(con.execute(f"""
    SELECT COUNT(*) AS groupes_en_doublon
    FROM (
        SELECT item_id, store_id, date
        FROM read_parquet('{path_sql}')
        GROUP BY item_id, store_id, date
        HAVING COUNT(*) > 1
    ) t
""").fetchdf().to_string(index=False))

# Examiner quelques lignes
print("\n--- Aperçu des données ---")
print(con.execute(f"""
    SELECT *
    FROM read_parquet('{path_sql}')
    ORDER BY date, item_id
    LIMIT 10
""").fetchdf().to_string(index=False))

con.close()
