
from pathlib import Path
import duckdb
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_DIR / "reports"
SILVER = PROJECT_DIR / "data" / "silver" / "sales_silver.parquet"

con = duckdb.connect()

# Historique des ventes
historical = con.execute(
    """
    SELECT
        CAST(date AS DATE) AS date,
        store_id,
        item_id,
        cat_id,
        dept_id,
        sales,
        sell_price,
        sales * sell_price AS actual_revenue
    FROM read_parquet(?)
    ORDER BY date
    """,
    [str(SILVER)],
).fetchdf()

con.close()

historical["date"] = pd.to_datetime(historical["date"])

historical.to_csv(
    REPORTS / "historical_sales_powerbi.csv",
    index=False,
)

# Prévisions futures
forecast_path = REPORTS / "forecast_28_days.csv"
forecast = pd.read_csv(forecast_path, parse_dates=["date"])

forecast["predicted_revenue"] = (
    forecast["predicted_sales"] * forecast["sell_price"]
)

forecast.to_csv(
    REPORTS / "forecast_powerbi.csv",
    index=False,
)

# Synthèse journalière
daily = forecast.groupby("date", as_index=False).agg(
    predicted_sales=("predicted_sales", "sum"),
    predicted_revenue=("predicted_revenue", "sum"),
    products_forecasted=("item_id", "nunique"),
)

daily.to_csv(
    REPORTS / "forecast_daily_powerbi.csv",
    index=False,
)

print("Exports Power BI terminés.")
print("Historique :", len(historical), "lignes")
print("Prévisions :", len(forecast), "lignes")
print("Résumé quotidien :", len(daily), "lignes")
