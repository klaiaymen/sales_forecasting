# Sales Forecasting — Data Engineering & Machine Learning

## Overview

An end-to-end sales forecasting project using the M5 Forecasting dataset. The project implements a local data engineering pipeline, machine learning model training, recursive forecasting, experiment tracking, and a Power BI dashboard.

## Objectives

- Build Bronze, Silver, and Gold data layers.
- Transform daily retail sales data into analytical datasets.
- Engineer lag and rolling-window features.
- Train and evaluate an XGBoost regression model against a baseline.
- Generate recursive 28-day sales forecasts.
- Visualize historical sales and forecast results in Power BI.

## Technology Stack

- Python
- DuckDB
- Parquet
- Pandas and NumPy
- scikit-learn
- XGBoost
- MLflow
- Power BI Desktop

## Data Pipeline

1. **Bronze:** Raw CSV files converted into Parquet.
2. **Silver:** Sales data transformed into a daily product-store dataset and enriched with calendar and price information.
3. **Gold:** Analytical features, including sales lags and rolling averages.
4. **Machine Learning:** Baseline evaluation and XGBoost training.
5. **Forecasting:** Recursive forecasts for 28 days.
6. **Business Intelligence:** Power BI dashboard for sales analysis.

## Current Scope

The current prototype uses 50 products from the CA_1 store.

- Historical data ends on April 24, 2016.
- Forecast horizon: April 25 to May 22, 2016.
- Forecast records generated: 1,400 (50 products × 28 days).

The forecast is a historical simulation, not a prediction of current sales.

## Project Structure

```text
sales-forecasting/
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── src/
├── models/
├── notebooks/
├── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the M5 dataset from Kaggle and place the required CSV files in `data/raw/`.

Run the scripts in the appropriate pipeline order. The dataset must be available locally before executing the transformations and forecasting scripts.

## Outputs

The pipeline generates forecast CSV files and evaluation reports under `reports/`. The Power BI report can be opened with Power BI Desktop.

## Dataset

M5 Forecasting Accuracy competition dataset:
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Please consult the dataset source and its terms before redistributing the data.

## Limitations

- Current modeling scope is restricted to 50 products from one store.
- Forecast accuracy should be reported using measured evaluation results.
- Recursive multi-day forecasts can accumulate prediction errors.
- Forecasts represent estimates, not guaranteed sales.

## Future Improvements

- Compare additional forecasting algorithms.
- Improve time-series cross-validation.
- Evaluate errors by product and date.
- Expand the pipeline to more stores and products.
- Improve the Power BI dashboard and automated reporting.
