# Team 015 - Synchrony Datathon 2026

Video: [link here]

## Setup

Python 3.9+. Install dependencies:

```
pip install pandas numpy scikit-learn scipy lightgbm matplotlib seaborn openpyxl
```

## How to run

Open the notebooks in `notebooks/` and run them in order:

1. `01_data_exploration.ipynb` — cleans the raw data and saves to `data/processed/`
2. `02_feature_engineering.ipynb` — feature selection experiments (uses the processed CSVs from step 1)
3. `datathon_v38.ipynb` — trains the final model and outputs `data/processed/forecast_v38.csv`

The final submission CSV is produced by step 3.

## Data

Raw data is in `data/raw/datathon_data.xlsx` (provided by Synchrony). The submission template is `data/raw/forecast_data.csv`.
