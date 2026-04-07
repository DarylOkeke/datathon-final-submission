# UIUC Statistics Datathon 2026: Synchrony Interval Forecasting

This repository contains the final forecasting pipeline for the UIUC Statistics Datathon 2026 Synchrony challenge.

## Problem

The task was to forecast August 2025 contact-center activity for 4 portfolios across 3 operational metrics:

- `Call Volume`
- `Customer Care Time (CCT)`
- `Abandon Rate`

The required submission contains one row per half-hour interval:

```text
4 portfolios x 3 metrics x 48 intervals x 31 days
```

The key modeling choice was to split the problem into two layers:

1. **Daily anchors:** establish reliable day-level totals or levels.
2. **Intraday disaggregation:** allocate those daily anchors across 48 half-hour intervals.

That framing avoided re-forecasting daily values that were already mostly provided and concentrated modeling effort on the interval shape problem.

## Repository Structure

```text
datathon_final/
  README.md
  requirements.txt
  run_pipeline.py
  data/
    raw/
      Data for Datathon (Revised).xlsx
    templates/
      template_forecast_v00.csv
    processed/
      *.csv
  notebooks/
    01_data_loading_and_preprocessing.ipynb
    02_cv_interval_shape_model.ipynb
    03_service_metric_profiles.ipynb
    04_final_submission_construction.ipynb
  src/
    pipeline.py
  outputs/
    forecast_v42.csv
    forecast_v42_pre_bias.csv
    cv_holdout_metrics.csv
    cv_feature_importance.csv
  archive/
    datathon_v42_original.ipynb
    forecast_v42_original.csv
```

## Data

The raw workbook contains:

- Daily sheets for portfolios `A-D`
- Interval sheets for portfolios `A-D`
- Daily staffing
- Definitions

The final pipeline uses:

- August 2025 daily `Call Volume`, `CCT`, and `Abandon Rate` as daily anchors
- Apr-Jun 2025 interval data to learn intraday shape
- Apr-Jun 2025 daily data where needed for CV repair and CCT scaling

Staffing and service level are not used in the final interval model because the final problem framing is intraday disaggregation, not a full daily forecasting stack.

## Pipeline Summary

### 1. Daily anchor preparation

The August daily layer was mostly complete. The only material issue was Portfolio `D` missing daily `Call Volume` and `CCT` for August 27-31, 2025.

- Portfolio `D` daily `Call Volume` was imputed from its previous-week share of observed `A+B+C` volume.
- Portfolio `D` daily `CCT` was imputed from its previous-week value, lightly adjusted by week-over-week `A+B+C` movement.
- Daily `Abandon Rate` was already present and kept as-is.

The completed daily anchor sheet is saved to:

```text
data/processed/august_daily_anchors.csv
```

### 2. Call-volume interval shape model

`Call Volume` is an additive count, so it was the cleanest metric for explicit interval modeling.

The training layer repairs Apr-Jun 2025 interval CV into 48-slot days:

- Drop blank non-data rows.
- Build a full 48-slot grid for each portfolio-day.
- Use daily `CV` as the trusted day total when present.
- Fill missing slots with historical portfolio/day-of-week/slot share priors.
- Estimate the daily total only when the daily anchor is missing but the interval day is still repairable.
- Drop days that are too incomplete to support a reliable reconstruction.

The model learns slot share:

```text
slot_share = interval_CV / daily_CV
```

The final CV disaggregation uses a 40/60 hybrid:

- 40% learned ML shape
- 60% historical profile shape

This keeps the stable weekday profile while allowing some adaptation when a day does not look like the typical historical pattern.

### 3. CCT and abandonment service profiles

`CCT` and abandonment were handled differently from `CV` because they did not reconcile cleanly between daily and interval sheets even on complete interval days.

For `CCT`:

- Build median `CCT` profiles by portfolio, day of week, and slot.
- Smooth the 48-slot profile.
- Scale the profile to the August daily `CCT` anchor.

For abandonment:

- Avoid modeling raw interval abandon rate directly.
- Convert historical interval abandoned calls into shares of each day’s observed abandoned calls.
- Build median abandoned-call share profiles by portfolio, day of week, and slot.
- Allocate August daily abandoned calls across slots.
- Compute interval abandon rate from interval abandoned calls divided by interval call volume.

### 4. Final submission construction

The final step recombines:

- interval `CV`
- interval `CCT`
- interval abandoned calls
- interval abandon rate

The pipeline writes two files:

```text
outputs/forecast_v42_pre_bias.csv
outputs/forecast_v42.csv
```

`forecast_v42_pre_bias.csv` is the anchor-reconciled forecast before competition post-processing.

`forecast_v42.csv` applies the final v42 guardrails:

- +5% post-processing bias to call volume
- +3% post-processing bias to CCT
- nonnegative counts
- abandoned calls capped at offered calls
- abandon rates bounded to `[0, 1]`

The bias was a competition-specific choice for the asymmetric workload penalty.

## How to Run

From the `datathon_final/` folder:

```bash
python run_pipeline.py
```

This regenerates:

- all `data/processed/*.csv`
- `outputs/cv_holdout_metrics.csv`
- `outputs/cv_feature_importance.csv`
- `outputs/forecast_v42_pre_bias.csv`
- `outputs/forecast_v42.csv`

If running from the parent repo with the existing Windows virtual environment:

```bash
cmd.exe /c winvenv.cmd datathon_final/run_pipeline.py
```

## Notebook Order

Run the notebooks in this order:

1. `notebooks/01_data_loading_and_preprocessing.ipynb`
2. `notebooks/02_cv_interval_shape_model.ipynb`
3. `notebooks/03_service_metric_profiles.ipynb`
4. `notebooks/04_final_submission_construction.ipynb`

The notebooks show the relevant implementation code for each stage and execute the shared implementation in `src/pipeline.py`.

## Final Output

The final competition-format file is:

```text
outputs/forecast_v42.csv
```

The original one-notebook v42 artifact is preserved for reference in:

```text
archive/datathon_v42_original.ipynb
archive/forecast_v42_original.csv
```
