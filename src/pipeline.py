from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "Data for Datathon (Revised).xlsx"
TEMPLATE_PATH = ROOT / "data" / "templates" / "template_forecast_v00.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
PORTFOLIOS = ["A", "B", "C", "D"]
SLOTS_PER_DAY = 48
TRAIN_START = pd.Timestamp("2025-04-01")
TRAIN_END = pd.Timestamp("2025-06-30")
AUGUST_START = pd.Timestamp("2025-08-01")
AUGUST_END = pd.Timestamp("2025-08-31")

CV_BIAS = 1.05
CCT_BIAS = 1.03
CV_MODEL_WEIGHT = 0.40
CV_PROFILE_WEIGHT = 0.60

CV_FEATURES = [
    "slot",
    "dow",
    "daily_cv",
    "is_peak",
    "slot_sin",
    "slot_cos",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
]


@dataclass
class RawData:
    daily: Dict[str, pd.DataFrame]
    intervals: Dict[str, pd.DataFrame]
    template: pd.DataFrame


@dataclass
class CVArtifacts:
    models: Dict[str, Dict[str, object]]
    profiles: pd.DataFrame
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _month_map() -> Dict[str, int]:
    return {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }


def slot_label(slot: int) -> str:
    hour = slot // 2
    minute = 30 if slot % 2 else 0
    return f"{hour}:{minute:02d}"


def normalize(arr: Sequence[float]) -> np.ndarray:
    out = np.clip(np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0), 0, None)
    total = float(out.sum())
    if total <= 0:
        return np.ones(len(out), dtype=float) / len(out)
    return out / total


def smooth_share(arr: Sequence[float], sigma: float = 0.7) -> np.ndarray:
    return normalize(gaussian_filter1d(normalize(arr), sigma=sigma))


def mape(actual: Sequence[float], pred: Sequence[float]) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    mask = actual_arr != 0
    return float(np.mean(np.abs((actual_arr[mask] - pred_arr[mask]) / actual_arr[mask])) * 100)


def load_raw_data() -> RawData:
    xlsx = pd.ExcelFile(RAW_DATA_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    daily: Dict[str, pd.DataFrame] = {}
    intervals: Dict[str, pd.DataFrame] = {}
    mmap = _month_map()

    for portfolio in PORTFOLIOS:
        daily_df = pd.read_excel(xlsx, f"{portfolio} - Daily")
        daily_df.columns = [str(c).strip() for c in daily_df.columns]
        daily_df["Date"] = pd.to_datetime(
            daily_df["Date"].astype(str).str.strip().str.rsplit(" ", n=1).str[0],
            format="%m/%d/%y",
        )
        for col in ["Call Volume", "CCT", "Service Level", "Abandon Rate"]:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
        daily_df["Portfolio"] = portfolio
        daily[portfolio] = daily_df.sort_values("Date").reset_index(drop=True)

        interval_df = pd.read_excel(xlsx, f"{portfolio} - Interval")
        interval_df.columns = [str(c).strip() for c in interval_df.columns]
        interval_df = interval_df.dropna(subset=["Interval"]).copy()
        interval_df["mnum"] = interval_df["Month"].map(mmap)
        interval_df["Day"] = pd.to_numeric(interval_df["Day"], errors="coerce").astype(int)
        interval_df["Date"] = pd.to_datetime(dict(year=2025, month=interval_df["mnum"], day=interval_df["Day"]))
        interval_df["slot"] = interval_df["Interval"].apply(lambda t: int(t.hour * 2 + t.minute // 30))
        for col in ["Call Volume", "Abandoned Calls", "Abandoned Rate", "CCT", "Service Level"]:
            interval_df[col] = pd.to_numeric(interval_df[col], errors="coerce")
        interval_df["Portfolio"] = portfolio
        intervals[portfolio] = interval_df.sort_values(["Date", "slot"]).reset_index(drop=True)

    return RawData(daily=daily, intervals=intervals, template=template)


def stack_daily(daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["Portfolio", "Date", "Call Volume", "CCT", "Service Level", "Abandon Rate"]
    return pd.concat([df[cols] for df in daily.values()], ignore_index=True).sort_values(["Portfolio", "Date"])


def stack_intervals(intervals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = [
        "Portfolio",
        "Date",
        "slot",
        "Interval",
        "Call Volume",
        "Abandoned Calls",
        "Abandoned Rate",
        "CCT",
        "Service Level",
    ]
    return pd.concat([df[cols] for df in intervals.values()], ignore_index=True).sort_values(["Portfolio", "Date", "slot"])


def patch_august_daily_anchors(daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    august = pd.date_range(AUGUST_START, AUGUST_END, freq="D")
    anchors = []
    for portfolio in PORTFOLIOS:
        p_aug = (
            daily[portfolio]
            .loc[(daily[portfolio]["Date"] >= AUGUST_START) & (daily[portfolio]["Date"] <= AUGUST_END)]
            .set_index("Date")
            .reindex(august)
            .copy()
        )
        p_aug["Portfolio"] = portfolio
        anchors.append(
            p_aug[["Portfolio", "Call Volume", "CCT", "Abandon Rate"]]
            .rename_axis("Date")
            .reset_index()
        )
    anchors_df = pd.concat(anchors, ignore_index=True)

    daily_map = {p: daily[p].set_index("Date") for p in PORTFOLIOS}
    missing_mask = (
        (anchors_df["Portfolio"] == "D")
        & (anchors_df["Date"] >= pd.Timestamp("2025-08-27"))
        & (anchors_df["Date"] <= pd.Timestamp("2025-08-31"))
    )

    for idx, row in anchors_df.loc[missing_mask].iterrows():
        dt = row["Date"]
        prev = dt - pd.Timedelta(days=7)
        abc_cv_now = sum(float(daily_map[p].loc[dt, "Call Volume"]) for p in ["A", "B", "C"])
        abc_cv_prev = sum(float(daily_map[p].loc[prev, "Call Volume"]) for p in ["A", "B", "C"])
        d_cv_prev = float(daily_map["D"].loc[prev, "Call Volume"])
        anchors_df.loc[idx, "Call Volume"] = round(d_cv_prev * abc_cv_now / abc_cv_prev)

        abc_cct_now = np.mean([float(daily_map[p].loc[dt, "CCT"]) for p in ["A", "B", "C"]])
        abc_cct_prev = np.mean([float(daily_map[p].loc[prev, "CCT"]) for p in ["A", "B", "C"]])
        d_cct_prev = float(daily_map["D"].loc[prev, "CCT"])
        adjusted = abc_cct_now + (d_cct_prev - abc_cct_prev)
        anchors_df.loc[idx, "CCT"] = round(0.7 * d_cct_prev + 0.3 * adjusted, 2)

    anchors_df["daily_abandoned_calls"] = anchors_df["Call Volume"] * anchors_df["Abandon Rate"]
    return anchors_df.sort_values(["Portfolio", "Date"]).reset_index(drop=True)


def create_full_interval_grid(portfolio: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    grid = pd.MultiIndex.from_product([dates, range(SLOTS_PER_DAY)], names=["Date", "slot"]).to_frame(index=False)
    grid["Portfolio"] = portfolio
    grid["dow"] = grid["Date"].dt.dayofweek
    grid["dom"] = grid["Date"].dt.day
    grid["is_peak"] = ((grid["slot"] // 2 >= 9) & (grid["slot"] // 2 <= 17)).astype(int)
    grid["slot_sin"] = np.sin(2 * np.pi * grid["slot"] / SLOTS_PER_DAY)
    grid["slot_cos"] = np.cos(2 * np.pi * grid["slot"] / SLOTS_PER_DAY)
    grid["dow_sin"] = np.sin(2 * np.pi * grid["dow"] / 7)
    grid["dow_cos"] = np.cos(2 * np.pi * grid["dow"] / 7)
    grid["dom_sin"] = np.sin(2 * np.pi * grid["dom"] / 31)
    grid["dom_cos"] = np.cos(2 * np.pi * grid["dom"] / 31)
    return grid


def build_cv_share_prior(observed_grid: pd.DataFrame) -> tuple[Dict[tuple[str, int], np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    train = observed_grid.dropna(subset=["daily_cv", "Call Volume"]).copy()
    train = train[(train["daily_cv"] > 0) & (train["Call Volume"] >= 0)]
    train["observed_share"] = train["Call Volume"] / train["daily_cv"]

    portfolio_slot = {}
    for portfolio in PORTFOLIOS:
        p = train[train["Portfolio"] == portfolio]
        med = p.groupby("slot")["observed_share"].median()
        arr = np.array([med.get(slot, np.nan) for slot in range(SLOTS_PER_DAY)], dtype=float)
        portfolio_slot[portfolio] = smooth_share(np.nan_to_num(arr, nan=0.0))

    global_med = train.groupby("slot")["observed_share"].median()
    global_profile = smooth_share([global_med.get(slot, 0.0) for slot in range(SLOTS_PER_DAY)])

    priors: Dict[tuple[str, int], np.ndarray] = {}
    for portfolio in PORTFOLIOS:
        for dow in range(7):
            sub = train[(train["Portfolio"] == portfolio) & (train["dow"] == dow)]
            med = sub.groupby("slot")["observed_share"].median()
            arr = np.array([med.get(slot, np.nan) for slot in range(SLOTS_PER_DAY)], dtype=float)
            fallback = portfolio_slot.get(portfolio, global_profile)
            arr = np.where(np.isfinite(arr), arr, fallback)
            priors[(portfolio, dow)] = smooth_share(arr)
    return priors, portfolio_slot, global_profile


def build_cv_training_layer(daily_all: pd.DataFrame, interval_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_daily = daily_all[(daily_all["Date"] >= TRAIN_START) & (daily_all["Date"] <= TRAIN_END)].copy()
    train_intervals = interval_all[(interval_all["Date"] >= TRAIN_START) & (interval_all["Date"] <= TRAIN_END)].copy()

    observed_grids = []
    for portfolio in PORTFOLIOS:
        grid = create_full_interval_grid(portfolio, TRAIN_START, TRAIN_END)
        observed = train_intervals[train_intervals["Portfolio"] == portfolio][
            ["Date", "slot", "Call Volume"]
        ].copy()
        p_daily = train_daily[train_daily["Portfolio"] == portfolio][["Date", "Call Volume"]].rename(columns={"Call Volume": "daily_cv"})
        grid = grid.merge(observed, on=["Date", "slot"], how="left")
        grid = grid.merge(p_daily, on="Date", how="left")
        observed_grids.append(grid)
    observed_grid = pd.concat(observed_grids, ignore_index=True)
    priors, _portfolio_slot, _global_profile = build_cv_share_prior(observed_grid)

    for (portfolio, dt), day in observed_grid.groupby(["Portfolio", "Date"], sort=True):
        day = day.sort_values("slot").reset_index(drop=True).copy()
        daily_cv = day["daily_cv"].iloc[0]
        prior = priors[(portfolio, int(day["dow"].iloc[0]))]
        valid = day["Call Volume"].notna() & (day["Call Volume"] >= 0)
        observed_sum = float(day.loc[valid, "Call Volume"].sum())
        observed_share_mass = float(prior[valid.to_numpy()].sum())
        observed_count = int(valid.sum())

        if pd.notna(daily_cv) and daily_cv > 0:
            total = float(daily_cv)
            repaired = np.zeros(SLOTS_PER_DAY, dtype=float)
            if observed_sum > total or observed_count == SLOTS_PER_DAY:
                repaired = np.clip(day["Call Volume"].fillna(0).to_numpy(dtype=float), 0, None)
                repaired = repaired * (total / repaired.sum()) if repaired.sum() > 0 else total * prior
                case = "A_scale_to_daily_anchor"
            else:
                repaired[valid.to_numpy()] = day.loc[valid, "Call Volume"].to_numpy(dtype=float)
                missing = ~valid.to_numpy()
                remainder = max(total - repaired.sum(), 0.0)
                missing_mass = float(prior[missing].sum())
                if missing.any() and missing_mass > 0:
                    repaired[missing] = remainder * prior[missing] / missing_mass
                case = "A_daily_anchor"
        elif observed_count >= 42 and observed_share_mass >= 0.80 and observed_sum > 0:
            total = observed_sum / observed_share_mass
            repaired = np.zeros(SLOTS_PER_DAY, dtype=float)
            repaired[valid.to_numpy()] = day.loc[valid, "Call Volume"].to_numpy(dtype=float)
            missing = ~valid.to_numpy()
            repaired[missing] = total * prior[missing]
            case = "B_estimated_daily_total"
        else:
            continue

        day["repaired_cv"] = repaired
        day["daily_cv"] = float(repaired.sum())
        day["slot_share"] = day["repaired_cv"] / day["daily_cv"]
        day["repair_case"] = case
        day["observed_slot_count"] = observed_count
        rows.append(day)

    return pd.concat(rows, ignore_index=True).sort_values(["Portfolio", "Date", "slot"])


def build_cv_share_profiles(cv_training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio in PORTFOLIOS:
        p = cv_training[cv_training["Portfolio"] == portfolio]
        fallback = smooth_share(p.groupby("slot")["slot_share"].median().reindex(range(SLOTS_PER_DAY), fill_value=0).to_numpy())
        for dow in range(7):
            sub = p[p["dow"] == dow]
            arr = sub.groupby("slot")["slot_share"].median().reindex(range(SLOTS_PER_DAY)).to_numpy(dtype=float)
            arr = np.where(np.isfinite(arr), arr, fallback)
            arr = smooth_share(arr)
            for slot, share in enumerate(arr):
                rows.append({"Portfolio": portfolio, "dow": dow, "slot": slot, "profile_share": share})
    return pd.DataFrame(rows)


def train_cv_models(cv_training: pd.DataFrame) -> CVArtifacts:
    models: Dict[str, Dict[str, object]] = {}
    metrics = []
    importances = []
    profiles = build_cv_share_profiles(cv_training)

    for portfolio in PORTFOLIOS:
        p = cv_training[cv_training["Portfolio"] == portfolio].copy()
        train = p[p["Date"] < pd.Timestamp("2025-06-01")].copy()
        test = p[p["Date"] >= pd.Timestamp("2025-06-01")].copy()
        if train.empty or test.empty:
            train = p.copy()
            test = p.copy()

        X_train = train[CV_FEATURES].to_numpy(dtype=float)
        y_train = train["slot_share"].to_numpy(dtype=float)
        X_test = test[CV_FEATURES].to_numpy(dtype=float)
        y_test_counts = test["repaired_cv"].to_numpy(dtype=float)

        models_to_try = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
            "Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=250,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=8,
                l2_regularization=1.0,
                random_state=42,
            ),
        }
        for name, model in models_to_try.items():
            model.fit(X_train, y_train)
            pred_counts = _daily_counts_from_share_predictions(test, model.predict(X_test))
            metrics.append({"Portfolio": portfolio, "model": name, "mape": mape(y_test_counts, pred_counts)})

        hgb = HistGradientBoostingRegressor(
            max_iter=250,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=42,
        )
        et = ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        hgb.fit(X_train, y_train)
        et.fit(X_train, y_train)
        ensemble_share = 0.5 * np.clip(hgb.predict(X_test), 0, None) + 0.5 * np.clip(et.predict(X_test), 0, None)
        pred_counts = _daily_counts_from_share_predictions(test, ensemble_share)
        metrics.append({"Portfolio": portfolio, "model": "HGB + ExtraTrees", "mape": mape(y_test_counts, pred_counts)})

        profile_lookup = profiles[profiles["Portfolio"] == portfolio].set_index(["dow", "slot"])["profile_share"]
        blend_counts = _blend_cv_predictions(test, ensemble_share, profile_lookup)
        metrics.append({"Portfolio": portfolio, "model": "40/60 ML-profile blend", "mape": mape(y_test_counts, blend_counts)})

        for feature, importance in zip(CV_FEATURES, et.feature_importances_):
            importances.append({"Portfolio": portfolio, "feature": feature, "importance": importance})

        # Refit on all Apr-Jun repaired data for the August forecast.
        X_all = p[CV_FEATURES].to_numpy(dtype=float)
        y_all = p["slot_share"].to_numpy(dtype=float)
        hgb.fit(X_all, y_all)
        et.fit(X_all, y_all)
        models[portfolio] = {"hgb": hgb, "et": et}

    return CVArtifacts(
        models=models,
        profiles=profiles,
        metrics=pd.DataFrame(metrics),
        feature_importance=pd.DataFrame(importances),
    )


def _daily_counts_from_share_predictions(df: pd.DataFrame, share_pred: Sequence[float]) -> np.ndarray:
    base = df.reset_index(drop=True)
    raw = np.asarray(share_pred, dtype=float)
    pred = np.zeros(len(base), dtype=float)
    for _dt, idx in base.groupby("Date").groups.items():
        idx = list(idx)
        shares = normalize(raw[idx])
        pred[idx] = shares * float(base.loc[idx[0], "daily_cv"])
    return pred


def _blend_cv_predictions(df: pd.DataFrame, ml_share_pred: Sequence[float], profile_lookup: pd.Series) -> np.ndarray:
    base = df.reset_index(drop=True)
    raw_ml = np.asarray(ml_share_pred, dtype=float)
    pred = np.zeros(len(base), dtype=float)
    for _dt, idx in base.groupby("Date").groups.items():
        idx = list(idx)
        ml_share = normalize(raw_ml[idx])
        prof_share = normalize([profile_lookup.loc[(int(base.loc[j, "dow"]), int(base.loc[j, "slot"]))] for j in idx])
        blended = normalize(CV_MODEL_WEIGHT * ml_share + CV_PROFILE_WEIGHT * prof_share)
        pred[idx] = blended * float(base.loc[idx[0], "daily_cv"])
    return pred


def forecast_august_cv(anchors: pd.DataFrame, artifacts: CVArtifacts) -> pd.DataFrame:
    rows = []
    profile_lookup = artifacts.profiles.set_index(["Portfolio", "dow", "slot"])["profile_share"]
    for portfolio in PORTFOLIOS:
        p_anchor = anchors[anchors["Portfolio"] == portfolio].sort_values("Date")
        for _, anchor in p_anchor.iterrows():
            dt = anchor["Date"]
            daily_cv = float(anchor["Call Volume"])
            grid = create_full_interval_grid(portfolio, dt, dt)
            grid["daily_cv"] = daily_cv
            X = grid[CV_FEATURES].to_numpy(dtype=float)
            hgb = artifacts.models[portfolio]["hgb"]
            et = artifacts.models[portfolio]["et"]
            ml_share = 0.5 * np.clip(hgb.predict(X), 0, None) + 0.5 * np.clip(et.predict(X), 0, None)
            prof_share = normalize([profile_lookup.loc[(portfolio, int(grid.loc[i, "dow"]), int(grid.loc[i, "slot"]))] for i in grid.index])
            share = normalize(CV_MODEL_WEIGHT * normalize(ml_share) + CV_PROFILE_WEIGHT * prof_share)
            cv = daily_cv * share
            for slot, value in enumerate(cv):
                rows.append(
                    {
                        "Portfolio": portfolio,
                        "Date": dt,
                        "slot": slot,
                        "interval_cv": float(value),
                        "cv_share": float(share[slot]),
                    }
                )
    return pd.DataFrame(rows)


def build_cct_profiles(daily_all: pd.DataFrame, interval_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scale_rows = []
    train_daily = daily_all[(daily_all["Date"] >= TRAIN_START) & (daily_all["Date"] <= TRAIN_END)]
    train_interval = interval_all[(interval_all["Date"] >= TRAIN_START) & (interval_all["Date"] <= TRAIN_END)].copy()
    train_interval["dow"] = train_interval["Date"].dt.dayofweek

    for portfolio in PORTFOLIOS:
        p_int = train_interval[
            (train_interval["Portfolio"] == portfolio)
            & train_interval["CCT"].notna()
            & train_interval["Call Volume"].fillna(0).ge(3)
        ].copy()
        p_daily = train_daily[train_daily["Portfolio"] == portfolio]
        avg_daily_cct = float(p_daily["CCT"].dropna().mean())
        scale_rows.append({"Portfolio": portfolio, "avg_train_daily_cct": avg_daily_cct})
        fallback = p_int.groupby("slot")["CCT"].median().reindex(range(SLOTS_PER_DAY)).interpolate().bfill().ffill()
        fallback_arr = gaussian_filter1d(fallback.to_numpy(dtype=float), sigma=0.7)
        for dow in range(7):
            sub = p_int[p_int["dow"] == dow]
            arr = sub.groupby("slot")["CCT"].median().reindex(range(SLOTS_PER_DAY)).to_numpy(dtype=float)
            arr = np.where(np.isfinite(arr), arr, fallback_arr)
            arr = gaussian_filter1d(arr, sigma=0.7)
            for slot, value in enumerate(arr):
                rows.append({"Portfolio": portfolio, "dow": dow, "slot": slot, "cct_profile": float(value)})
    return pd.DataFrame(rows), pd.DataFrame(scale_rows)


def build_abandonment_profiles(interval_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_interval = interval_all[(interval_all["Date"] >= TRAIN_START) & (interval_all["Date"] <= TRAIN_END)].copy()
    train_interval = train_interval[train_interval["Abandoned Calls"].notna()].copy()
    train_interval["dow"] = train_interval["Date"].dt.dayofweek
    daily_ab = train_interval.groupby(["Portfolio", "Date"])["Abandoned Calls"].sum().rename("daily_observed_abandoned_calls")
    train_interval = train_interval.merge(daily_ab.reset_index(), on=["Portfolio", "Date"], how="left")
    train_interval = train_interval[train_interval["daily_observed_abandoned_calls"] > 0].copy()
    train_interval["abd_share"] = train_interval["Abandoned Calls"] / train_interval["daily_observed_abandoned_calls"]

    for portfolio in PORTFOLIOS:
        p_int = train_interval[train_interval["Portfolio"] == portfolio]
        fallback = smooth_share(p_int.groupby("slot")["abd_share"].median().reindex(range(SLOTS_PER_DAY), fill_value=0).to_numpy())
        for dow in range(7):
            sub = p_int[p_int["dow"] == dow]
            arr = sub.groupby("slot")["abd_share"].median().reindex(range(SLOTS_PER_DAY)).to_numpy(dtype=float)
            arr = np.where(np.isfinite(arr), arr, fallback)
            arr = smooth_share(arr)
            for slot, value in enumerate(arr):
                rows.append({"Portfolio": portfolio, "dow": dow, "slot": slot, "abd_share_profile": float(value)})
    return pd.DataFrame(rows)


def forecast_service_metrics(anchors: pd.DataFrame, cct_profiles: pd.DataFrame, cct_scale: pd.DataFrame, abd_profiles: pd.DataFrame) -> pd.DataFrame:
    cct_lookup = cct_profiles.set_index(["Portfolio", "dow", "slot"])["cct_profile"]
    abd_lookup = abd_profiles.set_index(["Portfolio", "dow", "slot"])["abd_share_profile"]
    scale_lookup = cct_scale.set_index("Portfolio")["avg_train_daily_cct"]
    rows = []
    for _, anchor in anchors.sort_values(["Portfolio", "Date"]).iterrows():
        portfolio = anchor["Portfolio"]
        dt = anchor["Date"]
        dow = int(dt.dayofweek)
        daily_cct = float(anchor["CCT"])
        avg_train_cct = float(scale_lookup.loc[portfolio])
        daily_abd = float(anchor["daily_abandoned_calls"])
        cct_scale_factor = daily_cct / avg_train_cct if avg_train_cct > 0 else 1.0
        abd_share = normalize([abd_lookup.loc[(portfolio, dow, slot)] for slot in range(SLOTS_PER_DAY)])
        for slot in range(SLOTS_PER_DAY):
            rows.append(
                {
                    "Portfolio": portfolio,
                    "Date": dt,
                    "slot": slot,
                    "interval_cct": float(cct_lookup.loc[(portfolio, dow, slot)] * cct_scale_factor),
                    "interval_abandoned_calls": float(daily_abd * abd_share[slot]),
                    "abd_share": float(abd_share[slot]),
                }
            )
    return pd.DataFrame(rows)


def combine_forecast(cv_forecast: pd.DataFrame, service_forecast: pd.DataFrame, template: pd.DataFrame, apply_bias: bool = True) -> pd.DataFrame:
    merged = cv_forecast.merge(service_forecast, on=["Portfolio", "Date", "slot"], how="inner")
    if apply_bias:
        merged["interval_cv"] = merged["interval_cv"] * CV_BIAS
        merged["interval_cct"] = merged["interval_cct"] * CCT_BIAS
    merged["interval_cv"] = np.clip(merged["interval_cv"], 0, None)
    merged["interval_cct"] = np.clip(merged["interval_cct"], 0, None)
    merged["interval_abandoned_calls"] = np.clip(merged["interval_abandoned_calls"], 0, merged["interval_cv"])
    merged["interval_ar"] = np.where(merged["interval_cv"] > 0, merged["interval_abandoned_calls"] / merged["interval_cv"], 0.0)
    merged["interval_ar"] = np.clip(merged["interval_ar"], 0, 1)
    merged["Calls_Offered"] = np.round(merged["interval_cv"]).astype(int)
    merged["Abandoned_Calls"] = np.round(merged["interval_abandoned_calls"]).astype(int)

    rows = []
    august = pd.date_range(AUGUST_START, AUGUST_END, freq="D")
    for day_idx, dt in enumerate(august, start=1):
        for slot in range(SLOTS_PER_DAY):
            row = {"Month": "August", "Day": str(day_idx), "Interval": slot_label(slot)}
            for portfolio in PORTFOLIOS:
                rec = merged[(merged["Portfolio"] == portfolio) & (merged["Date"] == dt) & (merged["slot"] == slot)]
                if rec.empty:
                    raise ValueError(f"Missing forecast row for {portfolio} {dt.date()} slot {slot}")
                r = rec.iloc[0]
                calls = int(r["Calls_Offered"])
                abandoned = min(int(r["Abandoned_Calls"]), calls)
                row[f"Calls_Offered_{portfolio}"] = calls
                row[f"Abandoned_Calls_{portfolio}"] = abandoned
                row[f"Abandoned_Rate_{portfolio}"] = round(float(abandoned / calls if calls > 0 else 0.0), 6)
                row[f"CCT_{portfolio}"] = round(float(r["interval_cct"]), 2)
            rows.append(row)

    submission = pd.DataFrame(rows)[template.columns.tolist()]
    validate_submission(submission)
    return submission


def validate_submission(submission: pd.DataFrame) -> None:
    if submission.shape != (31 * SLOTS_PER_DAY, 19):
        raise AssertionError(f"Unexpected submission shape: {submission.shape}")
    if submission.isnull().any().any():
        raise AssertionError("Submission contains null values.")
    for portfolio in PORTFOLIOS:
        if (submission[f"Calls_Offered_{portfolio}"] < 0).any():
            raise AssertionError(f"Negative call volume in portfolio {portfolio}.")
        if (submission[f"Abandoned_Calls_{portfolio}"] < 0).any():
            raise AssertionError(f"Negative abandoned calls in portfolio {portfolio}.")
        if (submission[f"Abandoned_Calls_{portfolio}"] > submission[f"Calls_Offered_{portfolio}"]).any():
            raise AssertionError(f"Abandoned calls exceed offered calls in portfolio {portfolio}.")
        if not submission[f"Abandoned_Rate_{portfolio}"].between(0, 1).all():
            raise AssertionError(f"Abandon rate outside [0, 1] in portfolio {portfolio}.")
        if (submission[f"CCT_{portfolio}"] < 0).any():
            raise AssertionError(f"Negative CCT in portfolio {portfolio}.")


def save_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    raw = load_raw_data()
    daily_all = stack_daily(raw.daily)
    interval_all = stack_intervals(raw.intervals)
    anchors = patch_august_daily_anchors(raw.daily)
    cv_training = build_cv_training_layer(daily_all, interval_all)

    daily_all.to_csv(PROCESSED_DIR / "daily_clean.csv", index=False)
    interval_all.to_csv(PROCESSED_DIR / "interval_clean.csv", index=False)
    anchors.to_csv(PROCESSED_DIR / "august_daily_anchors.csv", index=False)
    cv_training.to_csv(PROCESSED_DIR / "cv_training_layer.csv", index=False)
    raw.template.to_csv(PROCESSED_DIR / "template_columns.csv", index=False)
    return daily_all, interval_all, anchors, cv_training


def run_cv_pipeline() -> tuple[CVArtifacts, pd.DataFrame]:
    ensure_dirs()
    cv_training = pd.read_csv(PROCESSED_DIR / "cv_training_layer.csv", parse_dates=["Date"])
    anchors = pd.read_csv(PROCESSED_DIR / "august_daily_anchors.csv", parse_dates=["Date"])
    artifacts = train_cv_models(cv_training)
    cv_forecast = forecast_august_cv(anchors, artifacts)
    artifacts.profiles.to_csv(PROCESSED_DIR / "cv_share_profiles.csv", index=False)
    artifacts.metrics.to_csv(OUTPUT_DIR / "cv_holdout_metrics.csv", index=False)
    artifacts.feature_importance.to_csv(OUTPUT_DIR / "cv_feature_importance.csv", index=False)
    cv_forecast.to_csv(OUTPUT_DIR / "cv_interval_forecast_unbiased.csv", index=False)
    return artifacts, cv_forecast


def run_service_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    daily_all = pd.read_csv(PROCESSED_DIR / "daily_clean.csv", parse_dates=["Date"])
    interval_all = pd.read_csv(PROCESSED_DIR / "interval_clean.csv", parse_dates=["Date"])
    anchors = pd.read_csv(PROCESSED_DIR / "august_daily_anchors.csv", parse_dates=["Date"])
    cct_profiles, cct_scale = build_cct_profiles(daily_all, interval_all)
    abd_profiles = build_abandonment_profiles(interval_all)
    service_forecast = forecast_service_metrics(anchors, cct_profiles, cct_scale, abd_profiles)
    cct_profiles.to_csv(PROCESSED_DIR / "cct_profiles.csv", index=False)
    cct_scale.to_csv(PROCESSED_DIR / "cct_profile_scaling.csv", index=False)
    abd_profiles.to_csv(PROCESSED_DIR / "abandonment_share_profiles.csv", index=False)
    service_forecast.to_csv(OUTPUT_DIR / "service_interval_forecast_unbiased.csv", index=False)
    return cct_profiles, cct_scale, abd_profiles, service_forecast


def run_final_submission(apply_bias: bool = True) -> pd.DataFrame:
    ensure_dirs()
    raw = load_raw_data()
    cv_forecast = pd.read_csv(OUTPUT_DIR / "cv_interval_forecast_unbiased.csv", parse_dates=["Date"])
    service_forecast = pd.read_csv(OUTPUT_DIR / "service_interval_forecast_unbiased.csv", parse_dates=["Date"])
    submission = combine_forecast(cv_forecast, service_forecast, raw.template, apply_bias=apply_bias)
    output_name = "forecast_v42.csv" if apply_bias else "forecast_v42_pre_bias.csv"
    submission.to_csv(OUTPUT_DIR / output_name, index=False)
    return submission
