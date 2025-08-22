
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from reporte_python.bdp_schema import AREAS, POSITIVE_AREAS, NEGATIVE_AREAS, coerce_date

def _day_key(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)

def _area_delta(a_today, a_prev):
    if pd.isna(a_today) or pd.isna(a_prev):
        return np.nan
    return float(a_today) - float(a_prev)

def compute_daily_areas(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    # Normalize date to day
    dcol = columns.get("fecha", "fecha")
    df = df.copy()
    df[dcol] = coerce_date(df[dcol])
    df["__day__"] = df[dcol].dt.normalize()

    # Aggregate per-day: mean of available values
    area_cols = [columns.get(a, a) for a in AREAS if columns.get(a, a) in df.columns]
    agg = df.groupby("__day__")[area_cols].mean(numeric_only=True)

    # Count per-day
    counts = df.groupby("__day__").size().rename("registros")
    out = agg.join(counts, how="outer").reset_index(names=["fecha"])
    return out.sort_values("fecha")

def compute_trends(df_daily: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    # day-over-day deltas & simple trend labels
    df = df_daily.copy().sort_values("fecha")
    area_cols = [c for c in df.columns if c in columns.values()]
    area_cols = [c for c in area_cols if c not in ["fecha", "registros", columns.get("categoria_dia", "categoria_dia")]]
    for c in area_cols:
        df[c + "_delta"] = df[c].diff()

        # 3-day slope (simple)
        df[c + "_slope3"] = df[c].rolling(3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) if pd.notna(x).all() else np.nan, raw=False)
        def lab(row):
            v = row[c + "_slope3"]
            if pd.isna(v):
                return "flat"
            if v > 0.05:
                return "up"
            if v < -0.05:
                return "down"
            return "flat"
        df[c + "_trend"] = df.apply(lab, axis=1)

    return df

def compute_composite_and_category(row: pd.Series, columns: Dict[str, str]) -> Tuple[float, int]:
    # Composite score: average(POSITIVE) - average(NEGATIVE)
    pos_vals = []
    for a in POSITIVE_AREAS:
        col = columns.get(a, a)
        if col in row and pd.notna(row[col]):
            pos_vals.append(float(row[col]))
    neg_vals = []
    for a in NEGATIVE_AREAS:
        col = columns.get(a, a)
        if col in row and pd.notna(row[col]):
            neg_vals.append(float(row[col]))

    if pos_vals:
        pos = sum(pos_vals)/len(pos_vals)
    else:
        pos = np.nan
    if neg_vals:
        neg = sum(neg_vals)/len(neg_vals)
    else:
        neg = np.nan

    comp = np.nan
    if not np.isnan(pos) and not np.isnan(neg):
        comp = pos - neg
    elif not np.isnan(pos):
        comp = pos
    elif not np.isnan(neg):
        comp = -neg

    # Category thresholds (tunable): map composite to 0..3
    # Assuming areas scale ~0..1 (or 0..100 normalized to 0..1 before call).
    # If your data is 0..100, pre-scale or tweak thresholds below.
    if np.isnan(comp):
        cat = 1  # default mildly negative if unknown
    else:
        if comp <= 0.25:
            cat = 0
        elif comp <= 0.45:
            cat = 1
        elif comp <= 0.65:
            cat = 2
        else:
            cat = 3
    return comp, cat
