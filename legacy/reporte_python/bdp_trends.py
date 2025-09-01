
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from reporte_python.bdp_schema import AREAS, POSITIVE_AREAS, NEGATIVE_AREAS, coerce_date

def compute_daily_areas(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    dcol = columns.get("fecha", "fecha")
    df = df.copy()
    df[dcol] = coerce_date(df[dcol])
    df["__day__"] = df[dcol].dt.normalize()
    area_cols = [columns.get(a, a) for a in AREAS if columns.get(a, a) in df.columns]
    agg = df.groupby("__day__")[area_cols].mean(numeric_only=True)
    counts = df.groupby("__day__").size().rename("registros")
    out = agg.join(counts, how="outer").reset_index(names=["fecha"])
    return out.sort_values("fecha")

def compute_trends(df_daily: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    df = df_daily.copy().sort_values("fecha")
    area_cols = [c for c in df.columns if c in columns.values()]
    area_cols = [c for c in area_cols if c not in ["fecha","registros", columns.get("categoria_dia","categoria_dia")]]
    for c in area_cols:
        df[c + "_delta"] = df[c].diff()
        df[c + "_slope3"] = df[c].rolling(3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) if pd.notna(x).all() else np.nan, raw=False)
        def lab(v):
            if pd.isna(v): return "flat"
            if v > 0.05: return "up"
            if v < -0.05: return "down"
            return "flat"
        df[c + "_trend"] = df[c + "_slope3"].map(lab)
    return df

def compute_composite_and_category(row: pd.Series, columns: Dict[str, str]) -> Tuple[float, int]:
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
    pos = sum(pos_vals)/len(pos_vals) if pos_vals else np.nan
    neg = sum(neg_vals)/len(neg_vals) if neg_vals else np.nan
    comp = np.nan
    if not np.isnan(pos) and not np.isnan(neg):
        comp = pos - neg
    elif not np.isnan(pos):
        comp = pos
    elif not np.isnan(neg):
        comp = -neg
    if np.isnan(comp):
        cat = 1
    else:
        if comp <= 0.25: cat = 0
        elif comp <= 0.45: cat = 1
        elif comp <= 0.65: cat = 2
        else: cat = 3
    return comp, cat
