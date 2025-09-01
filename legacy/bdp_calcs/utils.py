
# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
from .constants import EMA_ALPHA, EMA_MIN_PERIODS

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def safe_div(a, b):
    try:
        return float(a) / float(b) if b not in (0, 0.0, None, np.nan) else np.nan
    except Exception:
        return np.nan

def count_events(text):
    if pd.isna(text):
        return np.nan
    s = str(text)
    tokens = re.split(r"[;,/|]", s)
    tokens = [t.strip() for t in tokens if t.strip()]
    return float(len(tokens)) if tokens else np.nan

def winsorize_deltas(s: pd.Series, lower_q=0.02, upper_q=0.98) -> pd.Series:
    if s.isna().all():
        return s
    x = pd.to_numeric(s, errors="coerce")
    dx = x.diff()
    lo, hi = dx.quantile(lower_q), dx.quantile(upper_q)
    dx_w = dx.clip(lower=lo, upper=hi)
    first_valid_idx = x.first_valid_index()
    if first_valid_idx is None:
        return s
    xw = pd.Series(index=x.index, dtype=float)
    xw.loc[first_valid_idx] = x.loc[first_valid_idx]
    for i in range(x.index.get_loc(first_valid_idx) + 1, len(x.index)):
        idx = x.index[i]
        prev_idx = x.index[i-1]
        if pd.notna(dx_w.loc[idx]) and pd.notna(xw.loc[prev_idx]):
            xw.loc[idx] = xw.loc[prev_idx] + dx_w.loc[idx]
        else:
            xw.loc[idx] = x.loc[idx]
    return xw

def ema(series: pd.Series, alpha: float = EMA_ALPHA, min_periods: int = EMA_MIN_PERIODS) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    e = s.ewm(alpha=alpha, adjust=False).mean()
    valid_counts = s.notna().rolling(window=min_periods, min_periods=1).sum()
    return e.where(valid_counts >= min_periods)

def _norm_time(tok: str) -> str:
    if tok is None:
        return ""
    s = str(tok).strip().lower()
    if s in ("", "nan", "none"):
        return ""
    s = s.replace("a. m.", "am").replace("p. m.", "pm")
    s = s.replace("-", ":").replace(".", ":").replace(";", ":").replace(",", ":")
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"\d{3,6}", s) and ":" not in s and "am" not in s and "pm" not in s:
        if len(s) == 3:
            s = f"{s[0]}:{s[1:]}"
        elif len(s) == 4:
            s = f"{s[:2]}:{s[2:]}"
        elif len(s) == 5:
            s = f"{s[0]}:{s[1:3]}:{s[3:]}"
        elif len(s) == 6:
            s = f"{s[:2]}:{s[2:4]}:{s[4:]}"
    if "am" not in s and "pm" not in s:
        parts = s.split(":")
        if len(parts) == 1 and parts[0].isdigit():
            s = f"{int(parts[0]):02d}:00:00"
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            s = f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
        elif len(parts) == 3 and all(p.isdigit() for p in parts):
            s = f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return s

def to_datetime(fecha, hora=None, dayfirst=True):
    f = pd.to_datetime(fecha, errors="coerce", dayfirst=dayfirst)
    if hora is None:
        return f
    h = pd.Series(hora).apply(_norm_time)
    combo = f.astype("string").fillna("") + " " + h.astype("string").fillna("")
    dt = pd.to_datetime(combo.str.strip(), errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)
    return dt.fillna(f)

def hours_between(hora1, hora2, dayfirst=True):
    """Diferencia en horas (hora2 - hora1) manejando cruces de medianoche con fecha de hoy."""
    # usamos fecha arbitraria y ajustamos si cruza medianoche
    d0 = pd.to_datetime("2000-01-01 " + _norm_time(hora1), errors="coerce")
    d1 = pd.to_datetime("2000-01-01 " + _norm_time(hora2), errors="coerce")
    if pd.isna(d0) or pd.isna(d1):
        return np.nan
    if d1 < d0:  # cruza medianoche
        d1 = d1 + pd.Timedelta(days=1)
    return (d1 - d0).total_seconds() / 3600.0
