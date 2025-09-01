
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def quantile_bands(series: pd.Series, low=0.15, high=0.85):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (np.nan, np.nan)
    return (float(s.quantile(low)), float(s.quantile(high)))

def semaforo_estado(val: float, qlow: float, qhigh: float) -> str:
    if any(np.isnan([val, qlow, qhigh])):
        return "—"
    if val <= qlow:
        return "Caution"
    if val >= qhigh:
        return "Alert"
    return "OK"

def make_report_line(target: str, obs: float, ema14: float, pred_delta: float, obs_delta: float, sem: str, top_pos, top_neg) -> str:
    obs_s = "NaN" if pd.isna(obs) else f"{obs:.2f}"
    ema_s = "NaN" if pd.isna(ema14) else f"{ema14:.2f}"
    pd_s  = "NaN" if pd.isna(pred_delta) else f"{pred_delta:.2f}"
    od_s  = "NaN" if pd.isna(obs_delta) else f"{obs_delta:.2f}"
    pos = ", ".join((top_pos or [])[:3]) if top_pos else "—"
    neg = ", ".join((top_neg or [])[:3]) if top_neg else "—"
    return (
        f"- {target.capitalize()}: hoy={obs_s}, EMA14={ema_s}, Δpred={pd_s}, Δobs={od_s}, estado={sem}"
        f"  - Impulsores ↑: {pos}"
        f"  - Impulsores ↓: {neg}"
    )
