# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

MODIFIABLE_PREFIXES = [
    "horas_sueno","sueno_calidad_calc","sleep_debt","sleep_efficiency","circadian_alignment",
    "cafe_","alcohol_","stimulant_load","hydration_score","agua_litros","alimentacion",
    "movement_score","relaxation_score","morning_light_score","screen_night_score","exposicion_sol_","meditacion_","mov_",
    "social_","stressors_","estres_",
    "has_","adherencia_med",
]

def _features_for_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        if c == target:
            continue
        if c in ("animo","claridad","estres","activacion"):
            continue
        if c.endswith(("_lag1","_lag3","_roll3","_ema")) or any(c.startswith(p) for p in MODIFIABLE_PREFIXES):
            cols.append(c)
    X = df[cols].copy()
    # solo mantener numéricas
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols]
    # drop all-NaN
    keep = [c for c in X.columns if not X[c].isna().all()]
    return X[keep]

def drivers_del_dia(df: pd.DataFrame, target: str, top_k: int = 3) -> Dict:
    """
    Explica el cambio de hoy (Δy) con los cambios en drivers (ΔX):
        Entrena Δy ~ ΔX (lineal) en todo el historial disponible,
        estandariza ΔX, imputa medianas, y descompone la predicción de hoy
        como contrib_i = beta_i * Δz_i (ΔX estandarizado).
    Devuelve top_k positivos y negativos.
    """
    if target not in df.columns or len(df) < 4:
        return {"texto":"Sin suficientes datos.","tabla":pd.DataFrame()}

    # Target y features
    y = pd.to_numeric(df[target], errors="coerce")
    X = _features_for_target(df, target)
    if X.empty:
        return {"texto":"Sin features numéricos suficientes.","tabla":pd.DataFrame()}

    # Construir matriz de deltas (quitando la primera fila NaN tras diff)
    dY = y.diff().iloc[1:]  # longitud n-1
    dX = X.diff().iloc[1:]
    # Imputación por mediana de columnas (en deltas)
    med = dX.median()
    dX_f = dX.fillna(med)
    # Estandarización por columna (en deltas)
    ss = StandardScaler()
    dX_s = ss.fit_transform(dX_f)

    # Ajuste lineal simple
    mask = (~dY.isna()) & (~np.isnan(dX_s).any(axis=1))
    if not mask.any():
        return {"texto":"Insuficiente señal tras limpiar deltas.","tabla":pd.DataFrame()}
    lr = LinearRegression().fit(dX_s[mask], dY[mask])

    # Δ de hoy
    X_last = X.iloc[-1]
    X_prev = X.iloc[-2]
    dx_today = (X_last - X_prev).fillna(med).to_numpy().reshape(1, -1)
    dx_today_s = ss.transform(dx_today)

    betas = lr.coef_.ravel()
    dz = dx_today_s.ravel()

    contrib = [(col, float(b)*float(z)) for col, b, z in zip(X.columns, betas, dz) if not (np.isnan(b) or np.isnan(z))]
    contrib_df = pd.DataFrame(contrib, columns=["feature","contrib"]).sort_values("contrib", ascending=False)

    # Predicción y top
    delta_pred = float(np.sum([c for _, c in contrib]))
    delta_obs = float(y.iloc[-1] - y.iloc[-2]) if (pd.notna(y.iloc[-1]) and pd.notna(y.iloc[-2])) else np.nan
    top_pos = contrib_df.head(top_k)
    top_neg = contrib_df.tail(top_k).sort_values("contrib")

    texto = (
        f"{target.capitalize()}: Δobs={np.round(delta_obs,2) if delta_obs==delta_obs else 'NaN'}; "
        f"Δpred≈{np.round(delta_pred,2) if delta_pred==delta_pred else 'NaN'} | "
        f"↑ {', '.join(top_pos['feature'].tolist()) if len(top_pos) else '—'} | "
        f"↓ {', '.join(top_neg['feature'].tolist()) if len(top_neg) else '—'}"
    )

    return {"texto": texto, "tabla": contrib_df, "top_pos": top_pos['feature'].tolist(), "top_neg": top_neg['feature'].tolist(), "delta_pred": delta_pred, "delta_obs": delta_obs}
