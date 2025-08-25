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
    # solo numéricas
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols]
    keep = [c for c in X.columns if not X[c].isna().all()]
    return X[keep]

def _clean_deltas(dX: pd.DataFrame) -> pd.DataFrame:
    # quitar columnas todo NaN
    dX = dX.loc[:, dX.notna().sum(axis=0) > 0]
    if dX.empty:
        return dX
    # quitar columnas con mediana NaN (todo NaN igualmente)
    med = dX.median()
    dX = dX.loc[:, ~med.isna().values]
    if dX.empty:
        return dX
    # quitar columnas de varianza cero (deltas constantes)
    var = dX.var()
    dX = dX.loc[:, var > 0]
    return dX

def drivers_del_dia(df: pd.DataFrame, target: str, top_k: int = 3) -> Dict:
    """
    Explica el cambio de hoy (Δy) con ΔX:
    - Construye Δy ~ ΔX (lineal) con estandarización y mediana en deltas.
    - Descarta columnas sin señal (todo NaN o varianza cero).
    - Devuelve top_k ↑/↓ y Δpred.
    """
    if target not in df.columns or len(df) < 4:
        return {"texto":"Sin suficientes datos.","tabla":pd.DataFrame()}

    y = pd.to_numeric(df[target], errors="coerce")
    X = _features_for_target(df, target)
    if X.empty:
        return {"texto":"Sin features numéricos suficientes.","tabla":pd.DataFrame()}

    # Deltas
    dY = y.diff().iloc[1:]
    dX = X.diff().iloc[1:]
    dX = _clean_deltas(dX)
    if dX.empty:
        return {"texto":"Sin señal en ΔX (todas constantes o vacías).","tabla":pd.DataFrame()}

    # Imputación por mediana y estandarización
    med = dX.median()
    dX_f = dX.fillna(med)
    ss = StandardScaler()
    dX_s = ss.fit_transform(dX_f)

    # Más defensivo: si alguna fila queda con NaN/Inf, la descartamos
    row_ok = np.isfinite(dX_s).all(axis=1) & (~dY.isna().values)
    if not row_ok.any():
        return {"texto":"Insuficiente señal tras limpiar deltas.","tabla":pd.DataFrame()}

    lr = LinearRegression().fit(dX_s[row_ok], dY[row_ok])

    # Δ de hoy
    X_last = X.iloc[-1][dX.columns]
    X_prev = X.iloc[-2][dX.columns]
    dx_today = (X_last - X_prev).fillna(med).to_numpy().reshape(1, -1)
    dx_today_s = ss.transform(dx_today)

    betas = lr.coef_.ravel()
    dz = dx_today_s.ravel()

    contrib = [(col, float(b)*float(z)) for col, b, z in zip(dX.columns, betas, dz) if np.isfinite(b) and np.isfinite(z)]
    contrib_df = pd.DataFrame(contrib, columns=["feature","contrib"]).sort_values("contrib", ascending=False)

    delta_pred = float(np.sum([c for _, c in contrib])) if contrib else np.nan
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
