# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression

def _features_for_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        if c == target:
            continue
        if c in ("animo","claridad","estres","activacion"):
            continue
        # tomar drivers modifiables + lags/ema
        if c.endswith(("_lag1","_lag3","_roll3","_ema")) or any(c.startswith(p) for p in [
            "horas_sueno","sueno_calidad_calc","sleep_debt","sleep_efficiency","circadian_alignment",
            "cafe_","alcohol_","stimulant_load","hydration_score","agua_litros","alimentacion",
            "movement_score","relaxation_score","morning_light_score","screen_night_score","exposicion_sol_","meditacion_","mov_",
            "social_","stressors_","estres_",
            "has_","adherencia_med",
        ]):
            cols.append(c)
    X = df[cols].copy()
    # drop all-NaN
    keep = [c for c in cols if not X[c].isna().all()]
    return X[keep]

def drivers_del_dia(df: pd.DataFrame, target: str, top_k: int = 3) -> Dict:
    """
    Entrena un modelo lineal sobre features numéricas estandarizadas hasta t-1
    y calcula contribuciones aproximadas en t como:
        contrib_i = beta_i * (z_t_i - z_{t-1,i})
    donde z son features estandarizadas (imputadas por mediana).
    Devuelve top_k positivos y negativos con magnitud.
    """
    if target not in df.columns or len(df) < 3:
        return {"texto":"Sin suficientes datos.","tabla":pd.DataFrame()}
    X = _features_for_target(df, target)
    y = pd.to_numeric(df[target], errors="coerce")

    # Train up to t-1, explain delta at t
    Xtr, ytr = X.iloc[:-1], y.iloc[:-1]
    X_last = X.iloc[-1]
    X_prev = X.iloc[-2]

    # Keep only numeric columns for contributions
    num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    if not num_cols:
        return {"texto":"Sin features numéricos suficientes.","tabla":pd.DataFrame()}

    Ztr = Xtr[num_cols].apply(pd.to_numeric, errors="coerce")
    # Median imputation on train and apply to last/prev
    med = Ztr.median()
    Ztr_f = Ztr.fillna(med)

    # Standardize
    ss = StandardScaler()
    Ztr_s = ss.fit_transform(Ztr_f)

    # Fit linear regression for interpretability
    lr = LinearRegression().fit(Ztr_s, ytr)

    # Prepare last and prev in same pipeline
    z_last = pd.to_numeric(X_last[num_cols], errors="coerce").fillna(med).to_numpy().reshape(1, -1)
    z_prev = pd.to_numeric(X_prev[num_cols], errors="coerce").fillna(med).to_numpy().reshape(1, -1)
    z_last_s = ss.transform(z_last)
    z_prev_s = ss.transform(z_prev)

    delta_z = (z_last_s - z_prev_s).ravel()
    betas = lr.coef_.ravel()

    contrib = []
    for i, c in enumerate(num_cols):
        dz = float(delta_z[i])
        b = float(betas[i])
        if not (np.isnan(dz) or np.isnan(b)):
            contrib.append((c, b * dz))

    contrib_df = pd.DataFrame(contrib, columns=["feature","contrib"]).sort_values("contrib", ascending=False)

    top_pos = contrib_df.head(top_k)
    top_neg = contrib_df.tail(top_k).sort_values("contrib")

    delta_obs = (y.iloc[-1] - y.iloc[-2]) if (pd.notna(y.iloc[-1]) and pd.notna(y.iloc[-2])) else np.nan
    delta_pred = float(contrib_df["contrib"].sum()) if not contrib_df.empty else np.nan

    texto = (
        f"{target.capitalize()}: Δobs={np.round(delta_obs,2) if delta_obs==delta_obs else 'NaN'}; "
        f"Δpred≈{np.round(delta_pred,2) if delta_pred==delta_pred else 'NaN'} | "
        f"↑ {', '.join(top_pos['feature'].tolist()) if len(top_pos) else '—'} | "
        f"↓ {', '.join(top_neg['feature'].tolist()) if len(top_neg) else '—'}"
    )

    return {"texto": texto, "tabla": contrib_df, "top_pos": top_pos['feature'].tolist(), "top_neg": top_neg['feature'].tolist(), "delta_pred": delta_pred, "delta_obs": float(delta_obs) if delta_obs==delta_obs else np.nan}
