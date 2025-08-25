
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

HUMAN_LABELS = [
    ("horas_sueno", "horas de sueño"),
    ("sueno_calidad_calc", "calidad de sueño (calculada)"),
    ("sleep_debt", "deuda de sueño"),
    ("sleep_efficiency", "eficiencia de sueño"),
    ("circadian_alignment", "alineación circadiana"),
    ("cafe_", "café"),
    ("alcohol_", "alcohol"),
    ("stimulant_load", "carga de estimulantes"),
    ("hydration_score", "hidratación"),
    ("agua_litros", "agua"),
    ("alimentacion", "alimentación"),
    ("movement_score", "movimiento"),
    ("mov_", "movimiento"),
    ("relaxation_score", "respiración/meditación"),
    ("meditacion_", "respiración/meditación"),
    ("morning_light_score", "luz de mañana"),
    ("exposicion_sol_", "exposición al sol"),
    ("screen_night_score", "pantalla de noche (menor es mejor)"),
    ("social_", "interacciones"),
    ("stressors_", "estresores"),
    ("has_", "otras sustancias"),
    ("adherencia_med", "adherencia a medicación"),
]

def _human_label(feat: str) -> str:
    for pref, lab in HUMAN_LABELS:
        if feat.startswith(pref):
            return lab
    return feat

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
    # quitar columnas con mediana NaN
    med = dX.median()
    dX = dX.loc[:, ~med.isna().values]
    if dX.empty:
        return dX
    # quitar columnas de varianza cero
    var = dX.var()
    dX = dX.loc[:, var > 0]
    return dX

def _interpretacion_humanista(target: str, contrib_df: pd.DataFrame, delta_obs: float, delta_pred: float) -> str:
    # Selección de top 2 en cada sentido
    top_pos = contrib_df.sort_values("contrib", ascending=False).head(2)
    top_neg = contrib_df.sort_values("contrib", ascending=True).head(2)

    # Etiquetas humanas
    pos_labels = [_human_label(f) for f in top_pos["feature"].tolist()]
    neg_labels = [_human_label(f) for f in top_neg["feature"].tolist()]

    # Sentido del día
    def signo(x):
        if x != x:  # NaN
            return "estable"
        if x > 0.4: return "al alza"
        if x < -0.4: return "a la baja"
        return "estable"

    estado = signo(delta_obs)
    # Mensajes por target
    if target == "animo":
        base = f"Ánimo {estado}. "
        pos = f"Apoyaron: {', '.join(pos_labels)}. " if pos_labels else ""
        neg = f"Restaron: {', '.join(neg_labels)}. " if neg_labels else ""
        guia = "Para estabilizar: luz AM 10–15′, cortar café ≥6h antes de dormir, 5–10′ de respiración."
    elif target == "claridad":
        base = f"Claridad {estado}. "
        pos = f"Favorecieron: {', '.join(pos_labels)}. " if pos_labels else ""
        neg = f"Dificultaron: {', '.join(neg_labels)}. " if neg_labels else ""
        guia = "Enfoca higiene circadiana: sueño suficiente, luz de mañana y menos pantalla de noche."
    elif target == "estres":
        base = f"Estrés {estado}. "
        pos = f"Redujeron: {', '.join(pos_labels)}. " if pos_labels else ""
        neg = f"Elevadores: {', '.join(neg_labels)}. " if neg_labels else ""
        guia = "Si sube, prueba pausa breve + respiración 4-6 y delimita un siguiente paso acotado."
    elif target == "activacion":
        base = f"Activación {estado}. "
        pos = f"Impulsaron: {', '.join(pos_labels)}. " if pos_labels else ""
        neg = f"Atenuaron: {', '.join(neg_labels)}. " if neg_labels else ""
        guia = "Para afinar: algo de luz/movimiento temprano; evita alcohol tarde y acumular deuda de sueño."
    else:
        base = f"{target.capitalize()} {estado}. "
        pos = f"Positivos: {', '.join(pos_labels)}. " if pos_labels else ""
        neg = f"Negativos: {', '.join(neg_labels)}. "
        guia = ""

    # Nota de neutralidad si Δpred es chico
    neutral = ""
    if delta_pred == delta_pred and abs(delta_pred) < 0.3:
        neutral = " Día con cambios leves; confía en la EMA14 para referencia. "

    return base + pos + neg + neutral + guia

def drivers_del_dia(df: pd.DataFrame, target: str, top_k: int = 3, row_idx: int = -1) -> Dict:
    """
    Explica el cambio del día seleccionado (Δy) con ΔX:
      - Entrena Δy ~ ΔX con todo el historial disponible **hasta row_idx-1**.
      - row_idx: índice de fila del día a explicar (por defecto -1: último día).
      - Devuelve contribuciones y un texto humanista.
    """
    if target not in df.columns or len(df) < 4:
        return {"texto":"Sin suficientes datos.","tabla":pd.DataFrame()}

    # Resolver row_idx (soporta negativos)
    n = len(df)
    if row_idx < 0:
        row_idx = n + row_idx
    # Necesitamos al menos una delta previa y al menos 2 filas previas para entrenar
    if row_idx <= 1 or row_idx >= n:
        return {"texto":"row_idx fuera de rango para calcular Δ.", "tabla": pd.DataFrame()}

    y = pd.to_numeric(df[target], errors="coerce")
    X = _features_for_target(df, target)
    if X.empty:
        return {"texto":"Sin features numéricos suficientes.","tabla":pd.DataFrame()}

    # Deltas para entrenamiento: desde 1..row_idx-1
    dY = y.diff().iloc[1:row_idx]
    dX = X.diff().iloc[1:row_idx]
    dX = _clean_deltas(dX)
    if dX.empty or dY.dropna().empty:
        return {"texto":"Insuficiente señal tras limpiar deltas.","tabla":pd.DataFrame()}

    # Imputación y estandarización
    med = dX.median()
    dX_f = dX.fillna(med)
    ss = StandardScaler()
    dX_s = ss.fit_transform(dX_f)

    row_ok = np.isfinite(dX_s).all(axis=1) & (~dY.isna().values)
    if not row_ok.any():
        return {"texto":"Insuficiente señal tras limpiar deltas.","tabla":pd.DataFrame()}

    lr = LinearRegression().fit(dX_s[row_ok], dY[row_ok])

    # Δ del día a explicar: row_idx - (row_idx-1)
    X_today = X.iloc[row_idx][dX.columns]
    X_prev  = X.iloc[row_idx-1][dX.columns]
    dx = (X_today - X_prev).fillna(med).to_numpy().reshape(1, -1)
    dx_s = ss.transform(dx)

    betas = lr.coef_.ravel()
    dz = dx_s.ravel()

    contrib = [(col, float(b)*float(z)) for col, b, z in zip(dX.columns, betas, dz) if np.isfinite(b) and np.isfinite(z)]
    contrib_df = pd.DataFrame(contrib, columns=["feature","contrib"]).sort_values("contrib", ascending=False)

    delta_pred = float(np.sum([c for _, c in contrib])) if contrib else np.nan
    delta_obs = float(y.iloc[row_idx] - y.iloc[row_idx-1]) if (pd.notna(y.iloc[row_idx]) and pd.notna(y.iloc[row_idx-1])) else np.nan

    # Texto estándar + interpretación humanista
    top_pos = contrib_df.head(top_k)
    top_neg = contrib_df.tail(top_k).sort_values("contrib")
    texto = (
        f"{target.capitalize()}: Δobs={np.round(delta_obs,2) if delta_obs==delta_obs else 'NaN'}; "
        f"Δpred≈{np.round(delta_pred,2) if delta_pred==delta_pred else 'NaN'} | "
        f"↑ {', '.join(top_pos['feature'].tolist()) if len(top_pos) else '—'} | "
        f"↓ {', '.join(top_neg['feature'].tolist()) if len(top_neg) else '—'}"
    )
    narrativa = _interpretacion_humanista(target, contrib_df, delta_obs, delta_pred)

    return {
        "texto": texto,
        "narrativa": narrativa,
        "tabla": contrib_df,
        "top_pos": top_pos['feature'].tolist(),
        "top_neg": top_neg['feature'].tolist(),
        "delta_pred": delta_pred,
        "delta_obs": delta_obs,
        "row_idx": row_idx,
    }
