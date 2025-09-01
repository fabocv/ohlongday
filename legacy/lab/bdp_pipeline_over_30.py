
# -*- coding: utf-8 -*-
"""
BDP Baseline Pipeline (EMA + Predicción + Importancias)
Fecha: 2025-08-24

Descripción:
- Limpia y transforma tu tabla BDP (columnas en español).
- Aplica EMA (alpha=0.30, lookback≈14, warm-up≥7–10).
- Genera features temporales (lags, rolling) y derivadas (sleep_debt, stimulant_load, etc).
- Entrena modelos base (ElasticNetCV y RandomForestRegressor) con validación temporal.
- Calcula importancias por permutación y exporta métricas.
- Incluye función "drivers_del_dia" para explicar variaciones día a día respecto al EMA y al día anterior.

Requisitos:
- pandas, numpy, scikit-learn, scipy (opcional), matplotlib (opcional)
"""

import os
import sys
import math
import re
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")



def ensure_unique_column_names(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Garantiza que los nombres de columnas sean únicos.
    - Si hay duplicadas, conserva la PRIMERA y descarta las posteriores (para estabilidad del pipeline).
    - Informa cuáles se descartaron.
    """
    dup_mask = df.columns.duplicated(keep="first")
    if dup_mask.any():
        dups = df.columns[dup_mask]
        if verbose:
            print(f"[AVISO] Columnas duplicadas detectadas y descartadas (se conserva la primera): {list(dups)}")
        df = df.loc[:, ~dup_mask].copy()
    return df

# -----------------------------
# Configuración base
# -----------------------------
EMA_ALPHA = 0.30
EMA_LOOKBACK = 14
EMA_MIN_PERIODS = 7  # warm-up

# Columnas esperadas (no es obligatorio tener todas)
EXPECTED_COLS = [
    'animo', 'correo', 'activacion', 'conexion',
       'proposito', 'claridad', 'estres', 'sueno_calidad', 'hora_dormir',
       'hora_despertar', 'despertares_nocturnos', 'cafe_ultima_hora',
       'alcohol_ultima_hora', 'exposicion_sol_manana_min', 'mov_intensidad',
       'interacciones_calidad', 'tiempo_pantalla_noche_min',
       'tiempo_ejercicio', 'glicemia', 'autocuidado', 'ansiedad',
       'alimentacion', 'siesta_min', 'movimiento', 'irritabilidad',
       'meditacion_min', 'agua_litros', 'cafe_cucharaditas', 'alcohol_ud',
       'otras_sustancias', 'medicacion_tipo', 'medicacion_tomada',
       'exposicion_sol_min', 'dolor_fisico', 'eventos_estresores', 'tags',
       'notas', 'tiempo_pantallas', 'interacciones_significativas', 'fecha',
       'hora'
]

# Objetivos principales
TARGETS = ["estres", "animo", "claridad"]

# -----------------------------
# Utilidades
# -----------------------------
def coerce_numeric(s: pd.Series) -> pd.Series:
    """Intenta convertir a float; valores no convertibles -> NaN."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s

def _normalize_time_token(tok: str) -> str:
    """
    Normaliza horas con formatos variados a algo que pandas pueda entender:
    - Acepta "9", "9:3", "09:30", "09:30:05", "9.30", "09-30", "0930", "093005"
    - Acepta sufijos am/pm o "a. m."/"p. m." en español.
    Retorna string "HH:MM:SS" o el original si no se puede.
    """
    if tok is None:
        return ""
    s = str(tok).strip().lower()
    if s in ("", "nan", "none"):
        return ""
    # reemplaza separadores inusuales por ":"
    s = s.replace("a. m.", "am").replace("p. m.", "pm")
    s = s.replace("-", ":").replace(".", ":").replace(";", ":").replace(",", ":")
    s = re.sub(r'\s+', '', s)

    # Si viene en formato "0930" o "093005"
    if re.fullmatch(r'\d{3,6}', s) and not any(ch in s for ch in [":","am","pm"]):
        if len(s) == 3:   # HMM -> H:MM
            s = f"{s[0]}:{s[1:]}"
        elif len(s) == 4: # HHMM -> HH:MM
            s = f"{s[:2]}:{s[2:]}"
        elif len(s) == 5: # HMMSS -> H:MM:SS
            s = f"{s[0]}:{s[1:3]}:{s[3:]}"
        elif len(s) == 6: # HHMMSS -> HH:MM:SS
            s = f"{s[:2]}:{s[2:4]}:{s[4:]}"

    # Si trae am/pm, deja que pandas lo resuelva más adelante al concatenar con la fecha
    # Pad a HH:MM:SS si solo HH o HH:MM (sin am/pm)
    if "am" not in s and "pm" not in s:
        parts = s.split(":")
        if len(parts) == 1 and parts[0].isdigit():
            s = f"{int(parts[0]):02d}:00:00"
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            s = f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
        elif len(parts) == 3 and all(p.isdigit() for p in parts):
            s = f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return s

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columna 'dt' robusta a formatos variados de 'hora'."""
    df = df.copy()
    # Normaliza fecha (permite formatos latam)
    fecha = pd.to_datetime(df.get("fecha"), errors="coerce", dayfirst=True)
    hora_raw = df.get("hora")

    if hora_raw is not None:
        hora_norm = hora_raw.apply(_normalize_time_token)
        # Concatena fecha + hora (maneja am/pm y casos HH:MM:SS)
        combo = fecha.astype("string").fillna("") + " " + hora_norm.astype("string").fillna("")
        dt = pd.to_datetime(combo.str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        dt = fecha

    # Fallback: si sigue NaT, usa solo fecha
    fallback = pd.to_datetime(df.get("fecha"), errors="coerce", dayfirst=True)
    dt = dt.fillna(fallback)

    df["dt"] = dt
    # Ordena por tiempo y elimina duplicados conservando el último
    df = df.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)
    return df

def impute_last_valid(series: pd.Series, maxgap: int = 3) -> pd.Series:
    """
    Imputa por arrastre del último valor válido hasta 'maxgap' días.
    Si hay huecos mayores, deja NaN.
    """
    s = series.copy()
    # forward fill
    fwd = s.ffill()
    # calcular tamaño de huecos
    na_groups = s.isna().astype(int).groupby((s.notna()).cumsum()).cumcount()
    # si el hueco excede maxgap, mantenemos NaN en esas posiciones
    s_imputed = fwd.where(na_groups <= maxgap, np.nan)
    return s_imputed

def winsorize_deltas(s: pd.Series, lower_q=0.02, upper_q=0.98) -> pd.Series:
    """Winsoriza las diferencias día a día para atenuar outliers y reconstituye la serie suavizada."""
    if s.isna().all():
        return s
    x = s.copy()
    dx = x.diff()
    lo = dx.quantile(lower_q)
    hi = dx.quantile(upper_q)
    dx_w = dx.clip(lower=lo, upper=hi)
    # recrea serie a partir del primer valor observado
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
    """EMA con warm-up: retorna NaN hasta tener min_periods válidos previos."""
    s = series.copy()
    e = s.ewm(alpha=alpha, adjust=False).mean()
    # Warm-up: requiere min_periods válidos anteriores
    valid_counts = s.notna().rolling(window=min_periods, min_periods=1).sum()
    e = e.where(valid_counts >= min_periods)
    return e

def coalesce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convierte en lo posible las columnas a numéricas."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = coerce_numeric(out[c])
    return out

def count_events(text) -> float:
    """Cuenta eventos en un string separando por comas, punto y coma o pipes."""
    if pd.isna(text):
        return np.nan
    if not isinstance(text, str):
        text = str(text)
    tokens = [t.strip() for sep in [",", ";", "|", "/"] for t in text.split(sep)]
    tokens = [t for t in tokens if len(t) > 0]
    return float(len(tokens)) if tokens else np.nan

# -----------------------------
# Feature Engineering
# -----------------------------
DERIVED_NUMERIC = [
    "sueno_calidad", "horas_sueno", "siesta_min",
    "agua_litros", "cafe_cucharaditas", "alcohol_ud",
    "meditacion_min", "exposicion_sol_min", "movimiento",
    "autocuidado", "interacciones_significativas", "glicemia"
]

def add_basic_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas numéricas conocidas y crea recuentos de eventos."""
    df = df.copy()
    df = coalesce_numeric_columns(df, DERIVED_NUMERIC + [
        "animo","activacion","conexion","proposito","claridad",
        "estres","dolor_fisico","ansiedad","irritabilidad",
        "medicacion_tomada"
    ])
    # eventos_estresores -> recuento
    if "eventos_estresores" in df.columns:
        df["eventos_count"] = df["eventos_estresores"].apply(count_events)
    else:
        df["eventos_count"] = np.nan

    # otras_sustancias -> recuento (texto libre separado por comas/;/|)
    if "otras_sustancias" in df.columns:
        df["otras_sustancias_count"] = df["otras_sustancias"].apply(count_events)
    else:
        df["otras_sustancias_count"] = np.nan

    # tags: número de tags
    if "tags" in df.columns:
        df["tags_count"] = df["tags"].apply(count_events)
    else:
        df["tags_count"] = np.nan

    # medicacion_tomada: normalizar a 0/1
    if "medicacion_tomada" in df.columns:
        # cualquier no-NaN y >0 -> 1
        df["medicacion_tomada"] = (df["medicacion_tomada"].fillna(0).astype(float) > 0).astype(int)

    return df

def _clip_scales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "alimentacion" in df.columns:
        df["alimentacion"] = pd.to_numeric(df["alimentacion"], errors="coerce")
        df["alimentacion"] = df["alimentacion"].clip(lower=0, upper=10)
    return df


def add_ema_and_winsor(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Aplica winsorización de deltas + EMA a columnas numéricas."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(float)
        s_imp = impute_last_valid(s, maxgap=3)
        s_w = winsorize_deltas(s_imp)
        df[f"{c}_ema"] = ema(s_w, alpha=EMA_ALPHA, min_periods=EMA_MIN_PERIODS)
    return df

def add_lags_rolls(df: pd.DataFrame, cols: List[str], lags=(1,3), roll_win=3) -> pd.DataFrame:
    """Genera lags t-1, t-3 y rolling mean(3)."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
        df[f"{c}_roll{roll_win}"] = df[c].rolling(roll_win, min_periods=1).mean()
    return df

def add_derived_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas y algunas interacciones útiles."""
    df = df.copy()
    # sleep debt (tope inferior 0)
    if "horas_sueno" in df.columns:
        df["sleep_debt"] = (8.0 - df["horas_sueno"].astype(float)).clip(lower=0)
    else:
        df["sleep_debt"] = np.nan

    # stimulant load
    cafe = df["cafe_cucharaditas"].astype(float) if "cafe_cucharaditas" in df.columns else 0.0
    alcohol = df["alcohol_ud"].astype(float) if "alcohol_ud" in df.columns else 0.0
    df["stimulant_load"] = cafe + 0.5 * alcohol

    # interacciones
    if "estres" in df.columns:
        df["estres_x_sleep_debt"] = df["estres"].astype(float) * df["sleep_debt"]
    else:
        df["estres_x_sleep_debt"] = np.nan

    if "horas_sueno" in df.columns and "cafe_cucharaditas" in df.columns:
        df["cafe_x_sleep_deficit"] = df["cafe_cucharaditas"].astype(float) * (df["sleep_debt"] > 0).astype(int)
    else:
        df["cafe_x_sleep_deficit"] = np.nan

    if "movimiento" in df.columns and "dolor_fisico" in df.columns:
        df["mov_x_dolor_lag1"] = df["movimiento"].astype(float) * df["dolor_fisico"].shift(1).astype(float)
    else:
        df["mov_x_dolor_lag1"] = np.nan

    return df

def add_calendar_context(df: pd.DataFrame) -> pd.DataFrame:
    """Añade contexto de calendario: día de semana, mes, fin de semana."""
    df = df.copy()
    df["dow"] = df["dt"].dt.dayofweek  # 0=Monday
    df["month"] = df["dt"].dt.month
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    return df

def add_categoricals(df: pd.DataFrame, max_unique=15) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepara columnas categóricas de baja cardinalidad para One-Hot.
    NOTA: 'alimentacion' se fuerza a NUMÉRICA (0–10) y 'otras_sustancias' es TEXTO libre, no va a One-Hot.
    Solo se considera 'medicacion_tipo' si su cardinalidad es baja.
    """
    df = df.copy()
    cat_cols = []
    for c in ["medicacion_tipo"]:
        if c in df.columns:
            nun = df[c].dropna().astype(str).nunique()
            if 1 <= nun <= max_unique:
                cat_cols.append(c)
    return df, cat_cols

# -----------------------------
# Modelado y evaluación temporal
# -----------------------------
@dataclass
class ModelResult:
    target: str
    model_name: str
    mae: float
    rmse: float
    n_train: int
    n_test: int

def make_feature_set(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Construye X, y y listas de columnas numéricas/categóricas."""
    y = df[target].astype(float)

    # Candidatas numéricas (todas menos texto obvio)
    exclude_cols = set(["fecha","hora","dt","notas","eventos_estresores","tags","medicacion_tipo","otras_sustancias"] + TARGETS)
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Categóricas
    _, cat_cols = add_categoricals(df)

    # Mantén sólo filas con y no nula
    valid_idx = y.dropna().index
    X = df.loc[valid_idx, numeric_cols + cat_cols].copy()
    y = y.loc[valid_idx]

    # Elimina columnas completamente vacías (todo NaN), ya que el Imputer no puede calcular medianas ahí
    nunq_non_na = X.apply(lambda s: s.dropna().shape[0])
    keep_cols = nunq_non_na[nunq_non_na > 0].index.tolist()
    X = X[keep_cols]

    return X, y, numeric_cols, cat_cols

def time_series_splits(n: int, n_splits: int = 5, min_train: int = 10, test_size: int = 3):
    """
    Genera cortes de validación temporal (ventanas crecientes).
    """
    starts = []
    train_starts = 0
    while True:
        train_end = min_train + len(starts)*test_size
        test_end = train_end + test_size
        if test_end > n:
            break
        starts.append((0, train_end, train_end, test_end))
    if not starts:
        # Sin datos suficientes: un solo split 80/20
        train_end = int(n*0.8)
        return [(0, train_end, train_end, n)]
    return starts

def evaluate_models_timecv(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Entrena y evalúa ElasticNetCV y RandomForest con CV temporal.
    Retorna métricas combinadas y el mejor modelo entrenado sobre todo el set.
    """
    metrics = []
    models = {}

    numeric_cols = [c for c in X.columns if c not in set(cat_cols)]
    # Preprocesamiento con imputación
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=True, with_std=True))
    ])
    steps = [("num", num_pipe, numeric_cols)]
    if len(cat_cols) > 0:
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ])
        steps.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers=steps)

    # Definir pipelines
    pipe_en = Pipeline([
        ("pre", pre),
        ("en", ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=np.logspace(-3, 1, 20), max_iter=5000))
    ])
    pipe_rf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1))
    ])

    n = len(X)
    splits = time_series_splits(n, n_splits=5, min_train=10, test_size=3)

    for name, pipe in [("ElasticNetCV", pipe_en), ("RandomForest", pipe_rf)]:
        fold_results = []
        for (tr0, tr1, te0, te1) in splits:
            Xtr, Xte = X.iloc[tr0:tr1], X.iloc[te0:te1]
            ytr, yte = y.iloc[tr0:tr1], y.iloc[te0:te1]

            if len(Xtr) < 10 or len(Xte) < 3:
                continue

            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            mae = mean_absolute_error(yte, pred)
            rmse = mean_squared_error(yte, pred, squared=False)
            fold_results.append((mae, rmse))

        if fold_results:
            mae_mean = float(np.mean([m for m, _ in fold_results]))
            rmse_mean = float(np.mean([r for _, r in fold_results]))
            metrics.append({
                "model": name, "mae": mae_mean, "rmse": rmse_mean,
                "n_splits": len(fold_results), "n_samples": n
            })
        else:
            metrics.append({
                "model": name, "mae": np.nan, "rmse": np.nan,
                "n_splits": 0, "n_samples": n
            })

        # Entrena en todo el dataset (para importancias/permutaciones posteriores)
        pipe.fit(X, y)
        models[name] = pipe

    metrics_df = pd.DataFrame(metrics).sort_values("mae")
    # Selecciona mejor por MAE
    best_name = metrics_df.iloc[0]["model"]
    best_model = models[best_name]

    return metrics_df, { "best_name": best_name, "best_model": best_model, "all_models": models }

def compute_permutation_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Importancia por permutación en el pipeline entrenado (soporta ausencia de categóricas)."""
    pre = model.named_steps["pre"]
    # Recupera nombres originales por transformador
    numeric_names = []
    cat_out_names = []

    # ColumnTransformer guarda (name, transformer, columns) en .transformers_
    for name, trans, cols in pre.transformers_:
        if name == "num":
            numeric_names = list(cols)
        elif name == "cat":
            # 'trans' es un Pipeline con OneHot en el paso 'oh'
            try:
                ohe = trans.named_steps.get("oh", None)
                if ohe is not None:
                    cat_out_names = list(ohe.get_feature_names_out(cols))
                else:
                    # Si no hay 'oh', usa nombres originales
                    cat_out_names = list(cols)
            except Exception:
                cat_out_names = list(cols)

    feature_names = list(numeric_names) + list(cat_out_names)

    # Ejecuta permutación sobre el pipeline completo
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    # Alinear longitud por si feature_names está vacío (poco probable, pero seguro)
    k = result.importances_mean.shape[0]
    if len(feature_names) != k:
        # Si difiere, crear nombres genéricos
        feature_names = [f"f{i}" for i in range(k)]

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    return imp_df

# -----------------------------
# Drivers del día
# -----------------------------
def drivers_del_dia(df: pd.DataFrame, target: str, model: Pipeline, top_k: int = 5) -> Dict[str, object]:
    """
    Explica la variación del día más reciente vs (ayer y EMA) usando importancias + signo de cambio.
    Retorna un dict con texto y tablas.
    """
    X, y, num_cols, cat_cols = make_feature_set(df, target)
    if len(X) < 3:
        return {"texto": "No hay suficientes datos para generar explicación.", "tabla": None}

    imp_df = compute_permutation_importance(model, X, y, n_repeats=8)
    top_features = imp_df.head(20)["feature"].tolist()

    # Predicciones para las dos últimas fechas
    yhat = model.predict(X)
    last_idx = X.index[-1]
    prev_idx = X.index[-2]

    delta_pred = yhat[-1] - yhat[-2]
    delta_obs = (y.loc[last_idx] - y.loc[prev_idx]) if (last_idx in y.index and prev_idx in y.index) else np.nan

    # Construye contribuciones heurísticas: signo del cambio en cada feature top
    # (esto NO es SHAP; es una heurística orientativa usando deltas de features y peso de importancia)
    X_last = X.iloc[-1]
    X_prev = X.iloc[-2]
    contribs = []
    for feat in top_features:
        if feat in X.columns:
            d = X_last[feat] - X_prev[feat]
            w = imp_df.loc[imp_df["feature"] == feat, "importance_mean"].values[0]
            contribs.append((feat, d, w, d * w))
    contribs_df = pd.DataFrame(contribs, columns=["feature", "delta_feature", "weight", "score"]).sort_values("score", ascending=False)
    top_pos = contribs_df.head(top_k)
    top_neg = contribs_df.tail(top_k).sort_values("score")

    texto = (
        f"{target.upper()}: Predicción Δ≈{delta_pred:.2f} (obs Δ≈{delta_obs if not math.isnan(delta_obs) else np.nan}). "
        f"Posibles impulsores ↑: {', '.join(top_pos['feature'].tolist())}. "
        f"Frenos ↓: {', '.join(top_neg['feature'].tolist())}."
    )
    return {"texto": texto, "tabla": contribs_df}

# -----------------------------
# Flujo principal
# -----------------------------
def build_and_evaluate(df: pd.DataFrame, export_dir: str = "./outputs") -> Dict[str, object]:
    os.makedirs(export_dir, exist_ok=True)

    # Asegura nombres de columnas únicos
    df = ensure_unique_column_names(df)

    # 1) Tiempo
    df = parse_datetime(df)

    # 2) Numéricos + conteos
    df = add_basic_numeric(df)

    
    # Escala de 'alimentacion' restringida a [0,10]
    df = _clip_scales(df)
# 3) EMA (para métricas continuas y mediadores)
    ema_cols = list(set(DERIVED_NUMERIC + TARGETS + ["dolor_fisico","ansiedad","irritabilidad","glicemia"]))
    df = add_ema_and_winsor(df, ema_cols)

    # 4) Lags/Roll
    lag_cols = list(set(DERIVED_NUMERIC + TARGETS + ["dolor_fisico","ansiedad","irritabilidad","glicemia"]))
    df = add_lags_rolls(df, lag_cols, lags=(1,3), roll_win=3)

    # 5) Derivadas e interacciones
    df = add_derived_interactions(df)

    # 6) Calendario
    df = add_calendar_context(df)

    # 7) Modelado por target
    summary_rows = []
    best_models = {}

    for target in TARGETS:
        if target not in df.columns:
            continue
        X, y, num_cols, cat_cols = make_feature_set(df, target)
        if len(X) < 50:
            print(f"[AVISO] Muy pocos datos para {target}: {len(X)} filas. Se entrenará igual, pero la validación puede ser débil.")
        metrics_df, model_pack = evaluate_models_timecv(X, y, df, cat_cols)
        metrics_df["target"] = target
        metrics_df.to_csv(os.path.join(export_dir, f"{target}_cv_metrics.csv"), index=False, encoding="utf-8")

        # Importancias del mejor modelo
        best_model = model_pack["best_model"]
        best_name = model_pack["best_name"]
        imp_df = compute_permutation_importance(best_model, X, y, n_repeats=10)
        imp_df.to_csv(os.path.join(export_dir, f"{target}_feature_importance.csv"), index=False, encoding="utf-8")

        # Guarda resumen
        row = metrics_df.sort_values("mae").iloc[0].to_dict()
        row["target"] = target
        row["best_model"] = best_name
        summary_rows.append(row)

        best_models[target] = best_model

        # Drivers del día (si hay suficientes datos)
        expl = drivers_del_dia(df, target, best_model, top_k=5)
        if isinstance(expl.get("tabla"), pd.DataFrame):
            expl["tabla"].to_csv(os.path.join(export_dir, f"{target}_drivers_del_dia.csv"), index=False, encoding="utf-8")
        with open(os.path.join(export_dir, f"{target}_drivers_del_dia.txt"), "w", encoding="utf-8") as f:
            f.write(expl.get("texto","(sin texto)"))

    # 8) Exporta resumen general
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["target","mae"])
        summary_df.to_csv(os.path.join(export_dir, "resumen_modelos.csv"), index=False, encoding="utf-8")
    else:
        summary_df = pd.DataFrame()

    return {"df": df, "summary": summary_df, "best_models": best_models}

def load_csv_or_template(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8")
        df = ensure_unique_column_names(df)
    return df
    # Si no existe, crea un template vacío con columnas esperadas
    print(f"[INFO] No se encontró {path}. Se generará un template con columnas esperadas.")
    df = pd.DataFrame(columns=EXPECTED_COLS)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./bdp.csv", help="Ruta al CSV con tus datos BDP")
    parser.add_argument("--out", type=str, default="./outputs", help="Directorio de salida")
    args = parser.parse_args()

    df = load_csv_or_template(args.csv)
    results = build_and_evaluate(df, export_dir=args.out)

    # Guardo el dataframe enriquecido por si se quiere inspeccionar
    enr_path = os.path.join(args.out, "bdp_enriquecido.csv")
    results["df"].to_csv(enr_path, index=False, encoding="utf-8")
    print(f"[OK] Pipeline completo. Archivos en: {os.path.abspath(args.out)}")
