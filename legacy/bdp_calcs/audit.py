
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODIFIABLE_PREFIXES = [
    # Sueño
    "horas_sueno","sueno_calidad_calc","sleep_debt","sleep_efficiency","circadian_alignment",
    # Estimulantes/consumo
    "cafe_","alcohol_","stimulant_load","hydration_score","agua_litros","alimentacion",
    # Actividad/regulación/luz/pantalla
    "movement_score","relaxation_score","morning_light_score","screen_night_score","exposicion_sol_","meditacion_","mov_",
    # Psicosocial
    "social_","stressors_","estres_",
    # Sustancias y medicación
    "has_","adherencia_med",
]

def _as_series(df: pd.DataFrame, maybe_col: str) -> pd.Series:
    s = df.get(maybe_col)
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series([np.nan]*len(df), index=df.index, dtype=float)

def _is_modifiable(col: str) -> bool:
    return any(col.startswith(p) for p in MODIFIABLE_PREFIXES)

def _time_series_splits(n:int, n_splits:int=3, min_train:int=10, test_size:int=3):
    """Yield (tr0,tr1,te0,te1) indices for rolling-origin validation."""
    if n < min_train + test_size:
        return []
    splits = []
    start = 0
    while start + min_train + test_size <= n and len(splits) < n_splits:
        tr0, tr1 = 0, start + min_train
        te0, te1 = tr1, tr1 + test_size
        if te1 <= n:
            splits.append((tr0,tr1,te0,te1))
        start += test_size
    return splits

def _coverage(y: pd.Series) -> float:
    return float(y.notna().mean()) if len(y) else np.nan

def _reliability(y: pd.Series) -> Tuple[float,float]:
    """Rel (1 - MAD(Δ)/IQR) y AC1."""
    s = pd.to_numeric(y, errors="coerce")
    d = s.diff()
    mad = np.nanmedian(np.abs(d - np.nanmedian(d)))
    q75, q25 = np.nanpercentile(s, 75), np.nanpercentile(s, 25)
    iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else np.nan
    rel = np.nan
    if iqr and iqr > 0:
        rel = float(np.clip(1.0 - (mad / iqr), 0.0, 1.0))
    # AC1 autocorrelación lag-1
    ac1 = np.nan
    if s.notna().sum() >= 3:
        s0 = s.shift(1)
        mask = s.notna() & s0.notna()
        if mask.any():
            ac1 = float(np.corrcoef(s[mask], s0[mask])[0,1])
    return rel, ac1

def _features_for_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    # Candidate features: all numeric/categorical except the core targets
    y = pd.to_numeric(df[target], errors="coerce")
    # Choose modifiable drivers + their lags/ema if present
    cols = []
    for c in df.columns:
        if c == target: 
            continue
        if c in ("animo","claridad","estres","activacion"):
            continue
        if _is_modifiable(c) or c.endswith(("_lag1","_lag3","_roll3","_ema")):
            cols.append(c)
    X = df[cols].copy()
    # Build numeric/categorical sets
    num_cols, cat_cols = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    # Drop all-NaN columns
    keep = [c for c in cols if not X[c].isna().all()]
    X = X[keep]
    num_cols = [c for c in num_cols if c in keep]
    cat_cols = [c for c in cat_cols if c in keep]
    return X, y, num_cols, cat_cols

def _build_pipe(num_cols: List[str], cat_cols: List[str]):
    steps = []
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    steps.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
        steps.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(steps)
    model = ElasticNetCV(l1_ratio=[0.05,0.2,0.5,0.8,0.95,1.0], alphas=None, cv=3, n_jobs=None, random_state=42)
    pipe = Pipeline([("pre", pre), ("en", model)])
    return pipe

def core_audit(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """
    Devuelve una tabla con Coverage, Rel, AC1, MAE_model, MAE_baseline_naive, MAE_baseline_ema, Gain,
    Actionability% y Distinctiveness (1 - R2_parcial frente a otras core).
    """
    rows = []
    for target in targets:
        if target not in df.columns:
            continue
        X, y, num_cols, cat_cols = _features_for_target(df, target)
        n = len(df)
        # Coverage & reliability
        cov = _coverage(y)
        rel, ac1 = _reliability(y)

        # Temporal CV dynamic sizes
        min_train = max(5, int(n*0.6))
        test_size = max(2, int(n*0.2))
        splits = _time_series_splits(n, n_splits=3, min_train=min_train, test_size=test_size)

        # Baselines
        y_shift1 = y.shift(1)
        ema_col = f"{target}_ema"
        y_ema = pd.to_numeric(df.get(ema_col), errors="coerce") if ema_col in df.columns else pd.Series([np.nan]*n)

        maes_model, maes_naive, maes_ema, r2_parts = [], [], [], []

        if splits:
            for tr0,tr1,te0,te1 in splits:
                # Slices
                Xtr, Xte = X.iloc[tr0:tr1], X.iloc[te0:te1]
                ytr, yte = y.iloc[tr0:tr1], y.iloc[te0:te1]
                # Model
                pipe = _build_pipe(num_cols=[c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])],
                                   cat_cols=[c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])])
                try:
                    pipe.fit(Xtr, ytr)
                    pred = pipe.predict(Xte)
                    mae_m = mean_absolute_error(yte, pred)
                    maes_model.append(mae_m)
                except Exception:
                    pass
                # Naive
                pred_naive = y_shift1.iloc[te0:te1]
                mask_n = (~yte.isna()) & (~pred_naive.isna())
                if mask_n.any():
                    maes_naive.append(mean_absolute_error(yte[mask_n], pred_naive[mask_n]))
                # EMA
                pred_ema = y_ema.iloc[te0:te1]
                mask_e = (~yte.isna()) & (~pred_ema.isna())
                if mask_e.any():
                    maes_ema.append(mean_absolute_error(yte[mask_e], pred_ema[mask_e]))
        # Aggregation
        mae_model = float(np.mean(maes_model)) if maes_model else np.nan
        mae_naive = float(np.mean(maes_naive)) if maes_naive else np.nan
        mae_ema = float(np.mean(maes_ema)) if maes_ema else np.nan

        # Best baseline
        base_mae = np.nanmin([v for v in [mae_naive, mae_ema] if not np.isnan(v)]) if any(not np.isnan(v) for v in [mae_naive, mae_ema]) else np.nan
        gain = float(1 - (mae_model / base_mae)) if (base_mae and not np.isnan(base_mae) and mae_model==mae_model) else np.nan

        # Distinctiveness: R2_parcial frente a otras core (EMA/lag1 si existen)
        others = [t for t in targets if t != target and t in df.columns]
        Xd = []
        for t in others:
            s = _as_series(df, f"{t}_ema")
            if s.isna().all():
                s = _as_series(df, f"{t}_lag1")
            Xd.append(s)
        if Xd:
            M = np.vstack([np.array(s) for s in Xd]).T
            mask = (~np.isnan(M).any(axis=1)) & (~y.isna().values)
            if mask.any():
                # simple linear via np.linalg.lstsq
                A = np.c_[np.ones(mask.sum()), M[mask]]
                coef, *_ = np.linalg.lstsq(A, y[mask].values, rcond=None)
                yhat = A @ coef
                r2p = r2_score(y[mask].values, yhat)
                dist = float(1 - max(0.0, min(1.0, r2p)))
            else:
                dist = np.nan
        else:
            dist = np.nan

        # Actionability%: porcentaje de features (de X) que sean modificables (aprox). Si quieres, se puede refinar con top importancias.
        if len(X.columns):
            action_pct = float(np.mean([_is_modifiable(c) for c in X.columns]))
        else:
            action_pct = np.nan

        rows.append({
            "target": target,
            "coverage": cov,
            "rel": rel,
            "ac1": ac1,
            "mae_model": mae_model,
            "mae_naive": mae_naive,
            "mae_ema": mae_ema,
            "gain_vs_best_baseline": gain,
            "actionability_pct": action_pct,
            "distinctiveness": dist,
            "n_samples": n,
            "n_splits": len(splits) if splits else 0,
        })
    return pd.DataFrame(rows)
