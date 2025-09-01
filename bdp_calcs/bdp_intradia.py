# bdp_calcs/bdp_intradia.py
# ------------------------------------------------------------
# Agregador intradía robusto → serie diaria por variable.
# Política:
#  A) ≥3 tomas y span ≥4h → promedio ponderado por tiempo (intervalos acotados) + 30% cierre.
#  B) ≥2 tomas pero span <4h o sin hora → mezcla robusta (mediana 50% + cierre 30% + primera 20%).
#     Si son exactamente 2: 0.6 cierre + 0.4 primera.
#  C) 1 toma → 0.7 única + 0.3 EMA_prev (si hay), si no, única.
#  D) 0 tomas → EMA_prev (missing=True).
# Confianza intradía:
#  - Alta: n≥3 y span≥8h
#  - Media: n≥2 y span≥4h
#  - Baja: resto (incluye cluster severo, 1 o 0 tomas)
# Además, permite elegir estrategia por columna: 'mean' (default) o 'sum' (p. ej., café/alcohol/ejercicio).
# ------------------------------------------------------------

#from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import re

# -------------------- helpers de fecha/hora (sin warnings) --------------------

_DATE_DDMMYYYY = re.compile(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$")
_DATE_ISO      = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")

def _norm_date_ddmmyyyy(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace("/", "-", regex=False)
    def _pad(x: str) -> str:
        if _DATE_ISO.match(x):
            # ya viene ISO → lo dejamos (lo parseamos abajo con %Y-%m-%d)
            return x
        m = _DATE_DDMMYYYY.match(x)
        if not m: return x
        d, mth, y = m.groups()
        return f"{int(d):02d}-{int(mth):02d}-{y}"
    return s.map(_pad)

def _norm_time_hhmm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().replace({"nan": "", "NaT": ""})
    def _pad(t: str) -> str:
        m = re.match(r"^(\d{1,2}):(\d{1,2})$", t)
        if not m: return "00:00"
        h, mi = m.groups()
        return f"{int(h):02d}:{int(mi):02d}"
    return s.map(_pad)

def _timestamp_from_fecha_hora(df: pd.DataFrame, date_col="fecha", time_col="hora") -> pd.Series:
    if date_col not in df.columns:
        return pd.to_datetime(pd.Series([pd.NaT]*len(df), index=df.index))
    draw = df[date_col]
    d = _norm_date_ddmmyyyy(draw)
    # separa ISO vs DDMY
    iso_mask = d.str.match(_DATE_ISO)
    # hora (si hay)
    if time_col in df.columns:
        t = _norm_time_hhmm(df[time_col])
        ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        if iso_mask.any():
            ts.loc[iso_mask] = pd.to_datetime(
                d.loc[iso_mask] + " " + t.loc[iso_mask], format="%Y-%m-%d %H:%M", errors="coerce"
            )
        if (~iso_mask).any():
            ts.loc[~iso_mask] = pd.to_datetime(
                d.loc[~iso_mask] + " " + t.loc[~iso_mask], format="%d-%m-%Y %H:%M", errors="coerce"
            )
        return ts
    else:
        ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        if iso_mask.any():
            ts.loc[iso_mask] = pd.to_datetime(d.loc[iso_mask], format="%Y-%m-%d", errors="coerce")
        if (~iso_mask).any():
            ts.loc[~iso_mask] = pd.to_datetime(d.loc[~iso_mask], format="%d-%m-%Y", errors="coerce")
        return ts

# ------------------------------ dataclass salida ------------------------------

@dataclass
class DailyAgg:
    y: float
    n: int
    span_h: float
    quality: str
    first: Optional[float]
    last: Optional[float]
    median: Optional[float]
    peak: Optional[float]
    trough: Optional[float]
    missing: bool

# --------------------------- reglas de calidad/conf ---------------------------

def _daily_confidence(n: int, span_h: float) -> str:
    if n >= 3 and span_h >= 8: return "alta"
    if n >= 2 and span_h >= 4: return "media"
    return "baja"

# --------------------------- agregador por día/col ----------------------------

def _aggregate_day(
    vals: pd.Series,
    times: pd.Series,
    ema_prev: Optional[float] = None,
    closing_weight: float = 0.30,
    min_w_minutes: int = 15,
    max_w_hours: float = 6.0,
) -> DailyAgg:
    """
    vals: serie de valores (puede tener NaN). times: datetime en el mismo índice (puede NaT).
    """
    v = pd.to_numeric(vals, errors="coerce")
    t = pd.to_datetime(times, errors="coerce")

    # ordena por hora (mantiene NaT al final)
    order = t.argsort(kind="mergesort")
    v, t = v.iloc[order], t.iloc[order]

    # filtra valores válidos
    valid_mask = v.notna()
    v_valid = v[valid_mask]
    t_valid = t[valid_mask]

    n = int(v_valid.shape[0])
    if n == 0:
        y = float(ema_prev) if ema_prev is not None and np.isfinite(ema_prev) else float("nan")
        return DailyAgg(y=y, n=0, span_h=0.0, quality="baja",
                        first=None, last=None, median=None, peak=None, trough=None, missing=True)

    # métricas intradía
    first = float(v_valid.iloc[0])
    last  = float(v_valid.iloc[-1])
    median = float(np.nanmedian(v_valid.values))
    peak   = float(np.nanmax(v_valid.values))
    trough = float(np.nanmin(v_valid.values))

    # span en horas (si hay al menos 2 tiempos válidos)
    if n >= 2 and t_valid.notna().sum() >= 2:
        span = (t_valid.iloc[-1] - t_valid.iloc[0]).total_seconds() / 3600.0
    else:
        span = 0.0

    # ¿tenemos horas razonables para ponderar por tiempo?
    can_time_weight = (n >= 3 and span >= 4.0 and t_valid.notna().sum() == n)

    if can_time_weight:
        # pesos por intervalo: w_i = clamp(dt_i, 15min..6h). Última toma: 2h por defecto.
        mins = t_valid.dt.hour * 60 + t_valid.dt.minute
        mins = mins.astype(float).values
        w = []
        last_interval_h = 2.0
        for i in range(n):
            if i < n-1:
                dt_min = max(mins[i+1] - mins[i], float(min_w_minutes))
                w.append(dt_min / 60.0)
            else:
                w.append(last_interval_h)
        w = np.array(w, dtype=float)
        w = np.clip(w, min_w_minutes/60.0, max_w_hours)
        mu_time = float(np.nansum(v_valid.values * w) / np.nansum(w))
        y = 0.7 * mu_time + closing_weight * last + (0.3 - closing_weight) * mu_time  # equivale a 0.7*mu + 0.3*last
        # simplifica: y = 0.7*mu_time + 0.3*last
        y = 0.7*mu_time + 0.3*last
    else:
        # cluster o sin horas útiles → mezcla robusta
        if n >= 3:
            y = 0.5 * median + 0.3 * last + 0.2 * first
        elif n == 2:
            y = 0.6 * last + 0.4 * first
        else:  # n == 1
            if ema_prev is not None and np.isfinite(ema_prev):
                y = 0.7 * last + 0.3 * float(ema_prev)
            else:
                y = last

    quality = _daily_confidence(n, span)
    return DailyAgg(y=float(y), n=n, span_h=float(span), quality=quality,
                    first=first, last=last, median=median, peak=peak, trough=trough, missing=False)

# ------------------------ agregado secuencial por días ------------------------

def aggregate_intraday(
    df: pd.DataFrame,
    value_cols: List[str],
    *,
    date_col: str = "fecha",
    time_col: str = "hora",
    strategy_map: Optional[Dict[str, str]] = None,  # {'col': 'mean'|'sum'}
    closing_weight: float = 0.30,
    alpha_shrink: float = 0.30,  # usado para EMA_prev al avanzar días
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (daily_values, daily_meta).
    - daily_values: DataFrame indexado por fecha (date) con columnas = value_cols (agregado diario y_d).
    - daily_meta:   DataFrame con columnas auxiliares por col: n_{c}, span_h_{c}, quality_{c},
                    y columnas generales: n_tomas_total, span_h_total.
    Requiere que df esté filtrado por 'correo' si trabajas por usuario.
    """
    if strategy_map is None:
        strategy_map = {}

    # timestamp y orden estable
    ts = _timestamp_from_fecha_hora(df, date_col=date_col, time_col=time_col)
    df2 = df.copy()
    df2["__ts"] = ts
    df2 = df2.sort_values("__ts", kind="mergesort").reset_index(drop=True)

    # día (date) para agrupar
    day = pd.to_datetime(df2["__ts"], errors="coerce").dt.date
    df2["__day"] = day

    # contenedor de resultados
    unique_days = [d for d in pd.unique(day) if pd.notna(d)]
    unique_days = sorted(unique_days)
    if not unique_days:
        # sin fechas válidas: intentamos agrupar por la columna 'fecha' string normalizada
        if date_col in df2.columns:
            alt = _norm_date_ddmmyyyy(df2[date_col]).fillna("")
            unique_days = sorted(set(s for s in alt if s))
            # mapeamos luego a date al vuelo
        else:
            return (pd.DataFrame(columns=value_cols), pd.DataFrame())

    # acumuladores EMA_prev por columna
    ema_prev: Dict[str, Optional[float]] = {c: None for c in value_cols}

    rows_vals: List[Dict[str, float]] = []
    rows_meta: List[Dict[str, object]] = []

    for d in unique_days:
        # subset del día
        if isinstance(d, str):
            mask = _norm_date_ddmmyyyy(df2[date_col]).eq(d)
        else:
            mask = df2["__day"] == d
        day_df = df2.loc[mask].copy()

        # métricas generales del día
        n_tomas_total = int(day_df.shape[0])
        times = pd.to_datetime(day_df["__ts"], errors="coerce")
        if n_tomas_total >= 2 and times.notna().sum() >= 2:
            span_total = (times.max() - times.min()).total_seconds() / 3600.0
        else:
            span_total = 0.0

        out_vals: Dict[str, float] = {}
        out_meta: Dict[str, object] = {
            "date": d,
            "n_tomas_total": n_tomas_total,
            "span_h_total": float(span_total),
        }

        for c in value_cols:
            strat = (strategy_map.get(c) or "mean").lower()
            series_c = pd.to_numeric(day_df.get(c, pd.Series(index=day_df.index, dtype=float))), 

            s = pd.to_numeric(day_df.get(c, pd.Series(index=day_df.index, dtype=float)), errors="coerce")

            if strat == "sum":
                y = float(np.nansum(s.values)) if s.notna().any() else np.nan
            elif strat == "max":
                y = float(np.nanmax(s.values)) if s.notna().any() else np.nan
            elif strat == "last":
                # day_df ya está ordenado por __ts; toma el último no-NaN del día
                y = float(s.dropna().iloc[-1]) if s.dropna().size else np.nan
            elif strat == "first":  # 👈 NUEVO
                y = float(s.dropna().iloc[0]) if s.dropna().size else np.nan
            else:  # 'mean' por defecto
                agg = _aggregate_day(vals=s, times=times, ema_prev=ema_prev[c], closing_weight=closing_weight)
                y = agg.y
                if np.isfinite(y):
                    if ema_prev[c] is None or not np.isfinite(ema_prev[c]):
                        ema_prev[c] = float(y)
                    else:
                        ema_prev[c] = float(alpha_shrink*y + (1-alpha_shrink)*ema_prev[c])

            out_vals[c] = agg.y
            out_meta[f"n_{c}"] = agg.n
            out_meta[f"span_h_{c}"] = agg.span_h
            out_meta[f"quality_{c}"] = agg.quality
            out_meta[f"missing_{c}"] = agg.missing

        rows_vals.append(out_vals)
        rows_meta.append(out_meta)

    daily_values = pd.DataFrame(rows_vals, index=pd.to_datetime([r["date"] for r in rows_meta]))
    daily_values.index.name = "date"
    daily_meta = pd.DataFrame(rows_meta, index=daily_values.index)
    return daily_values, daily_meta
