from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

# -------------------------------
# 1) Helpers de tiempo y logística
# -------------------------------

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

def _series_minutes(df: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve una Serie float (minutos) alineada a df.index.
    Si la columna no existe, retorna Serie de NaN."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        return s.reindex(df.index)
    else:
        return pd.Series(np.nan, index=df.index, dtype=float)

@dataclass
class BlockCfg:
    name: str; start_h: int; end_h: int; m50: float; k: float; free_min: float; cap: float; weight: float

DEFAULT_CFGS: Dict[str, BlockCfg] = {
    "manana": BlockCfg("manana", 6, 12, m50=90, k=0.06, free_min=20, cap=240, weight=0.25),
    "tarde":  BlockCfg("tarde", 12, 20, m50=75, k=0.07, free_min=15, cap=240, weight=0.55),
    "noche":  BlockCfg("noche", 20, 24, m50=45, k=0.08, free_min=10, cap=240, weight=1.00),
}

def _logistic_penalty(minutes: float, m50: float, k: float, free_min: float, cap: float) -> float:
    if np.isnan(minutes): return np.nan
    x = float(np.clip(minutes, 0, cap))
    if x <= free_min: return 0.0
    return 1.0 / (1.0 + np.exp(-k * (x - m50)))

def _to_time(s):
    if pd.isna(s): return None
    t = pd.to_datetime(str(s), errors="coerce")
    if pd.isna(t): return None
    return pd.Timestamp("1970-01-01") + pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

def _minutes_overlap(a_start, a_end, b_start, b_end) -> float:
    if a_end < a_start: a_end = a_start
    if b_end < b_start: b_end = b_start
    start = max(a_start, b_start); end = min(a_end, b_end)
    return max(0.0, (end - start).total_seconds() / 60.0)

def _blue_light_multiplier(row: pd.Series, noche_minutes: float, last_hours: float = 2.0, gamma: float = 0.6) -> float:
    h_sleep = _to_time(row.get("hora_dormir"))
    if h_sleep is None or not isinstance(noche_minutes, (int, float)) or np.isnan(noche_minutes) or noche_minutes <= 0:
        return 1.0
    base_day = pd.Timestamp("1970-01-01")
    night_start = base_day + pd.Timedelta(hours=DEFAULT_CFGS["noche"].start_h)
    sleep_day = base_day if h_sleep.hour >= DEFAULT_CFGS["noche"].start_h else base_day + pd.Timedelta(days=1)
    night_end = sleep_day + pd.Timedelta(hours=h_sleep.hour, minutes=h_sleep.minute, seconds=h_sleep.second)
    night_duration_min = max(1.0, (night_end - night_start).total_seconds() / 60.0)
    last_start = night_end - pd.Timedelta(hours=last_hours)
    ratio = _minutes_overlap(night_start, night_end, last_start, night_end) / night_duration_min
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return 1.0 + gamma * ratio

def build_ptn_from_screens(
    df: pd.DataFrame,
    col_fecha: str = "fecha",
    col_noche_explicit: str = "tiempo_pantalla_noche_min",
    col_manana: str = "tiempo_pantalla_manana_min",
    col_tarde: str = "tiempo_pantalla_tarde_min",
    col_noche_block: str = "tiempo_pantalla_noche_bloque_min",
    apply_shift_d1: bool = True,
    cfgs: Dict[str, BlockCfg] = DEFAULT_CFGS,
    gamma_luz_azul: float = 0.6,
) -> pd.DataFrame:
    out = df.copy()

    # Fecha y orden
    if col_fecha not in out.columns:
        raise ValueError(f"Falta columna '{col_fecha}'")
    if not pd.api.types.is_datetime64_any_dtype(out[col_fecha]):
        out[col_fecha] = pd.to_datetime(out[col_fecha], dayfirst=True, errors="coerce")
    out = out.sort_values(col_fecha).reset_index(drop=True)

    # Series de minutos (¡siempre Series!)
    m_m = _series_minutes(out, col_manana)
    m_t = _series_minutes(out, col_tarde)
    if col_noche_explicit in out.columns:
        m_n = _series_minutes(out, col_noche_explicit)
    else:
        m_n = _series_minutes(out, col_noche_block)

    # Penalizaciones por bloque
    p_m = m_m.apply(lambda x: _logistic_penalty(x, cfgs["manana"].m50, cfgs["manana"].k, cfgs["manana"].free_min, cfgs["manana"].cap))
    p_t = m_t.apply(lambda x: _logistic_penalty(x, cfgs["tarde"].m50,  cfgs["tarde"].k,  cfgs["tarde"].free_min,  cfgs["tarde"].cap))
    p_n_base = m_n.apply(lambda x: _logistic_penalty(x, cfgs["noche"].m50,  cfgs["noche"].k,  cfgs["noche"].free_min,  cfgs["noche"].cap))

    # Multiplicador luz azul (usa la columna de noche que esté activa)
    use_noche_col = col_noche_explicit if col_noche_explicit in out.columns else col_noche_block
    multipliers = out.apply(lambda r: _blue_light_multiplier(r, r.get(use_noche_col, np.nan), last_hours=2.0, gamma=gamma_luz_azul), axis=1)
    p_n = p_n_base * multipliers

    # Ponderación por bloque
    out["ptn_manana"] = p_m * cfgs["manana"].weight
    out["ptn_tarde"]  = p_t * cfgs["tarde"].weight
    out["ptn_noche"]  = p_n * cfgs["noche"].weight
    out["ptn_noche_multiplier"] = multipliers

    # Suma diaria (NaN no penaliza: se trata como 0 en la suma)
    out["PTN_raw"] = out["ptn_manana"].fillna(0) + out["ptn_tarde"].fillna(0) + out["ptn_noche"].fillna(0)

    # Efecto día siguiente
    out["PTN_d1"] = out["PTN_raw"].shift(1) if apply_shift_d1 else out["PTN_raw"]
    return out


# -------------------------------
# 4) WBN = WB - MB - PTN_d1
# -------------------------------

def compute_WBN(df: pd.DataFrame,
                col_WB: str = "WB",
                col_MB: str = "MB",
                col_PTN_d1: str = "PTN_d1",
                clip_0_10: bool = True) -> pd.Series:
    """
    Calcula WBN = WB - MB - PTN_d1.
    Si faltan columnas, supone MB=0. PTN_d1 faltante => 0 (no penaliza).
    """
    WB = df[col_WB]
    MB = df[col_MB]
    PTN = df[col_PTN_d1].fillna(0)
    WBN = WB - MB - PTN
    if clip_0_10:
        WBN = WBN.clip(lower=0, upper=10)
    return WBN
