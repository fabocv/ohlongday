import numpy as np
import pandas as pd

# ---- Helpers robustos ----
def _as_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float).reindex(df.index) if col in df.columns \
           else pd.Series(default, index=df.index, dtype=float)

def _gauss_peak(x: pd.Series, mu: float = 6.5, sigma: float = 2.0, floor: float = 0.6) -> pd.Series:
    """Campana centrada en 6.5 (intensidad ideal 6–7). Devuelve 0.6..1.0."""
    x = pd.to_numeric(x, errors="coerce")
    g = np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)  # 0..1
    return (floor + (1 - floor) * g).astype(float)

def _logistic_benefit(minutes: pd.Series, m50=35, k=0.12, free_min=8, cap=120) -> pd.Series:
    """Beneficio 0..1 por minutos de ejercicio (satura ~120′). Libre hasta 8′."""
    m = pd.to_numeric(minutes, errors="coerce")
    x = m.clip(lower=0, upper=cap)
    base = 1.0 / (1.0 + np.exp(-k * (x - m50)))
    base = base.where(x > free_min, 0.0)
    base[m.isna()] = np.nan
    return base.astype(float)

def _tod_factor(df: pd.DataFrame, col_hora_ej: str = "hora_ejercicio", col_hora_dormir: str = "hora_dormir",
                near_sleep_hours: float = 2.5, low: float = 0.85, high: float = 1.05) -> pd.Series:
    """
    Factor horario 0.85..1.05: penaliza si ejercicio cae <2.5h antes de dormir (~0.85),
    da +5% si lejos del sueño. Si no hay horas -> 1.0.
    """
    if col_hora_ej not in df.columns or col_hora_dormir not in df.columns:
        return pd.Series(1.0, index=df.index, dtype=float)

    he = pd.to_datetime(df[col_hora_ej], errors="coerce").dt.hour + \
         pd.to_datetime(df[col_hora_ej], errors="coerce").dt.minute/60.0
    hd = pd.to_datetime(df[col_hora_dormir], errors="coerce").dt.hour + \
         pd.to_datetime(df[col_hora_dormir], errors="coerce").dt.minute/60.0

    # distancia en horas (considera wrap nocturno simple)
    dist = (hd - he).mod(24)
    # muy cerca del sueño => low; muy lejos => high
    fact = np.where(dist <= near_sleep_hours, low, high)
    fact = pd.Series(fact, index=df.index).fillna(1.0)
    return fact.clip(lower=0.85, upper=1.05).astype(float)

# ---- Núcleo: ejercicio -> EX_raw, EX_dir, mitigadores ----
def compute_exercise_effect(
    df: pd.DataFrame,
    col_min: str = "tiempo_ejercicio",      # minutos de ejercicio total del día
    col_int: str = "mov_intensidad",        # 0..10 (o NaN)
    col_hora_ej: str = "hora_ejercicio",    # opcional HH:MM
    col_hora_dormir: str = "hora_dormir",   # opcional HH:MM
    alpha_ex: float = 0.4,                  # bonus directo máx (puntos en 0..10)
    rho_today: float = 0.35,                # cuánto amortigua PTN hoy
    rho_d1: float = 0.25                    # cuánto amortigua PTN mañana
) -> pd.DataFrame:
    out = df.copy()
    minutes = _as_series(out, col_min, default=np.nan)
    intensity = _as_series(out, col_int, default=np.nan)

    b_min = _logistic_benefit(minutes, m50=35, k=0.12, free_min=8, cap=120)     # 0..1
    b_int = _gauss_peak(intensity, mu=6.5, sigma=2.0, floor=0.6)                # 0.6..1.0
    b_tod = _tod_factor(out, col_hora_ej, col_hora_dormir, near_sleep_hours=2.5, low=0.85, high=1.05)

    EX_raw = (b_min * b_int * b_tod).clip(0, 1)                                  # 0..1
    out["EX_raw"] = EX_raw
    out["EX_dir"] = alpha_ex * EX_raw                                            # 0..alpha_ex
    out["EX_mitig_today"] = (rho_today * EX_raw).clip(0, 0.95)                   # 0..rho_today
    out["EX_mitig_d1"] = (rho_d1 * EX_raw).clip(0, 0.95)                         # 0..rho_d1
    return out

# ---- Integración en tu pipeline WBN existente ----
def apply_exercise_to_WBN(
    daily: pd.DataFrame,
    *,
    col_WB: str = "WB",
    col_MB: str = "MB",
    col_PTN_today: str = "PTN_today",
    col_PTN_d1: str = "PTN_d1",
    relief_MB_mu: float = 0.25,     # baja MB hasta 0.25 * EX_raw (opcional, pon 0 para desactivar)
    clip_0_10: bool = True
) -> pd.DataFrame:
    out = daily.copy()

    WB  = _as_series(out, col_WB, default=np.nan)
    MB  = _as_series(out, col_MB, default=0.0).fillna(0.0)
    PT0 = _as_series(out, col_PTN_today, default=0.0).fillna(0.0)
    PT1 = _as_series(out, col_PTN_d1, default=0.0).fillna(0.0)

    EX_dir = _as_series(out, "EX_dir", default=0.0).fillna(0.0)
    EXm0   = _as_series(out, "EX_mitig_today", default=0.0).fillna(0.0)
    EXm1   = _as_series(out, "EX_mitig_d1",    default=0.0).fillna(0.0)
    EXraw  = _as_series(out, "EX_raw", default=0.0).fillna(0.0)

    # Mitigación en PTN
    PT0_adj = PT0 * (1.0 - EXm0)
    PT1_adj = PT1 * (1.0 - EXm1)

    # Alivio metabólico opcional
    MB_adj = (MB - relief_MB_mu * EXraw).clip(lower=0.0)

    WBN_ex = WB - MB_adj - (PT0_adj + PT1_adj) + EX_dir
    if clip_0_10:
        WBN_ex = WBN_ex.clip(0, 10)

    out["PTN_today_adj"] = PT0_adj
    out["PTN_d1_adj"]    = PT1_adj
    out["MB_adj"]        = MB_adj
    out["WBN_ex"]        = WBN_ex
    return out
