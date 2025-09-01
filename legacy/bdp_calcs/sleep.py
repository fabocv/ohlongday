
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .utils import hours_between, clip01, safe_div
from .constants import SLEEP_WEIGHTS

def _mid_sleep(h_ini: str, h_fin: str):
    h = hours_between(h_ini, h_fin)
    if np.isnan(h):
        return np.nan
    # punto medio desde la hora de dormir
    return (h / 2.0)

def _circadian_alignment(h_ini: str, h_fin: str):
    """Score 0–1 con óptimo en ~03:30 de mid-sleep (triangular 00:30–03:30–06:30)."""
    h = _mid_sleep(h_ini, h_fin)
    if np.isnan(h):
        return np.nan
    # mid-sleep relativo a 24h; tomamos 0 en 0:30 y 6:30, 1 en 3:30
    # distancia en horas a 3.5 horas
    dist = abs(h - 3.5)
    # 3 horas a cada lado hasta 0
    return clip01(1 - dist / 3.0)

def calc_sueno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega: TIB (h), sleep_efficiency (0–1), sleep_debt (h),
            circadian_alignment (0–1),
            sueno_calidad_calc (0–10) y sueno_calidad_gap.
    """
    out = df.copy()
    h_dormir = out.get("hora_dormir")
    h_despe = out.get("hora_despertar")

    # TIB y eficiencia
    out["TIB_h"] = np.where(h_dormir.notna() & h_despe.notna(),
                            hours_between(h_dormir, h_despe), np.nan)
    out["sleep_efficiency"] = clip01(out["horas_sueno"] / out["TIB_h"]) if "horas_sueno" in out.columns else np.nan

    # Sleep debt
    out["sleep_debt"] = np.where(out.get("horas_sueno").notna(), (8.0 - out["horas_sueno"]).clip(lower=0), np.nan)

    # Continuidad (despertares)
    if "despertares_nocturnos" in out.columns:
        out["sleep_continuity"] = clip01(1 - (pd.to_numeric(out["despertares_nocturnos"], errors="coerce") / 4.0))
    else:
        out["sleep_continuity"] = np.nan

    # Alineación circadiana
    out["circadian_alignment"] = np.where(h_dormir.notna() & h_despe.notna(),
                                          _circadian_alignment(h_dormir, h_despe), np.nan)

    # Timing cafeína y alcohol
    if "cafe_ultima_hora" in out.columns and "hora_dormir" in out.columns:
        out["coffee_hours_before_bed"] = out.apply(lambda r: hours_between(r.get("cafe_ultima_hora"), r.get("hora_dormir")), axis=1)
        out["caffeine_timing_score"] = clip01(out["coffee_hours_before_bed"] / 6.0)
    else:
        out["coffee_hours_before_bed"] = np.nan
        out["caffeine_timing_score"] = np.nan

    if "alcohol_ultima_hora" in out.columns and "hora_dormir" in out.columns:
        out["alcohol_hours_before_bed"] = out.apply(lambda r: hours_between(r.get("alcohol_ultima_hora"), r.get("hora_dormir")), axis=1)
        # penaliza fuerte si <3h; modulada por dosis si hay alcohol_ud
        dose = pd.to_numeric(out.get("alcohol_ud"), errors="coerce").fillna(0)
        penalty = (1 - clip01(out["alcohol_hours_before_bed"] / 3.0)) * clip01(dose / 2.0)
        out["alcohol_timing_score"] = clip01(1 - penalty)
    else:
        out["alcohol_hours_before_bed"] = np.nan
        out["alcohol_timing_score"] = np.nan

    # Luz mañana y pantalla noche
    out["morning_light_score"] = clip01(pd.to_numeric(out.get("exposicion_sol_manana_min"), errors="coerce") / 15.0)
    out["screen_night_score"] = 1 - clip01(pd.to_numeric(out.get("tiempo_pantalla_noche_min"), errors="coerce") / 60.0)

    # Duración
    duration_score = clip01(1 - (abs(pd.to_numeric(out.get("horas_sueno"), errors="coerce") - 8.0) / 3.0))

    # Composición sueno_calidad_calc
    comps = {
        "duration": duration_score,
        "continuity": out["sleep_continuity"],
        "efficiency": out["sleep_efficiency"],
        "circadian_alignment": out["circadian_alignment"],
        "caffeine_timing": out["caffeine_timing_score"],
        "alcohol_timing": out["alcohol_timing_score"],
        "morning_light": out["morning_light_score"],
        "screen_night": out["screen_night_score"],
    }
    # normaliza pesos por disponibilidad
    weights = SLEEP_WEIGHTS.copy()
    wsum = 0.0
    acc = np.zeros(len(out))
    for k, w in weights.items():
        v = pd.to_numeric(comps[k], errors="coerce")
        mask = v.notna()
        if mask.any():
            acc[mask] = acc[mask] + w * v[mask]
            wsum += w
    out["sueno_calidad_calc"] = np.where(wsum > 0, 10.0 * acc / wsum, np.nan)

    # Gap subjetivo vs calculado (si existe sueno_calidad)
    if "sueno_calidad" in out.columns:
        out["sueno_calidad_gap"] = pd.to_numeric(out["sueno_calidad"], errors="coerce") - out["sueno_calidad_calc"]
    else:
        out["sueno_calidad_gap"] = np.nan

    return out
