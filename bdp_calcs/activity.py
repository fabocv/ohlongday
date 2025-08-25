
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .utils import clip01

def calc_actividad_luz_pantalla(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza movimiento, meditación, luz y pantalla a scores 0–1."""
    out = df.copy()

    # movimiento: si hay intensidad 0–10 úsala, si no, normaliza minutos de forma simple
    if "mov_intensidad" in out.columns:
        out["movement_score"] = clip01(pd.to_numeric(out["mov_intensidad"], errors="coerce") / 7.0)
    elif "movimiento" in out.columns:
        mv = pd.to_numeric(out["movimiento"], errors="coerce")
        # normalización simple por percentil 75 personal
        p75 = mv.quantile(0.75) if mv.notna().any() else np.nan
        out["movement_score"] = clip01(mv / p75) if p75 and p75 > 0 else clip01(mv / 60.0)
    else:
        out["movement_score"] = np.nan

    # meditación
    out["relaxation_score"] = clip01(pd.to_numeric(out.get("meditacion_min"), errors="coerce") / 15.0)

    # luz de mañana y pantalla de noche
    out["morning_light_score"] = clip01(pd.to_numeric(out.get("exposicion_sol_manana_min"), errors="coerce") / 15.0)
    out["screen_night_score"] = 1 - clip01(pd.to_numeric(out.get("tiempo_pantalla_noche_min"), errors="coerce") / 60.0)

    return out
