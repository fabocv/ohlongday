
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .utils import hours_between, clip01

def calc_estimulantes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula timings y flags de cafeína/alcohol e hidratación básica."""
    out = df.copy()

    # café después de las 14:00
    if "cafe_ultima_hora" in out.columns:
        # 14:00 -> horas desde medianoche
        def after_14(h):
            try:
                hh = hours_between("00:00", h)
                return int(hh >= 14.0) if not np.isnan(hh) else 0
            except Exception:
                return 0
        out["cafe_despues_14"] = out["cafe_ultima_hora"].apply(after_14)
    else:
        out["cafe_despues_14"] = 0

    # alcohol <3h antes de dormir
    if "alcohol_ultima_hora" in out.columns and "hora_dormir" in out.columns:
        out["alcohol_hours_before_bed"] = out.apply(lambda r: hours_between(r.get("alcohol_ultima_hora"), r.get("hora_dormir")), axis=1)
        out["alcohol_tarde"] = (out["alcohol_hours_before_bed"] < 3).astype(int)
    else:
        out["alcohol_tarde"] = 0

    # carga estimulantes
    cafe = pd.to_numeric(out.get("cafe_cucharaditas"), errors="coerce").fillna(0.0)
    alcohol = pd.to_numeric(out.get("alcohol_ud"), errors="coerce").fillna(0.0)
    out["stimulant_load"] = cafe + 0.5 * alcohol

    # hidratación simple
    agua = pd.to_numeric(out.get("agua_litros"), errors="coerce")
    out["hydration_score"] = clip01(agua / 2.0) if agua is not None else 0.0

    return out
