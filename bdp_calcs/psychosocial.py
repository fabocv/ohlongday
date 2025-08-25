
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .utils import count_events, clip01

STRESSOR_TAGS = {
    "estres_plazo": ["plazo", "deadline"],
    "estres_conflicto": ["conflicto", "pelea", "disputa"],
    "estres_multitarea": ["multitarea", "muchas cosas", "overload"],
}

def calc_psicosocial(df: pd.DataFrame) -> pd.DataFrame:
    """Conteos y flags simples de interacciones y estresores."""
    out = df.copy()
    out["social_count"] = out.get("interacciones_significativas").apply(count_events) if "interacciones_significativas" in out.columns else np.nan
    out["social_quality_score"] = clip01(pd.to_numeric(out.get("interacciones_calidad"), errors="coerce") / 10.0)

    # conteo de estresores
    out["stressors_count"] = out.get("eventos_estresores").apply(count_events) if "eventos_estresores" in out.columns else np.nan

    # flags por palabras clave
    base = out.get("eventos_estresores")
    low = base.fillna("").astype(str).str.lower() if base is not None else None
    for col, words in STRESSOR_TAGS.items():
        if low is None:
            out[col] = 0
        else:
            out[col] = low.apply(lambda s: int(any(w in s for w in words)))

    return out
