
# -*- coding: utf-8 -*-
import pandas as pd

SUBSTANCE_KEYWORDS = {
    "has_nicotina": ["nicotina", "tabaco", "cigarro", "cigarrillo", "cig"],
    "has_thc": ["thc", "marihuana", "cannabis", "weed", "hierba"],
    "has_benzodiazepina": ["benzo", "benzodiazepina", "clonazepam", "alprazolam", "diazepam", "lorazepam"],
    "has_modafinilo": ["modafinilo", "modafinil", "moda"],
    "has_psilocibina": ["psilocibina", "psilocybin", "hongos", "setas"],
}

def calc_medicacion_sustancias(df: pd.DataFrame) -> pd.DataFrame:
    """Crea flags para sustancias y normaliza adherencia de medicación."""
    out = df.copy()
    base = out.get("otras_sustancias")
    if base is None:
        for k in SUBSTANCE_KEYWORDS.keys():
            out[k] = 0
    else:
        low = base.fillna("").astype(str).str.lower()
        for k, words in SUBSTANCE_KEYWORDS.items():
            out[k] = low.apply(lambda s: int(any(w in s for w in words)))

    if "medicacion_tomada" in out.columns:
        out["adherencia_med"] = (pd.to_numeric(out["medicacion_tomada"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        out["adherencia_med"] = 0
    return out
