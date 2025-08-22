
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

# Canonical Spanish area names used in this project
AREAS = [
    "animo",
    "activacion",
    "sueno",
    "conexion",
    "proposito",
    "claridad",
    "estres"
]

POSITIVE_AREAS = ["animo", "activacion", "sueno", "conexion", "proposito", "claridad"]
NEGATIVE_AREAS = ["estres"]

# Default column mapping. Override as needed when calling the APIs.
DEFAULT_COLUMNS = {
    "fecha": "fecha",               # date/time of entry
    "notas": "notas",               # free text note
    "estresores": "estresores",     # list or comma-separated string
    # Area signals (float). Provide any subset; missing will be ignored.
    "animo": "animo",
    "activacion": "activacion",
    "sueno": "sueno",
    "conexion": "conexion",
    "proposito": "proposito",
    "claridad": "claridad",
    "estres": "estres",
    # Optional precomputed fields
    "categoria_dia": "categoria_dia"  # ("Muy negativo", "Débil", "Aceptable", "Positivo") or 0-3
}

@dataclass
class DayCard:
    fecha: str
    registros: int
    resumen_areas: Dict[str, float]
    mensajes_humanos: List[str]
    notas: List[str]
    estresores: List[str]
    comparacion_prev: Optional[str]
    interpretacion_dia: str
    faltante: bool = False  # marks missing day

def coerce_date(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.tz_localize(None) if s.dt.tz is not None else s

def ensure_str_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    # split by commas/semicolons if string
    if isinstance(x, str):
        # Allow JSON-like lists
        t = x.strip()
        if (t.startswith('[') and t.endswith(']')):
            try:
                arr = eval(t, {"__builtins__": {}}, {})
                return [str(i).strip() for i in arr if str(i).strip()]
            except Exception:
                pass
        for sep in [";", ",", "|", "•"]:
            if sep in x:
                return [i.strip() for i in x.split(sep) if i.strip()]
        return [x.strip()] if x.strip() else []
    return [str(x).strip()]
