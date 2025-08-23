
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

AREAS = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]
POSITIVE_AREAS = ["animo","activacion","sueno","conexion","proposito","claridad"]
NEGATIVE_AREAS = ["estres"]

DEFAULT_COLUMNS = {
    "fecha": "fecha",
    "hora": "hora",
    "notas": "notas",
    "estresores": "estresores",
    "animo": "animo",
    "activacion": "activacion",
    "sueno": "sueno",
    "conexion": "conexion",
    "proposito": "proposito",
    "claridad": "claridad",
    "estres": "estres",
    "categoria_dia": "categoria_dia"
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
    faltante: bool = False

def coerce_date(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s

def ensure_str_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
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
