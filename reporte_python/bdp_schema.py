
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterable, Union
import math

import pandas as pd

# Áreas núcleo (promedios/deltas)
AREAS: List[str] = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]

# Áreas con signo (para tendencias: + es mejor en positivas; - es mejor en negativas)
POSITIVE_AREAS: List[str] = ["animo","activacion","sueno","conexion","proposito","claridad"]
NEGATIVE_AREAS: List[str] = ["estres"]

# Variables numéricas extra que algunos cálculos/etiquetas usan
EXTRA_NUM_VARS: List[str] = [
    "horas_sueno","siesta_min","autocuidado","alimentacion","movimiento","dolor_fisico",
    "ansiedad","irritabilidad","meditacion_min","exposicion_sol_min","agua_litros",
    "cafe_cucharaditas","alcohol_ud","glicemia"
]

# Mapeo de nombres lógicos -> columnas reales del CSV
DEFAULT_COLUMNS: Dict[str, str] = {
    # claves mínimas
    "fecha": "fecha",
    "hora": "hora",
    "notas": "notas",
    "estresores": "eventos_estresores",
    "tags": "tags",

    # áreas núcleo
    "animo": "animo",
    "activacion": "activacion",
    "sueno": "sueno_calidad",
    "conexion": "conexion",
    "proposito": "proposito",
    "claridad": "claridad",
    "estres": "estres",

    # métricas/variables extra
    "horas_sueno": "horas_sueno",
    "siesta_min": "siesta_min",
    "autocuidado": "autocuidado",
    "alimentacion": "alimentacion",
    "movimiento": "movimiento",
    "dolor_fisico": "dolor_fisico",
    "ansiedad": "ansiedad",
    "irritabilidad": "irritabilidad",
    "meditacion_min": "meditacion_min",
    "exposicion_sol_min": "exposicion_sol_min",
    "agua_litros": "agua_litros",
    "cafe_cucharaditas": "cafe_cucharaditas",
    "alcohol_ud": "alcohol_ud",
    "medicacion_tomada": "medicacion_tomada",
    "medicacion_tipo": "medicacion_tipo",
    "otras_sustancias": "otras_sustancias",
    "interacciones_significativas": "interacciones_significativas",
    "glicemia": "glicemia",
}

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def ensure_str_list(x: Any) -> List[str]:
    """
    Normaliza campo potencialmente múltiple (tags/estresores) a lista de str limpia.
    - None/NaN -> []
    - str -> split por coma/; / |  y trim
    - Iterable -> coerción a str de cada elemento y limpieza
    """
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan","none","null"}:
            return []
        # separadores comunes
        parts = [p.strip() for p in re_split(r"[;,|]", s) if p.strip()]
        return parts
    if isinstance(x, Iterable) and not isinstance(x, (bytes, bytearray)):
        out: List[str] = []
        for it in x:
            if it is None:
                continue
            s = str(it).strip()
            if not s or s.lower() in {"nan","none","null"}:
                continue
            out.append(s)
        return out
    s = str(x).strip()
    if not s or s.lower() in {"nan","none","null"}:
        return []
    return [s]

import re as _re
def re_split(pat: str, s: str) -> List[str]:
    try:
        return _re.split(pat, s)
    except Exception:
        return [s]

def coerce_date(series_or_values: Union[pd.Series, Iterable, str]) -> pd.Series:
    """
    Convierte a pandas datetime con dayfirst=True (soporta 'dd-MM-YYYY').
    Si se recibe un string único, devuelve Series de longitud 1.
    """
    if isinstance(series_or_values, pd.Series):
        return pd.to_datetime(series_or_values, errors="coerce", dayfirst=True)
    # iterable o escalar
    try:
        return pd.to_datetime(series_or_values, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime([series_or_values], errors="coerce", dayfirst=True)

# ------------------------------------------------------------------
# estructura usada por bdp_cards (y luego convertida a dict en report)
# ------------------------------------------------------------------
@dataclass
class DayCard:
    fecha: str
    registros: int = 0
    resumen_areas: Dict[str, float] = field(default_factory=dict)
    mensajes_humanos: List[str] = field(default_factory=list)
    notas: List[str] = field(default_factory=list)
    estresores: List[str] = field(default_factory=list)
    comparacion_prev: str = ""
    interpretacion_dia: str = ""
    faltante: bool = False
