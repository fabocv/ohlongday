
from __future__ import annotations
from typing import Dict, Any
import json
from pathlib import Path
import math
import pandas as pd

# Carga humanización de dominios
CFG_DOM = Path(__file__).parent / "dominio.json"
with open(CFG_DOM, "r", encoding="utf-8") as f:
    DOM_CFG = json.load(f)["dominios_humanizacion"]

DECIMALES = int(DOM_CFG.get("decimales", 2))

# Carga config general para dominios (inversión de estrés, etc.)
CFG_GEN = Path(__file__).parent / "bdp_config.json"
try:
    _GEN = json.loads(CFG_GEN.read_text(encoding="utf-8"))
    _ESTRES_INPUT_INVERTIDO = bool(_GEN.get("dominios_config", {}).get("estres_input_invertido", False))
except Exception:
    _ESTRES_INPUT_INVERTIDO = False  # por defecto asumimos estrés "alto=peor" (no invertido)

# Añade cerca del inicio:
CORE_KEYS = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]

def _infer_scale(row: dict) -> str:
    # Si alguna área del día > 1.5 asumimos 0–10 para TODO el día
    for k in CORE_KEYS:
        try:
            if float(row.get(k, 0)) > 1.5:
                return "0-10"
        except Exception:
            pass
    return "0-1"

def _normalize_ctx(v, scale: str):
    if v is None: return float("nan")
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if scale == "0-10":
        x = x / 10.0
    elif x > 10.5:
        x = x / 100.0
    return max(0.0, min(1.0, x))


def _normalize01(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if x > 1.5 and x <= 10.5:
        x = x / 10.0
    elif x > 10.5:
        x = x / 100.0
    return max(0.0, min(1.0, x))

def compute_dominios_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    """Calcula H, V, C, P, S- a partir de áreas ya promediadas del día."""
    scale = _infer_scale(row)

    def g(k): return _normalize_ctx(row.get(k), scale)

    animo      = g("animo")
    activacion = g("activacion")
    sueno      = g("sueno")
    conexion   = g("conexion")
    proposito  = g("proposito")
    claridad   = g("claridad")
    estres     = g("estres")

    def mean(vals):
        vals = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
        return sum(vals)/len(vals) if vals else float("nan")
    def wmean(pairs):
        vals = [(v,w) for (v,w) in pairs if v is not None and not (isinstance(v,float) and math.isnan(v))]
        sw = sum(w for _,w in vals)
        return sum(v*w for v,w in vals)/sw if vals and sw>0 else float("nan")

    H = mean([animo])
    V = wmean([(activacion,0.5),(sueno,0.5)])
    C = mean([conexion])
    P = wmean([(proposito,0.6),(claridad,0.4)])

    # Estrés invertido S-: si la entrada ya es "bajo=mejor", entonces S- = estrés_promedio;
    # si la entrada es "alto=peor", entonces S- = 1 - estrés_promedio.
    if estres == estres:  # not NaN
        if _ESTRES_INPUT_INVERTIDO:
            S_inv = estres   # ya invertido en la entrada (alto = menos estrés)
        else:
            S_inv = 1.0 - estres
    else:
        S_inv = float("nan")

    S_inv = (1.0 - estres) if estres == estres and not _ESTRES_INPUT_INVERTIDO else estres
    
    return {"H": H, "V": V, "C": C, "P": P, "S-": S_inv}

def _humanize_single(code: str, value: float, cfg: Dict[str, Any]) -> str:
    doms = cfg["dominios"]
    if code not in doms or value is None or (isinstance(value,float) and pd.isna(value)):
        return f"{code}: — sin datos"
    spec = doms[code]
    nombre = spec.get("nombre", code)
    icono = spec.get("icono", "")
    msg = "— sin datos"
    for r in spec.get("rangos", []):
        lo, hi = float(r["min"]), float(r["max"])
        lo_inc = bool(r.get("min_inclusive", True))
        hi_inc = bool(r.get("max_inclusive", True))
        ok_lo = (value > lo) or (lo_inc and abs(value - lo) < 1e-9) or (value == lo if lo_inc else False)
        ok_hi = (value < hi) or (hi_inc and abs(value - hi) < 1e-9) or (value == hi if hi_inc else False)
        if ok_lo and ok_hi:
            msg = r.get("mensaje", msg)
            break
    val_txt = f"{value:.{DECIMALES}f}"
    return f"{code} ({nombre}): {val_txt} — {msg}"

def interpret_dominios(dom_values: Dict[str, float]) -> Dict[str, str]:
    """Devuelve un dict code -> string humanizada usando dominio.json."""
    out = {}
    for k, v in dom_values.items():
        out[k] = _humanize_single(k, v, DOM_CFG)
    return out
