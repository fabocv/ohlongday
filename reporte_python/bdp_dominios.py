
from __future__ import annotations
from typing import Dict, Any
import json
from pathlib import Path
import math

# Load dominio.json sitting next to this file
_CFG_PATH = Path(__file__).parent / "dominio.json"
with open(_CFG_PATH, "r", encoding="utf-8") as f:
    DOM_CFG = json.load(f)["dominios_humanizacion"]

def _mean(vals):
    nums = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(nums) / len(nums) if nums else float("nan")

def _wmean(pairs: Dict[str, float], row: Dict[str, float]):
    num = 0.0
    den = 0.0
    for k, w in pairs.items():
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)): 
            continue
        num += float(v) * float(w)
        den += float(w)
    return num / den if den > 0 else float("nan")

def compute_dominios(row: Dict[str, float]) -> Dict[str, float]:
    """
    row: dict con áreas diarias promedio (0..1): animo, activacion, sueno, conexion, proposito, claridad, estres
    return: dict con H, V, C, P, S-
    """
    H = _mean([row.get("animo")])
    V = _wmean({"activacion": 0.5, "sueno": 0.5}, row)
    C = _mean([row.get("conexion")])
    P = _wmean({"proposito": 0.6, "claridad": 0.4}, row)
    estres = row.get("estres")
    S_inv = (1.0 - float(estres)) if (estres is not None and not (isinstance(estres, float) and math.isnan(estres))) else float("nan")
    return {"H": H, "V": V, "C": C, "P": P, "S-": S_inv}

def interpret_dominios(dom_values: Dict[str, float], cfg: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    """
    dom_values: dict H,V,C,P,S-
    return: dict {code: {"valor": float, "mensaje": str, "nombre": str, "icono": str}}
    """
    cfg = cfg or DOM_CFG
    dec = int(cfg.get("decimales", 2))
    out: Dict[str, Dict[str, Any]] = {}
    dominios = cfg["dominios"]
    for code, meta in dominios.items():
        v = dom_values.get(code)
        nombre = meta.get("nombre", code)
        icono = meta.get("icono", "")
        msg = ""
        # pick range
        for r in meta.get("rangos", []):
            lo, hi = float(r["min"]), float(r["max"])
            lo_ok = (v >= lo) if r.get("min_inclusive", True) else (v > lo)
            hi_ok = (v <= hi) if r.get("max_inclusive", False) else (v < hi)
            if v is not None and not (isinstance(v, float) and math.isnan(v)) and lo_ok and hi_ok:
                msg = r["mensaje"]
                break
        out[code] = {
            "valor": None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), dec),
            "mensaje": msg,
            "nombre": nombre,
            "icono": icono,
        }
    return out

def format_chip(code: str, info: Dict[str, Any], cfg: Dict[str, Any] = None) -> str:
    cfg = cfg or DOM_CFG
    plantilla = cfg.get("formato_chip", "{codigo} ({nombre}): {valor} — {mensaje}")
    val = info.get("valor")
    sval = "—" if val is None else f"{val:.{int(cfg.get('decimales',2))}f}"
    return plantilla.format(codigo=code, nombre=info.get("nombre",""), valor=sval, mensaje=info.get("mensaje",""))
