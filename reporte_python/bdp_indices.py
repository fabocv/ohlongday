
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd

CORE_KEYS = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]

def _nz(x):
    return x if x is not None and not (isinstance(x,float) and pd.isna(x)) else None

def _infer_scale(vals: Dict[str, float] | None) -> str:
    if not vals: return "0-1"
    for k in CORE_KEYS:
        try:
            if float(vals.get(k, 0)) > 1.5:
                return "0-10"
        except Exception:
            pass
    return "0-1"

def _n01(v: Optional[float], scale: str) -> Optional[float]:
    if v is None: return None
    try:
        x = float(v)
    except Exception:
        return None
    if scale == "0-10":
        x = x / 10.0
    elif x > 10.5:
        x = x / 100.0
    if x < 0: x = 0.0
    if x > 1: x = 1.0
    return x

def _cafe_mg_from_cdta(cdta):
    try:
        return float(cdta) * 30.0
    except Exception:
        return None

def _cafe_mg_from_any(day_row):
    # Prioridad: mg directo si existe; si no, cucharaditas * 30 mg
    try:
        if day_row.get('cafeina_mg') not in (None, ''):
            return float(day_row.get('cafeina_mg'))
    except Exception:
        pass
    return _cafe_mg_from_cdta(day_row.get('cafe_cucharaditas'))

    try:
        return float(cdta) * 30.0
    except Exception:
        return None

def detect_indices(day_row: Dict[str,float], prev_row: Dict[str,float] | None = None) -> List[Dict[str,str]]:
    get = lambda k: _nz(day_row.get(k)) if day_row else None
    scale = _infer_scale(day_row)

    animo = _n01(get("animo"), scale)
    activ = _n01(get("activacion"), scale)
    sueno = _n01(get("sueno"), scale)
    conex = _n01(get("conexion"), scale)
    propo = _n01(get("proposito"), scale)
    clari = _n01(get("claridad"), scale)
    estres = _n01(get("estres"), scale)

    autoc = _n01(get("autocuidado"), scale)
    alim  = _n01(get("alimentacion"), scale)
    movim = _n01(get("movimiento"), scale)
    dolor = _n01(get("dolor_fisico"), scale)
    ans   = _n01(get("ansiedad"), scale)
    irri  = _n01(get("irritabilidad"), scale)
    med_min   = get("meditacion_min")
    sol_min   = get("exposicion_sol_min")
    agua_l    = get("agua_litros")
    cafe_cdta = get("cafe_cucharaditas")
    alc_ud    = get("alcohol_ud")
    cafe_mg = _cafe_mg_from_any(day_row)
    sueno_h   = get("sueno_horas")

    notes = day_row.get("__notas__", [])
    if isinstance(notes, list):
        txt = " ".join(str(x) for x in notes).lower()
        if "ansios" in txt:
            ans = max(ans or 0.0, 0.80)
        if "hipomani" in txt:
            activ = max(activ or 0.0, 0.80)
        if "estres" in txt or "estrés" in txt:
            estres = max(estres or 0.0, 0.70)

    d_activ = None
    if prev_row:
        pa = _n01(prev_row.get("activacion"), scale)
        if pa is not None and activ is not None:
            d_activ = activ - pa

    HI_T = 0.60
    LO_T = 0.35

    out: List[Dict[str,str]] = []

    risk = 0
    if cafe_mg and cafe_mg >= 180:
        risk += 1
    if isinstance(alc_ud, (int,float)) and alc_ud >= 2:
        risk += 1
    if isinstance(med_min, (int,float)) and med_min >= 10:
        risk -= 1
    if isinstance(sol_min, (int,float)) and sol_min >= 15:
        risk -= 1
    if isinstance(agua_l, (int,float)) and agua_l < 1.2:
        risk += 1
    if movim is not None and movim < 0.3:
        risk += 1
    if dolor is not None and dolor > 0.6:
        risk += 1

    hi = max(0.55, min(0.75, HI_T - 0.03*risk))
    lo = max(0.25, min(0.45, LO_T + 0.02*risk))

    if (animo is not None and animo < lo) and ((sueno is not None and sueno < lo) or (estres is not None and estres > hi)):
        out.append({"id":"DI","nombre":"Depresivo","mensaje":"Señales depresivas: prioriza descanso y micro-logros amables hoy."})
    elif (dolor is not None and dolor > 0.6) and ((autoc is not None and autoc < 0.4) or (alim is not None and alim < 0.4)):
        out.append({"id":"DI","nombre":"Depresivo","mensaje":"Dolor y autocuidado bajos: pasos pequeños y compasivos hoy."})

    if (ans is not None and ans > hi):
        out.append({"id":"AI","nombre":"Ansiedad","mensaje":"Ansiedad elevada: pauta 2-3 pausas de respiración guiada."})
    elif (activ is not None and activ > hi) and ((estres is not None and estres >= 0.60) or (sueno is not None and sueno < lo)):
        out.append({"id":"AI","nombre":"Ansiedad","mensaje":"Señales ansiosas: respiración pausada y micro-pausas pueden ayudar."})

    if (activ is not None and activ > hi) and (sueno is not None and sueno < lo) and ((clari is not None and clari < 0.5) or (propo is not None and propo > 0.6)):
        out.append({"id":"HI","nombre":"Hipomanía","mensaje":"Energía alta con poco sueño: acota tareas y fija límites suaves."})

    if (animo is not None and animo < lo) and (activ is not None and activ > hi) and ((estres is not None and estres > hi) or (irri is not None and irri > hi)):
        out.append({"id":"MI","nombre":"Mixto","mensaje":"Mixto: contención emocional + descarga física breve."})

    poco_sueno_horas = (isinstance(sueno_h,(int,float)) and sueno_h < 6)
    cond_base = (activ is not None and activ > hi) and ((clari is not None and clari < 0.45) or (sueno is not None and sueno < 0.45)) and (estres is not None and estres >= 0.5)
    cond_delta = (d_activ is not None and d_activ > 0.15) and (clari is not None and clari < 0.5)
    cond_habitos = ((cafe_mg and cafe_mg >= 180) or (isinstance(alc_ud,(int,float)) and alc_ud >= 2) or poco_sueno_horas) and (activ and activ > 0.55)
    if cond_base or cond_delta or cond_habitos:
        out.append({"id":"IPI","nombre":"Impulsividad","mensaje":"Impulsividad: aplica la regla de los 10 minutos antes de decidir."})

    if (conex is not None and conex < lo) and ((estres is not None and estres > 0.5) or (irri is not None and irri > hi)):
        out.append({"id":"RI","nombre":"Relacional","mensaje":"Relacional: nombra necesidades y establece límites amables."})

    if not out:
        out.append({"id":"SI","nombre":"Estabilidad","mensaje":"Eutimia: reconoce lo que sostiene tu bienestar hoy."})

    return out
