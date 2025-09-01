
from __future__ import annotations
from typing import Dict, List, Optional

ICONS = {
    "animo": "😊",
    "activacion": "⚡",
    "sueno": "🌙",
    "conexion": "🤝",
    "proposito": "🎯",
    "claridad": "🔍",
    "estres": "🔥",
}

AREA_LABEL = {
    "animo": "Ánimo",
    "activacion": "Activación",
    "sueno": "Sueño",
    "conexion": "Conexión",
    "proposito": "Propósito",
    "claridad": "Claridad",
    "estres": "Estrés",
}

CORE_KEYS = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]

def _infer_scale(vals: Dict[str, float] | None) -> str:
    if not vals: return "0-1"
    for k in CORE_KEYS:
        try:
            if float(vals.get(k, 0)) > 1.5:
                return "0-10"
        except Exception:
            pass
    return "0-1"

def _norm01(v: Optional[float], scale: str) -> Optional[float]:
    if v is None:
        return None
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

def _level_label_generic(v: Optional[float], scale: str) -> str:
    x = _norm01(v, scale)
    if x is None:
        return ""
    if x >= 2/3:
        return "alto"
    if x <= 1/3:
        return "bajo"
    return "medio"

def stack_human_messages(area_trends: Dict[str, str], current_values: Dict[str, float] | None = None) -> List[str]:
    order = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]
    msgs: List[str] = []
    vals = current_values or {}
    scale = _infer_scale(vals)
    for k in order:
        trend = area_trends.get(k, "flat")
        icon = ICONS.get(k, "•")
        name = AREA_LABEL.get(k, k.capitalize())
        lvl = _level_label_generic(vals.get(k), scale)
        if trend == "flat":
            msgs.append(f"{icon} {name}: se mantiene {lvl if lvl else 'estable'} ✔️")
        elif trend == "up":
            msgs.append(f"{icon} {name}: al alza ⬆️{f' ({lvl})' if lvl else ''}")
        else:  # down
            msgs.append(f'{icon} {name}: bajó ⬇️{f" ({lvl})" if lvl else ""}')
    return msgs

def kind_for_category(cat: int) -> str:
    if cat <= 0:
        return "Día muy desafiante: mereces pausas y cuidado extra."
    if cat == 1:
        return "Hubo baches, sí. Aún así, estás aquí y eso ya es avance. Celebra lo que sí se pudo."
    if cat == 3:
        return "Vas en camino. Sostén lo que funciona y date crédito por tu constancia."
    return "Día equilibrado: sigue con pasos pequeños y amables."
