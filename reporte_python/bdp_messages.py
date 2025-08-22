
from __future__ import annotations
from typing import Dict, List

# Escala fenomenológica 0–3
LABELS = {
    0: "Muy negativo / Bloqueo fuerte",
    1: "Débil / Retroceso",
    2: "Aceptable / En camino",
    3: "Positivo / Avance claro",
}

# Amabilidad por categoría diaria
KIND_BY_CAT = {
    0: "Día desafiante. Te mereces descanso y cuidado suave. Un paso pequeño vale oro hoy.",
    1: "Hubo baches, sí. Aún así, estás aquí y eso ya es avance. Celebra lo que sí se pudo.",
    2: "Vas en camino. Sostén lo que funciona y date crédito por tu constancia.",
    3: "¡Bien! Notable claridad y regulación. Agradece el cuerpo y comparte un poquito de esa luz.",
}

# Iconos por área
ICONS = {
    "animo": "😊",
    "activacion": "⚡",
    "sueno": "🌙",
    "conexion": "🤝",
    "proposito": "🎯",
    "claridad": "🔍",
    "estres": "🔥"
}

# Etiquetas y iconos de tendencia
TREND_LABELS = {"up": "al alza", "down": "a la baja", "flat": "estable"}
TREND_ICONS = {"up": "⬆️", "down": "⬇️", "flat": "✔️"}

def stack_human_messages(area_trends: Dict[str, str]) -> List[str]:
    msgs = []
    for area, trend in area_trends.items():
        label = TREND_LABELS.get(trend, "estable")
        ticon = TREND_ICONS.get(trend, "✔️")
        aicon = ICONS.get(area, "•")
        # Normaliza el nombre del área para display
        area_name = area.capitalize() if isinstance(area, str) else str(area)
        msgs.append(f"{aicon} {area_name}: {label} {ticon}")
    return msgs

def kind_for_category(cat: int) -> str:
    return KIND_BY_CAT.get(cat, "Hoy eres suficiente. Escúchate con cariño.")
