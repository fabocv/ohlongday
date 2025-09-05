# ========= Bin & helpers =========
import random
from bdp.calcs.espiritual import *

def bucket_0a10_idx(v: float) -> int:
    if v is None: return 0
    v = max(0.0, min(10.0, float(v)))
    return min(5, int(v // 2))  # 0..1.99→0, ..., 10→5

def _fmt_score(v): 
    return "s/d" if v is None else f"{v:.2f}/10"

# ========= Themes (título/emojis y tono) =========
THEMES = {
    "calido": {
        "emojis": {"activacion":"⚡","sueno":"🛌","conexion":"🤝","espiritual":"🕊️"},
        "style": {"emoji": True, "imperative": False}
    },
    "tecnico": {
        "emojis": {"activacion":"","sueno":"","conexion":"","espiritual":""},
        "style": {"emoji": False, "imperative": False}
    },
    "clinico": {
        "emojis": {"activacion":"","sueno":"","conexion":"","espiritual":""},
        "style": {"emoji": False, "imperative": True}
    },
    "minimal": {
        "emojis": {"activacion":"","sueno":"","conexion":"","espiritual":"🌿"},
        "style": {"emoji": True, "imperative": True}
    }
}

def _title(tipo, theme):
    t = theme if theme in THEMES else "calido"
    e = THEMES[t]["emojis"].get(tipo, "")
    return (e+" " if e else "") + tipo.capitalize()

# ========= Micro-frases por nivel (0..5) =========
PHRASE = {
  "activacion": {
    0: ["Agotamiento: baja la carga."],
    1: ["Energía baja: activa suave."],
    2: ["En ascenso: consolida rutina."],
    3: ["Estable: buen ritmo."],
    4: ["Alta: regula estímulos."],
    5: ["Sobre-activación: prioriza descanso."]
  },
  "sueno": {
    0: ["Muy corto: higiene urgente."],
    1: ["Induce sueño temprano hoy."],
    2: ["Mejorando: fija horario."],
    3: ["Estable: protege rutina."],
    4: ["Óptimo: 7–8 h constantes."],
    5: ["Excesivo: consulta si persiste."]
  },
  "conexion": {
    0: ["Aislamiento: 1 contacto seguro."],
    1: ["Baja conexión: prioriza 1:1 breve."],
    2: ["Subiendo: interacciones cortas."],
    3: ["Estable: calidad sobre cantidad."],
    4: ["Muy social: cuida límites."],
    5: ["Exceso social: dosifica."]
  },
  "espiritual": {
    0: ["Desconexión: 5′ de respiración."],
    1: ["Práctica baja: 10′ diarios."],
    2: ["En ascenso: horario fijo."],
    3: ["Creciente conexión espiritual."],
    4: ["Movida: prioriza calidad."],
    5: ["Sólida: sin rigidez."]
  },
  "carga_metab": {
    0: ["F0."],
    1: ["F1"],
    2: ["F2"],
    3: ["F3"],
    4: ["F4"],
    5: ["F5"]
  },
}

def frase_bloque(tipo:str, score:float, seed=None):
    idx = bucket_0a10_idx(score)
    opciones = PHRASE.get(tipo, {}).get(idx, ["Sin datos"])
    if seed is not None: random.seed(seed)
    return random.choice(opciones)

# ========= Insights base por banda+trend =========
def _band(score):
    if score is None: return 'nd'
    return 'bajo' if score < 3 else ('medio' if score < 6 else ('bueno' if score < 8 else 'alto'))

def _trend(delta):
    if delta is None: return 'estable'
    if delta >= 0.6: return 'fuerte_up'
    if delta >= 0.2: return 'suave_up'
    if delta > -0.2: return 'estable'
    if delta > -0.6: return 'suave_down'
    return 'fuerte_down'

INSIGHT_BASE = {
  "activacion": {
    "bajo": "Activa suave 10–15′ y cuida descansos.",
    "medio": "Mantén ritmo y evita sobrecarga nocturna.",
    "bueno": "Buen tono: micro-movimiento y cierre claro del día.",
    "alto": "Regula estímulos y planifica descargas controladas."
  },
  "sueno": {
    "bajo": "Rutina estricta: horario fijo y ambiente oscuro.",
    "medio": "Consolida rutina y siestas <20′ antes de 17:00.",
    "bueno": "Mantén horarios estables y luz tenue nocturna.",
    "alto": "Sostén hábitos; revisa excesos si aparece somnolencia."
  },
  "conexion": {
    "bajo": "Agenda 1 interacción segura de 10–20′.",
    "medio": "Prioriza calidad sobre cantidad; evita maratones.",
    "bueno": "Buen balance: mantén límites y pausas.",
    "alto": "Dosifica exposición y resguarda recuperación."
  },
  "espiritual": {
    "bajo": "Practicar gratitud y conexión espiritual.",
    "medio": "Sostén 10′ y añade 1 práctica distinta/semana.",
    "bueno": "Buena práctica: calidad y foco.",
    "alto": "Mantén sin rigidez; integra gratitud breve."
  },
  "carga_metab": {
    "bajo": "CM BAJA.",
    "medio": "CM MEDIA.",
    "bueno": "CM BUENA",
    "alto": "CM ALTA"
  }
}

ADJUST_TREND = {
  "suave_up": "Vas al alza: consolida con rutina simple.",
  "fuerte_up": "Gran subida: evita excesos; mantén consistencia.",
  "suave_down": "Ligera baja: reduce carga y vuelve a básicos.",
  "fuerte_down": "Baja marcada: protege descanso 2 noches."
}

# ========= Cross-heuristics (usa otras variables/tendencias) =========
def _cross_hints(tipo, score, ctx: dict):
    """
    ctx: dict opcional con promedios/flags últimos 7d, e.g.:
      sueno_score, activacion_score, estres_score,
      tiempo_pantalla_noche_min, cafe_tarde_bool, alcohol_ud,
      mov_intensidad, consistencia
    """
    if not ctx: return []

    hints = []
    pantallas = ctx.get("tiempo_pantalla_noche_min")
    cafe_tarde = ctx.get("cafe_tarde_bool")  # True si tomaste café después de 16:00 en ≥1 de los últimos 3 días
    alcohol = ctx.get("alcohol_ud")
    mov = ctx.get("mov_intensidad")
    sueno = ctx.get("sueno_score")
    act = ctx.get("activacion_score")
    estres = ctx.get("estres_score")
    consist = ctx.get("consistencia")

    if tipo == "sueno":
        if pantallas is not None and pantallas > 60: hints.append("Pantallas >60′ noche: recórtalas a 30′.")
        if cafe_tarde: hints.append("Sin cafeína después de las 16:00.")
        if alcohol and alcohol > 0: hints.append("Evita alcohol 3–4 h antes de dormir.")
    elif tipo == "activacion":
        if sueno is not None and sueno < 5: hints.append("Prioriza 2 noches de recuperación (sueño).")
        if mov is not None and score is not None and score < 3 and mov < 4: hints.append("Suma 10–15′ de caminata con sol.")
        if cafe_tarde: hints.append("Reduce cafeína vespertina.")
    elif tipo == "conexion":
        if estres is not None and estres >= 6 and score is not None and score < 6:
            hints.append("Prefiere 1:1 breve en entorno tranquilo.")
        if act is not None and act < 4 and score is not None and score < 4:
            hints.append("Encuentros cortos; evita multitudes.")
    elif tipo == "espiritual":
        if consist is not None and consist < 0.4: hints.append("Fija una hora diaria y repite 3 días seguidos.")
        if estres is not None and estres >= 6: hints.append("Incluye 3–5′ de respiración 4–6 al cierre del día.")
    return hints

# ========= Generador de insight con theme + cruces =========
def insight_bloque_tema(tipo:str, score:float, delta:float=None, consistencia:float=None, ctx:dict=None, theme:str="calido"):
    band = _band(score)
    tr = _trend(delta)
    base = INSIGHT_BASE.get(tipo, {}).get(band, "Sin datos.")
    adj = ADJUST_TREND.get(tr, None)
    cross = _cross_hints(tipo, score, {**(ctx or {}), "consistencia": consistencia})
    parts = [base]
    if adj: parts.append(adj)
    if cross: parts.append(" ".join(cross))
    insight = " ".join(parts)

    title = _title(tipo, theme)
    return title, insight

# ========= Bloque listo para usar =========
def render_bloque(tipo:str, score:float, delta:float=None, consistencia:float=None, ctx:dict=None, theme:str="calido"):
    titulo = _title(tipo, theme)
    score_txt = _fmt_score(score)
    frase = frase_bloque(tipo if tipo!="sueno" else "sueno", score)  # clave 'sueno' coincide con PHRASE
    _, insight = insight_bloque_tema(tipo if tipo!="sueno" else "sueno", score, delta, consistencia, ctx, theme)
    return {
        "title": titulo,           # ej. "🕊️ Espiritual"
        "score_txt": score_txt,    # ej. "5.90/10"
        "frase": frase,            # micro-frase por nivel
        "insight": insight         # insight con tendencia + cruces
    }


# --- Helpers ya definidos arriba: bucket_0a10_idx, PHRASE, THEMES, render_bloque,
# media_semanal_ultimos7, aplicar_fecha_logica, insight_bloque_tema, etc. ---

def weekly_score(df, col, modo='fair'):
    """Promedio 0–10 últimos 7 días móviles para la columna `col`."""
    prom, _, _, _ = media_semanal_ultimos7(df, col=col, modo=modo)
    return prom

def _last7_index(df):
    last_day = df['fecha_logica'].max()
    return pd.date_range(last_day - pd.Timedelta(days=6), last_day, freq='D')

def build_ctx(df):
    """Contexto cruzado (últimos 7 días) para insights."""
    idx7 = _last7_index(df)
    g = (df.groupby('fecha_logica').agg({
        # ajusta nombres si tus columnas difieren
        'tiempo_pantalla_noche_min': 'mean' if 'tiempo_pantalla_noche_min' in df.columns else 'mean',
        'alcohol_ud': 'sum' if 'alcohol_ud' in df.columns else 'sum',
        'mov_intensidad': 'mean' if 'mov_intensidad' in df.columns else 'mean',
        # puntajes 0–10 si existen
        'sueno': 'mean' if 'sueno' in df.columns else 'mean',
        'activacion': 'mean' if 'activacion' in df.columns else 'mean',
        'estres': 'mean' if 'estres' in df.columns else 'mean',
        'cafe_ultima_hora': 'max' if 'cafe_ultima_hora' in df.columns else 'max'
    }).reindex(idx7))
    # Safe gets
    pantallas = float(g.get('tiempo_pantalla_noche_min').mean()) if 'tiempo_pantalla_noche_min' in g else None
    alcohol   = float(g.get('alcohol_ud').sum()) if 'alcohol_ud' in g else None
    carga_metabolica = float(g.get('carga_metabolica').mean()) if 'carga_metabolica' in g else None
    mov       = float(g.get('mov_intensidad').mean()) if 'mov_intensidad' in g else None
    sueno_sc  = float(g.get('sueno').mean()) if 'sueno' in g else None
    act_sc    = float(g.get('activacion').mean()) if 'activacion' in g else None
    estres_sc = float(g.get('estres').mean()) if 'estres' in g else None
    cafe_tarde = bool((g.get('cafe_ultima_hora')>=16).fillna(False).any()) if 'cafe_ultima_hora' in g else None

    return {
        "tiempo_pantalla_noche_min": pantallas,
        "alcohol_ud": alcohol,
        "mov_intensidad": mov,
        "sueno_score": sueno_sc,
        "activacion_score": act_sc,
        "estres_score": estres_sc,
        "cafe_tarde_bool": cafe_tarde,
        "carga_metabolica":carga_metabolica
    }

def render_dashboard(df, theme="calido", modo='fair', consistencias=None, deltas=None):
    """
    Devuelve dict con 4 bloques listos: activación, sueño, conexión, espiritual.
    - `consistencias` y `deltas` son dict opcionales por clave.
    """
    df = aplicar_fecha_logica(df)
    ctx = build_ctx(df)

    # Detecta columna de sueño (ajusta si usas otra normalizada 0–10)
    sueno_col = 'sueno' if 'sueno' in df.columns else None

    scores = {
        "activacion": weekly_score(df, 'activacion', modo) if 'activacion' in df.columns else None,
        "sueno": weekly_score(df, sueno_col, modo) if sueno_col else None,
        "conexion": weekly_score(df, 'conexion', modo) if 'conexion' in df.columns else None,
        "espiritual": weekly_score(df, 'E_d', modo) if 'E_d' in df.columns else None,
        "carga_metabolica": weekly_score(df, "carga_metabolica",modo)
    }

    out = {}
    for k in ["activacion", "sueno", "conexion", "espiritual", "carga_metabolica"]:
        sc = scores[k]
        delta = (deltas or {}).get(k)   # puedes pasar EMA-slope si ya lo calculas
        cons  = (consistencias or {}).get(k, None)
        out[k] = render_bloque(k, sc, delta=delta, consistencia=cons, ctx=ctx, theme=theme)
    return out
