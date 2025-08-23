
from __future__ import annotations
import os, json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Paquete local
from reporte_python.bdp_schema import DEFAULT_COLUMNS
from reporte_python.bdp_cards import build_daily_cards
from reporte_python.bdp_dominios import compute_dominios_from_row, interpret_dominios
from reporte_python.bdp_messages import stack_human_messages
from reporte_python.bdp_indices import detect_indices

# ------------------------ Config desde JSON ------------------------
CONFIG_PATH = Path(__file__).parent / "bdp_config.json"

_CFG: Dict[str,Any] = {}

## Etiquetas humanas para áreas (con acentos)
AREA_LABEL = {
    "animo": "ánimo",
    "activacion": "activación",
    "sueno": "sueño",
    "conexion": "conexión",
    "proposito": "propósito",
    "claridad": "claridad",
    "estres": "estrés",
}

def _labelize(areas: list[str]) -> list[str]:
    return [AREA_LABEL.get(a, a) for a in areas]

def _title_es(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]

def _load_config(config: Dict[str,Any] | None) -> Dict[str,Any]:
    base: Dict[str,Any] = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)
    except FileNotFoundError:
        base = {}
    if config:
        merged = dict(base)
        merged.update(config)
        return merged
    return base or {}

# ------------------------ HTML base ------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Informe BDP - Coaching</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #0f1115;
      --panel:#151923;
      --muted:#9aa4b2;
      --text:#e6e9ef;
      --chip:#232a36;
      --border:#2a3140;
    }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
           margin: 0; background: var(--bg); color: var(--text); }
    .wrap { max-width: 1000px; margin: 36px auto; padding: 0 16px; }
    h1 { margin: 8px 0 4px; font-size: 28px; }
    p.lead { color: var(--muted); margin: 0 0 16px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }
    .missing { opacity: .7; }
    .date { font-weight: 700; font-size: 15px; }
    .meta { color: var(--muted); font-size: 13px; }
    .section-title { font-weight: 700; font-size: 13px; margin-top: 8px; }
    .areas { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .chip { background: var(--chip); border: 1px solid var(--border);
            padding: 6px 8px; border-radius: 999px; font-size: 12px; }
    .list { margin: 8px 0 0 0; padding-left: 18px; }
    
    .section-title.notes { font-size: 12px; margin-top: 6px; }
    .list.notes { font-size: 12px; line-height: 1.3; margin-top: 4px; }
.footer { margin-top: 10px; border-top: 1px dashed var(--border); padding-top: 10px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Informe BDP (Coaching)</h1>
    <p class="lead">Resumen diario con interpretaciones amables, dominios humanizados, tendencias y comparación con el día anterior.</p>
    <div class="grid">
      %%CARDS_HTML%%
    </div>
  </div>
</body>
</html>
"""

# ------------------------ Helpers ------------------------

# Áreas núcleo ponderadas para balance de bienestar (0..1); estrés resta.
WEIGHTS_WB = {
    "animo": 0.27,
    "sueno": 0.26,
    "conexion": 0.14,
    "proposito": 0.12,
    "claridad": 0.09,
    "activacion": 0.03,
    "estres": -0.35
}

def _wb_score(ctx: Dict[str,Any] | None) -> float | None:
    if not ctx:
        return None
    # inferir escala del día
    scale = "0-1"
    for k in ["animo","activacion","sueno","conexion","proposito","claridad","estres"]:
        try:
            if float(ctx.get(k, 0)) > 1.5:
                scale = "0-10"; break
        except Exception:
            pass
    def n01(v):
        try:
            x = float(v)
        except Exception:
            return None
        if scale == "0-10": x /= 10.0
        elif x > 10.5: x /= 100.0
        return 0.0 if x < 0 else 1.0 if x > 1 else x
    total = 0.0; have = False
    for k, w in WEIGHTS_WB.items():
        v = n01(ctx.get(k))
        if v is None: continue
        have = True
        total += w * v
    return total if have else None

def _fmt_num(x: float, cfg: Dict[str,Any], digits: int = 2, signed: bool = True) -> str:
    """Formatea números respetando locale.decimal_comma."""
    try:
        f = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    except Exception:
        return str(x)
    if cfg.get("locale", {}).get("decimal_comma"):
        f = f.replace(".", ",")
    return f


def _fmt_glic_chip(val: float, cfg: Dict[str,Any]) -> str:
    m = cfg.get("metabolico", {})
    es_dm = bool(m.get("es_diabetico", True))
    max_norm = float(m.get("glicemia_max_normal", 100))
    acept_dm_max = float(m.get("glicemia_aceptable_diabetico_max", 170))
    alta_umbral = float(m.get("glicemia_alta_umbral", 180))
    try:
        v = float(val)
    except Exception:
        return ""
    if v < 70:                col = "#5aa9ff"  # baja
    elif v <= max_norm:       col = "#22c55e"  # normal
    elif es_dm and v <= acept_dm_max: col = "#f59e0b"  # aceptable DM
    elif v >= alta_umbral:    col = "#ef4444"  # alta
    else:                     col = "#fb923c"  # elevada
    return f'<span class="chip">🩸 Glic. <span style="color:{col};font-weight:700">{v:.0f}</span> mg/dL</span>'


def _movido_line(deltas: Dict[str,float], cfg: Dict[str,Any]) -> str:
    thr = cfg["umbrales"]["delta_significativo"]
    min_areas = cfg["umbrales"]["min_areas_para_movido"]
    min_pct = cfg["umbrales"]["min_porcentaje_movido"]
    core_deltas = {k: v for k, v in deltas.items() if k in CORE_AREAS}
    cand = [a for a, v in core_deltas.items() if v == v and abs(float(v)) >= thr]
    total = len(core_deltas)
    if not cand:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    is_movido = len(cand) >= min_areas or (len(cand)/max(1, total)) >= min_pct
    if not is_movido:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
        cand_sorted = sorted(cand, key=lambda a: abs(core_deltas[a]), reverse=True)[:4]
    # Etiquetas humanas + capitalización (Sueño, Ánimo, …)
    pretty = [_title_es(x) for x in _labelize(cand_sorted)]
    return cfg["mensajes_movimiento"]["intro_mas_movido"].format(areas_list=", ".join(pretty))

CORE_AREAS = ["animo","activacion","sueno","conexion","proposito","claridad","estres"]

# --- Comparación: métricas de contexto a listar (con íconos y etiquetas) ---
CONTEXT_ITEMS = [
    ("agua_litros",       "💧", "Agua_litros"),
    ("glicemia",          "🩸", "Glicemia"),
    ("meditacion_min",    "🧘", "Meditacion_min"),
    ("exposicion_sol_min","☀️", "Exposicion_sol_min"),
    ("ansiedad",          "😟", "Ansiedad"),
]


# (las demás variables se consideran contexto, no para comparación genérica)
# ------------------------ Helpers ------------------------

def _movido_line(deltas: Dict[str,float], cfg: Dict[str,Any]) -> str:
    thr = cfg["umbrales"]["delta_significativo"]
    min_areas = cfg["umbrales"]["min_areas_para_movido"]
    min_pct = cfg["umbrales"]["min_porcentaje_movido"]
    core_deltas = {k: v for k, v in deltas.items() if k in CORE_AREAS}
    cand = [a for a, v in core_deltas.items() if v == v and abs(float(v)) >= thr]
    total = len(core_deltas)
    if not cand:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    is_movido = len(cand) >= min_areas or (len(cand)/max(1, total)) >= min_pct
    if not is_movido:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    cand_sorted = sorted(cand, key=lambda a: abs(core_deltas[a]), reverse=True)[:4]
    return cfg["mensajes_movimiento"]["intro_mas_movido"].format(areas_list=", ".join(cand_sorted))

import unicodedata

def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.strip().lower()

def _resolve_columns(df: pd.DataFrame, expected: Dict[str,str]) -> pd.DataFrame:
    """
    Intenta mapear/renombrar columnas reales a los nombres esperados en DEFAULT_COLUMNS,
    usando comparación insensible a mayúsculas/acentos/espacios.
    No elimina columnas; sólo renombra si encuentra equivalentes.
    """
    if df is None or df.empty:
        return df
    # Normaliza encabezados reales
    real_cols = list(df.columns)
    norm_to_real = {_norm(c): c for c in real_cols}
    rename_map = {}
    for k, exp in expected.items():
        if not isinstance(exp, str): 
            continue
        if exp in df.columns:
            continue  # ya está
        # buscar por forma normalizada
        real = norm_to_real.get(_norm(exp))
        if real:
            rename_map[real] = exp
        else:
            # sinónimo básico para fecha
            if k == "fecha":
                for alt in ["fecha", "fechas", "dia", "día", "date", "fecha_registro"]:
                    real = norm_to_real.get(_norm(alt))
                    if real and real not in rename_map:
                        rename_map[real] = exp
                        break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _fmt_date_es(val) -> str:
    meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    try:
        ts = pd.to_datetime(val, errors="coerce", dayfirst=True)
        if pd.isna(ts): raise ValueError
        return f"{ts.day} de {meses[int(ts.month)-1].capitalize()} de {ts.year}"
    except Exception:
        return str(val)

def _norm01_autoscale(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if x > 10.5: x = x/100.0
    elif x > 1.5: x = x/10.0
    if x < 0: x = 0.0
    if x > 1: x = 1.0
    return x

def _level_label(v: Any, ctx: Dict[str,Any] | None = None) -> str:
    scale = "0-1"
    if ctx:
        try:
            for k in ["animo","activacion","sueno","conexion","proposito","claridad","estres"]:
                if float(ctx.get(k, 0)) > 1.5:
                    scale = "0-10"; break
        except Exception:
            pass
    try:
        x = float(v)
    except Exception:
        return ""
    if scale == "0-10":
        x = x/10.0
    elif x > 10.5:
        x = x/100.0
    if pd.isna(x): return ""
    if x >= 2/3: return "alto"
    if x <= 1/3: return "bajo"
    return "medio"

def _frase_general(deltas: Dict[str,float], cfg: Dict[str,Any],
                   curr_ctx: Dict[str,Any] | None = None,
                   prev_ctx: Dict[str,Any] | None = None) -> str:
    if prev_ctx is None:
        return cfg["frases_generales"]["sin_prev"]
    s_curr = _wb_score(curr_ctx)
    s_prev = _wb_score(prev_ctx)
    if s_curr is None or s_prev is None:
        return cfg["frases_generales"]["sin_datos"]
    delta = s_curr - s_prev
    thr_f = cfg.get("umbrales_balance", {}).get("delta_fuerte", 0.15)
    thr_l = cfg.get("umbrales_balance", {}).get("delta_leve", 0.05)
    if   delta >=  thr_f: return cfg["frases_generales"]["mejora_fuerte"]
    elif delta >=  thr_l: return cfg["frases_generales"]["mejora"]
    elif delta <= -thr_f: return cfg["frases_generales"]["empeora_fuerte"]
    elif delta <= -thr_l: return cfg["frases_generales"]["empeora"]
    # delta pequeño: ¿mixto?
    pos = neg = 0
    for a, dv in deltas.items():
        if a.startswith("__") or dv is None: continue
        try: x = float(dv)
        except: continue
        if a == "estres":
            if x < 0: pos += 1
            elif x > 0: neg += 1
        else:
            if x > 0: pos += 1
            elif x < 0: neg += 1
    return cfg["frases_generales"]["mixto"] if (pos and neg) else cfg["frases_generales"]["estable"]


def _movido_line(deltas: Dict[str,float], cfg: Dict[str,Any]) -> str:
    thr = cfg["umbrales"]["delta_significativo"]
    min_areas = cfg["umbrales"]["min_areas_para_movido"]
    min_pct = cfg["umbrales"]["min_porcentaje_movido"]
    core_deltas = {k: v for k, v in deltas.items() if k in CORE_AREAS}
    cand = [a for a, v in core_deltas.items() if v == v and abs(float(v)) >= thr]
    total = len(core_deltas)
    if not cand:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    is_movido = len(cand) >= min_areas or (len(cand)/max(1, total)) >= min_pct
    if not is_movido:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    cand_sorted = sorted(cand, key=lambda a: abs(core_deltas[a]), reverse=True)[:4]
    return cfg["mensajes_movimiento"]["intro_mas_movido"].format(areas_list=", ".join(cand_sorted))

def _detalle_areas(deltas: Dict[str,float], cfg: Dict[str,Any], curr_res: Dict[str,float] | None = None) -> List[str]:
    """
    Muestra viñetas de métricas de contexto (con ícono) y agrega
    la línea de Sueño al final. Omite entradas con Δ NaN.
    """
    import pandas as pd

    out: List[str] = []
    thr = cfg["umbrales"]["delta_significativo"]
    big = cfg["umbrales"]["delta_fuerte"]

    def _texto_delta(dv: float) -> str:
        ad = abs(float(dv))
        if ad >= big:
            return "subió mucho" if dv > 0 else "bajó mucho"
        elif ad >= thr:
            return "al alza" if dv > 0 else "bajó"
        else:
            return "estable"

    # --- 1) Viñetas de contexto (en el orden pedido) ---
    for key, icon, label in CONTEXT_ITEMS:
        dv = deltas.get(key, float("nan"))
        if pd.isna(dv):            # <- no mostramos NaN
            continue
        txt = _texto_delta(dv)
        out.append(f"{icon} {label}: {txt} (Δ {_fmt_num(dv, cfg)}).")

    # --- 2) Línea de Sueño con ícono (si hay Δ) ---
    dv_su = deltas.get("sueno", float("nan"))
    if not pd.isna(dv_su):
        txt_su = _texto_delta(dv_su)
        out.append(f"🌙 Sueño: {txt_su} (Δ {_fmt_num(dv_su, cfg)}).")

    return out


def _balance_and_label(deltas: Dict[str,float], cfg: Dict[str,Any], curr_ctx: Dict[str,Any] | None = None, prev_ctx: Dict[str,Any] | None = None) -> str:
    fg = cfg["frases_generales"]
    if prev_ctx is None or curr_ctx is None:
        return f"<strong>Balance global:</strong> {fg['estable']}"
    s_curr = _wb_score(curr_ctx)
    s_prev = _wb_score(prev_ctx)
    if s_curr is None or s_prev is None:
        return f"<strong>Balance global:</strong> {fg['estable']}"
    delta = s_curr - s_prev
    thr = cfg.get("umbrales_balance", {})
    thr_f = float(thr.get("delta_fuerte", 0.15))
    thr_l = float(thr.get("delta_leve", 0.05))
    if   delta >=  thr_f: return f"<strong>Balance global:</strong> {fg['mejora_fuerte']}"
    elif delta >=  thr_l: return f"<strong>Balance global:</strong> {fg['mejora']}"
    elif delta <= -thr_f: return f"<strong>Balance global:</strong> {fg['empeora_fuerte']}"
    elif delta <= -thr_l: return f"<strong>Balance global:</strong> {fg['empeora']}"
    else:                 return f"<strong>Balance global:</strong> {fg['estable']}"

def _comparacion(prev_res: Dict[str,float] | None, curr_res: Dict[str,float], cfg: Dict[str,Any]) -> Dict[str,Any]:
    if prev_res is None:
        return { "frase_general": cfg["frases_generales"]["sin_prev"],
                 "mensaje_movimiento": "", "lista_detalle_areas": [],
                 "resumen_compuesto": "<strong>Balance global:</strong> ✔️ estable" }
    if not curr_res:
        return { "frase_general": cfg["frases_generales"]["sin_datos"],
                 "mensaje_movimiento": "", "lista_detalle_areas": [],
                 "resumen_compuesto": "<strong>Balance global:</strong> ✔️ estable" }
    keys = sorted(set(prev_res.keys()) | set(curr_res.keys()))
    deltas = { k: (curr_res.get(k) - prev_res.get(k) if pd.notna(curr_res.get(k)) and pd.notna(prev_res.get(k)) else float('nan')) for k in keys }
    frase_general = _frase_general(deltas, cfg, curr_ctx=curr_res, prev_ctx=prev_res)
    mensaje_movimiento = _movido_line(deltas, cfg)
    detalle = _detalle_areas(deltas, cfg, curr_res=curr_res)
    resumen = _balance_and_label(deltas, cfg, curr_ctx=curr_res, prev_ctx=prev_res)
    return { "frase_general": frase_general, "mensaje_movimiento": mensaje_movimiento, "lista_detalle_areas": detalle, "resumen_compuesto": resumen }

def _contexto_chips(day_summary: Dict[str,Any], cfg: Dict[str,Any]) -> List[str]:
    # Respeta config 'render.mostrar_contexto'
    if not cfg.get("render", {}).get("mostrar_contexto", False):
        return []
    chips: List[str] = []
    ctx = day_summary or {}
    # (dejar mínimo para que no sea redundante; se puede reactivar más tarde)
    cdta = ctx.get("cafe_cucharaditas")
    if isinstance(cdta, (int,float)):
        mg = cdta * 30.0
        chips.append(f'<span class="chip">☕ {cdta:.0f} cdta (~{mg:.0f} mg)</span>')
    alc = ctx.get("alcohol_ud")
    if isinstance(alc, (int,float)):
        chips.append(f'<span class="chip">🍺 {alc:.0f} ud</span>')
    # Glicemia (si existe)
    gc = ctx.get("glicemia")
    try:
        if gc is not None and str(gc).strip() != "" and not pd.isna(float(gc)):
            chips.append(_fmt_glic_chip(gc, cfg))
    except Exception:
        pass

# ------------------------ Render de una card ------------------------

def _render_card(day: Dict[str,Any], prev_real: Dict[str,Any] | None, cfg: Dict[str,Any]) -> str:
    classes = "card" + (" missing" if day.get("faltante") else "")
    html: List[str] = [f'<div class="{classes}">']

    html.append(f'<div class="date">{_fmt_date_es(day.get("fecha"))}</div>')
    regs = int(day.get("registros", 0))
    html.append(f'<div class="meta"><b>Registros:</b> {regs}</div>')

    areas = day.get("resumen_areas") or {}
    icons = cfg["iconos_areas"]
    if areas:
        html.append('<div class="section-title">Áreas (promedios del día):</div>')
        html.append('<div class="areas">')
        for k in ["animo","activacion","sueno","conexion","proposito","claridad","estres"]:
            if k in areas and pd.notna(areas[k]):
                icon = icons.get(k,"•")
                html.append(f'<span class="chip">{icon} {k}: {areas[k]:.2f}</span>')
        html.append('</div>')

    if areas:
        dom_vals = compute_dominios_from_row(areas)
        dom_human = interpret_dominios(dom_vals)
        html.append('<div class="section-title">Dominios (promedios del día):</div>')
        html.append('<div class="areas">')
        for code in ["H","V","C","P","S-"]:
            if code in dom_human:
                html.append(f'<span class="chip">{dom_human[code]}</span>')
        html.append('</div>')

    ctx_chips = _contexto_chips(areas, cfg)
    if ctx_chips:
        html.append('<div class="section-title">Contexto del día</div>')
        html.append('<div class="areas">' + " ".join(ctx_chips) + '</div>')

    if day.get("notas"):
        html.append('<div class="section-title notes">Notas</div>')
        # limpia vacías/nan y dedup
        seen = set(); cleaned = []
        for it in day["notas"]:
            s = str(it).strip()
            payload = s.split("]", 1)[1].strip() if "]" in s else s
            if payload == "" or payload.lower() in {"nan","none","null"}:
                continue
            if s in seen: continue
            seen.add(s); cleaned.append(s)
        if cleaned:
            html.append('<ul class="list notes">')
            for n in cleaned:
                html.append(f'<li>{n}</li>')
            html.append('</ul>')



    raw_stress = day.get("estresores", [])
    seen_s = set()
    cleaned_s = []
    for it in raw_stress:
        s = str(it).strip()
        if not s or s.lower() in {"nan","none","null"}:
            continue
        if s in seen_s:
            continue
        seen_s.add(s)
        cleaned_s.append(s)
    if cleaned_s:
        html.append('<div class="section-title">Estresores</div>')
        html.append('<ul class="list">')
        for e in cleaned_s:
            html.append(f'<li>{e}</li>')
        html.append('</ul>')

    # Mensaje humano (tendencias) — ahora usamos niveles para TODAS las áreas
    if areas and day.get("mensajes_humanos"):
        # mensajes_humanos ya fue construido con current values
        html.append('<div class="section-title">Mensaje humano (tendencias)</div>')
        html.append('<ul class="list notes">')
        for m in day["mensajes_humanos"]:
            html.append(f'<li>{m}</li>')
        html.append('</ul>')

    indices = []
    if areas:
        prev_res = prev_real.get("resumen_areas") if prev_real else None
        indices = detect_indices({**areas, "__notas__": day.get("notas", [])}, prev_res)
    if indices:
        html.append('<div class="section-title">📊 Índices clave</div>')
        html.append('<ul class="list notes">')
        for it in indices:
            html.append(f'<li><strong>{it["id"]} ({it["nombre"]}):</strong> {it["mensaje"]}</li>')
        html.append('</ul>')

    inter = day.get("interpretacion_dia")
    if inter:
        html.append('<div class="footer">')
        html.append(f'<div><b>{_load_config(None)["nombres"]["footer_key"]}:</b> {inter}</div>')
        html.append('</div>')

    comp = _comparacion(prev_real.get("resumen_areas") if prev_real else None, areas, _load_config(None))
    html.append('<div class="footer">')
    html.append(f'<div><b>{_load_config(None)["nombres"]["comparacion_key"]}:</b> {comp["frase_general"]}</div>')
    if comp.get("mensaje_movimiento"):
        html.append(f'<div>{comp["mensaje_movimiento"]}</div>')
    if comp.get("lista_detalle_areas"):
        html.append('<ul class="list notes">')
        for line in comp["lista_detalle_areas"][:6]:
            html.append(f'<li>{line}</li>')
        html.append('</ul>')
    html.append(f'<div class="section-title">{comp["resumen_compuesto"]}</div>')
    html.append('</div>')  # footer

    html.append('</div>')  # card
    return "\n".join(html)

# ------------------------ Generador principal ------------------------

def generate_report_coach(input_csv: str,
                          output_html: str,
                          config: Dict[str,Any] | None = None,
                          start_date: str | None = None,
                          end_date: str | None = None,
                          tag_filter: str | None = None) -> str:
    cfg = _load_config(config)
    global _CFG
    _CFG = cfg

    df = pd.read_csv(input_csv, dtype={"hora": "string"}, sep=None, engine="python")
    # normaliza encabezados crudos
    df.columns = [str(c).strip() for c in df.columns]
    # renombra para que coincidan con DEFAULT_COLUMNS cuando sea posible
    df = _resolve_columns(df, DEFAULT_COLUMNS)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    if start_date:
        sd = pd.to_datetime(start_date, errors="coerce", dayfirst=True)
        df = df[df["fecha"] >= sd]
    if end_date:
        ed = pd.to_datetime(end_date, errors="coerce", dayfirst=True)
        df = df[df["fecha"] <= ed]

    if tag_filter and "tags" in df.columns:
        df = df[df["tags"].astype(str).str.contains(tag_filter, case=False, na=False)]

    cards = build_daily_cards(df, DEFAULT_COLUMNS, fill_missing_days=True)

    def to_dict(c):
        if isinstance(c, dict): return c
        return dict(
            fecha=getattr(c, "fecha", ""),
            registros=getattr(c, "registros", 0),
            resumen_areas=getattr(c, "resumen_areas", {}),
            mensajes_humanos=getattr(c, "mensajes_humanos", []),
            notas=getattr(c, "notas", []),
            estresores=getattr(c, "estresores", []),
            interpretacion_dia=getattr(c, "interpretacion_dia", ""),
            faltante=getattr(c, "faltante", False),
        )
    cards = [to_dict(c) for c in cards]

    html_cards: List[str] = []
    prev_real = None
    for c in cards:
        html_cards.append(_render_card(c, prev_real, cfg))
        if not c.get("faltante"):
            prev_real = c

    final_html = HTML_TEMPLATE.replace("%%CARDS_HTML%%", "\n".join(html_cards))

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    return os.path.abspath(output_html)
