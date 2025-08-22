
import os, json, math
import pandas as pd
from typing import List, Dict, Any
from reporte_python.bdp_schema import DEFAULT_COLUMNS, AREAS
from reporte_python.bdp_cards import build_daily_cards
from datetime import datetime

DEFAULT_CFG = {"nombres": {"footer_key": "Interpretación del día", "comparacion_key": "Comparación con el día anterior"}, "umbrales": {"delta_significativo": 0.05, "delta_fuerte": 0.15, "min_areas_para_movido": 2, "min_porcentaje_movido": 0.4}, "peso_areas": {"animo": 1.0, "activacion": 1.0, "sueno": 1.0, "conexion": 0.8, "proposito": 0.8, "claridad": 1.0, "estres": 1.2}, "mapeo_tendencia": {"up": "al alza", "down": "a la baja", "flat": "estable"}, "iconos": {"animo": "😊", "activacion": "⚡", "sueno": "🌙", "conexion": "🤝", "proposito": "🎯", "claridad": "🔍", "estres": "🔥"}, "frases_generales": {"sin_prev": "Primera medición del período: no hay día anterior para comparar.", "sin_datos": "Hoy no hubo registros; se mantiene la última referencia conocida.", "mejora": "En comparación al día de ayer, tu día mostró una <strong>mejora general</strong>.", "empeora": "En comparación al día de ayer, tu día tuvo <strong>mayor dificultad</strong>.", "estable": "Comparado con ayer, el panorama se mantuvo <strong>relativamente estable</strong>.", "mixto": "Hubo <strong>cambios mixtos</strong> respecto a ayer: mejoras en algunas áreas y retrocesos en otras."}, "mensajes_movimiento": {"intro_mas_movido": "En comparación al día de ayer, tu día estuvo <strong>más movido</strong> por: {areas_list}.", "intro_menos_movido": "Respecto a ayer, hoy se percibe <strong>menos movimiento</strong> general.", "criterio": "Se considera 'más movido' si el número de áreas con cambios significativos supera {min_areas} o {min_pct}% del total medido."}, "plantillas_detalle": {"area_mejora_fuerte": "{icon} {area}: <strong>mejoró mucho</strong> (Δ +{delta}).", "area_mejora": "{icon} {area}: <strong>mejoró</strong> (Δ +{delta}).", "area_empeora_fuerte": "{icon} {area}: <strong>empeoró notablemente</strong> (Δ −{delta}).", "area_empeora": "{icon} {area}: <strong>descendió</strong> (Δ −{delta}).", "area_estable": "{icon} {area}: <strong>estable</strong> (Δ {delta}).", "estres_inversion_mejora": "{icon} Estrés: <strong>disminuyó</strong> (Δ −{delta}), lo cual favorece el balance.", "estres_inversion_empeora": "{icon} Estrés: <strong>aumentó</strong> (Δ +{delta}), podría influir en la percepción general."}, "resumen_compuesto": {"mejora_fuerte": "Balance global: <strong>mejoró con claridad</strong>.", "mejora": "Balance global: <strong>ligera mejora</strong>.", "estable": "Balance global: <strong>sin cambios relevantes</strong>.", "empeora": "Balance global: <strong>ligero retroceso</strong>.", "empeora_fuerte": "Balance global: <strong>retroceso marcado</strong>."}, "seleccion_areas_destacadas": {"max_listar": 4, "orden": "por_magnitud_desc", "notas": "Listar primero por |Δ|."}, "renderizado": {"bloques": ["frase_general", "mensaje_movimiento", "lista_detalle_areas", "resumen_compuesto"], "unir_lineas_con": " "}}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Informe BDP - Coaching</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #151923;
      --muted: #9aa4b2;
      --text: #e6e9ef;
      --ok: #0aa57f;
      --warn: #f2a365;
      --bad: #e86d6d;
      --chip: #232a36;
      --border: #2a3140;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; 
           margin: 0; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 960px; margin: 40px auto; padding: 0 16px; }}
    h1 {{ margin: 8px 0 4px; font-size: 28px; }}
    p.lead {{ color: var(--muted); margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }}
    .date {{ font-weight: 700; font-size: 15px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .areas {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .chip {{ background: var(--chip); border: 1px solid var(--border); padding: 6px 8px; border-radius: 999px; font-size: 12px; }}
    .list {{ margin: 8px 0 0 0; padding-left: 18px; }}
    .footer {{ margin-top: 10px; border-top: 1px dashed var(--border); padding-top: 10px; font-size: 13px; }}
    .missing {{ opacity: 0.6; }}
    .section-title {{ font-weight: 700; font-size: 13px; margin-top: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Informe BDP (Coaching)</h1>
    <p class="lead">Resumen diario con interpretaciones amables, tendencias por área y comparación con el día anterior.</p>
    <div class="grid">
      {cards_html}
    </div>
  </div>
</body>
</html>
"""

def _load_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    return config if config is not None else DEFAULT_CFG

def _balance_and_label(deltas: Dict[str, float], cfg: Dict[str, Any]) -> str:
    # returns HTML string with <strong>Balance global:</strong> + iconified label
    weights = cfg["peso_areas"]
    score = 0.0
    mag = 0.0
    for a, dv in deltas.items():
        if a.startswith("__"): 
            continue
        w = float(weights.get(a, 1.0))
        if pd.isna(dv):
            continue
        sgn = (-1 if dv > 0 else (1 if dv < 0 else 0)) if a == "estres" else (1 if dv > 0 else (-1 if dv < 0 else 0))
        score += w * sgn * abs(dv)
        mag += w * abs(dv)
    if mag < max(1e-9, cfg["umbrales"]["delta_significativo"]):
        return "<strong>Balance global:</strong> ✔️ estable ✔️"
    norm = score / mag if mag > 1e-6 else 0.0
    if norm >= 0.66: return cfg["resumen_compuesto"]["mejora_fuerte"]
    if norm >= 0.2:  return cfg["resumen_compuesto"]["mejora"]
    if norm <= -0.66: return cfg["resumen_compuesto"]["empeora_fuerte"]
    if norm <= -0.2:  return cfg["resumen_compuesto"]["empeora"]
    return "<strong>Balance global:</strong> ✔️ estable ✔️"

def _frase_general(deltas: Dict[str,float], cfg: Dict[str,Any]) -> str:
    if deltas.get("__no_prev__", False):
        return cfg["frases_generales"]["sin_prev"]
    if deltas.get("__sin_datos__", False):
        return cfg["frases_generales"]["sin_datos"]
    pos = neg = 0
    for a, dv in deltas.items():
        if a.startswith("__") or pd.isna(dv) or dv == 0:
            continue
        if a == "estres":
            if dv < 0: pos += 1
            elif dv > 0: neg += 1
        else:
            if dv > 0: pos += 1
            elif dv < 0: neg += 1
    if pos and not neg: return cfg["frases_generales"]["mejora"]
    if neg and not pos: return cfg["frases_generales"]["empeora"]
    if pos and neg:     return cfg["frases_generales"]["mixto"]
    return cfg["frases_generales"]["estable"]

def _movido_line(deltas: Dict[str,float], cfg: Dict[str,Any]) -> str:
    thr = cfg["umbrales"]["delta_significativo"]
    min_areas = cfg["umbrales"]["min_areas_para_movido"]
    min_pct = cfg["umbrales"]["min_porcentaje_movido"]
    cand = [a for a,v in deltas.items() if not a.startswith("__") and pd.notna(v) and abs(float(v)) >= thr]
    total = len([a for a in deltas.keys() if not a.startswith("__")])
    if not cand:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    is_movido = len(cand) >= min_areas or (len(cand)/max(1,total)) >= min_pct
    if not is_movido:
        return cfg["mensajes_movimiento"]["intro_menos_movido"]
    topn = cfg["seleccion_areas_destacadas"]["max_listar"]
    cand_sorted = sorted(cand, key=lambda a: abs(deltas[a]), reverse=True)[:topn]
    pretty = ", ".join(cand_sorted)
    return cfg["mensajes_movimiento"]["intro_mas_movido"].format(areas_list=pretty)

def _detalle_areas(deltas: Dict[str,float], cfg: Dict[str,Any]):
    out = []
    thr = cfg["umbrales"]["delta_significativo"]
    big = cfg["umbrales"]["delta_fuerte"]
    icons = cfg["iconos"]
    tpl = cfg["plantillas_detalle"]
    items = sorted(((k,v) for k,v in deltas.items() if not k.startswith("__")), key=lambda kv: abs(kv[1] if kv[1] is not None else 0), reverse=True)
    for a, dv in items:
        if pd.isna(dv) or abs(float(dv)) < 0.0001:
            out.append(tpl["area_estable"].format(icon=icons.get(a,"•"), area=a, delta="0.00"))
            continue
        ad = abs(float(dv))
        if a == "estres":
            if dv < -thr:
                msg = tpl["estres_inversion_mejora"].format(icon=icons.get(a,"•"), delta=f"{abs(dv):.2f}")
            elif dv > thr:
                msg = tpl["estres_inversion_empeora"].format(icon=icons.get(a,"•"), delta=f"{abs(dv):.2f}")
            else:
                msg = tpl["area_estable"].format(icon=icons.get(a,"•"), area=a, delta=f"{dv:+.2f}")
            out.append(msg)
            continue
        if dv > thr:
            msg = (tpl["area_mejora_fuerte"] if ad >= big else tpl["area_mejora"]).format(icon=icons.get(a,"•"), area=a, delta=f"{ad:.2f}")
        elif dv < -thr:
            msg = (tpl["area_empeora_fuerte"] if ad >= big else tpl["area_empeora"]).format(icon=icons.get(a,"•"), area=a, delta=f"{ad:.2f}")
        else:
            msg = tpl["area_estable"].format(icon=icons.get(a,"•"), area=a, delta=f"{dv:+.2f}")
        out.append(msg)
    return out

def _comparacion_texto(prev_resumen, curr_resumen, cfg):
    if prev_resumen is None:
        return { "frase_general": cfg["frases_generales"]["sin_prev"], "mensaje_movimiento": "", "lista_detalle_areas": [], "resumen_compuesto": cfg["resumen_compuesto"]["estable"] }
    if not curr_resumen:
        return { "frase_general": cfg["frases_generales"]["sin_datos"], "mensaje_movimiento": "", "lista_detalle_areas": [], "resumen_compuesto": cfg["resumen_compuesto"]["estable"] }
    keys = sorted(set(prev_resumen.keys()) | set(curr_resumen.keys()))
    deltas = { k: (curr_resumen.get(k) - prev_resumen.get(k) if pd.notna(curr_resumen.get(k)) and pd.notna(prev_resumen.get(k)) else float('nan')) for k in keys }
    frase_general = _frase_general(deltas, cfg)
    mensaje_movimiento = _movido_line(deltas, cfg)
    detalle = _detalle_areas(deltas, cfg)
    resumen = _balance_and_label(deltas, cfg)
    return { "frase_general": frase_general, "mensaje_movimiento": mensaje_movimiento, "lista_detalle_areas": detalle, "resumen_compuesto": resumen }

def _render_card(day, prev_card, cfg):
    comp_text = _comparacion_texto(prev_card.get("resumen_areas") if prev_card else None, day["resumen_areas"], cfg)
    interpretacion_label = cfg["nombres"]["footer_key"]
    comp_key = cfg["nombres"]["comparacion_key"]
    classes = "card" + (" missing" if day.get("faltante") else "")
    html = [f'<div class="{classes}">']
    html.append(f'<div class="date">{day["fecha"]}</div>')
    regs = day.get("registros", 0)
    html.append(f'<div class="meta"><b>Registros:</b> {regs}</div>')
    if day.get("resumen_areas"):
        html.append('<div class="section-title">Áreas (promedios del día):</div>')
        html.append('<div class="areas">')
        for k, v in day["resumen_areas"].items():
            if pd.isna(v): continue
            html.append(f'<span class="chip">{k}: {v:.2f}</span>')
        html.append('</div>')
    if day.get("notas"):
            html.append('<div class="section-title">Notas</div>')
            html.append('<ul class="list">')
            for n in day["notas"]:
                html.append(f'<li>{n}</li>')
            html.append('</ul>')
    if day.get("estresores"):
            html.append('<div class="section-title">Estresores</div>')
            html.append('<ul class="list">')
            for s in day["estresores"]:
                html.append(f'<li>{s}</li>')
            html.append('</ul>')
    if day.get("mensajes_humanos"):
        html.append('<div class="section-title">Mensaje humano (tendencias)</div>')
        html.append('<ul class="list">')
        for m in day["mensajes_humanos"]:
            html.append(f'<li>{m}</li>')
        html.append('</ul>')
    if day.get("interpretacion_dia"):
        html.append('<div class="footer">')
        html.append(f'<div><b>{interpretacion_label}:</b> {day["interpretacion_dia"]}</div>')
        html.append('</div>')
    html.append('<div class="footer">')
    html.append(f'<div><b>{comp_key}:</b> {comp_text["frase_general"]}</div>')
    if comp_text.get("mensaje_movimiento"):
        html.append(f'<div>{comp_text["mensaje_movimiento"]}</div>')
    if comp_text.get("lista_detalle_areas"):
        html.append('<ul class="list">')
        for line in comp_text["lista_detalle_areas"][:6]:
            html.append(f'<li>{line}</li>')
        html.append('</ul>')
    html.append(f'<div class="meta">{comp_text["resumen_compuesto"]}</div>')
    html.append('</div>')
    html.append('</div>')
    return "\n".join(html)

def generate_report_coach(input_csv: str, output_html: str, config: dict = None, start_date: str = None, end_date: str = None, tag_filter: str = None) -> str:
    cfg = _load_config(config)
    df = pd.read_csv(input_csv, parse_dates=["fecha"], dayfirst=True)
    if start_date:
        df = df[df["fecha"] >= pd.to_datetime(start_date, dayfirst=True)]
    if end_date:
        df = df[df["fecha"] <= pd.to_datetime(end_date, dayfirst=True)]
    if tag_filter and "tag" in df.columns:
        df = df[df["tag"].astype(str).str.contains(tag_filter, case=False, na=False)]
    cards = build_daily_cards(df, DEFAULT_COLUMNS, fill_missing_days=True)
    def to_dict(card):
        if isinstance(card, dict): return card
        return dict(
            fecha=getattr(card, "fecha", ""),
            registros=getattr(card, "registros", 0),
            resumen_areas=getattr(card, "resumen_areas", {}),
            mensajes_humanos=getattr(card, "mensajes_humanos", []),
            notas=getattr(card, "notas", []),
            estresores=getattr(card, "estresores", []),
            comparacion_prev=getattr(card, "comparacion_prev", ""),
            interpretacion_dia=getattr(card, "interpretacion_dia", ""),
            faltante=getattr(card, "faltante", False),
        )
    cards = [to_dict(c) for c in cards]
    html_cards = []
    prev_real = None
    for c in cards:
        html_cards.append(_render_card(c, prev_real, cfg))
        if not c.get("faltante"):
            prev_real = c
    html = HTML_TEMPLATE.format(cards_html="\n".join(html_cards))
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(output_html)
