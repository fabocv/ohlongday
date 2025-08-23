
import os, json
import pandas as pd
from typing import List, Dict, Any
from reporte_python.bdp_schema import DEFAULT_COLUMNS
from reporte_python.bdp_cards import build_daily_cards
from reporte_python.bdp_dominios import compute_dominios, interpret_dominios, format_chip

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
    <p class="lead">Resumen diario con interpretaciones amables, dominios, tendencias y comparación con el día anterior.</p>
    <div class="grid">
      {cards_html}
    </div>
  </div>
</body>
</html>
"""

def _fmt_date_es(val):
    import pandas as pd, datetime as _dt, re as _re
    meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    try:
        ts = pd.to_datetime(val, errors="coerce", dayfirst=False)
        if pd.isna(ts): raise ValueError
        return f"{ts.day} de {meses[int(ts.month)-1].capitalize()} de {ts.year}"
    except Exception:
        s = str(val)
        m = _re.match(r"^(\d{2})[-/](\d{2})[-/](\d{4})$", s)
        if m:
            d, m_, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{d} de {meses[m_-1].capitalize()} de {y}"
        return s


def _balance_and_label_html(deltas: Dict[str, float]) -> str:
    # Simple net direction scoring with stress inverted
    if not deltas:
        return "<strong>Balance global:</strong> ✔️ <strong>estable</strong> ✔️"
    pos = neg = 0
    mag = 0.0
    for a, dv in deltas.items():
        if a.startswith("__"): continue
        if dv is None or (isinstance(dv,float) and pd.isna(dv)): continue
        if a == "estres":
            dv = -dv
        mag += abs(dv)
        if dv > 0: pos += 1
        elif dv < 0: neg += 1
    if mag < 0.05:
        return "<strong>Balance global:</strong> ✔️ <strong>estable</strong> ✔️"
    ratio = (pos - neg) / max(1, (pos + neg))
    if ratio >= 0.66: return "<strong>Balance global:</strong> 🌟 <strong>mejora fuerte</strong> 🌟"
    if ratio >= 0.2:  return "<strong>Balance global:</strong> 🌱 <strong>ligera mejora</strong> 🌱"
    if ratio <= -0.66: return "<strong>Balance global:</strong> ⛔ <strong>retroceso marcado</strong> ⛔"
    if ratio <= -0.2:  return "<strong>Balance global:</strong> ⚠️ <strong>ligero retroceso</strong> ⚠️"
    return "<strong>Balance global:</strong> ✔️ <strong>estable</strong> ✔️"

def _comparacion_texto(prev_resumen: Dict[str,float], curr_resumen: Dict[str,float]) -> Dict[str,Any]:
    if prev_resumen is None:
        return {"frase_general": "Primera medición del período: no hay día anterior para comparar.",
                "mensaje_movimiento": "", "lista_detalle_areas": [],
                "resumen_compuesto": "<strong>Balance global:</strong> ✔️ <strong>estable</strong> ✔️"}
    if not curr_resumen:
        return {"frase_general": "Hoy no hubo registros; se mantiene la última referencia conocida.",
                "mensaje_movimiento": "", "lista_detalle_areas": [],
                "resumen_compuesto": "<strong>Balance global:</strong> ✔️ <strong>estable</strong> ✔️"}
    keys = sorted(set(prev_resumen.keys()) | set(curr_resumen.keys()))
    deltas = {k: ((curr_resumen.get(k) - prev_resumen.get(k)) if (k in curr_resumen and k in prev_resumen) else None) for k in keys}
    # movement line
    thr = 0.05
    changed = [a for a,v in deltas.items() if v is not None and abs(v) >= thr]
    if changed:
        top = ", ".join(sorted(changed, key=lambda a: abs(deltas[a]), reverse=True)[:4])
        mov = f"En comparación al día de ayer, tu día estuvo <strong>más movido</strong> por: {top}."
    else:
        mov = "Respecto a ayer, hoy se percibe <strong>menos movimiento</strong> general."
    # detail lines
    detail = []
    for a in sorted(changed, key=lambda k: abs(deltas[k]), reverse=True):
        dv = deltas[a]
        if a == "estres":
            if dv < 0:
                detail.append(f"🔥 Estrés: <strong>disminuyó</strong> (Δ −{abs(dv):.2f}), lo cual favorece el balance.")
            else:
                detail.append(f"🔥 Estrés: <strong>aumentó</strong> (Δ +{abs(dv):.2f}), podría influir en la percepción general.")
        else:
            if dv > 0:
                detail.append(f"{a}: <strong>subió</strong> (Δ +{abs(dv):.2f})")
            else:
                detail.append(f"{a}: <strong>bajó</strong> (Δ −{abs(dv):.2f})")
    return {"frase_general": ("En comparación al día de ayer, tu día mostró una <strong>mejora general</strong>." if sum(1 for a,v in deltas.items() if v and ((a!='estres' and v>0) or (a=='estres' and v<0))) > 
                               sum(1 for a,v in deltas.items() if v and ((a!='estres' and v<0) or (a=='estres' and v>0))) else
                               "Hubo <strong>cambios mixtos</strong> respecto a ayer: mejoras en algunas áreas y retrocesos en otras.") if changed else "Comparado con ayer, el panorama se mantuvo <strong>relativamente estable</strong>.",
            "mensaje_movimiento": mov,
            "lista_detalle_areas": detail,
            "resumen_compuesto": _balance_and_label_html(deltas)}

def _render_card(day, prev_card):
    classes = "card" + (" missing" if day.get("faltante") else "")
    html = [f'<div class="{classes}">']
    date_txt = _fmt_date_es(day["fecha"])
    html.append(f'<div class="date">{date_txt}</div>')
    regs = day.get("registros", 0)
    html.append(f'<div class="meta"><b>Registros:</b> {regs}</div>')

    if day.get("resumen_areas"):
        html.append('<div class="section-title">Áreas (promedios del día):</div>')
        html.append('<div class="areas">')
        # include icons for known keys
        icon_map = {"animo":"😊","activacion":"⚡","sueno":"🌙","conexion":"🤝","proposito":"🎯","claridad":"🔍","estres":"🔥"}
        for k, v in day["resumen_areas"].items():
            if pd.isna(v): continue
            icon = icon_map.get(k,"")
            html.append(f'<span class="chip">{icon} {k}: {v:.2f}</span>')
        html.append('</div>')

        # Dominios
        dom_vals = compute_dominios(day.get("resumen_areas", {}))
        dom_infos = interpret_dominios(dom_vals)
        html.append('<div class="section-title">Dominios (promedios del día):</div>')
        html.append('<div class="areas">')
        for code, info in dom_infos.items():
            html.append(f'<span class="chip">{format_chip(code, info)}</span>')
        html.append('</div>')

    # Notas (siempre que existan)
    if day.get("notas"):
        html.append('<div class="section-title">Notas</div>')
        html.append('<ul class="list">')
        for n in day["notas"]:
            html.append(f'<li>{n}</li>')
        html.append('</ul>')

    # Estresores si 2+
    if regs >= 2 and day.get("estresores"):
        html.append('<div class="section-title">Estresores</div>')
        html.append('<ul class="list">')
        for s in day["estresores"]:
            html.append(f'<li>{s}</li>')
        html.append('</ul>')

    # Mensajes humanos
    if day.get("mensajes_humanos"):
        html.append('<div class="section-title">Mensaje humano (tendencias)</div>')
        html.append('<ul class="list">')
        for m in day["mensajes_humanos"]:
            html.append(f'<li>{m}</li>')
        html.append('</ul>')

    if day.get("interpretacion_dia"):
        html.append('<div class="footer">')
        html.append(f'<div><b>Interpretación del día:</b> {day["interpretacion_dia"]}</div>')
        html.append('</div>')

    comp = _comparacion_texto(prev_card.get("resumen_areas") if prev_card else None, day["resumen_areas"])
    html.append('<div class="footer">')
    html.append(f'<div><b>Comparación con el día anterior:</b> {comp["frase_general"]}</div>')
    if comp.get("mensaje_movimiento"):
        html.append(f'<div>{comp["mensaje_movimiento"]}</div>')
    if comp.get("lista_detalle_areas"):
        html.append('<ul class="list">')
        for line in comp["lista_detalle_areas"][:6]:
            html.append(f'<li>{line}</li>')
        html.append('</ul>')
    html.append(f'<div class="meta">{comp["resumen_compuesto"]}</div>')
    html.append('</div>')

    html.append('</div>')
    return "\n".join(html)

def generate_report_coach(input_csv: str, output_html: str, config: dict = None, start_date: str = None, end_date: str = None, tag_filter: str = None) -> str:
    df = pd.read_csv(input_csv, dtype={"hora": "string"})  # fuerza hora como texto
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

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
        html_cards.append(_render_card(c, prev_real))
        if not c.get("faltante"):
            prev_real = c

    html = HTML_TEMPLATE.format(cards_html="\n".join(html_cards))
    if os.path.exists(output_html): os.remove(output_html)
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(output_html)
