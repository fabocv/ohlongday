
from __future__ import annotations
import os, json, re
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


# ------------------------ Overview (Resumen de días) ------------------------
_INDICATOR_LABEL = {
    "AI": "Ansiedad",
    "HI": "Hipomanía",
    "MI": "Mixto",
    "ALI": "Labilidad",
    "IPI": "Impulsividad",
    "RI": "Relacional",
    "SI": "Estabilidad",
}
_LEVEL_RANK = {"bajo": 0, "moderado": 1, "alto": 2, "ok": 1}

def _fmt_fecha_es(fecha_str: str) -> str:
    meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    f = None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            f = datetime.strptime(fecha_str, fmt)
            break
        except Exception:
            continue
    if not f:
        return fecha_str
    return f"{f.day} de {meses[f.month-1].capitalize()} de {f.year}"

def _indices_to_map(items):
    out = {}
    for it in (items or []):
        code = str(it.get("id") or it.get("code") or "").strip()
        if not code:
            continue
        lvl = it.get("nivel"); 
        lvl = str(lvl).lower() if isinstance(lvl, str) else None
        out[code] = (lvl, _INDICATOR_LABEL.get(code, it.get("nombre", code)))
    return out

def _state_between(prev, curr, code):
    p = prev.get(code) if prev else None
    c = curr.get(code) if curr else None
    if not p and not c:
        return None
    if not p and c:
        return ("☁️", "emergió", "state-new")
    if p and not c:
        return ("👋", "cedió", "state-cede")
    pl, cl = p[0], c[0]
    if not pl or not cl:
        return ("🗿", "se mantiene", "state-keep")
    rp = _LEVEL_RANK.get(pl, 0); rc = _LEVEL_RANK.get(cl, 0)
    if rc > rp:   return ("⤴️", "subió", "state-up")
    if rc < rp:   return ("⤵️", "bajó", "state-down")
    if rc >= 1:   return ("🗿", "se mantiene", "state-keep")
    return None

def _build_overview(cards: List[Dict[str,Any]], cfg: Dict[str,Any]) -> str:
    if not cards:
        return ""
    html = []
    html.append('<section class="days-overview">')
    html.append('<div class="overview-grid">')
    prev_idx_map = {}
    order = cfg.get("indices_prioridad", ["AI","HI","MI","ALI","IPI","RI","SI"])
    for c in cards:
        if c.get("faltante"):
            continue
        fecha = _fmt_fecha_es(str(c.get("fecha","")))
        areas = c.get("resumen_areas") or {}
        notas = c.get("notas") or []
        curr_indices = detect_indices({**areas, "__notas__": notas}, None)
        curr_map = _indices_to_map(curr_indices)
        chips_all = []
        for code in order:
            st = _state_between(prev_idx_map, curr_map, code)
            if not st:
                continue
            emoji, label, css = st
            if code == "SI" and css not in {"state-new","state-cede"}:
                continue
            nombre = curr_map.get(code, prev_idx_map.get(code))[1] if (curr_map.get(code) or prev_idx_map.get(code)) else _INDICATOR_LABEL.get(code, code)
            chip_class = {
                "state-up": "chip-up",
                "state-down": "chip-down",
                "state-new": "chip-new",
                "state-cede": "chip-cede",
                "state-keep": "chip-keep",
            }.get(css, "chip-keep")
            chips_all.append(f'<span class="chip {chip_class}" title="{nombre}: {label}" aria-label="{nombre}: {label}"><span class="k">{nombre}</span> <span class="s">{emoji}</span></span>')
        extra = 0
        if len(chips_all) > 4:
            extra = len(chips_all) - 4
            chips = chips_all[:4]
        else:
            chips = chips_all
        items_html = (" ".join(chips) if chips else 'Sin cambios relevantes en índices.')
        if extra:
            items_html += f' <span class="more">+{extra} más</span>'
        html.append('<article class="overview-card">')
        html.append(f'<div class="date">{fecha}</div>')
        html.append(f'<p class="overview-line">{items_html}</p>')
        html.append('</article>')
        prev_idx_map = curr_map
    html.append('</div></section>')
    return "\n".join(html)
# ------------------------ Config desde JSON ------------------------
CONFIG_PATH = Path(__file__).parent / "bdp_config.json"

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
    .section-title { font-weight: 700; font-size: 13px; margin-top: 8px; margin-bottom: 8px}
    .areas { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .chip { background: var(--chip); border: 1px solid var(--border);
            padding: 6px 8px; border-radius: 999px; font-size: 12px; }
    .list { margin: 8px 0 0 0; padding-left: 18px; }
    .footer { margin-top: 10px; border-top: 1px dashed var(--border); padding-top: 10px; font-size: 13px; }
  
  /* ---- Resumen de días (overview) ---- */
  .days-overview { margin: 14px 0 18px; }
  .overview-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  @media (max-width: 1200px) { .overview-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 900px)  { .overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 640px)  { .overview-grid { grid-template-columns: 1fr; } }

  .overview-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 8px 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08);
  }
  .overview-card .date { font-weight: 700; margin-bottom: 6px; font-size: 13px; }
  .overview-line { margin: 0; font-size: 12px; line-height: 1.45; color: var(--text); }
  .overview-line .item .k { font-weight: 600; }
  .overview-line .item .v { font-weight: 600; }

  /* chips */
  .overview-line .chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 12px;
    line-height: 1.4;
    margin: 2px 6px 2px 0;
    background: #f3f4f6;
    border: 1px solid var(--border);
    cursor: help;
  }
  .overview-line .chip .k { font-weight: 600; }
  .overview-line .chip .s { font-weight: 600; margin-left: 4px; }
  .overview-line .chip-up    { background: #ecfdf5; border-color: #bbf7d0; }   /* up */
  .overview-line .chip-down  { background: #fffbeb; border-color: #fcd34d; }   /* down */
  .overview-line .chip-new   { background: #eff6ff; border-color: #bfdbfe; }   /* new */
  .overview-line .chip-cede  { background: #f8fafc; border-color: #e5e7eb; }   /* cede */
  .overview-line .chip-keep  { background: #f1f5f9; border-color: #e2e8f0; }   /* keep */
  .overview-line .more { color: #6b7280; font-weight: 600; }

  .text-sm { font-size:12px}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Informe BDP diario</h1>
    <p class="lead">Resumen diario con interpretaciones amables, dominios humanizados, tendencias y comparación con el día anterior.</p>
    <h3> Resumen de días </h3>
    %%OVERVIEW_HTML%%
    <h3> Detalles de la semana </h3>
    <div class="grid">
      %%CARDS_HTML%%
    </div>
  </div>
</body>
</html>
"""

# ------------------------ Helpers ------------------------

def _fmt_date_es(val) -> str:
    meses = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre"]
    try:
        ts = pd.to_datetime(val, errors="coerce", dayfirst=False)
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
    cand_sorted = sorted(cand, key=lambda a: abs(deltas[a]), reverse=True)[:4]
    return cfg["mensajes_movimiento"]["intro_mas_movido"].format(areas_list=", ".join(cand_sorted))

def _detalle_areas(deltas: Dict[str,float], cfg: Dict[str,Any], curr_res: Dict[str,float] | None = None) -> List[str]:
    out = []
    thr = cfg["umbrales"]["delta_significativo"]
    big = cfg["umbrales"]["delta_fuerte"]
    icons = cfg["iconos_areas"]
    tpl = cfg["plantillas_detalle"]
    items = sorted(((k,v) for k,v in deltas.items() if not k.startswith("__")),
                   key=lambda kv: abs(kv[1] if kv[1] is not None else 0), reverse=True)
    for a, dv in items:
        curr_v = None if curr_res is None else curr_res.get(a)
        lvl = _level_label(curr_v, curr_res)
        if pd.isna(dv) or abs(float(dv)) < thr:
            # Δ pequeño -> "se mantiene <nivel>"
            if a == "estres" and lvl:
                out.append(f'{icons.get(a,"•")} Estrés: <strong>se mantiene {lvl}</strong> (Δ {dv:+.2f})')
            else:
                nombre = a.capitalize() if a != "sueno" else "Sueño"
                if lvl:
                    out.append(f'{icons.get(a,"•")} {nombre}: <strong>se mantiene {lvl}</strong> (Δ {dv:+.2f})')
                else:
                    out.append(tpl["area_estable"].format(icon=icons.get(a,"•"), area=nombre, delta=f"{dv:+.2f}"))
            continue

            # Grandes cambios: usar plantillas
        ad = abs(float(dv))
        nombre = a.capitalize() if a != "sueno" else "Sueño"
        if a == "estres":
            if dv < -thr:
                msg = tpl["estres_disminuye"].format(icon=icons.get(a,"•"), delta=f"{abs(dv):.2f}")
            elif dv > thr:
                msg = tpl["estres_aumenta"].format(icon=icons.get(a,"•"), delta=f"{abs(dv):.2f}")
            out.append(msg)
            continue

        if dv > 0:
            msg = (tpl["area_mejora_fuerte"] if ad >= big else tpl["area_mejora"]).format(icon=icons.get(a,"•"), area=nombre, delta=f"{ad:.2f}")
        else:
            msg = (tpl["area_empeora_fuerte"] if ad >= big else tpl["area_empeora"]).format(icon=icons.get(a,"•"), area=nombre, delta=f"{ad:.2f}")
        out.append(msg)
    return out

def _balance_and_label(deltas: Dict[str,float], cfg: Dict[str,Any]) -> str:
    weights = cfg["peso_areas"]
    score = 0.0
    mag = 0.0
    for a, dv in deltas.items():
        if a.startswith("__") or pd.isna(dv):
            continue
        w = float(weights.get(a, 1.0))
        sgn = (-1 if dv > 0 else (1 if dv < 0 else 0)) if a == "estres" else (1 if dv > 0 else (-1 if dv < 0 else 0))
        score += w * sgn * abs(dv)
        mag += w * abs(dv)
    if mag <= max(1e-9, cfg["umbrales"]["delta_significativo"]):
        return "<strong>Balance global:</strong> ✔️ estable ✔️"
    norm = score / mag if mag > 1e-6 else 0.0
    if norm >= 0.66:  return "<strong>Balance global:</strong> 🌟 mejora fuerte 🌟"
    if norm >= 0.20:  return "<strong>Balance global:</strong> 🌱 ligera mejora 🌱"
    if norm <= -0.66: return "<strong>Balance global:</strong> ⛔ retroceso marcado ⛔"
    if norm <= -0.20: return "<strong>Balance global:</strong> ⚠️ ligero retroceso ⚠️"
    return "<strong>Balance global:</strong> ✔️ estable ✔️"

def _comparacion(prev_res: Dict[str,float] | None, curr_res: Dict[str,float], cfg: Dict[str,Any]) -> Dict[str,Any]:
    if prev_res is None:
        return { "frase_general": cfg["frases_generales"]["sin_prev"],
                 "mensaje_movimiento": "", "lista_detalle_areas": [],
                 "resumen_compuesto": "<strong>Balance global:</strong> ✔️ estable ✔️" }
    if not curr_res:
        return { "frase_general": cfg["frases_generales"]["sin_datos"],
                 "mensaje_movimiento": "", "lista_detalle_areas": [],
                 "resumen_compuesto": "<strong>Balance global:</strong> ✔️ estable ✔️" }
    keys = sorted(set(prev_res.keys()) | set(curr_res.keys()))
    deltas = { k: (curr_res.get(k) - prev_res.get(k) if pd.notna(curr_res.get(k)) and pd.notna(prev_res.get(k)) else float('nan')) for k in keys }
    frase_general = _frase_general(deltas, cfg)
    mensaje_movimiento = _movido_line(deltas, cfg)
    detalle = _detalle_areas(deltas, cfg, curr_res=curr_res)
    resumen = _balance_and_label(deltas, cfg)
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
    return chips

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
                html.append(f'<span class="chip text-sm" >{dom_human[code]}</span>')
        html.append('</div>')

    ctx_chips = _contexto_chips(areas, cfg)
    if ctx_chips:
        html.append('<div class="section-title">Contexto del día</div>')
        html.append('<div class="areas">' + " ".join(ctx_chips) + '</div>')

    if day.get("notas"):
        html.append('<div class="section-title">Notas</div>')
        html.append('<ul class="list">')
        for n in day["notas"]:
            html.append(f'<li class="text-sm">{n}</li>')
        html.append('</ul>')

    if regs >= 2 and day.get("estresores"):
        html.append('<div class="section-title">Estresores</div>')
        html.append('<ul class="list">')
        for s in day["estresores"]:
            html.append(f'<li class="text-sm">{s}</li>')
        html.append('</ul>')

    # Mensaje humano (tendencias) — ahora usamos niveles para TODAS las áreas
    if areas and day.get("mensajes_humanos"):
        # mensajes_humanos ya fue construido con current values
        html.append('<div class="section-title">Mensaje humano (tendencias)</div>')
        html.append('<ul class="list">')
        for m in day["mensajes_humanos"]:
            html.append(f'<li class="text-sm">{m}</li>')
        html.append('</ul>')

    indices = []
    if areas:
        prev_res = prev_real.get("resumen_areas") if prev_real else None
        indices = detect_indices({**areas, "__notas__": day.get("notas", [])}, prev_res)
    if indices:
        html.append('<div class="section-title">📊 Índices clave</div>')
        html.append('<ul class="list">')
        for it in indices:
            html.append(f'<li class="text-sm"><strong>{it["id"]} ({it["nombre"]}):</strong> {it["mensaje"]}</li>')
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
        html.append('<ul class="list">')
        for line in comp["lista_detalle_areas"][:6]:
            html.append(f'<li>{line}</li>')
        html.append('</ul>')
    html.append(f'<div class="meta">{comp["resumen_compuesto"]}</div>')
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

    
    df = pd.read_csv(input_csv, dtype={"hora": "string"})
    df.columns = [str(c).strip() for c in df.columns]
    # mapea a nombres esperados
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

    # construir Resumen de días (overview)
    overview_html = _build_overview(cards, cfg)
    final_html = HTML_TEMPLATE.replace("%%OVERVIEW_HTML%%", overview_html).replace("%%CARDS_HTML%%", "\n".join(html_cards))

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)

    return os.path.abspath(output_html)