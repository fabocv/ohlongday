# bdp_report_coach.py
from datetime import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import json
import random

# Cargar micro motivaciones desde archivo externo
with open("reporte_python/micro_motivaciones.json", "r", encoding="utf-8") as f:
    MICRO_MOTIVATIONS = json.load(f)

def zscore(series: pd.Series):
    s = series.astype(float)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - mean) / std

def compute_indices(df: pd.DataFrame, w_sueno: float = 1.2):
    for col in ["animo","activacion","conexion","proposito","claridad","estres","sueno_calidad"]:
        if col in df.columns:
            df[f"z_{col}"] = zscore(df[col])
    v = df.get("z_activacion", 0) + w_sueno*df.get("z_sueno_calidad", 0)
    p = df.get("z_proposito", 0) + df.get("z_claridad", 0)
    df["H_t"] = df.get("z_animo", 0)
    df["V_t"] = v
    df["C_t"] = df.get("z_conexion", 0)
    df["P_t"] = p
    df["S_t_neg"] = -df.get("z_estres", 0)
    df["BDP_score"] = (df["H_t"] + df["V_t"] + df["C_t"] + df["P_t"] + df["S_t_neg"]) / 5.0
    bins = [-np.inf, -0.5, 0.0, 0.5, np.inf]
    labels = [0, 1, 2, 3]
    df["BDP_feno_0_3"] = pd.cut(df["BDP_score"], bins=bins, labels=labels).astype(int)
    return df

def fig_to_base64(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    plt.close(figure)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def simple_line_plot(dates, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(dates, values)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    encoded = fig_to_base64(fig)
    return f'<img alt="{title}" src="data:image/png;base64,{encoded}"/>'

def icon_for_level(x):
    mapping = {0: "⬛️ Bloqueo fuerte", 1: "🟨 Débil/retroceso", 2: "🟩 En camino", 3: "🟦 Avance claro"}
    return mapping.get(int(x), "⬜️ N/A")

def micro_motivation(level:int) -> str:
    frases = MICRO_MOTIVATIONS.get(str(level), ["Confía en tu proceso."])
    return random.choice(frases)


def _pick_message(messages: dict, key: str, level: str, fallback: str) -> str:
    try:
        pool = messages.get(key, {}).get(level, [])
        if pool:
            return random.choice(pool)
    except Exception:
        pass
    return fallback

def _load_messages(config: dict | None = None) -> dict:
    # Path can be provided via config["report"]["messages_path"]; else try default next to CSV/runtime.
    default_candidates = []
    if config and isinstance(config, dict):
        p = (config.get("report") or {}).get("messages_path")
        if p:
            default_candidates.append(p)
    default_candidates += ["./mensajes.json", "/mnt/data/mensajes.json"]
    for cand in default_candidates:
        try:
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            continue
    # Minimal fallback
    return {
        "animo_bajo": {"leve": ["He notado señales suaves de ánimo bajo."],
                       "moderado": ["Varios días con ánimo bajo."],
                       "alto": ["Ánimo muy frágil últimamente."]},
        "estres_alto": {"leve": ["Algo de tensión en el cuerpo."],
                        "moderado": ["Estrés presente en varios registros."],
                        "alto": ["Estrés muy intenso estos días."]},
        "sueno_alterado": {"leve": ["Pequeñas variaciones en el sueño."],
                           "moderado": ["Sueño irregular."],
                           "alto": ["Sueño muy alterado."]},
        "autocuidado_bajo": {"leve": ["Autocuidado algo bajo."],
                             "moderado": ["Autocuidado en segundo plano."],
                             "alto": ["Autocuidado muy bajo."]},
        "tendencia_positiva": {"leve": ["Inicio de mejora."],
                               "moderado": ["Tendencia positiva clara."],
                               "alto": ["Mejora fuerte y sostenida."]}
    }

def build_week_findings(df: pd.DataFrame, thresholds: dict, config: dict | None = None) -> list:
    messages = _load_messages(config)
    findings = []

    # --- Ánimo bajo (posible DI) ---
    low_mask = df["z_animo"] < thresholds.get("animo_bajo_z", -0.3)
    streak = 0
    max_streak_low = 0
    for v in low_mask:
        if v:
            streak += 1
            if streak > max_streak_low:
                max_streak_low = streak
        else:
            streak = 0
    if low_mask.any():
        mean_low = float(df.loc[low_mask, "z_animo"].mean()) if low_mask.any() else 0.0
        lvl = "leve"
        if max_streak_low >= thresholds.get("animo_bajo_streak", 3) or mean_low < -0.4:
            lvl = "moderado"
        if max_streak_low >= thresholds.get("animo_bajo_streak", 3) + 2 or mean_low < -0.6:
            lvl = "alto"
        msg = _pick_message(messages, "animo_bajo", lvl,
                            "He notado que tu ánimo ha estado algo bajo.")
        findings.append(f"Indicador: posible ánimo bajo ({lvl}) — {msg}")

    # --- Estrés alto (posible AI) ---
    high_stress = df["z_estres"] > thresholds.get("estres_alto_z", 0.6)
    prop = float(high_stress.mean()) if len(df) else 0.0
    if high_stress.any():
        lvl = "leve"
        if prop >= thresholds.get("estres_alto_ratio_min", 0.3):
            lvl = "moderado"
        if prop >= 0.6:
            lvl = "alto"
        msg = _pick_message(messages, "estres_alto", lvl,
                            "Parece que el estrés estuvo más intenso últimamente.")
        findings.append(f"Indicador: posible estrés alto ({lvl}) — {msg}")

    # --- Sueño alterado ---
    std_sueno = float(df["horas_sueno"].std(ddof=0)) if "horas_sueno" in df.columns else 0.0
    mean_z_sueno = float(df.get("z_sueno_calidad", pd.Series([0]*len(df))).mean())
    if (std_sueno > 0) or (mean_z_sueno < 0):
        lvl = "leve"
        if std_sueno > thresholds.get("sueno_irregular_std_horas", 1.2) or mean_z_sueno < thresholds.get("sueno_calidad_media_min", -0.2):
            lvl = "moderado"
        if std_sueno > thresholds.get("sueno_irregular_std_horas", 1.2) + 0.8 or mean_z_sueno < -0.6:
            lvl = "alto"
        msg = _pick_message(messages, "sueno_alterado", lvl,
                            "Tu descanso se ve algo irregular.")
        findings.append(f"Indicador: posible sueño alterado ({lvl}) — {msg}")

    # --- Autocuidado bajo ---
    if "autocuidado" in df.columns and not df["autocuidado"].isna().all():
        mean_auto = float(df["autocuidado"].mean())
        if mean_auto < thresholds.get("autocuidado_media_min", 5):
            lvl = "moderado"
            if mean_auto < thresholds.get("autocuidado_media_min", 5) - 2:
                lvl = "alto"
            elif mean_auto >= thresholds.get("autocuidado_media_min", 5) - 1:
                lvl = "leve"
            msg = _pick_message(messages, "autocuidado_bajo", lvl,
                                "El autocuidado quedó un poco atrás.")
            findings.append(f"Indicador: posible autocuidado bajo ({lvl}) — {msg}")

    # --- Tendencia positiva ---
    if len(df) >= 6:
        first = float(df["BDP_score"].head(3).mean())
        last = float(df["BDP_score"].tail(3).mean())
        delta = last - first
        if delta > 0:
            lvl = "leve"
            if delta > thresholds.get("tendencia_positiva_delta", 0.3):
                lvl = "moderado"
            if delta > thresholds.get("tendencia_positiva_delta", 0.3) + 0.4:
                lvl = "alto"
            msg = _pick_message(messages, "tendencia_positiva", lvl,
                                "Se nota una tendencia positiva estos días.")
            findings.append(f"Indicador: posible tendencia positiva ({lvl}) — {msg}")

    return findings


def build_messages_timeline(df: pd.DataFrame, days:int=7) -> str:
    df = df.copy()
    df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce")
    df = df.sort_values(["fecha_dt", "hora", "entry_id"])
    if not df["fecha_dt"].isna().all():
        cutoff = df["fecha_dt"].max() - pd.Timedelta(days=days-1)
        df = df[df["fecha_dt"] >= cutoff]
    cards = []
    for _, row in df.iterrows():
        msgs = []
        if isinstance(row.get("interacciones_significativas", ""), str) and row["interacciones_significativas"].strip():
            msgs.append(f"👥 <strong>Interacciones:</strong> {row['interacciones_significativas']}")
        if isinstance(row.get("eventos_estresores", ""), str) and row["eventos_estresores"].strip():
            msgs.append(f"⚠️ <strong>Estresores:</strong> {row['eventos_estresores']}")
        if isinstance(row.get("notas", ""), str) and row["notas"].strip():
            msgs.append(f"📝 <strong>Notas:</strong> {row['notas']}")
        if not msgs:
            msgs.append("—")
        content = "<br/>".join(msgs)
        level = int(row.get("BDP_feno_0_3", 2))
        micro = micro_motivation(level)
        cards.append(f"""
        <div class="msg-card">
          <div class="msg-head">{row['fecha']} · {row['hora']} <span class="muted">id: {row['entry_id']}</span></div>
          <div class="msg-body">{content}</div>
          <div class="msg-micro">💬 <em>{micro}</em></div>
        </div>
        """)
    return "\\n".join(cards) if cards else "<p class='muted'>No hay mensajes.</p>"

def generate_report_coach(input_csv: str, output_html: str, config: dict | None = None, start_date: str | None = None, end_date: str | None = None, tag_filter: str | None = None) -> str:
    df = pd.read_csv(input_csv, encoding="utf-8")
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce").dt.strftime("%d-%m-%Y")
    for c in ["animo","activacion","conexion","proposito","claridad","estres","sueno_calidad",
              "horas_sueno","siesta_min","autocuidado","alimentacion","movimiento","dolor_fisico",
              "ansiedad","irritabilidad","meditacion_min","exposicion_sol_min","agua_litros",
              "cafeina_mg","alcohol_ud"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    days_window = (config or {}).get("report", {}).get("days_window", 7)
    if start_date:
        sd = pd.to_datetime(start_date, format="%d-%m-%Y", errors="coerce")
        df = df[pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce") >= sd]
    if end_date:
        ed = pd.to_datetime(end_date, format="%d-%m-%Y", errors="coerce")
        df = df[pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce") <= ed]
    if not start_date and not end_date and not df.empty:
        dates = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce")
        cutoff = dates.max() - pd.Timedelta(days=days_window-1)
        df = df[dates >= cutoff]
    if tag_filter and "tags" in df.columns:
        df = df[df["tags"].fillna("").str.contains(tag_filter, case=False, na=False)]
    w = (config or {}).get("weights", {}).get("sueno_en_vitalidad", 1.2)
    df = compute_indices(df, w_sueno=w)

    ordered = df.copy()
    ordered["fecha_dt"] = pd.to_datetime(ordered["fecha"], format="%d-%m-%Y", errors="coerce")
    ordered = ordered.sort_values(["fecha_dt","hora"])
    dates = ordered["fecha_dt"]

    imgs = []
    imgs.append(simple_line_plot(dates, ordered["BDP_score"], "BDP Score", "z-compuesto"))
    imgs.append(simple_line_plot(dates, ordered["H_t"], "H_t (Humor)", "z"))
    imgs.append(simple_line_plot(dates, ordered["V_t"], "V_t (Vitalidad)", "z"))
    imgs.append(simple_line_plot(dates, ordered["P_t"], "P_t (Prop/Claridad)", "z"))
    imgs.append(simple_line_plot(dates, ordered["C_t"], "C_t (Conexión)", "z"))
    imgs.append(simple_line_plot(dates, ordered["S_t_neg"], "S_t⁻ (Estrés invertido)", "z"))
    imgs_html = "\\n".join(imgs)

    small = ordered[["fecha","hora","H_t","V_t","C_t","P_t","S_t_neg","BDP_score","BDP_feno_0_3"]].copy()
    small["Estado"] = small["BDP_feno_0_3"].apply(icon_for_level)
    styled_table = (
        small.rename(columns={
            "fecha":"Fecha","hora":"Hora",
            "BDP_score":"BDP Score","BDP_feno_0_3":"Fenomenología (0–3)"
        }).to_html(index=False, float_format=lambda x: f"{x:.2f}")
    )

    thresholds = (config or {}).get("thresholds", {})
    findings = build_week_findings(ordered, thresholds=thresholds, config=config)
    timeline_html = build_messages_timeline(ordered, days=days_window)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BDP – Informe (mensajes + coaching)</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Noto Sans', Arial; margin: 24px; }}
  h1, h2 {{ margin: 0.2rem 0 0.6rem; }}
  .grid {{ display:grid; grid-template-columns: 1fr; gap: 16px; }}
  .card {{ border:1px solid #ddd; border-radius:12px; padding:16px; box-shadow: 0 2px 8px rgba(0,0,0,.04);}}
  .muted {{ color:#666; font-size: 12px; }}
  img {{ max-width: 100%; height:auto; border-radius:8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
  th {{ background: #fafafa; }}
  .msg-card {{ border:1px solid #eee; border-radius:10px; padding:12px; margin-bottom:10px; }}
  .msg-head {{ font-weight:600; margin-bottom:6px; }}
  .msg-body {{ line-height:1.4; }}
  .msg-micro {{ margin-top:6px; color:#333; }}
</style>
</head>
<body>
  <h1>BDP – Informe con mensajes y coaching</h1>
  <div class="muted">Generado: {now}</div>

  <div class="grid">
    <div class="card">
      <h2>Resumen cuantitativo</h2>
      <div>{styled_table}</div>
    </div>

    <div class="card">
      <h2>Tendencias</h2>
      {imgs_html}
    </div>

    <div class="card">
      <h2>Hallazgos automáticos</h2>
      {"<ul>" + "".join(f"<li>{f}</li>" for f in findings) + "</ul>" if findings else "<p class='muted'>Sin hallazgos destacados.</p>"}
    </div>

    <div class="card">
      <h2>Mensajes + micro-motivaciones ({days_window} días)</h2>
      {timeline_html}
    </div>

    <div class="card">
      <h2>Notas</h2>
      <p>Las micro-motivaciones se generan según la escala fenomenológica 0–3 del registro.</p>
      <p class="muted">Este informe no reemplaza a profesionales de salud mental.</p>
    </div>
  </div>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    return output_html
