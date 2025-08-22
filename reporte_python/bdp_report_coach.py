# bdp_report_coach.py
from datetime import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import random

areas = [("BDP_score", "BDP Score (Compuesto)"),
        ("H_t", "H_t (Ánimo)"),
        ("V_t", "V_t (Vitalidad)"),
        ("P_t", "P_t (Propósito/Claridad)"),
        ("C_t", "C_t (Conexión)"),
        ("S_t_neg", "S_t⁻ (Estrés invertido)")]

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


def _numeric_x(dates):
    # Convert datetime index to numeric days since first non-nan
    import numpy as np
    d = pd.to_datetime(dates)
    mask = ~d.isna()
    if not mask.any():
        return np.arange(len(d)), mask
    base = d[mask].iloc[0]
    x = (d - base).dt.total_seconds() / (24*3600)
    x = x.fillna(0).to_numpy()
    return x, mask

def compute_trend_metrics(dates, series):
    """
    Returns dict with slope_per_day, r2, delta_last3, delta_pct,
    volatility (std), and n.
    """
    import numpy as np
    s = pd.to_numeric(series, errors="coerce")
    x, mask = _numeric_x(dates)
    y = s.to_numpy()
    y = np.where(np.isnan(y), np.nan, y)
    valid = mask.to_numpy() & ~np.isnan(y)
    n = int(valid.sum())
    if n < 3:
        return {"slope_per_day": 0.0, "r2": 0.0, "delta_last3": 0.0, "volatility": float(np.nan), "n": n}
    xv, yv = x[valid], y[valid]
    # Linear regression
    slope, intercept = np.polyfit(xv, yv, 1)
    yhat = slope * xv + intercept
    ss_res = float(((yv - yhat) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)
    # Delta last 3 vs first 3
    first3 = float(pd.Series(yv).head(3).mean())
    last3 = float(pd.Series(yv).tail(3).mean())
    delta_last3 = last3 - first3
    volatility = float(pd.Series(yv).std(ddof=0))
    return {"slope_per_day": float(slope), "r2": float(r2), "delta_last3": float(delta_last3), "volatility": volatility, "n": n}

def interpret_area(area_key: str, metrics: dict) -> tuple[str, str, str]:
    """
    Returns (technical, human) interpretation based on trend metrics.
    """
    slope = metrics.get("slope_per_day", 0.0)
    r2 = metrics.get("r2", 0.0)
    vol = metrics.get("volatility", 0.0)
    n = metrics.get("n", 0)
    # Thresholds (heuristic, z-units per day)
    # Over ~7 days, 0.02 per day ~ 0.14 change, noticeable.
    S_UP = 0.02
    S_DOWN = -0.02
    R_OK = 0.25
    VOL_HIGH = 0.7  # z-level variability

    arrow = "→"
    label = "Estable"
    if slope >= S_UP and r2 >= R_OK:
        arrow = "↗"
        label = "Tendencia al alza"
    elif slope <= S_DOWN and r2 >= R_OK:
        arrow = "↘"
        label = "Tendencia a la baja"
    elif abs(slope) < S_UP and vol is not None and vol >= VOL_HIGH:
        arrow = "≈"
        label = "Variable"

    technical = f"{label} {arrow} · pendiente={slope:.3f} z/día · R²={r2:.2f} · n={n}"
    # Human message per area
    area_human = {
        "BDP_score": {
            "up": "Se ve una mejora general en tu balance. Reconoce lo que te ayudó y repítelo con suavidad.",
            "down": "El balance se ve más desafiante. Quizá convenga reducir exigencias y priorizar descanso.",
            "flat": "Tu balance se mantiene estable. Sostén hábitos que te hacen bien.",
            "var": "El balance se ve cambiante. Rituales simples pueden darte anclaje."
        },
        "H_t": {
            "up": "El ánimo muestra señales de mejora. Celebra pequeños momentos de claridad o calma.",
            "down": "El ánimo parece decaer. Date espacio suave y busca algo que te reconforte.",
            "flat": "Ánimo estable. Mantén los apoyos que te sirven.",
            "var": "Ánimo variable. Respira y observa sin juicio; pasará."
        },
        "V_t": {
            "up": "La vitalidad sube. Cuida el ritmo, evita sobrecargarte.",
            "down": "La vitalidad baja. Prioriza sueño y pausas breves.",
            "flat": "Energía estable. Sostén lo que te nutre.",
            "var": "Energía cambiante. Ritmo amable y agua pueden ayudar."
        },
        "P_t": {
            "up": "Más claridad/propósito. Aprovecha para ordenar prioridades.",
            "down": "Menos claridad/propósito. Tareas pequeñas y concretas pueden ayudar.",
            "flat": "Claridad estable. Mantén tu método actual.",
            "var": "Claridad variable. Anotar antes de actuar puede ayudarte."
        },
        "C_t": {
            "up": "Se fortalece la conexión. Alimenta los vínculos que te cuidan.",
            "down": "Conexión a la baja. Busca apoyo seguro o un gesto de cercanía.",
            "flat": "Conexión estable. Reconoce la red que ya tienes.",
            "var": "Conexión variable. Límites y cuidado personal primero."
        },
        "S_t_neg": {
            "up": "Estrés en descenso (mejor). Sostén micro-pausas.",
            "down": "Estrés en alza. Baja el ritmo y pide apoyo cuando puedas.",
            "flat": "Estrés estable. Protege tus espacios de pausa.",
            "var": "Estrés variable. Pequeñas anclas pueden estabilizar."
        }
    }
    key = "flat"
    if label == "Tendencia al alza":
        key = "up"
    elif label == "Tendencia a la baja":
        key = "down"
    elif label == "Variable":
        key = "var"
    human = area_human.get(area_key, area_human["BDP_score"]).get(key, "Observa tu proceso con amabilidad.")
    return technical, human, arrow

def interpretative_plot(dates, series, title):
    # Plot with regression line and return <img> HTML
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(dates, series, marker="o", linewidth=1)
    # Trendline
    x, mask = _numeric_x(dates)
    y = pd.to_numeric(series, errors="coerce").to_numpy()
    valid = mask.to_numpy() & ~np.isnan(y)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        xline = np.linspace(x[valid].min(), x[valid].max(), 50)
        yline = slope * xline + intercept
        # Convert xline back to datetimes for plotting
        base = pd.to_datetime(dates)[mask].iloc[0]
        dline = pd.to_datetime(base) + pd.to_timedelta(xline, unit="D")
        ax.plot(dline, yline, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("z")
    ax.grid(True, linestyle="--", alpha=0.5)
    encoded = fig_to_base64(fig)
    return f'<img alt="{title}" src="data:image/png;base64,{encoded}"/>'


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
    phrases = {
        0: "Hoy toca ir con mucha amabilidad: delega lo que puedas y pide apoyo.",
        1: "Un paso pequeñito vale. No tiene que salir perfecto.",
        2: "Sostén lo que te funciona. Constancia suave antes que intensidad.",
        3: "Reconoce el avance de hoy. Anota un pequeño logro."
    }
    return phrases.get(int(level), "Sigue a tu ritmo: amabilidad primero.")


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
    interp_rows = []
    metrics = {k: compute_trend_metrics(dates, ordered[k]) for k, _ in areas}
    for key, label in areas:
        res = interpret_area(key, metrics[key])
        if isinstance(res, tuple) and len(res) == 3:
            tech, hum, arrow = res
        else:
            tech, hum = res
            arrow = "→"
        interp_rows.append(f"<tr><td><strong>{arrow} {label}</strong></td><td>{tech}</td><td>{hum}</td></tr>")
    interp_table = "<table><thead><tr><th>Área</th><th>Interpretación técnica</th><th>Mensaje humano</th></tr></thead><tbody>" + "".join(interp_rows) + "</tbody></table>"

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
      <div style="margin-top:10px;">{interp_table}</div>
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
