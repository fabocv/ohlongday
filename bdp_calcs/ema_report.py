# Generate the HTML again, this time with inline narrative functions to avoid import issues
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import importlib.util
from bdp_calcs.drivers import drivers_del_dia

# === Helpers: Hábitos y Patrones ==================================================
def _parse_hhmm(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([None]*0)
    s = s.astype(str).str.strip().str.replace(".", ":", regex=False).str.replace(";", ":", regex=False)
    def fix(x):
        if not x or str(x).lower() in {"nan","nat"}: return None
        parts = str(x).split(":")
        if len(parts)==1 and parts[0].isdigit(): return f"{int(parts[0])%24:02d}:00"
        if len(parts)>=2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0])%24:02d}:{int(parts[1])%60:02d}"
        return None
    return s.map(fix)

def _daypart(hhmm: str) -> str:
    if not isinstance(hhmm, str) or ":" not in hhmm: return "desconocido"
    hh = int(hhmm.split(":")[0])
    if 5<=hh<12: return "mañana"
    if 12<=hh<18: return "tarde"
    if 18<=hh<24: return "noche"
    return "madrugada"

def habit_kpis(df: pd.DataFrame, goals=None, win_short=7, win_prev=7, var_win=14):
    if goals is None:
        goals = {
            "agua_litros": ("min", 1.6),
            "tiempo_ejercicio_min": ("min", 30),
            "meditacion_min": ("min", 10),
            "exposicion_sol_min": ("min", 15),
            "tiempo_pantalla_noche_min": ("max", 60),
            "alcohol_ud": ("max", 1),
            "cafe_cucharaditas": ("max", 3),
            "horas_sueno": ("min", 7),
            "sueno_calidad": ("min", 6),
        }

    out_rows = []
    if "fecha" in df.columns:
        dfd = df.copy()
        dfd["_fecha"] = pd.to_datetime(dfd["fecha"], errors="coerce", dayfirst=True)
    else:
        dfd = df.copy()
        dfd["_fecha"] = pd.to_datetime(dfd.index)

    for col, (mode, goal) in goals.items():
        if col not in dfd.columns: 
            continue
        s = pd.to_numeric(dfd[col], errors="coerce")
        daily = s.groupby(dfd["_fecha"].dt.date).sum(min_count=1)
        if len(daily.dropna()) == 0:
            continue

        last = daily.dropna().iloc[-1] if len(daily.dropna()) else np.nan
        recent = daily.tail(win_short)
        prev = daily.tail(win_short+win_prev).head(win_prev)
        mean_recent = recent.mean()
        mean_prev = prev.mean() if len(prev) else np.nan
        delta_pct = np.nan
        if pd.notna(mean_recent) and pd.notna(mean_prev) and mean_prev != 0:
            delta_pct = (mean_recent - mean_prev) / abs(mean_prev)

        def ok(v):
            if pd.isna(v): return False
            return v >= goal if mode=="min" else v <= goal
        adherencia = np.mean([ok(v) for v in recent]) if len(recent) else np.nan

        streak = 0
        for v in reversed(daily.values.tolist()):
            if ok(v): streak += 1
            else: break

        var = daily.tail(var_win).std()

        estado = "irregular"
        if pd.notna(delta_pct) and pd.notna(adherencia):
            if delta_pct >= 0.10 and adherencia >= 0.60: estado = "construyendo"
            elif abs(delta_pct) < 0.10 and adherencia >= 0.60: estado = "mantenimiento"
            elif delta_pct <= -0.10 or adherencia < 0.30: estado = "en retroceso"

        out_rows.append({
            "habito": col,
            "ultimo": last,
            "prom_7d": mean_recent,
            "prom_7d_prev": mean_prev,
            "delta_pct": delta_pct,
            "adherencia_7d": adherencia,
            "streak_dias": streak,
            "variabilidad_14d": var,
            "estado": estado,
            "meta": f"{mode} {goal}",
        })

    summary = pd.DataFrame(out_rows).sort_values(["estado","habito"])

    dayparts = pd.DataFrame()
    if "hora" in df.columns and "fecha" in df.columns:
        horas = _parse_hhmm(df["hora"])
        parts = horas.map(_daypart)
        dfx = df.copy()
        dfx["_fecha"] = pd.to_datetime(dfx["fecha"], errors="coerce", dayfirst=True).dt.date
        dfx["_part"] = parts
        if len(dfx["_fecha"].dropna())>0:
            last_day = dfx["_fecha"].dropna().iloc[-1]
            dd = dfx[dfx["_fecha"]==last_day]
        else:
            dd = dfx
        collect = []
        for col in (goals.keys()):
            if col not in dd.columns: continue
            vals = pd.to_numeric(dd[col], errors="coerce")
            g = vals.groupby(dd["_part"]).sum(min_count=1)
            for k,v in g.items():
                collect.append({"habito": col, "franja": k if pd.notna(k) else "desconocido", "valor": v})
        dayparts = pd.DataFrame(collect)

    return {"summary": summary, "dayparts": dayparts}

def _z(s: pd.Series, win: int = 30) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    r = x.rolling(win, min_periods=max(10, win//3))
    return (x - r.mean()) / r.std()

def _clip_pos(s: pd.Series) -> pd.Series:
    return s.clip(lower=0).fillna(0)

def pattern_scores(df: pd.DataFrame, win: int = 30) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    zmap = {}
    for c in ["animo","claridad","estres","activacion","horas_sueno","sueno_calidad",
              "tiempo_ejercicio_min","exposicion_sol_min","tiempo_pantalla_noche_min",
              "agua_litros","cafe_cucharaditas","alcohol_ud","ansiedad","irritabilidad"]:
        if c in df.columns:
            zmap[c] = _z(df[c], win)
    z = lambda k: zmap.get(k, pd.Series(index=df.index, dtype=float))

    stim = df.get("otras_sustancias", pd.Series([""]*len(df))).astype(str).str.contains("psicoestimulantes", case=False, na=False).astype(int)

    a = (0.30*_clip_pos(z("estres")) +
         0.35*_clip_pos(z("ansiedad")) +
         0.25*_clip_pos(-z("claridad")) +
         0.25*_clip_pos(-z("horas_sueno")) +
         0.15*_clip_pos(z("tiempo_pantalla_noche_min")) +
         0.10*_clip_pos(z("cafe_cucharaditas")))
    ans_signal = (a / 1.40).clip(upper=1.0)

    h = (0.35*_clip_pos(z("activacion")) +
         0.20*_clip_pos(z("animo")) +
         0.25*_clip_pos(-z("horas_sueno")) +
         0.15*_clip_pos(z("tiempo_ejercicio_min")) +
         0.10*stim)
    hypo_signal = (h / 1.05).clip(upper=1.0)

    d = (0.40*_clip_pos(-z("animo")) +
         0.25*_clip_pos(-z("claridad")) +
         0.15*_clip_pos(-z("tiempo_ejercicio_min")) +
         0.10*_clip_pos(-z("exposicion_sol_min")) +
         0.10*_clip_pos(z("tiempo_pantalla_noche_min")) +
         0.10*_clip_pos(-z("sueno_calidad")))
    dep_signal = (d / 1.10).clip(upper=1.0)

    b = (0.35*_clip_pos(z("estres")) +
         0.25*_clip_pos(-z("sueno_calidad")) +
         0.20*_clip_pos(-z("horas_sueno")) +
         0.10*_clip_pos(z("tiempo_pantalla_noche_min")) +
         0.10*_clip_pos(-z("agua_litros")))
    burn_signal = (b / 1.00).clip(upper=1.0)

    out["signal_ansiedad"] = ans_signal
    out["signal_hipomania_like"] = hypo_signal
    out["signal_depresivo"] = dep_signal
    out["signal_burnout"] = burn_signal
    return out

def pattern_summary_text(df: pd.DataFrame, signals: pd.DataFrame, row_idx: int = -1, thr: float = 0.7) -> str:
    row = signals.iloc[row_idx]
    msgs = []
    def mk(label, s):
        if pd.isna(s): return
        if s >= thr:
            msgs.append(f"{label}: señal alta ({s:.2f}).")
        elif s >= 0.5:
            msgs.append(f"{label}: señal moderada ({s:.2f}).")
    mk("Ansiedad", row.get("signal_ansiedad", np.nan))
    mk("Hipomanía-like", row.get("signal_hipomania_like", np.nan))
    mk("Sesgo depresivo", row.get("signal_depresivo", np.nan))
    mk("Sobrecarga/burnout", row.get("signal_burnout", np.nan))
    base = "Patrones sugeridos (no diagnósticos): "
    tail = " — Señales vagas o incompletas." if not msgs else ""
    return base + " ".join(msgs) + tail + " Usa esto como orientación para conversación terapéutica."

def habits_section_html(df: pd.DataFrame, goals=None) -> str:
    res = habit_kpis(df, goals=goals)
    summ = res["summary"]
    dp = res["dayparts"]
    if summ is None or summ.empty:
        return "<div class='section'><h2>Hábitos y cambios</h2><div class='small'>Sin datos suficientes.</div></div>"
    rows = []
    state_cls = {"construyendo":"good","mantenimiento":"ok","irregular":"warn","en retroceso":"bad"}
    for r in summ.itertuples(index=False):
        estado = getattr(r, "estado")
        cls = state_cls.get(estado, "ok")
        hab = getattr(r, "habito")
        ultimo = getattr(r, "ultimo")
        p7 = getattr(r, "prom_7d")
        p7p = getattr(r, "prom_7d_prev")
        d = getattr(r, "delta_pct")
        adh = getattr(r, "adherencia_7d")
        streak = getattr(r, "streak_dias")
        var = getattr(r, "variabilidad_14d")
        meta = getattr(r, "meta")
        def fmt(x, nd=2):
            return "" if pd.isna(x) else f"{x:.{nd}f}"
        def pct(x):
            return "" if pd.isna(x) else f"{x*100:+.0f}%"
        def pc(x):
            return "" if pd.isna(x) else f"{x*100:.0f}%"
        rows.append(f"""
        <tr>
          <td><strong>{hab}</strong><div class="small">meta: {meta}</div></td>
          <td>{fmt(ultimo)}</td>
          <td>{fmt(p7)}</td>
          <td>{fmt(p7p)}</td>
          <td>{pct(d)}</td>
          <td>{pc(adh)}</td>
          <td>{streak}</td>
          <td>{fmt(var)}</td>
          <td><span class="badge {cls}">{estado}</span></td>
        </tr>""")
    table = f"""
    <table class="table">
      <thead>
        <tr><th>Hábito</th><th>Último</th><th>Prom. 7d</th><th>Prom. 7d prev</th><th>Δ%</th><th>Adherencia 7d</th><th>Racha</th><th>Variab. 14d</th><th>Estado</th></tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""

    chips = []
    if dp is not None and not dp.empty:
        for (hab), g in dp.groupby("habito"):
            tags = " ".join(f"<span class='tag'>{row['franja']}: {row['valor']:.2f}</span>"
                            for _, row in g.dropna(subset=["valor"]).iterrows() if row["valor"]>0)
            if tags:
                chips.append(f"<div><div class='small'><strong>{hab}</strong> — último día</div>{tags}</div>")
    dplay = "<div class='small'>—</div>" if not chips else "".join(chips)

    return f"""
    <div class="section">
      <h2>Hábitos y cambios</h2>
      <div class="card">{table}</div>
      <div class="section">
        <div class="card">
          <div class="small">Distribución por franja (mañana/tarde/noche, último día):</div>
          {dplay}
        </div>
      </div>
    </div>
    """

def patterns_section_html(df: pd.DataFrame) -> str:
    sig = pattern_scores(df)
    txt = pattern_summary_text(df, sig, row_idx=-1, thr=0.7)
    row = sig.iloc[-1] if len(sig) else pd.Series(dtype=float)
    labels = [
        ("Ansiedad", "signal_ansiedad"),
        ("Hipomanía-like", "signal_hipomania_like"),
        ("Sesgo depresivo", "signal_depresivo"),
        ("Sobrecarga/burnout", "signal_burnout"),
    ]
    cards = []
    for lab, key in labels:
        v = float(row.get(key, np.nan)) if len(row) else np.nan
        vw = 0 if (pd.isna(v) or v<0) else min(100, round(v*100))
        vtxt = "" if pd.isna(v) else f"{v:.2f}"
        cards.append(f"""
        <div class="card">
          <div class="kpi"><strong>{lab}</strong><span class="badge">{vtxt}</span></div>
          <div class="bar"><div style="width:{vw}%;"></div></div>
        </div>
        """)
    grid = f"<div class='grid' style='grid-template-columns:repeat(2,1fr);gap:12px'>{''.join(cards)}</div>"
    return f"""
    <div class="section">
      <h2>Patrones sugeridos</h2>
      <div class="card"><div class="small">{txt}</div></div>
      <div class="section">{grid}</div>
      <div class="small" style="margin-top:8px">Heurísticas personalizadas (no diagnósticos). Usa esto para guiar la conversación clínica.</div>
    </div>
    """

targets = ["animo","claridad","estres","activacion"]
labels = {"animo":"Ánimo","claridad":"Claridad","estres":"Estrés","activacion":"Activación"}
icons = {"animo":"😊","claridad":"🧠","estres":"😵","activacion":"⚡"}

def wellbeing_index(df, targets=targets):
    def z(s):
        s = pd.to_numeric(s, errors="coerce")
        m, v = s.mean(), s.std()
        return (s - m) / v if (pd.notna(v) and v != 0) else pd.Series([float("nan")]*len(s), index=s.index)
    parts = []
    for c in targets:
        if c in df.columns:
            zc = z(df[c])
            if c == "estres":
                zc = -zc
            parts.append(zc)
    if not parts:
        return pd.Series([float("nan")]*len(df), index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)

def last_delta(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    if len(s.dropna()) < 2:
        return float("nan")
    return float(s.iloc[-1] - s.iloc[-2])

def trend_label(delta: float):
    if isinstance(delta, float) and not math.isnan(delta):
        if delta > 0.4: return "al alza"
        if delta < -0.4: return "a la baja"
        return "estable"
    return "sin dato"

def narrative_summary_inline(df):
    wb = wellbeing_index(df, targets=targets)
    dwb = wb.diff().iloc[-1] if len(wb.dropna())>=2 else float("nan")
    wb_lbl = trend_label(dwb)
    pts = []
    for t in targets:
        if t in df.columns and len(pd.to_numeric(df[t], errors="coerce").dropna())>=2:
            dy = pd.to_numeric(df[t], errors="coerce").diff().iloc[-1]
            lbl = trend_label(dy)
            if t == "estres" and lbl == "al alza": lbl += " (desfavorable)"
            if t == "estres" and lbl == "a la baja": lbl += " (favorable)"
            pts.append(f"{labels[t]} {lbl} (Δ{dy:+.2f})")
    out = [f"Bienestar compuesto (WB) {wb_lbl}."]
    if pts: out.append("Estado por target → " + " | ".join(pts))
    return " ".join(out)

def descriptive_summary_inline(df, row_idx=-1, top_k=2):
    wb = wellbeing_index(df, targets=targets)
    dwb = wb.diff().iloc[-1] if len(wb.dropna())>=2 else float("nan")
    wb_lbl = trend_label(dwb)
    tgt_lines = []
    for t in targets:
        if t in df.columns and len(pd.to_numeric(df[t], errors="coerce").dropna())>=2:
            dy = pd.to_numeric(df[t], errors="coerce").diff().iloc[-1]
            lbl = trend_label(dy)
            if t == "estres" and lbl == "al alza": lbl += " (desfavorable)"
            if t == "estres" and lbl == "a la baja": lbl += " (favorable)"
            tgt_lines.append(f"{labels[t].lower()} {lbl} (Δ{dy:+.2f})")
    reasons = []
    for t in targets:
        res = drivers_del_dia(df, t, top_k=top_k, row_idx=row_idx)
        tab = res.get("tabla")
        if isinstance(tab, pd.DataFrame) and not tab.empty:
            pos = tab.sort_values("contrib", ascending=False).head(top_k)["feature"].tolist()
            neg = tab.sort_values("contrib", ascending=True).head(top_k)["feature"].tolist()
            pos_h = ", ".join(pos) if pos else "—"
            neg_h = ", ".join(neg) if neg else "—"
            reasons.append(f"{labels[t]}: apoyaron {pos_h}; restaron {neg_h}.")
    out = [f"Su bienestar compuesto está {wb_lbl}."]
    if tgt_lines: out.append("Por targets: " + "; ".join(tgt_lines) + ".")
    if reasons: out.append("Motivos probables del día: " + " ".join(reasons))
    return " ".join(out)

def day_label(df):
    if "fecha" in df.columns:
        try:
            d = pd.to_datetime(df["fecha"],  dayfirst=True, errors="coerce").dt.strftime("%d-%m-%Y (%a)").iloc[-1]
            if isinstance(d, str) and d.strip():
                return d
        except Exception:
            pass
    return str(df.index[-1]) if len(df) else "—"

def choose_model_tag():
    return "EMA"

def report_variables(df, targets):
    # Deltas for cards
    deltas = {t: last_delta(df[t]) if t in df.columns else float("nan") for t in targets}

    # Build report pieces
    short_sum = narrative_summary_inline(df)
    verbose_sum = descriptive_summary_inline(df, row_idx=-1, top_k=2)
    habits_html = habits_section_html(df)
    patterns_html = patterns_section_html(df)
    wb = wellbeing_index(df, targets=targets)
    wb_last = float(wb.iloc[-1]) if len(wb.dropna()) else float("nan")

    # Drivers tables
    driver_tables = {}
    for t in targets:
        res = drivers_del_dia(df, t, top_k=3, row_idx=-1)
        tab = res.get("tabla")
        if isinstance(tab, pd.DataFrame) and not tab.empty:
            pos = tab.sort_values("contrib", ascending=False).head(3)
            neg = tab.sort_values("contrib", ascending=True).head(3)
            show = pd.concat([pos, neg]).drop_duplicates()
            driver_tables[t] = show
        else:
            driver_tables[t] = pd.DataFrame(columns=["feature","contrib"])
    return deltas, short_sum, verbose_sum, habits_html, patterns_html, wb_last, driver_tables

def trend_badge(t, val):
    cls = "neutral"
    arrow = "→"
    if isinstance(val, float) and not math.isnan(val):
        thr = 0.4
        if val > thr:
            cls = "up"; arrow = "↑"
        elif val < -thr:
            cls = "down"; arrow = "↓"
        else:
            cls = "neutral"; arrow = "→"
    if t == "estres":
        if cls == "up": cls = "bad"
        elif cls == "down": cls = "good"
    return f'<span class="badge {cls}">{arrow} {val:+.2f}</span>' if isinstance(val, float) and not math.isnan(val) else '<span class="badge neutral">s/d</span>'

# HTML/CSS
css = """
<style>
:root {
  --bg: #0b1020;
  --card: #121833;
  --muted: #9aa4c7;
  --text: #eaf0ff;
  --good: #1db954;
  --bad: #ff4d4f;
  --up: #2dd4bf;
  --down: #f59e0b;
  --accent: #7c83ff;
}
* { box-sizing: border-box; }
body { margin:0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
.container { max-width: 1100px; margin: 40px auto; padding: 0 16px; }
.header { display:flex; align-items:center; justify-content:space-between; margin-bottom: 18px; }
.h1 { font-size: 26px; font-weight: 800; letter-spacing: 0.2px; }
.sub { color: var(--muted); font-size: 14px; }
.grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.card { background: linear-gradient(180deg, rgba(124,131,255,0.06), rgba(18,24,51,0.9)); border: 1px solid rgba(124,131,255,0.15); border-radius: 16px; padding: 14px; box-shadow: 0 6px 24px rgba(0,0,0,0.25); }
.card h3 { margin: 0 0 8px; font-size: 16px; font-weight: 700; display:flex; align-items:center; gap:8px; }
.card .big { font-size: 20px; margin: 6px 0; }
.badge { display:inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; background: rgba(154,164,199,0.15); color: var(--muted); }
.badge.up { background: rgba(45,212,191,0.15); color: var(--up); }
.badge.down { background: rgba(245,158,11,0.15); color: var(--down); }
.badge.good { background: rgba(29,185,84,0.15); color: var(--good); }
.badge.bad { background: rgba(255,77,79,0.15); color: var(--bad); }
.kicker { color: var(--muted); font-size: 12px; margin-top: 4px; }
.section { margin-top: 28px; }
.section h2 { font-size: 18px; margin: 0 0 12px; font-weight: 800; letter-spacing: 0.3px; }
.textblock { background: rgba(18,24,51,0.6); border: 1px solid rgba(124,131,255,0.13); padding: 14px; border-radius: 12px; line-height: 1.45; color: #dfe6ff; }
.table { width:100%; border-collapse: collapse; font-size: 13px; }
.table th, .table td { padding: 8px 10px; border-bottom: 1px solid rgba(124,131,255,0.1); text-align:left; }
.table th { color: var(--muted); font-weight: 600; }
.chip { padding: 3px 8px; border-radius: 999px; background: rgba(124,131,255,0.15); color: var(--accent); font-weight: 700; font-size: 12px; }
.footer { margin: 24px 0 60px; color: var(--muted); font-size: 12px; text-align:center; }
.icon { font-size: 18px; }
.small { font-size:12px; color: var(--muted); }

.tag{display:inline-block;margin:0 6px 6px 0;padding:3px 8px;border-radius:999px;background:rgba(124,131,255,.12);color:#cbd1ff;font-size:12px}
.kpi{display:flex;align-items:center;gap:8px}
.bar{height:10px;background:rgba(124,131,255,.15);border-radius:999px;overflow:hidden}
.bar > div{height:10px;background:linear-gradient(90deg,#7c83ff,#2dd4bf)}
</style>
"""

def df_to_table(dfx: pd.DataFrame):
    if dfx is None or dfx.empty:
        return '<div class="small">Sin señal suficiente para hoy.</div>'
    rows = "\n".join(
        f"<tr><td>{r.feature}</td><td>{r.contrib:+.3f}</td></tr>"
        for r in dfx.itertuples(index=False)
    )
    return f"""
    <table class="table">
      <thead><tr><th>Driver</th><th>Contribución Δ</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """

def build_ema_report(df, target_mail, targets=targets):
    df = df[df['correo'] == target_mail]
    deltas, short_sum, verbose_sum, habits_html, patterns_html, wb_last, driver_tables = report_variables(df, targets)

    if (len(df) == 0):
        print("Correo no encontrado")
        return
    cards_html = ""
    for t in targets:
        icon = icons.get(t, "•")
        val = pd.to_numeric(df[t], errors="coerce").iloc[-1] if t in df.columns else float("nan")
        delta = deltas.get(t, float("nan"))
        badge = trend_label(delta)
        # map badge to styled span
        def badge_html(t, d):
            cls = "neutral"; arrow = "→"
            if isinstance(d, float) and not math.isnan(d):
                thr = 0.4
                if d > thr: cls, arrow = "up", "↑"
                elif d < -thr: cls, arrow = "down", "↓"
                else: cls, arrow = "neutral", "→"
            if t == "estres":
                if cls == "up": cls = "bad"
                elif cls == "down": cls = "good"
            return f'<span class="badge {cls}">{arrow} {d:+.2f}</span>' if isinstance(d, float) and not math.isnan(d) else '<span class="badge neutral">s/d</span>'
        cards_html += f"""
        <div class="card">
        <h3><span class="icon">{icon}</span> {labels[t]}</h3>
        <div class="big">Último: <strong>{'' if isinstance(val,float) and math.isnan(val) else f'{val:.2f}'}</strong> {badge_html(t, delta)}</div>
        <div class="kicker small">Δ vs. día anterior (umbral ±0.4)</div>
        </div>
        """

    drivers_sections = ""
    for t in targets:
        drivers_sections += f"""
        <div class="section">
        <h2>{labels[t]}: Drivers del día</h2>
        {df_to_table(driver_tables.get(t))}
        </div>
        """

    wb_last_str = "" if math.isnan(wb_last := (wellbeing_index(df, targets=targets).iloc[-1] if len(df) else float("nan"))) else f"{wb_last:.2f}"

    html = f"""<!DOCTYPE html>
    <html lang="es">
    <head>
    <meta charset="utf-8" />
    <title>Informe BDP – EMA</title>
    {css}
    </head>
    <body>
    <div class="container">
        <div class="header">
        <div>
            <div class="h1">Informe BDP <span class="chip">EMA</span></div>
            <div class="sub">{(pd.to_datetime(df['fecha'], dayfirst=True ,errors='coerce').dt.strftime('%d-%m-%Y (%a)').iloc[-1] if 'fecha' in df.columns else str(df.index[-1]))} • Índice de bienestar (WB) último: {wb_last_str}</div>
        </div>
        <div class="sub">Generado: {datetime.now().strftime("%d-%m-%Y %H:%M")}</div>
        </div>

        <div class="grid">
        {cards_html}
        </div>

        <div class="section">
        <h2>Resumen breve</h2>
        <div class="textblock">{short_sum}</div>
        </div>

        <div class="section">
        <h2>Interpretación descriptiva</h2>
        <div class="textblock">{verbose_sum}</div>
        </div>

        {habits_html}

        {patterns_html}

        {drivers_sections}

        <div class="footer">
        Informe BDP — generado automáticamente. Este reporte es orientativo y no reemplaza evaluación profesional.
        </div>
    </div>
    </body>
    </html>
    """

    OUTDIR = Path("output"); OUTDIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTDIR / "informeBDP_EMA.html"
    outfile.write_text(html, encoding="utf-8")
    print("Saved:", outfile)
    return df
