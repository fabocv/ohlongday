# Generate the HTML again, this time with inline narrative functions to avoid import issues
import math
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import importlib.util
from bdp_calcs.relations import narrative_summary, descriptive_summary
from bdp_calcs.drivers import drivers_del_dia


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


def day_label(df):
    if "fecha" in df.columns:
        try:
            d = pd.to_datetime(df["fecha"], errors="coerce").dt.strftime("%d-%m-%Y (%a)").iloc[-1]
            if isinstance(d, str) and d.strip():
                return d
        except Exception:
            pass
    return str(df.index[-1]) if len(df) else "—"

def choose_model_tag():
    return "EMA"



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

def build_ema_report(df, targets=targets):
    # Deltas for cards
    deltas = {t: last_delta(df[t]) if t in df.columns else float("nan") for t in targets}

    # Build report pieces
    short_sum = narrative_summary(df, targets=targets, top_k=3)
    verbose_sum = descriptive_summary(df, targets=targets, row_idx=-1, top_k=2)
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
            <div class="sub">{(pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce').dt.strftime('%d-%m-%Y (%a)').iloc[-1] if 'fecha' in df.columns else str(df.index[-1]))} • Índice de bienestar (WB) último: {wb_last_str}</div>
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
