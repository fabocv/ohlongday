# Generate the HTML again, this time with inline narrative functions to avoid import issues
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
import importlib.util
from bdp_calcs.drivers import drivers_del_dia
from bdp_calcs.variables import report_variables
import json

import re


def _norm_date_ddmmyyyy(s: pd.Series) -> pd.Series:
    # reemplaza / por -, recorta espacios
    s = s.astype(str).str.strip().str.replace("/", "-", regex=False)
    # fuerza 2 dígitos día/mes si vienen como 1-1-2025
    pat = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")
    def _pad(x):
        m = pat.match(x)
        if not m: return x
        d, mth, y = m.groups()
        return f"{int(d):02d}-{int(mth):02d}-{y}"
    return s.map(_pad)

def _norm_time_hhmm(s: pd.Series) -> pd.Series:
    # admite "7:5" -> "07:05", y valores vacíos -> "00:00"
    s = s.astype(str).str.strip().replace({"nan":"", "NaT":""})
    def _pad(t):
        if not t: return "00:00"
        m = re.match(r"^(\d{1,2}):(\d{1,2})$", t)
        if not m: return "00:00"
        h, mi = m.groups()
        return f"{int(h):02d}:{int(mi):02d}"
    return s.map(_pad)

def _timestamp_from_fecha_hora(df: pd.DataFrame,
                               date_col: str = "fecha",
                               time_col: str = "hora") -> pd.Series:
    """
    Devuelve una serie datetime con formato explícito:
    - si hay fecha y hora -> '%d-%m-%Y %H:%M'
    - si sólo hay fecha   -> '%d-%m-%Y'
    """
    if date_col not in df.columns:
        # sin columna fecha: devolvemos índice como NaT y no rompemos
        return pd.to_datetime(pd.Series([pd.NaT]*len(df), index=df.index))
    d = _norm_date_ddmmyyyy(df[date_col])
    if time_col in df.columns:
        t = _norm_time_hhmm(df[time_col])
        dt = pd.to_datetime(d + " " + t, format="%d-%m-%Y %H:%M", errors="coerce")
    else:
        dt = pd.to_datetime(d, format="%d-%m-%Y", errors="coerce")
    return dt


# === Ejes fisiológicos (proxies) ==================================================
def _znorm_roll(s: pd.Series, win: int = 30) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    r = x.rolling(win, min_periods=max(10, win//3))
    z = (x - r.mean()) / r.std()
    return z

def _sigmoid01(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return 1 / (1 + np.exp(-x.clip(-6, 6)))

def _clamp_idx(n: int, i: int) -> int:
    if n <= 0: return -1
    return max(0, min(int(i), n-1))

def _col_any(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(index=df.index, dtype=float)

def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)

def _sleep_hours_series(df: pd.DataFrame) -> pd.Series:
    # 1) explícita
    s = _col_any(df, ["horas_sueno","sueno_horas"])
    if s.notna().any():
        return s
    # 2) estimar con hora_dormir / hora_despertar (o hora_DSPT)
    hd = df.get("hora_dormir")
    hr = df.get("hora_despertar", df.get("hora_DSPT"))
    if hd is None or hr is None:
        return pd.Series(index=df.index, dtype=float)
    def _to_minutes(x):
        try:
            t = pd.to_datetime(_norm_time_hhmm(x.astype(str)), format="%H:%M", errors="coerce")
            return t.dt.hour * 60 + t.dt.minute
        except Exception:
            return pd.Series([np.nan]*len(df), index=df.index, dtype=float)
    m_sleep  = _to_minutes(hd)
    m_wake   = _to_minutes(hr)
    dur = (m_wake - m_sleep) % (24*60)  # envuelve medianoche
    return (dur / 60.0).astype(float)

def _series_or_zero(df: pd.DataFrame, key: str) -> pd.Series:
    if key in df.columns:
        return pd.to_numeric(df[key], errors="coerce")
    return pd.Series(index=df.index, dtype=float)

def _screen_load_z(df, win=30):
    zmap = {}
    if "tiempo_pantallas" in df.columns:
        zmap["tiempo_pantallas"] = _znorm_roll(df["tiempo_pantallas"], win)
    if "tiempo_pantalla_noche_min" in df.columns:
        zmap["tiempo_pantalla_noche_min"] = _znorm_roll(df["tiempo_pantalla_noche_min"], win)
    if not zmap:
        return pd.Series(index=df.index, dtype=float)
    return pd.concat(zmap.values(), axis=1).max(axis=1, skipna=True)

def _morning_sun_z(df, win=30):
    if "exposicion_sol_manana_min" in df.columns:
        return _znorm_roll(df["exposicion_sol_manana_min"], win)
    return _znorm_roll(_series_or_zero(df, "exposicion_sol_min"), win)

def _irregular_bedtime(df, win=14):
    s = df.get("hora_dormir")
    if s is None:
        return pd.Series(index=df.index, dtype=float)
    ss = s.astype(str).str.strip().str.replace(".", ":", regex=False).str.replace(";", ":", regex=False)
    def to_min(v):
        try:
            if not v or v.lower() in {"nan","nat"}: return np.nan
            hh, mm = v.split(":")[:2]
            return (int(hh)%24)*60 + (int(mm)%60)
        except Exception:
            return np.nan
    mins = ss.map(to_min)
    r = mins.rolling(win, min_periods=max(7, win//2))
    return r.std()
def hormones_proxies(df: pd.DataFrame, win: int = 30, row_idx: int | None = None):
    # --- Señales base (con aliases) ---
    z = {}
    z["animo"]       = _znorm_roll(_col_any(df, ["animo"]), win)
    z["claridad"]    = _znorm_roll(_col_any(df, ["claridad"]), win)
    z["estres"]      = _znorm_roll(_col_any(df, ["estres"]), win)
    z["activacion"]  = _znorm_roll(_col_any(df, ["activacion"]), win)

    horas_sueno = _sleep_hours_series(df)
    z["horas_sueno"]    = _znorm_roll(horas_sueno, win)
    z["sueno_calidad"]  = _znorm_roll(_col_any(df, ["sueno_calidad"]), win)

    z["tiempo_ejercicio_min"] = _znorm_roll(_col_any(df, ["tiempo_ejercicio_min","tiempo_ejercicio"]), win)
    z["exposicion_sol_min"]   = _znorm_roll(_col_any(df, ["exposicion_sol_min","exposicion_sol_manana_min"]), win)
    z["agua_litros"]          = _znorm_roll(_col_any(df, ["agua_litros"]), win)
    z["cafe_cucharaditas"]    = _znorm_roll(_col_any(df, ["cafe_cucharaditas"]), win)
    z["alcohol_ud"]           = _znorm_roll(_col_any(df, ["alcohol_ud"]), win)
    z["ansiedad"]             = _znorm_roll(_col_any(df, ["ansiedad"]), win)
    z["irritabilidad"]        = _znorm_roll(_col_any(df, ["irritabilidad"]), win)
    z["glicemia"]             = _znorm_roll(_col_any(df, ["glicemia"]), win)
    z["mov_intensidad"]       = _znorm_roll(_col_any(df, ["mov_intensidad","movimiento"]), win)
    z["despertares_nocturnos"]= _znorm_roll(_col_any(df, ["despertares_nocturnos","DSPTes_nocturnos"]), win)

    z["screen"]       = _screen_load_z(df, win)
    z["sun_morning"]  = _morning_sun_z(df, win)
    bed_irreg_raw     = _irregular_bedtime(df)
    z["bed_irreg"]    = _znorm_roll(bed_irreg_raw, win)

    # --- Proxies (signos coherentes) ---
    cort_z = (0.30*z["estres"] +
              0.25*z["activacion"] +
              0.20*(-z["horas_sueno"]) +
              0.15*z["cafe_cucharaditas"] +
              0.10*(-z["sun_morning"]) +
              0.10*z["despertares_nocturnos"])

    mel_z = (0.35*z["sueno_calidad"] +
             0.25*(-z["screen"]) +
             0.15*z["exposicion_sol_min"] +
             0.15*(-z["alcohol_ud"]) +
             0.10*(-z["bed_irreg"]))

    ins_z = (0.40*(-z["glicemia"]) +
             0.25*z["tiempo_ejercicio_min"] +
             0.15*z["horas_sueno"] +
             0.10*(-z["alcohol_ud"]) +
             0.10*_znorm_roll(_series_or_zero(df, "alimentacion"), win))

    anx_mix = pd.concat([z["estres"], z["ansiedad"]], axis=1).max(axis=1, skipna=True)
    ap_z = (0.40*z["horas_sueno"] +
            0.20*z["sueno_calidad"] +
            0.20*(-anx_mix) +
            0.10*z["tiempo_ejercicio_min"] +
            0.10*(-z["screen"]))

    stim_flag = df.get("otras_sustancias", pd.Series([""]*len(df))).astype(str)\
        .str.contains("psicoestimulantes|nicotina", case=False, na=False).astype(int)
    stim_z = _znorm_roll(stim_flag, win)
    move_proxy = z["mov_intensidad"]

    cat_z = (0.35*z["estres"] +
             0.20*z["ansiedad"] +
             0.20*(0.6*z["cafe_cucharaditas"] + 0.4*stim_z) +
             0.15*(move_proxy) +
             0.10*(-z["sueno_calidad"]))

    scores = pd.DataFrame({
        "cortisol_proxy":     _sigmoid01(cort_z),
        "melatonina_proxy":   _sigmoid01(mel_z),
        "insulin_sens_proxy": _sigmoid01(ins_z),
        "apetito_proxy":      _sigmoid01(ap_z),
        "catecolaminas_proxy":_sigmoid01(cat_z),
    }, index=df.index)

    # --- Explicaciones top en el día pedido ---
    explains = {}
    if len(scores):
        ridx = _clamp_idx(len(scores), row_idx if row_idx is not None else len(scores)-1)

        def contribs(terms):
            vals = []
            for label, series, sign in terms:
                s = pd.to_numeric(series, errors="coerce")
                v = (s.iloc[ridx] if len(s) and ridx < len(s) else np.nan)
                vals.append((label, (v * sign) if pd.notna(v) else np.nan))
            up = sorted([x for x in vals if pd.notna(x[1]) and x[1] > 0], key=lambda t: -t[1])[:3]
            dn = sorted([x for x in vals if pd.notna(x[1]) and x[1] < 0], key=lambda t:  t[1])[:3]
            return up, dn

        explains["cortisol_proxy"] = contribs([
            ("estrés", z["estres"], +0.30),
            ("activación", z["activacion"], +0.25),
            ("café", z["cafe_cucharaditas"], +0.15),
            ("despertares", z["despertares_nocturnos"], +0.10),
            ("poca luz matinal", -z["sun_morning"], +0.10),
            ("poco sueño", -z["horas_sueno"], +0.20),
        ])
        explains["melatonina_proxy"] = contribs([
            ("calidad de sueño", z["sueno_calidad"], +0.35),
            ("menos pantallas", -z["screen"], +0.25),
            ("luz diurna", z["exposicion_sol_min"], +0.15),
            ("menos alcohol", -z["alcohol_ud"], +0.15),
            ("regularidad horario", -z["bed_irreg"], +0.10),
        ])
        explains["insulin_sens_proxy"] = contribs([
            ("glicemia baja", -z["glicemia"], +0.40),
            ("ejercicio", z["tiempo_ejercicio_min"], +0.25),
            ("horas de sueño", z["horas_sueno"], +0.15),
            ("menos alcohol", -z["alcohol_ud"], +0.10),
            ("alimentación", _znorm_roll(_series_or_zero(df, "alimentacion"), win), +0.10),
        ])
        explains["apetito_proxy"] = contribs([
            ("más sueño", z["horas_sueno"], +0.40),
            ("calidad de sueño", z["sueno_calidad"], +0.20),
            ("menos estrés/ansiedad", -anx_mix, +0.20),
            ("ejercicio", z["tiempo_ejercicio_min"], +0.10),
            ("menos pantallas tarde", -z["screen"], +0.10),
        ])
        explains["catecolaminas_proxy"] = contribs([
            ("estrés", z["estres"], +0.35),
            ("ansiedad", z["ansiedad"], +0.20),
            ("estimulantes", 0.6*z["cafe_cucharaditas"] + 0.4*stim_z, +0.20),
            ("mov. intensidad", move_proxy, +0.15),
            ("poco sueño reparador", -z["sueno_calidad"], +0.10),
        ])

    return scores, explains

def hormones_section_html(df: pd.DataFrame, row_idx: int = -1) -> str:
    scores, explains = hormones_proxies(df, win=30, row_idx=row_idx)

    if scores is None or scores.empty:
        return "<div class='section'><h2>Ejes fisiológicos (proxies)</h2><div class='small'>Sin datos suficientes.</div></div>"

    ridx = _clamp_idx(len(scores), row_idx if row_idx is not None else len(scores)-1)
    row = scores.iloc[ridx]

    labels = [
        ("Cortisol-like",     "cortisol_proxy",     "carga/activación matinal"),
        ("Melatonina-like",   "melatonina_proxy",   "presión de sueño/ritmo nocturno"),
        ("Sens. a insulina",  "insulin_sens_proxy", "respuesta metabólica"),
        ("Control de apetito","apetito_proxy",      "equilibrio hambre/saciedad"),
        ("Catecolaminas",     "catecolaminas_proxy","tono simpático"),
    ]

    css_local = """
<style>
#horm-prox .grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px }
#horm-prox .card{ border:1px solid rgba(0,0,0,.08); border-radius:12px; padding:12px }
#horm-prox .kpi{ display:flex; align-items:center; justify-content:space-between; margin-bottom:6px }
#horm-prox .badge{ font-weight:700; padding:2px 8px; border-radius:999px; background:rgba(0,0,0,.06) }
#horm-prox .small{ opacity:.8 }
#horm-prox .bar{ height:8px; background:rgba(0,0,0,.08); border-radius:8px; overflow:hidden; margin-top:6px }
#horm-prox .bar > div{ height:100%; width:0; background:#22a063 }
@media (prefers-color-scheme: dark){
  #horm-prox .card{ border-color: rgba(255,255,255,.12) }
  #horm-prox .badge{ background: rgba(255,255,255,.10) }
  #horm-prox .bar{ background: rgba(255,255,255,.12) }
}
</style>
""".strip()

    cards = []
    for title, key, sub in labels:
        v = row.get(key, np.nan)
        vw = 0 if (pd.isna(v) or v < 0) else min(100, int(round(float(v)*100)))
        vtxt = "s/d" if pd.isna(v) else f"{float(v):.2f}"
        cards.append(f"""
        <div class="card">
          <div class="kpi"><strong>{title}</strong><span class="badge">{vtxt}</span></div>
          <div class="small">{sub}</div>
          <div class="bar"><div style="width:{vw}%;"></div></div>
        </div>
        """.strip())

    grid = f"<div class='grid'>{''.join(cards)}</div>"

    def mk_text(key):
        up_dn = explains.get(key, ([], []))
        if not isinstance(up_dn, tuple) or len(up_dn) != 2:
            return "—"
        up, dn = up_dn
        def fmt(lst, pref):
            if not lst: return ""
            txt = ", ".join(l for (l, _) in lst)
            return f"{pref}: {txt}. "
        base = {
            "cortisol_proxy":     "Cortisol-like: mayor = más carga/activación. ",
            "melatonina_proxy":   "Melatonina-like: mayor = mejor presión de sueño. ",
            "insulin_sens_proxy": "Sensibilidad a insulina: mayor = mejor. ",
            "apetito_proxy":      "Control de apetito: mayor = mejor equilibrio hambre/saciedad. ",
            "catecolaminas_proxy":"Catecolaminas: mayor = más tono simpático. ",
        }.get(key, "")
        return base + fmt(up, "Impulsaron") + fmt(dn, "Atenuaron")

    items = "".join(f"<li>{mk_text(key)}</li>" for _, key, _ in labels)

    how_calc = (
        "<div class='small'>Cómo se estima: z-scores móviles (30d) de variables relevantes "
        "(↑/↓ según fisiología), combinados y mapeados a 0–1 con función sigmoide. "
        "Ej.: cortisol-like ↑ con más estrés, activación, café y despertares; ↓ con más sueño y luz matinal. "
        "Estos son <strong>proxies orientativos</strong>; no equivalen a exámenes clínicos.</div>"
    )

    return f"""
    <div id="horm-prox" class="section">
      {css_local}
      <h2>Ejes fisiológicos (proxies)</h2>
      {grid}
      <div class="section">
        <div class="card"><ul class="small">{items}</ul></div>
        {how_calc}
      </div>
    </div>
    """.strip()


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
            d = pd.to_datetime(df["fecha"],  dayfirst=True, errors="coerce").dt.strftime("%d-%m-%Y (%a)").iloc[-1]
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
/* Secciones a pantalla completa */
.fullscreen-section{
  min-height: 100svh; /* 100% alto de pantalla, robusto en mobile */
  display: flex;
  flex-direction: column;
  justify-content: flex-start;   /* o 'center' si quieres centrar */
  gap: 12px;
  padding: 16px 0;               /* respiro vertical */
  box-sizing: border-box;
  overflow: auto;                 /* si el contenido excede la pantalla, scrollea dentro */
}
.container.snap-y { scroll-snap-type: y proximity; }
.fullscreen-section { scroll-snap-align: start; }
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

def build_ema_report(
    df: pd.DataFrame,
    target_email: str,
    targets,
    target_day: str | None = None,   # "25-08-2025"
    lookback_days: int = 14
) -> str:
    if "fecha" in df.columns:
        _f = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
        df = df.iloc[_f.argsort(kind="mergesort")].reset_index(drop=True)
        fechas = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce").dt.date
    else:
        fechas = pd.to_datetime(df.index, errors="coerce").date

    # 2) Resolver índice del día objetivo
    # target_day es "%d-%m-%Y"
    def _find_row_idx_for_day(df, target_day: str) -> int:
        fechas = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce").dt.date
        td = pd.to_datetime(target_day, format="%d-%m-%Y", errors="coerce").date()
        exact = np.where(fechas.values == td)[0]
        if exact.size: return int(exact[-1])
        leq = np.where((pd.notna(fechas.values)) & (fechas.values <= td))[0]
        return int(leq[-1]) if leq.size else len(df)-1

    row_idx = _find_row_idx_for_day(df, target_day)

    #row_idx = (len(df) - 1) if not target_day else _find_row_idx_for_day(target_day)

    # 3) Piezas del informe (todas con row_idx consistente)
    deltas, short_sum, verbose_sum, habits_html, patterns_html, kpi_html, wb_day, driver_tables, guia_html = \
        report_variables(df, targets, lookback_days=lookback_days, row_idx=row_idx)

    # 4) Proxies fisiológicos del día elegido
    hormones_html = hormones_section_html(df, row_idx=row_idx)

    # 5) Tarjetas de targets con Δ del día elegido
    def _delta_at(series: pd.Series, idx: int) -> float:
        s = pd.to_numeric(series, errors="coerce")
        if idx <= 0 or idx >= len(s): return float("nan")
        a, b = s.iloc[idx], s.iloc[idx-1]
        return float(a - b) if (pd.notna(a) and pd.notna(b)) else float("nan")

    cards_html = ""
    for t in targets:
        icon = icons.get(t, "•")
        s = pd.to_numeric(df.get(t, pd.Series(dtype=float)), errors="coerce")
        val = s.iloc[row_idx] if len(s) else float("nan")
        delta = _delta_at(s, row_idx)
        def badge_html(tname, d):
            cls = "neutral"; arrow = "→"
            if isinstance(d, float) and not math.isnan(d):
                thr = 0.4
                if d > thr: cls, arrow = "up", "↑"
                elif d < -thr: cls, arrow = "down", "↓"
                else: cls, arrow = "neutral", "→"
            if tname == "estres":
                if cls == "up": cls = "bad"
                elif cls == "down": cls = "good"
            return f'<span class="badge {cls}">{arrow} {d:+.2f}</span>' if isinstance(d, float) and not math.isnan(d) else '<span class="badge neutral">s/d</span>'

        cards_html += f"""
        <div class="card">
          <h3><span class="icon">{icon}</span> {labels[t]}</h3>
          <div class="big">Último: <strong>{'' if (isinstance(val, float) and math.isnan(val)) else f'{val:.2f}'}</strong> {badge_html(t, delta)}</div>
          <div class="kicker small">Δ vs. día anterior (umbral ±0.4)</div>
        </div>
        """

    # 6) Drivers (tabla) – ya están calculados con row_idx en report_variables si los usas más abajo
    drivers_sections = ""
    for t in targets:
        tab = driver_tables.get(t)
        if isinstance(tab, pd.DataFrame) and not tab.empty:
            rows = "\n".join(f"<tr><td>{r.feature}</td><td>{r.contrib:+.3f}</td></tr>" for r in tab.itertuples(index=False))
            table_html = f"""<table class="table"><thead><tr><th>Driver</th><th>Contribución Δ</th></tr></thead><tbody>{rows}</tbody></table>"""
        else:
            table_html = '<div class="small">Sin señal suficiente para hoy.</div>'
        drivers_sections += f"""
        <div class="section">
          <h2>{labels[t]}: Drivers del día</h2>
          {table_html}
        </div>
        """

    # 7) Cabecera (fecha del día elegido + WB de ese día)
    if "fecha" in df.columns:
        head_date = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce").dt.strftime("%d-%m-%Y (%a)").iloc[row_idx]
    else:
        head_date = str(row_idx)

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
          <div class="sub">{head_date} • Índice de bienestar (WB) día: {'' if (isinstance(wb_day, float) and math.isnan(wb_day)) else f'{wb_day:.2f}'}</div>
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

      <div style="padding:50px"></div>
      {habits_html}
      {patterns_html}
      {guia_html}
      {kpi_html}
      {hormones_html}
      {drivers_sections}

      <div class="footer">Informe BDP — generado automáticamente. Este reporte es orientativo y no reemplaza evaluación profesional.</div>
    </div>
    </body>
    </html>"""

    OUTDIR = Path("output"); OUTDIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTDIR / "informeBDP_EMA.html"
    outfile.write_text(html, encoding="utf-8")
    print("Saved:", outfile)
    return str(outfile)



def _build_ema_report(df, target_mail, targets=targets, row_idx=-1):
    df = df[df['correo'] == target_mail]
    deltas, short_sum, verbose_sum, habits_html, patterns_html, kpi_html, wb_last, driver_tables, guia_html = report_variables(df, targets, row_idx)
    hormones_html = hormones_section_html(df)
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
        
        <div id='intro-section' class='fullscreen-section'>
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
        </div>

        <div id='habits-section' class='fullscreen-section'>
            {habits_html}
        </div>

        <div id='patterns-section' class='fullscreen-section'>
            {patterns_html}
        </div>

        {guia_html}

        <div id='kpi-section' class='fullscreen-section'>
            {kpi_html}
        </div>

        {hormones_html}

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
    return str(outfile)
