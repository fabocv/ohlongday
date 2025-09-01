# Generate the HTML again, this time with inline narrative functions to avoid import issues
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import importlib.util
from bdp_calcs.drivers import drivers_del_dia
import json
from bdp_calcs.mini_guia_neuroconductual import render_mini_guia_section


with open('bdp_calcs/goals.json') as f:
    goals = json.load(f)
    goals = goals["default"]

# === Helpers: Hábitos y Patrones ==================================================
# ========= Narrativa humana para targets + drivers =========
import numpy as np
import pandas as pd

_FRIENDLY = {
    "animo":"Ánimo","claridad":"Claridad","estres":"Estrés","activacion":"Activación",
    "meditacion_min":"meditación","alcohol_ud":"alcohol",
    "mov_intensidad":"intensidad de movimiento","glicemia":"glicemia",
    "cafe_cucharaditas":"café","exposicion_sol_min":"exposición a sol",
    "despertares_nocturnos":"despertares nocturnos","tiempo_ejercicio_min":"ejercicio",
    "agua_litros":"agua","horas_sueno":"horas de sueño","sueno_calidad":"sueño (calidad /10)",
}
_UNITS = {
    "meditacion_min":"min","exposicion_sol_min":"min","tiempo_ejercicio_min":"min",
    "cafe_cucharaditas":"cucharaditas","alcohol_ud":"ud",
    "pantallas_min":"min","tiempo_pantalla_noche_min":"min",
    "despertares_nocturnos":"despertares","agua_litros":"L",
    "horas_sueno":"h","mov_intensidad":"/10","sueno_calidad":"/10",
    "glicemia":"mg/dL",
}
_ROUND = {"despertares_nocturnos":1,"agua_litros":1,"horas_sueno":1,"mov_intensidad":1,"sueno_calidad":1}

def _lbl(c): return _FRIENDLY.get(c, c.replace("_"," "))
def _unit(c): return _UNITS.get(c,"")
def _rd(x, c=None):
    if x is None or pd.isna(x): return None
    nd = _ROUND.get(c, 0)
    try: return round(float(x), nd)
    except: return None

def _num(df, c):
    s = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    return s if len(s) else pd.Series([np.nan]*len(df))

def _trend_target(df, c, ridx, thr=0.4):
    s = _num(df, c)
    if len(s) < 2: return "s/d", None
    ridx = int(np.clip(ridx, 0, len(s)-1))
    dy = s.diff().iloc[ridx]
    if pd.isna(dy): return "s/d", None
    if dy >= thr: return "al alza", float(dy)
    if dy <= -thr: return "a la baja", float(dy)
    return "estable", float(dy)

def _today_and_vs7d(df, c, ridx, win=7):
    s = _num(df, c)
    if len(s)==0: return None, None
    ridx = int(np.clip(ridx, 0, len(s)-1))
    v = s.iloc[ridx]
    start = max(0, ridx - (win-1))
    base = s.iloc[start:ridx].mean(skipna=True) if ridx>start else np.nan
    dv = (v - base) if not (pd.isna(v) or pd.isna(base)) else np.nan
    return (None if pd.isna(v) else float(v),
            None if pd.isna(dv) else float(dv))

def _fmt_val(v, c):
    u = _unit(c)
    vv = _rd(v, c)
    if vv is None: return "s/d"
    return f"{vv} {u}".strip()

def _fmt_delta7d(dv, c, zero_thr=1e-6):
    if dv is None: return "→ 0 (vs. 7d)"
    if abs(dv) < zero_thr: return "→ 0 (vs. 7d)"
    sign = "↑" if dv>0 else "↓"
    val = _rd(abs(dv), c)
    u = _unit(c)
    return f"{sign} {val} {u} (vs. 7d)".replace("  "," ")

def _describe_driver(df, c, ridx, contrib_sign, win=7):
    """Texto tipo: 'meditación: 25 min; ↑ 10 min (vs. 7d) — aportó ↑'"""
    v, dv = _today_and_vs7d(df, c, ridx, win=win)
    val_txt = _fmt_val(v, c)
    delta_txt = _fmt_delta7d(dv, c)
    aporto = "aportó ↑" if contrib_sign>0 else "aportó ↓"
    return f"{_lbl(c)}: {val_txt}; {delta_txt} — {aporto}"

def narrativa_humana_targets_v2(
    df: pd.DataFrame,
    targets=("animo","claridad","estres","activacion"),
    row_idx: int = -1,
    top_k: int = 2,
    drivers_func=None,   # p.ej. drivers_del_dia
    win_base: int = 7,
    thr_trend: float = 0.4
) -> str:
    ridx = int(np.clip(row_idx if row_idx is not None else len(df)-1, 0, len(df)-1))
    bloques = []
    for t in targets:
        trend, dy = _trend_target(df, t, ridx, thr=thr_trend)
        dy_txt = "s/d" if dy is None else f"{dy:+.2f}"
        cab = f"{_lbl(t)}: {trend} (Δ{dy_txt})"

        pos, neg = [], []
        if callable(drivers_func):
            res = drivers_func(df, t, top_k=top_k, row_idx=ridx) or {}
            tab = res.get("tabla")
            if isinstance(tab, pd.DataFrame) and not tab.empty:
                tab = tab.copy()
                tab["contrib"] = pd.to_numeric(tab["contrib"], errors="coerce")
                tab = tab.dropna(subset=["contrib"])
                pos = tab.sort_values("contrib", ascending=False).head(top_k)[["feature","contrib"]].values.tolist()
                neg = tab.sort_values("contrib", ascending=True).head(top_k)[["feature","contrib"]].values.tolist()

        # fallback simple si no hay tabla
        if not pos and not neg:
            pool = [("meditacion_min",+1),("exposicion_sol_min",+1),("tiempo_ejercicio_min",+1),
                    ("mov_intensidad",-1),("glicemia",-1),("cafe_cucharaditas",-1),("alcohol_ud",-1)]
            pos = pool[:top_k]; neg = pool[top_k:top_k*2]

        pos_txt = "; ".join(_describe_driver(df, c, ridx, +1 if s>0 else -1, win=win_base) for c,s in pos)
        neg_txt = "; ".join(_describe_driver(df, c, ridx, +1 if s>0 else -1, win=win_base) for c,s in neg)

        if t == "estres":
            bloque = (f"{cab}.\n"
                      f"Factores que lo elevaron (desfavorable): {pos_txt or '—'}.\n"
                      f"Factores que lo redujeron (favorable): {neg_txt or '—'}.")
        else:
            bloque = (f"{cab}.\n"
                      f"Factores que lo elevaron: {pos_txt or '—'}.\n"
                      f"Factores que lo redujeron: {neg_txt or '—'}.")
        bloques.append(bloque)
    return "\n\n".join(bloques)



THR_DELTA = 0.35  # antes 0.40

def _delta_at(series: pd.Series, idx: int) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if idx <= 0 or idx >= len(s):
        return float("nan")
    a, b = s.iloc[idx], s.iloc[idx-1]
    return float(a - b) if (pd.notna(a) and pd.notna(b)) else float("nan")

def _wb_delta_at(df: pd.DataFrame, idx: int) -> float:
    wb = wellbeing_index(df, targets=targets)
    if idx <= 0 or idx >= len(wb):
        return float("nan")
    a, b = wb.iloc[idx], wb.iloc[idx-1]
    return float(a - b) if (pd.notna(a) and pd.notna(b)) else float("nan")


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
            "agua_litros": ("min", 2),
            "tiempo_ejercicio_min": ("min", 30),
            "meditacion_min": ("max", 120),
            "exposicion_sol_min": ("min", 15),
            "tiempo_pantalla_noche_min": ("max", 60),
            "alcohol_ud": ("max", 1),
            "cafe_cucharaditas": ("min", 0),
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
        return "<div class='section'><br><h1>Hábitos y cambios</h1><div class='small'>Sin datos suficientes.</div></div>"
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

def patterns_section_html(df: pd.DataFrame, row_idx=-1) -> str:
    sig = pattern_scores(df)
    txt = pattern_summary_text(df, sig, row_idx=row_idx, thr=0.7)
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

def _daily_sum(df, col, day):
    s = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
    if "fecha" in df.columns:
        d = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True).dt.date
        return s[d == day].sum(min_count=1)
    return pd.NA

def quick_kpis_section_html(df: pd.DataFrame, goals: dict) -> str:
    # Día objetivo: último con 'fecha'
    if "fecha" not in df.columns or df["fecha"].dropna().empty:
        return ""
    last_day = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True).dt.date.dropna().iloc[-1]

    # Pantallas total y noche
    total_screens = _daily_sum(df, "tiempo_pantallas", last_day)
    night_screens = _daily_sum(df, "tiempo_pantalla_noche_min", last_day)

    # Meta
    goal_screen = None
    if isinstance(goals, dict) and "tiempo_pantallas" in goals:
        mode, val = goals["tiempo_pantallas"]
        if mode == "max":
            goal_screen = val

    # Badges y texto
    def badge(v, thr):
        if pd.isna(v): return "<span class='badge'>–</span>"
        if thr is None: return f"<span class='badge'>{v:.0f} min</span>"
        cls = "good" if v <= thr else "bad"
        return f"<span class='badge {cls}'>{v:.0f} min</span>"

    goal_txt = "" if goal_screen is None else f"<div class='small'>meta diaria: ≤ {goal_screen:.0f} min</div>"
    total_card = f"""
    <div class="card">
      <div class="kpi"><strong>Pantallas (día)</strong>{badge(total_screens, goal_screen)}</div>
      {goal_txt}
    </div>
    """
    night_card = ""
    if not pd.isna(night_screens):
        night_card = f"""
        <div class="card">
          <div class="kpi"><strong>Pantallas (noche)</strong><span class="badge">{night_screens:.0f} min</span></div>
          <div class="small">últimas horas antes de dormir</div>
        </div>
        """

    grid = f"<div class='grid' style='grid-template-columns:repeat(2,1fr);gap:12px'>{total_card}{night_card}</div>"
    return f"""
    <div class="section">
      <h2>Indicadores rápidos</h2>
      {grid}
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



def trend_label(delta: float, thr: float = 0.40) -> str:
    if isinstance(delta, float) and not math.isnan(delta):
        if delta >  THR_DELTA: return "al alza"
        if delta < -THR_DELTA: return "a la baja"
    return "estable"

def level_label(x: float) -> str:
    if not isinstance(x, float) or math.isnan(x): return "sin dato"
    if x >=  0.50: return "alto"
    if x <= -0.50: return "bajo"
    return "medio"


def narrative_summary_inline(df: pd.DataFrame, row_idx: int = -1):
    wb_series = wellbeing_index(df, targets=targets)
    wb_val = float(wb_series.iloc[row_idx]) if (len(wb_series) and 0 <= row_idx < len(wb_series)) else float("nan")
    dwb    = _wb_delta_at(df, row_idx)

    wb_lvl_lbl = level_label(wb_val)
    wb_trd_lbl = trend_label(dwb, thr=0.40)

    pts = []
    for t in targets:
        if t in df.columns and len(pd.to_numeric(df[t], errors="coerce").dropna()) >= 2:
            dy = _delta_at(df[t], row_idx)
            lbl = trend_label(dy, thr=0.40)
            if t == "estres" and lbl == "al alza": lbl += " (desfavorable)"
            if t == "estres" and lbl == "a la baja": lbl += " (favorable)"
            pts.append(f"{labels[t]} {lbl} (Δ{dy:+.2f})")

    out = [f"Bienestar compuesto (WB) {wb_lvl_lbl} y {wb_trd_lbl} (ΔWB {dwb:+.2f})."]
    if pts: out.append("Estado por target → " + " | ".join(pts))
    return " ".join(out)


def descriptive_summary_inline(df: pd.DataFrame, row_idx: int = -1, top_k: int = 2):
    # Nivel + tendencia del día elegido
    wb_series = wellbeing_index(df, targets=targets)
    wb_val = float(wb_series.iloc[row_idx]) if (len(wb_series) and 0 <= row_idx < len(wb_series)) else float("nan")
    dwb    = _wb_delta_at(df, row_idx)
    wb_hdr = f"Su bienestar compuesto está <b>{trend_label(dwb, 0.40)} </b> (nivel {level_label(wb_val)}, ΔWB {dwb:+.2f})."

    # Línea por target (usando el mismo umbral)
    tgt_lines = []
    for t in targets:
        if t in df.columns and len(pd.to_numeric(df[t], errors="coerce").dropna()) >= 2:
            dy = _delta_at(df[t], row_idx)
            lbl = trend_label(dy, 0.40)
            if t == "estres" and lbl == "al alza": lbl += " (desfavorable)"
            if t == "estres" and lbl == "a la baja": lbl += " (favorable)"
            tgt_lines.append(f"{labels[t].lower()} {lbl} (Δ{dy:+.2f})")

    # Drivers: “elevaron” vs “redujeron”. Para ESTRES, aclaramos que elevar = peor.
    reasons = []
    for t in targets:
        res = drivers_del_dia(df, t, top_k=top_k, row_idx=row_idx)
        tab = res.get("tabla")
        if isinstance(tab, pd.DataFrame) and not tab.empty:
            pos = tab.sort_values("contrib", ascending=False).head(top_k)["feature"].tolist()
            neg = tab.sort_values("contrib", ascending=True).head(top_k)["feature"].tolist()
            pos_h = ", ".join(pos) if pos else "—"
            neg_h = ", ".join(neg) if neg else "—"
            if t == "estres":
                reasons.append(f"<br><b>Estrés</b>: factores que lo elevaron (desfavorable): {pos_h}; que lo redujeron (favorable): {neg_h}. <br>")
            else:
                reasons.append(f"<br><b>{labels[t]}</b>: factores que lo elevaron: {pos_h}; que lo redujeron: {neg_h}.<br>")

    out = [wb_hdr]
    if tgt_lines: out.append("<br>Por targets: " + "; ".join(tgt_lines) + ".")

    texto = narrativa_humana_targets_v2(
        df=df,                       
        targets=["animo","claridad","estres","activacion"],
        row_idx= -1,                         
        top_k=2,
        drivers_func=lambda df,t,**kw: drivers_del_dia(df, t, **kw),  # si la tienes
        win_base=7,
    )

    if texto:   out.append("<br><h3>Motivos probables del día:</h3> " + texto + "<br>")
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

def report_variables(df: pd.DataFrame, targets, lookback_days: int = 14, row_idx: int = -1, goals=None):
    # Deltas para tarjetas en el día elegido
    deltas = {t: (_delta_at(df[t], row_idx) if t in df.columns else float("nan")) for t in targets}

    # Piezas del reporte (todas consistentes con row_idx)
    short_sum   = narrative_summary_inline(df, row_idx=row_idx)
    verbose_sum = descriptive_summary_inline(df, row_idx=row_idx, top_k=2)
    habits_html   = habits_section_html(df, goals)          # puede seguir usando último día disponible
    patterns_html = patterns_section_html(df)               # idem
    kpi_html      = quick_kpis_section_html(df, goals)      # idem

    wb = wellbeing_index(df, targets=targets)
    wb_last = float(wb.iloc[row_idx]) if (len(wb) and row_idx < len(wb)) else float("nan")

    # Drivers del día por target (tabla ya se obtiene en descriptive; aquí podemos devolverlos también si hace falta)
    driver_tables = {}
    for t in targets:
        res = drivers_del_dia(df, t, top_k=5, row_idx=row_idx)
        driver_tables[t] = res.get("tabla", pd.DataFrame())

    # Mini guía (si tu función la usa con ventana, pásale lookback_days si corresponde dentro de su implementación)
    guia_html = render_mini_guia_section(df, alpha=0.30, lookback_days=lookback_days,
                                       row_idx=row_idx, add_style=False)

    return deltas, short_sum, verbose_sum, habits_html, patterns_html, kpi_html, wb_last, driver_tables, guia_html


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
