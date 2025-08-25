
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import List, Dict, Any

CORE_DEFAULT = ["animo","claridad","estres","activacion"]

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, v = s.mean(), s.std()
    if not np.isfinite(v) or v == 0:
        return pd.Series([np.nan]*len(s), index=s.index, dtype=float)
    return (s - m) / v

def wellbeing_index(df: pd.DataFrame, targets: List[str] = None) -> pd.Series:
    if targets is None:
        targets = CORE_DEFAULT
    cols = [c for c in targets if c in df.columns]
    if not cols:
        return pd.Series([np.nan]*len(df), index=df.index, dtype=float)
    parts = []
    for c in cols:
        zc = _z(df[c])
        if c == "estres":
            zc = -zc
        parts.append(zc)
    M = pd.concat(parts, axis=1)
    return M.mean(axis=1)

def core_correlations(df: pd.DataFrame, targets: List[str] = None) -> Dict[str, pd.DataFrame]:
    if targets is None:
        targets = CORE_DEFAULT
    sub = df[ [c for c in targets if c in df.columns] ].apply(pd.to_numeric, errors="coerce")
    corr_levels = sub.corr()
    corr_deltas = sub.diff().corr()
    return {"levels": corr_levels, "deltas": corr_deltas}

def crosslag_influence(df: pd.DataFrame, targets: List[str] = None) -> pd.DataFrame:
    if targets is None:
        targets = CORE_DEFAULT
    T = [c for c in targets if c in df.columns]
    if len(T) < 2:
        return pd.DataFrame()
    D = df[T].apply(pd.to_numeric, errors="coerce").diff()
    Dz = D.apply(lambda s: (s - s.mean())/s.std() if s.std() not in (0, None, np.nan) and pd.notna(s.std()) else s, axis=0)
    rows = []
    for j in T:
        y = Dz[j].shift(0)
        X = pd.concat([Dz[x].shift(1) for x in T], axis=1)
        X = X.rename(columns={c: f"{c}_lag1" for c in T})
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < 10:
            continue
        yv = data[j]
        Xv = data.drop(columns=[j])
        A = np.c_[np.ones(len(Xv)), Xv.values]
        coef, *_ = np.linalg.lstsq(A, yv.values, rcond=None)
        names = ["intercept"] + list(Xv.columns)
        coefs = dict(zip(names, coef))
        for name, val in coefs.items():
            if name.endswith("_lag1") and not name.startswith(f"{j}_"):
                src = name.replace("_lag1","")
                rows.append({"src": src, "dst": j, "coef_std": float(val)})
    if not rows:
        return pd.DataFrame()
    infl = pd.DataFrame(rows)
    return infl.pivot_table(index="src", columns="dst", values="coef_std", aggfunc="mean")

def aggregate_drivers_history(df: pd.DataFrame, drivers_func, targets: List[str] = None, top_k: int = 3) -> Dict[str, Any]:
    if targets is None:
        targets = CORE_DEFAULT
    n = len(df)
    start = 3
    avg_contrib = {}
    abs_contrib = {}
    freq_sign = {}
    for t in targets:
        acc = {}
        counts = {}
        pos_counts = {}
        for idx in range(start, n):
            res = drivers_func(df, t, top_k=top_k, row_idx=idx)
            tab = res.get("tabla")
            if isinstance(tab, pd.DataFrame) and not tab.empty:
                for _, row in tab.iterrows():
                    f = row["feature"]
                    v = float(row["contrib"])
                    acc[f] = acc.get(f, 0.0) + v
                    counts[f] = counts.get(f, 0) + 1
                    pos_counts[f] = pos_counts.get(f, 0) + (1 if v > 0 else 0)
        if counts:
            feats = sorted(counts.keys())
            mean = pd.Series({f: acc.get(f,0.0)/counts.get(f,1) for f in feats}).sort_values(ascending=False)
            mean_abs = pd.Series({f: abs(acc.get(f,0.0))/counts.get(f,1) for f in feats}).sort_values(ascending=False)
            freq = pd.Series({f: pos_counts.get(f,0)/counts.get(f,1) for f in feats}).sort_values(ascending=False)
            avg_contrib[t] = mean
            abs_contrib[t] = mean_abs
            freq_sign[t] = freq
        else:
            avg_contrib[t] = pd.Series(dtype=float)
            abs_contrib[t] = pd.Series(dtype=float)
            freq_sign[t] = pd.Series(dtype=float)
    return {"avg_contrib": avg_contrib, "abs_contrib": abs_contrib, "freq_sign": freq_sign}

def wellbeing_synergy(agg: Dict[str, Any], positive_targets: List[str] = ["animo","claridad","activacion"], negative_targets: List[str] = ["estres"]) -> pd.Series:
    scores = {}
    keys = set()
    for d in agg["avg_contrib"].values():
        keys.update(d.index.tolist())
    for f in keys:
        score = 0.0
        cnt = 0
        for t, s in agg["avg_contrib"].items():
            if f in s.index:
                w = 1.0 if t in positive_targets else -1.0 if t in negative_targets else 0.0
                score += w * s.loc[f]
                cnt += 1
        if cnt > 0:
            scores[f] = score / cnt
    if not scores:
        return pd.Series(dtype=float)
    return pd.Series(scores).sort_values(ascending=False)

# --- Humanist summary utilities ---

_HUMAN_LABELS = [
    ("horas_sueno", "horas de sueño"),
    ("sueno_calidad_calc", "calidad de sueño (calculada)"),
    ("sleep_debt", "deuda de sueño"),
    ("sleep_efficiency", "eficiencia de sueño"),
    ("circadian_alignment", "alineación circadiana"),
    ("cafe_", "café"),
    ("alcohol_", "alcohol"),
    ("stimulant_load", "carga de estimulantes"),
    ("hydration_score", "hidratación"),
    ("agua_litros", "agua"),
    ("alimentacion", "alimentación"),
    ("movement_score", "movimiento"),
    ("mov_", "movimiento"),
    ("relaxation_score", "respiración/meditación"),
    ("meditacion_", "respiración/meditación"),
    ("morning_light_score", "luz de mañana"),
    ("exposicion_sol_", "exposición al sol"),
    ("screen_night_score", "pantalla de noche"),
    ("social_", "interacciones"),
    ("stressors_", "estresores"),
    ("has_", "otras sustancias"),
    ("adherencia_med", "adherencia a medicación"),
]

def _humanize_feature(name: str) -> str:
    for pref, lab in _HUMAN_LABELS:
        if name.startswith(pref):
            return lab
    return name

def _pair_from_corr(df_corr: pd.DataFrame):
    if df_corr is None or df_corr.empty:
        return None
    m = df_corr.copy()
    for c in m.columns:
        m.loc[c, c] = np.nan
    s = m.stack()
    if s.empty:
        return None
    idx = s.abs().idxmax()
    return (idx[0], idx[1], float(s.loc[idx]))

def _trend_label(delta: float) -> str:
    if delta != delta:
        return "sin dato"
    if delta > 0.4:
        return "al alza"
    if delta < -0.4:
        return "a la baja"
    return "estable"

def narrative_summary(df: pd.DataFrame, targets: List[str] = None, top_k: int = 3) -> str:
    if targets is None:
        targets = CORE_DEFAULT
    wb = wellbeing_index(df, targets=targets)
    wb_trend = "estable"
    if len(wb.dropna()) >= 2:
        dwb = wb.diff().iloc[-1]
        if pd.notna(dwb):
            if dwb > 0.4:
                wb_trend = "al alza"
            elif dwb < -0.4:
                wb_trend = "a la baja"
            else:
                wb_trend = "estable"
    cors = core_correlations(df, targets=targets)
    pair = _pair_from_corr(cors.get("deltas"))
    corr_line = ""
    if pair:
        a, b, v = pair
        dir_txt = "se mueven juntos" if v > 0 else "tienden a moverse en direcciones opuestas"
        corr_line = f"{a} y {b} {dir_txt} en cambios diarios (corr≈{v:.2f})."
    infl = crosslag_influence(df, targets=targets)
    infl_line = ""
    if isinstance(infl, pd.DataFrame) and not infl.empty:
        pairs = []
        for src in infl.index:
            for dst in infl.columns:
                val = infl.loc[src, dst]
                if pd.notna(val):
                    pairs.append((src, dst, float(val)))
        if pairs:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pairs = pairs[:2]
            parts = []
            for src, dst, val in top_pairs:
                sign = "favorece" if val > 0 else "reduce"
                parts.append(f"{src} {sign} {dst} al día siguiente (β≈{val:.2f})")
            infl_line = "; ".join(parts) + "."
    # Aggregate drivers & synergy
    try:
        from .drivers import drivers_del_dia
        agg = aggregate_drivers_history(df, drivers_func=drivers_del_dia, targets=targets, top_k=top_k)
    except Exception:
        agg = {"avg_contrib": {}}
    syn = wellbeing_synergy(agg) if isinstance(agg, dict) else pd.Series(dtype=float)
    pos_line = neg_line = ""
    if isinstance(syn, pd.Series) and not syn.empty:
        pos = [f for f, v in syn.items() if v > 0]
        neg = [f for f, v in syn.items() if v < 0]
        top_pos = pos[:top_k]
        top_neg = neg[-top_k:][::-1] if neg else []
        if top_pos:
            pos_line = "Apoyan el bienestar: " + ", ".join({_humanize_feature(f) for f in top_pos}) + "."
        if top_neg:
            neg_line = "Tienden a restar: " + ", ".join({_humanize_feature(f) for f in top_neg}) + "."

    lines = []
    lines.append(f"Bienestar compuesto (WB) {wb_trend}.")
    # Per-target last-day trends
    per_target = []
    for tgt in targets:
        if tgt in df.columns and len(pd.to_numeric(df[tgt], errors="coerce").dropna()) >= 2:
            dy = pd.to_numeric(df[tgt], errors="coerce").diff().iloc[-1]
            lbl = _trend_label(dy)
            if tgt == "estres" and lbl == "al alza":
                lbl += " (desfavorable)"
            if tgt == "estres" and lbl == "a la baja":
                lbl += " (favorable)"
            pretty = {"animo":"ánimo","claridad":"claridad","estres":"estrés","activacion":"activación"}.get(tgt, tgt)
            per_target.append(f"{pretty}: {lbl} (Δ{dy:+.2f})" if dy==dy else f"{pretty}: {lbl}")
    if per_target:
        lines.append("Estado por target → " + " | ".join(per_target))
    if corr_line:
        lines.append(corr_line)
    if infl_line:
        lines.append(infl_line)
    if pos_line:
        lines.append(pos_line)
    if neg_line:
        lines.append(neg_line)
    if not pos_line and not neg_line:
        lines.append("Aumenta la señal registrando luz AM, respiración y horarios de café/alcohol.")
    if wb_trend == "al alza":
        lines.append("Sugerencia: consolida lo que ya funciona (luz AM, sueño suficiente, pantallas moderadas).")
    elif wb_trend == "a la baja":
        lines.append("Sugerencia: prioriza higiene de sueño hoy, corta cafeína temprano y agenda 10–15′ de respiración.")
    else:
        lines.append("Sugerencia: día estable; mantén 1–2 micro-hábitos clave sin sobrecargar cambios.")
    return " ".join(lines)


def _delta(s, idx=-1):
    s = pd.to_numeric(s, errors="coerce")
    if len(s.dropna()) < 2:
        return np.nan
    if idx < 0:
        idx = len(s) + idx
    if idx <= 0 or idx >= len(s):
        return np.nan
    return float(s.iloc[idx] - s.iloc[idx-1])

def _fmt_sign(x, unit=""):
    if x != x:
        return "s/d"
    return f"{x:+.2f}{unit}" if unit else f"{x:+.2f}"

def _not_empty(x):
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    return str(x).strip() != "" and str(x).strip().lower() != "nan"

def descriptive_summary(df: pd.DataFrame, targets: List[str] = None, row_idx: int = -1, top_k: int = 2) -> str:
    """
    Informe descriptivo (estilo humano/terapeuta):
    - Estado de WB (a la baja/al alza/estable)
    - Por target: sube/baja/estable (con Δ)
    - Drivers clave del día (top_k) como razones
    - Variables de contexto (agua, sueño, glicemia, estresores/interacciones)
    - Recomendaciones y mensaje para terapeuta
    """
    if targets is None:
        targets = CORE_DEFAULT
    # WB trend
    wb = wellbeing_index(df, targets=targets)
    wb_d = wb.diff().iloc[-1] if len(wb.dropna())>=2 else np.nan
    wb_lbl = _trend_label(wb_d)

    # Per-target deltas
    pretty = {"animo":"ánimo","claridad":"claridad","estres":"estrés","activacion":"activación"}
    tgt_lines = []
    for t in targets:
        if t in df.columns:
            dy = _delta(df[t], row_idx)
            lbl = _trend_label(dy)
            if t == "estres" and lbl == "al alza": lbl += " (desfavorable)"
            if t == "estres" and lbl == "a la baja": lbl += " (favorable)"
            tgt_lines.append(f"{pretty.get(t,t)} {lbl} (Δ{_fmt_sign(dy)})")

    # Drivers del día por target (razones)
    reasons = []
    try:
        from .drivers import drivers_del_dia
        for t in targets:
            if t in df.columns:
                res = drivers_del_dia(df, t, top_k=top_k, row_idx=row_idx)
                tab = res.get("tabla")
                if isinstance(tab, pd.DataFrame) and not tab.empty:
                    pos = tab.sort_values("contrib", ascending=False).head(top_k)["feature"].tolist()
                    neg = tab.sort_values("contrib", ascending=True).head(top_k)["feature"].tolist()
                    # human labels
                    def hum(n): return _humanize_feature(n)
                    pos_h = ", ".join(hum(x) for x in pos) if pos else "—"
                    neg_h = ", ".join(hum(x) for x in neg) if neg else "—"
                    reasons.append(f"{pretty.get(t,t).capitalize()}: apoyaron {pos_h}; restaron {neg_h}.")
    except Exception:
        pass

    # Contexto del día (deltas y eventos textuales)
    rows = {}
    for col, unit in [("agua_litros","L"), ("sueno_calidad_calc",""), ("sleep_efficiency",""),
                      ("horas_sueno","h"), ("glicemia",""), ("stimulant_load",""),
                      ("morning_light_score",""), ("screen_night_score","")]:
        if col in df.columns:
            rows[col] = _delta(df[col], row_idx)
    eventos = df.get("eventos_estresores")
    interac = df.get("interacciones_significativas")
    eventos_txt = str(eventos.iloc[row_idx]) if eventos is not None and len(eventos)>=abs(row_idx) and _not_empty(eventos.iloc[row_idx]) else ""
    interac_txt = str(interac.iloc[row_idx]) if interac is not None and len(interac)>=abs(row_idx) and _not_empty(interac.iloc[row_idx]) else ""

    # Recomendaciones basadas en señales
    recs = []
    # hidratación
    if "agua_litros" in rows and rows["agua_litros"] == rows["agua_litros"] and rows["agua_litros"] < -0.2:
        recs.append("Sube la hidratación (objetivo ≥ 2 vasos extra hoy).")
    # sueño
    if ("sueno_calidad_calc" in rows and rows["sueno_calidad_calc"]==rows["sueno_calidad_calc"] and rows["sueno_calidad_calc"] < -0.2) or        ("sleep_efficiency" in rows and rows["sleep_efficiency"]==rows["sleep_efficiency"] and rows["sleep_efficiency"] < -0.2):
        recs.append("Prioriza higiene de sueño (hora fija de dormir, ambiente oscuro, pantallas bajas).")
    # glicemia
    if "glicemia" in rows and rows["glicemia"]==rows["glicemia"] and rows["glicemia"] > 0.0:
        recs.append("Favorece comidas balanceadas y horarios regulares para estabilizar glicemia.")
    # luz/pantalla
    if "morning_light_score" in rows and rows["morning_light_score"]==rows["morning_light_score"] and rows["morning_light_score"] < 0:
        recs.append("10–15′ de luz natural en la mañana.")
    if "screen_night_score" in rows and rows["screen_night_score"]==rows["screen_night_score"] and rows["screen_night_score"] > 0.0:
        recs.append("Baja pantalla en la noche (idealmente 60′ antes de dormir).")
    # estrés
    if _not_empty(eventos_txt):
        recs.append("Procesa el/los estresor(es) del día con una pausa + respiración 4-6 (3–5 min).")

    # Mensaje para terapeuta (según patrones)
    therapist_msgs = []
    if any("estrés" in line and "al alza" in line for line in tgt_lines) or _not_empty(eventos_txt):
        therapist_msgs.append("Explorar estresores recientes y estrategias de afrontamiento que ayudaron/ no ayudaron.")
    if any("claridad a la baja" in line for line in tgt_lines):
        therapist_msgs.append("Indagar en sueño, rumiación y carga cognitiva; evaluar horarios de pantalla y cafeína.")
    if any("ánimo a la baja" in line for line in tgt_lines):
        therapist_msgs.append("Registrar actividades con sentido y micro-victorias; revisar autodiálogo.")
    if any("activación al alza" in line for line in tgt_lines):
        therapist_msgs.append("Diferenciar activación útil vs. ansiosa; revisar respiración y pausas programadas.")

    # Armar texto
    out = []
    out.append(f"{pd.to_datetime(df['fecha'], errors='coerce').dt.strftime('%d-%m-%Y (%a)').iloc[row_idx] if 'fecha' in df.columns else str(df.index[row_idx])}: su bienestar compuesto está {wb_lbl}.")
    if tgt_lines:
        out.append("Por targets: " + "; ".join(tgt_lines) + ".")
    if reasons:
        out.append("Motivos probables del día: " + " ".join(reasons))
    ctx_bits = []
    if "agua_litros" in rows and rows["agua_litros"]==rows["agua_litros"]:
        ctx_bits.append(f"agua { _fmt_sign(rows['agua_litros'],'L') }")
    if "horas_sueno" in rows and rows["horas_sueno"]==rows["horas_sueno"]:
        ctx_bits.append(f"horas de sueño { _fmt_sign(rows['horas_sueno'],'h') }")
    if "sueno_calidad_calc" in rows and rows["sueno_calidad_calc"]==rows["sueno_calidad_calc"]:
        ctx_bits.append(f"calidad de sueño { _fmt_sign(rows['sueno_calidad_calc']) }")
    if "glicemia" in rows and rows["glicemia"]==rows["glicemia"]:
        ctx_bits.append(f"glicemia { _fmt_sign(rows['glicemia']) }")
    if ctx_bits or _not_empty(eventos_txt) or _not_empty(interac_txt):
        ctx = ", ".join(ctx_bits)
        if _not_empty(eventos_txt):
            ctx += f"; estresores: {eventos_txt}"
        if _not_empty(interac_txt):
            ctx += f"; interacciones: {interac_txt}"
        out.append("Contexto: " + ctx + ".")
    if recs:
        out.append("Se recomienda: " + " ".join(recs))
    if therapist_msgs:
        out.append("Mensaje para terapeuta: " + " ".join(therapist_msgs))

    return " ".join(out)
