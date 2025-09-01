
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
    """
    Índice simple de bienestar (WB) en z-scores:
    WB = mean( z(animo), z(claridad), z(activacion), -z(estres) )
    Devuelve un Series (puede tener NaN si faltan columnas).
    """
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
    """Correlaciones en nivel y en delta (lag-0) entre variables core."""
    if targets is None:
        targets = CORE_DEFAULT
    sub = df[ [c for c in targets if c in df.columns] ].apply(pd.to_numeric, errors="coerce")
    corr_levels = sub.corr()
    corr_deltas = sub.diff().corr()
    return {"levels": corr_levels, "deltas": corr_deltas}

def crosslag_influence(df: pd.DataFrame, targets: List[str] = None) -> pd.DataFrame:
    """
    Estimación simple de influencia cruzada (estandarizada):
    Para cada j, ajusta Δj_t ~ Δj_{t-1} + sum_{i!=j} Δi_{t-1} y reporta coeficientes de i→j.
    Coefs son estándar (X e Y zscore en deltas), sin p-val.
    """
    if targets is None:
        targets = CORE_DEFAULT
    T = [c for c in targets if c in df.columns]
    if len(T) < 2:
        return pd.DataFrame()
    D = df[T].apply(pd.to_numeric, errors="coerce").diff()
    # z-score columnas
    Dz = D.apply(lambda s: (s - s.mean())/s.std() if s.std() not in (0, None, np.nan) and pd.notna(s.std()) else s, axis=0)
    rows = []
    for j in T:
        y = Dz[j].shift(0)
        X = pd.concat([Dz[x].shift(1) for x in T], axis=1)
        X = X.rename(columns={c: f"{c}_lag1" for c in T})
        # Agrega Δj_{t-1} como control explícito (ya está en X)
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < 10:
            continue
        yv = data[j]
        Xv = data.drop(columns=[j])
        # Resolver por mínimos cuadrados
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
    # matriz ancha src x dst
    return infl.pivot_table(index="src", columns="dst", values="coef_std", aggfunc="mean")

def aggregate_drivers_history(df: pd.DataFrame, drivers_func, targets: List[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """
    Recorre la historia y acumula contribuciones de drivers_del_dia para cada target.
    drivers_func: función (df, target, top_k, row_idx) -> dict con 'tabla' (feature, contrib).
    Devuelve:
      - avg_contrib[target]: DataFrame (feature -> media de contribución)
      - abs_contrib[target]: DataFrame (feature -> media de |contrib|)
      - freq_sign[target]: DataFrame (feature -> % de veces con signo positivo)
    """
    if targets is None:
        targets = CORE_DEFAULT
    n = len(df)
    start = 3  # empezamos desde la 3ra diferencia posible
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
    """
    Combina contribuciones promedio por feature ponderando por signo de bienestar:
      +1 para targets positivos (animo, claridad, activacion)
      -1 para targets negativos (estres)
    Devuelve una serie ordenada: features que mejoran/empeoran el bienestar compuesto a lo largo del tiempo.
    """
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
    # Return top absolute off-diagonal pair (name1, name2, value)
    if df_corr is None or df_corr.empty:
        return None
    m = df_corr.copy()
    for c in m.columns:
        m.loc[c, c] = np.nan
    # stack and get max abs
    s = m.stack()
    if s.empty:
        return None
    idx = s.abs().idxmax()
    return (idx[0], idx[1], float(s.loc[idx]))

def narrative_summary(df: pd.DataFrame, targets: List[str] = None, top_k: int = 3) -> str:
    """
    Genera un resumen humanista (5–7 líneas) con:
    - Tendencia reciente del bienestar (WB)
    - Par de targets más acoplados en Δ
    - Influencias cruzadas más fuertes (lag-1)
    - Top drivers pro-bienestar y anti-bienestar (histórico)
    """
    if targets is None:
        targets = CORE_DEFAULT
    # WB
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
    # correlations
    cors = core_correlations(df, targets=targets)
    pair = _pair_from_corr(cors.get("deltas"))
    corr_line = ""
    if pair:
        a, b, v = pair
        dir_txt = "se mueven juntos" if v > 0 else "tienden a moverse en direcciones opuestas"
        corr_line = f"{a} y {b} {dir_txt} en cambios diarios (corr≈{v:.2f})."
    # crosslag
    infl = crosslag_influence(df, targets=targets)
    infl_line = ""
    if isinstance(infl, pd.DataFrame) and not infl.empty:
        # flatten to list of (src, dst, val) and get top-2 by |val|
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
    # drivers synergy
    agg = aggregate_drivers_history(df, drivers_func=lambda d, t, top_k, row_idx: {"tabla": pd.DataFrame()}, targets=targets, top_k=top_k)
    # Nota: para no recorrer día por día con drivers (que puede ser costoso), si el caller no ha pasado por el runner de drivers,
    # agg quedará vacío. Aquí intentamos usar wellbeing_synergy si ya existe un 'agg' real; de lo contrario, lo recalculamos liviano
    # llamando drivers_del_dia a través del runner externo.
    try:
        from .drivers import drivers_del_dia
        agg = aggregate_drivers_history(df, drivers_func=drivers_del_dia, targets=targets, top_k=top_k)
    except Exception:
        pass
    syn = wellbeing_synergy(agg)
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

    # Sugerencia genérica basada en tendencia WB
    if wb_trend == "al alza":
        lines.append("Sugerencia: consolida lo que ya funciona (luz AM, sueño suficiente, pantallas moderadas).")
    elif wb_trend == "a la baja":
        lines.append("Sugerencia: prioriza higiene de sueño hoy, corta cafeína temprano y agenda 10–15′ de respiración.")
    else:
        lines.append("Sugerencia: día estable; mantén 1–2 micro-hábitos clave sin sobrecargar cambios.")

    return " ".join(lines)
