
"""
mini_guia_neuroconductual.py
--------------------------------
Sección HTML: "Mini-guía de intervención por diana neuro-conductual (para tu tablero)"
Lista para integrarse al pipeline BDP.

NOVEDADES
- Compatibilidad con nombres actualizados:
  * hora_despertar (antes: hora_DSPT)
  * despertares_nocturnos (antes: DSPTes_nocturnos)
- Estas señales ahora contribuyen a la tarjeta de "Disregulación autonómica":
  * Variabilidad de la hora de despertar (desorden de ritmo circadiano)
  * Promedio de despertares nocturnos (sueño fragmentado)

USO RÁPIDO
----------
from mini_guia_neuroconductual import render_mini_guia_section
html = render_mini_guia_section(df, alpha=0.30, lookback_days=14)
# -> devuelve un <section>...</section> listo para insertar.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import html
import math
import re

# ---------------------------- utilidades numéricas ----------------------------
# --- Mini-guía anclada al día elegido ---------------------------------------

from datetime import timedelta

def _first_text_by_day(raw_df: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw_df.columns:
        return pd.Series(dtype=object)
    tmp = raw_df.dropna(subset=["__day"])[["__day", col]].copy()
    tmp[col] = tmp[col].astype(str).str.strip()
    # primer no-vacío del día
    return tmp.groupby("__day")[col].apply(
        lambda s: next((x for x in s if x and x.lower() not in ("nan","nat","none","null")), "")
    )


def _coerce_hhmm_latam_ampm(s: pd.Series) -> pd.Series:
    """
    Convierte '10:00:00 a. m.' / '9:30:00 p. m.' / '9:30 am' / '09:30' → 'HH:MM'.
    Inválidos -> '' (string vacío). Sin warnings.
    """
    s = s.astype(str).str.replace("\xa0", " ", regex=False).str.strip().str.lower()
    # unifica 'a. m.' / 'a.m.' / 'am' y 'p. m.' / 'pm'
    s = s.str.replace(r"\s*a\.?\s*m\.?\s*", " am ", regex=True)
    s = s.str.replace(r"\s*p\.?\s*m\.?\s*", " pm ", regex=True)
    # normaliza separadores y espacios
    s = s.str.replace(".", ":", regex=False).str.replace(",", ":", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    out = pd.Series("", index=s.index, dtype=object)

    # 12h con segundos: 'H:MM:SS am/pm'
    m = s.str.match(r"^\d{1,2}:\d{2}:\d{2}\s*(am|pm)$")
    if m.any():
        t = pd.to_datetime(s[m], format="%I:%M:%S %p", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    # 12h sin segundos: 'H:MM am/pm'
    m = s.str.match(r"^\d{1,2}:\d{2}\s*(am|pm)$")
    if m.any():
        t = pd.to_datetime(s[m], format="%I:%M %p", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    # 24h con segundos: 'HH:MM:SS'
    m = s.str.match(r"^\d{1,2}:\d{2}:\d{2}$")
    if m.any():
        t = pd.to_datetime(s[m], format="%H:%M:%S", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    # 24h sin segundos: 'HH:MM'
    m = s.str.match(r"^\d{1,2}:\d{2}$")
    if m.any():
        t = pd.to_datetime(s[m], format="%H:%M", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    # sólo dígitos tipo '730' → 07:30
    m = s.str.match(r"^\d{3,4}$")
    if m.any():
        d = s[m]
        h = d.str[:-2].astype(int)
        mi = d.str[-2:].astype(int)
        ok = (h.between(0,23) & mi.between(0,59))
        out.loc[m & ok] = h[ok].map("{:02d}".format) + ":" + mi[ok].map("{:02d}".format)

    # fracción de día / horas decimales (p. ej., '0.42' o '7.5')
    m = s.str.match(r"^\d+([\.:]\d+)?$")
    if m.any():
        dec = s[m].str.replace(":", ".", regex=False)
        with np.errstate(all="ignore"):
            f = pd.to_numeric(dec, errors="coerce")
        # Excel day fraction (0..1)
        frac = (f >= 0) & (f <= 1)
        if frac.any():
            total_min = (f[frac] * 24 * 60).round().astype(int)
            hh = (total_min // 60).clip(0,23)
            mm = (total_min % 60).clip(0,59)
            out.loc[m[frac].index] = hh.map("{:02d}".format) + ":" + mm.map("{:02d}".format)
        # horas decimales (0..24)
        hrs = (f > 0) & (f < 24)
        if hrs.any():
            h = f[hrs].astype(int)
            mm = ((f[hrs] - h) * 60).round().astype(int).clip(0,59)
            out.loc[m[hrs].index] = h.map("{:02d}".format) + ":" + mm.map("{:02d}".format)

    return out


def _hhmm_to_minutes_series(s: pd.Series) -> pd.Series:
    hhmm = _coerce_hhmm_latam_ampm(s)
    t = pd.to_datetime(hhmm, format="%H:%M", errors="coerce")  # formato explícito
    return (t.dt.hour * 60 + t.dt.minute).astype(float)



def _num_series(df: pd.DataFrame, cols, default=np.nan) -> pd.Series:
    """Señales de ESTADO (0–10, etc.): faltante = NaN (no inventar 0)."""
    if isinstance(cols, str): cols = [cols]
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)

def _num0(df: pd.DataFrame, cols) -> pd.Series:
    """Señales de DOSIS/TIEMPO (minutos, cucharaditas, etc.): faltante = 0."""
    if isinstance(cols, str): cols = [cols]
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)



def _str_series(df: pd.DataFrame, cols, default: str = "") -> pd.Series:
    """
    Devuelve Serie de strings alineada a df.index.
    """
    if isinstance(cols, str):
        cols = [cols]
    for c in cols:
        if c in df.columns:
            return df[c].astype(str).fillna("").str.strip()
    return pd.Series(default, index=df.index, dtype=object)

def _hhmm_to_minutes_series(s: pd.Series) -> pd.Series:
    """Convierte 'HH:MM' → minutos (float). Vacíos → NaN."""
    t = pd.to_datetime(s.astype(str), format="%H:%M", errors="coerce")
    return (t.dt.hour * 60 + t.dt.minute).astype(float)


def _clamp_row_idx(df: pd.DataFrame, row_idx: int | None) -> int:
    if len(df) == 0:
        return -1
    if row_idx is None:
        return len(df) - 1
    return max(0, min(int(row_idx), len(df) - 1))

def _safe_num(v, fmt="{:.2f}", sd="—"):
    try:
        x = float(v)
        if not np.isfinite(x):
            return sd
        return fmt.format(x)
    except Exception:
        return sd

def _safe_mean_str(s, fmt="{:.2f}", sd="—"):
    m = pd.to_numeric(s, errors="coerce").mean()
    return _safe_num(m, fmt=fmt, sd=sd)


def _window_by_days(df: pd.DataFrame, row_idx: int, lookback_days: int) -> tuple[pd.DataFrame, dict]:
    """
    Devuelve (df_win, meta) para la ventana [row_idx-(lookback_days-1), row_idx].
    meta = {'start': date|None, 'end': date|None, 'n': int, 'warmup': bool}
    """
    if len(df) == 0:
        return df.copy(), {"start": None, "end": None, "n": 0, "warmup": True}

    ridx = _clamp_row_idx(df, row_idx)

    # Sin fecha → recorte por índice
    if "fecha" not in df.columns:
        start_idx = max(0, ridx - (lookback_days - 1))
        win = df.iloc[start_idx:ridx + 1].copy()
        meta = {"start": None, "end": None, "n": len(win), "warmup": len(win) < 7}
        return win, meta

    # Con fecha/hora → ordenar y recortar por timestamp
    try:
        ts = pd.to_datetime(df["fecha"].astype(str) + " " + df.get("hora", "00:00").astype(str),
                            dayfirst=True, errors="coerce")
    except Exception:
        ts = pd.to_datetime(df["fecha"].astype(str), dayfirst=True, errors="coerce")

    df2 = df.copy()
    df2["__ts"] = ts
    df2 = df2.sort_values("__ts")

    ridx = _clamp_row_idx(df2, ridx)
    end_ts = df2["__ts"].iloc[ridx]
    if pd.isna(end_ts):
        start_idx = max(0, ridx - (lookback_days - 1))
        win = df2.iloc[start_idx:ridx + 1].copy()
        meta = {"start": None, "end": None, "n": len(win), "warmup": len(win) < 7}
        return win.drop(columns=["__ts"]), meta

    start_ts = end_ts - pd.Timedelta(days=lookback_days - 1)
    mask = (df2["__ts"] >= start_ts) & (df2["__ts"] <= end_ts)
    win = df2.loc[mask].copy()
    if win.empty:
        start_idx = max(0, ridx - (lookback_days - 1))
        win = df2.iloc[start_idx:ridx + 1].copy()
        meta = {"start": None, "end": None, "n": len(win), "warmup": len(win) < 7}
        return win.drop(columns=["__ts"]), meta

    meta = {
        "start": start_ts.normalize().date(),
        "end":   end_ts.normalize().date(),
        "n":     len(win),
        "warmup": len(win) < 7,
    }
    return win.drop(columns=["__ts"]), meta

def _delta_ansiedad_post_expo(ans_series: pd.Series, exp_bin: pd.Series) -> float:
    """
    Promedia (ans[i+1] - ans[i]) sobre los días con exposición (exp_bin[i]==1)
    usando pares válidos dentro de la ventana. Devuelve NaN si no hay pares.
    """
    s = pd.to_numeric(ans_series, errors="coerce")
    e = pd.to_numeric(exp_bin, errors="coerce").fillna(0).astype(int)
    if len(s) == 0 or len(e) == 0:
        return float("nan")
    n = min(len(s), len(e))
    s, e = s.iloc[:n], e.iloc[:n]

    deltas = []
    for i in range(n - 1):
        if e.iloc[i] == 1 and pd.notna(s.iloc[i]) and pd.notna(s.iloc[i+1]):
            deltas.append(float(s.iloc[i+1] - s.iloc[i]))
    return float(np.nanmean(deltas)) if deltas else float("nan")


def _to_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.to_numeric(pd.Series(series), errors="coerce")

def _scale_01(x: pd.Series, lo: float, hi: float) -> pd.Series:
    x = _to_numeric(x)
    return (x.clip(lower=lo, upper=hi) - lo) / max(1e-9, (hi - lo))

def _ema(x: pd.Series, alpha: float) -> pd.Series:
    x = _to_numeric(x)
    return x.ewm(alpha=alpha, adjust=False).mean()

def _last_valid(series: pd.Series) -> Optional[float]:
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return None

def _count_nonempty(series: pd.Series) -> int:
    if series is None:
        return 0
    return int(series.fillna("").astype(str).str.strip().replace("nan","").ne("").sum())

# ---------------------------- alias de columnas -------------------------------

def _col(df: pd.DataFrame, primary: str, fallback: Optional[str]=None, default_value=np.nan) -> pd.Series:
    if primary in df.columns:
        return df[primary]
    if fallback and (fallback in df.columns):
        return df[fallback]
    return pd.Series([default_value]*len(df), index=df.index)

def _time_to_minutes(series: pd.Series) -> pd.Series:
    """
    Convierte "HH:MM" (o "H:MM") a minutos desde medianoche. Valores inválidos -> NaN.
    """
    s = series.fillna("").astype(str).str.strip()
    out = []
    for v in s:
        try:
            if v == "" or v.lower() == "nan":
                out.append(np.nan)
                continue
            parts = v.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            out.append(h*60 + m)
        except Exception:
            out.append(np.nan)
    return pd.Series(out, index=series.index, dtype="float")

# -------------------------- dataclass de la tarjeta ---------------------------

@dataclass
class DianaCard:
    key: str
    titulo: str
    subtitulo: str
    emoji: str
    protocolo: str
    medir: str
    score: float           # 0–100 (alto = mayor prioridad de intervención)
    estado: str            # "verde" | "ambar" | "rojo"
    confiabilidad: str     # "alta" | "media" | "baja"
    metricas: Dict[str, str]

# -------------------------- cálculo de los 4 targets -------------------------

import unicodedata

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def _interacciones_flag_series(df: pd.DataFrame, cols=None) -> pd.Series:
    """
    Devuelve una Serie 0/1 por fila indicando si hubo interacciones significativas.
    - Considera la PRIMERA columna disponible en `cols` (o un set por defecto).
    - Detección por:
        * NÚMEROS > 0 en el texto (p. ej., '2', '3 de 5').
        * PALABRAS positivas: sí/ok/bien/algunas/varias/muchas/alguna/algun/pocas/alta/media/baja/hubo/positivo/true/verdadero.
        * ICONOS ✓ ✔ ✅.
      Niega si aparecen palabras negativas claras: no/ninguna/ninguno/cero/nada/false/falso
      (el número, si existe, tiene prioridad).
    - Retorna Serie dtype int (0/1) alineada al índice del df.
    """
    if cols is None:
        cols = ["interacciones_significativas", "interacciones_calidad", "interacciones"]

    # Toma la primera columna existente
    col = next((c for c in (cols if isinstance(cols, (list, tuple)) else [cols]) if c in df.columns), None)
    if col is None:
        return pd.Series(0, index=df.index, dtype=int)

    s = df[col].astype(str)

    # Normaliza: minúsculas + quita acentos + colapsa espacios
    s_norm = s.map(lambda x: _strip_accents(x.strip().lower()) if isinstance(x, str) else "")
    s_norm = s_norm.str.replace(r"\s+", " ", regex=True)

    # 1) ¿hay número > 0?
    #    Extrae el primer número entero (p. ej., "2", "3").
    num = pd.to_numeric(s_norm.str.extract(r"(-?\d+)", expand=False), errors="coerce")
    flag_num = num.fillna(0).astype(float).gt(0)

    # 2) Palabras/íconos
    # Positivas (presencia >0). Incluye 'pocas', 'algun/a', 'hubo', niveles 'alta/media/baja'
    pos_words = [
        "si","ok","bien","algunas","algunos","alguna","algun","varias","varios",
        "muchas","muchos","pocas","pocos","alta","media","baja","hubo","positivo",
        "true","verdadero","presente","hecho","realizado","realizada"
    ]
    neg_words = [
        "no","ninguna","ninguno","cero","0","nada","false","falso","ausente","ningun"
    ]

    # Íconos ✓ ✔ ✅ (y similares)
    pos_icons_re = r"[✓✔✅]"

    # Regex de palabras con bordes
    pos_re = r"\b(" + "|".join(map(re.escape, pos_words)) + r")\b"
    neg_re = r"\b(" + "|".join(map(re.escape, neg_words)) + r")\b"

    has_pos_word  = s_norm.str.contains(pos_re, regex=True, na=False)
    has_neg_word  = s_norm.str.contains(neg_re, regex=True, na=False)
    has_pos_icon  = s_norm.str.contains(pos_icons_re, regex=True, na=False)

    # Regla textual: positivo si (palabra positiva O icono) y NO hay negación explícita
    flag_txt = ( (has_pos_word | has_pos_icon) & ~has_neg_word )

    # 3) Combinación: el número manda si existe; si no, usa texto.
    flag = flag_num | flag_txt

    return flag.astype(int)

def _compute_sesgo_amenaza(df: pd.DataFrame, alpha: float, lookback_days: int) -> Tuple[float, Dict[str,str]]:
    # 1) EMAs y ventana
    ans_full = _ema(_num_series(df, "ansiedad"), alpha)
    est_full = _ema(_num_series(df, "estres"),   alpha)

    ans = ans_full.tail(lookback_days)
    est = est_full.tail(lookback_days)

    ans01 = _scale_01(ans, 0, 10)
    est01 = _scale_01(est, 0, 10)

    # 2) Flags de "exposición" (robustos)
    #    a) interacciones significativas: usar flag diario si existe; si no, parsear texto
    if "interacciones_ok" in df.columns:
        inter_ok_full = df["interacciones_ok"].astype(int) > 0
    else:
        # usa el helper que parsea texto: "sí", "varias", "2", etc.
        inter_ok_full = (_interacciones_flag_series(df) > 0)

    #    b) eventos/etiquetas de exposición por texto (aliases)
    evt_txt_full = _str_series(df, ["eventos_estresores", "evento_estresor", "exposicion", "expo_grad"])
    evt_ok_full  = evt_txt_full.str.strip().ne("")

    #    c) minutos de exposición explícitos (si los registras)
    expo_min_ok_full = (_num0(df, ["exposicion_min", "expo_min", "exposicion_grad_min"]) > 0)

    exp_bin_full = (inter_ok_full | evt_ok_full | expo_min_ok_full).astype(int)

    # 3) Alinear exp_bin con la MISMA ventana de 'ans'
    exp_bin = exp_bin_full.reindex(ans.index).fillna(0).astype(int)

    # 4) Carga de amenaza (↑10–20% si hubo exposición ese día)
    carga = float(((ans01 * 0.6 + est01 * 0.4) * (1 + 0.2*exp_bin)).mean() * 100.0)

    # 5) Δ ansiedad post-exposición (día+1 − día0) dentro de la ventana
    #    Usa helper que promedia pares válidos sólo en días con exp_bin==1.
    d_anx = _delta_ansiedad_post_expo(ans, exp_bin)  # devuelve NaN si no hay pares → mostramos "s/d"

    metricas = {
        "Ansiedad EMA (0–10)": _safe_num(_last_valid(ans), sd="—"),
        "Estrés EMA (0–10)":   _safe_num(_last_valid(est), sd="—"),
        "Días con exposición": str(int(exp_bin.sum())),
        "Δ Ansiedad post-exposición (día+1 − día0)": _safe_num(d_anx, fmt="{:+.2f}", sd="s/d"),
    }
    return carga, metricas

# Pesos por defecto (prior)
W_PLACER = {"animo": 0.6, "autocuidado": 0.4}
W_ENERG  = {"activacion": 0.6, "claridad": 0.4}

def _blend_weights(prior: dict, learned: dict | None, n: int,
                   n_min: int = 14, n_max: int = 60) -> dict:
    if not learned:
        return prior
    # λ crece desde 0 (n<=n_min) hasta 1 (n>=n_max)
    lam = 0.0 if n <= n_min else (1.0 if n >= n_max else (n - n_min) / (n_max - n_min))
    out = {}
    for k in prior:
        lp = float(prior[k])
        ld = float(max(0.0, learned.get(k, lp)))  # sin pesos negativos
        out[k] = (1 - lam) * lp + lam * ld
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}

def _compute_anhedonia(
    df: pd.DataFrame,
    alpha: float = 0.30,
    lookback_days: int = 14,
    row_idx: int | None = None,
    min_ex_minutes: int = 10,
    min_med_minutes: int = 5,
    reward_cols_bounds: dict = None,   # {"animo": (0,10), "autocuidado": (0,10), "activacion": (0,10), "claridad": (0,10)}
    penalty_weight: float = 0.30,
) -> Tuple[float, Dict[str, str]]:

    # 0) Ventana anclada si se pide (asumiendo df ordenado por fecha asc)
    if row_idx is not None and "fecha" in df.columns and len(df) > 0:
        fechas = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
        ridx = max(0, min(int(row_idx), len(df)-1))
        end_day = fechas.iloc[ridx]
        start_day = end_day - pd.Timedelta(days=lookback_days-1)
        df = df[(fechas >= start_day) & (fechas <= end_day)].copy()

    # 1) EMA de componentes de recompensa/energía
    ani = _ema(_num_series(df, "animo"),       alpha).tail(lookback_days)
    aut = _ema(_num_series(df, "autocuidado"), alpha).tail(lookback_days)
    act = _ema(_num_series(df, "activacion"),  alpha).tail(lookback_days)
    cla = _ema(_num_series(df, "claridad"),    alpha).tail(lookback_days)

    # 2) Bounds (por defecto 0–10)
    rb = reward_cols_bounds or {"animo": (0,10), "autocuidado": (0,10), "activacion": (0,10), "claridad": (0,10)}
    placer  = (_scale_01(ani, *rb["animo"])*0.6 + _scale_01(aut, *rb["autocuidado"])*0.4)
    energia = (_scale_01(act, *rb["activacion"])*0.6 + _scale_01(cla, *rb["claridad"])*0.4)

    # 3) Adherencia micro-tareas (booleans por día) + robustez de aliases
    ej_ok  = (_num0(df, ["tiempo_ejercicio_min","tiempo_ejercicio"]) >= min_ex_minutes)
    med_ok = (_num0(df, ["meditacion_min","meditacion"])            >= min_med_minutes)
    int_ok = (_num0(df, ["interacciones_significativas","interacciones_calidad","interacciones"]) > 0)

    # Asegura ventana y calcula adherencia como proporción de días con ≥1 micro-tarea cumplida
    adh_frame = pd.concat([ej_ok, med_ok, int_ok], axis=1).tail(lookback_days)
    if len(adh_frame) == 0:
        adher = 0.0
    else:
        adher = float(adh_frame.any(axis=1).mean())

    # 4) Score (alto = más anhedonia)
    base = float(((placer + energia)/2).mean())
    if not np.isfinite(base):
        base = 0.0
    w = float(max(0.0, min(1.0, penalty_weight)))
    penal = (1.0 - adher) * w
    score = float(max(0.0, min(1.0, 1.0 - base + penal)) * 100.0)

    # 5) Métricas explicativas
    metricas = {
        "Placer (EMA 0–1)": _safe_mean_str(placer),
        "Energía (EMA 0–1)": _safe_mean_str(energia),
        "Adherencia micro-tareas (%)": f"{adher*100:.0f}%",
        "Ánimo EMA (0–10)": _safe_num(_last_valid(ani), sd="—"),
        "Activación EMA (0–10)": _safe_num(_last_valid(act), sd="—"),
    }
    return score, metricas



def _compute_rumiacion(
    df: pd.DataFrame,
    alpha: float,
    lookback_days: int,
    *,
    mitig_cap: float = 0.30,        # tope absoluto de mitigación (antes 0.35)
    mitig_frac_cap: float = 0.70,   # la meditación no puede “comerse” >70% de la base
    pantallas_hi: int = 180         # escala 0–pantallas_hi (puedes subir a 240 si sueles tener >180min)
) -> Tuple[float, Dict[str,str]]:
    # 1) Estado (EMA)
    cla = _ema(_num_series(df, "claridad"),      alpha).tail(lookback_days)
    ans = _ema(_num_series(df, "ansiedad"),      alpha).tail(lookback_days)
    irr = _ema(_num_series(df, "irritabilidad"), alpha).tail(lookback_days)

    # 2) Dosis/tiempo
    pant = _num0(df, ["tiempo_pantalla_noche_min","pantallas_tarde_min","pantallas_min_tarde","pantallas_min"]).tail(lookback_days)
    med  = _num0(df, ["meditacion_min","meditacion"]).tail(lookback_days)

    # 3) Escalado
    clar_low = 1.0 - _scale_01(cla, 0, 10)
    ans01    = _scale_01(ans, 0, 10)
    irr01    = _scale_01(irr, 0, 10) if not irr.dropna().empty else pd.Series([0.0]*len(clar_low), index=clar_low.index)
    pant01   = _scale_01(pant, 0, pantallas_hi)

    base_series = clar_low*0.45 + ans01*0.25 + irr01*0.15 + pant01*0.15
    base = float(np.nanmean(base_series.values)) if len(base_series) else 0.0
    if not np.isfinite(base): base = 0.0

    # 4) Mitigación limitada
    med_mean = float(np.nanmean(pd.to_numeric(med, errors="coerce").values)) if len(med) else np.nan
    mitig_raw = (med_mean/60.0) * mitig_cap if np.isfinite(med_mean) else 0.0
    mitig = min(mitig_cap, mitig_raw, base * mitig_frac_cap)  # 👈 límite relativo a la base

    # 5) Score
    score = float(np.clip(base - mitig, 0.0, 1.0) * 100.0)

    metricas = {
        "Claridad EMA (0–10)":          _safe_num(_last_valid(cla), sd="—"),
        "Ansiedad EMA (0–10)":          _safe_num(_last_valid(ans), sd="—"),
        "Irritabilidad EMA (0–10)":     _safe_num(_last_valid(irr), sd="—"),
        "Pantallas noche (min, media)": _safe_num(pant.mean(), fmt="{:.0f}", sd="—"),
        "Meditación (min, media)":      _safe_num(med.mean(),  fmt="{:.0f}", sd="—"),
        # si quieres ver el balance:
        # "Base (0–1)": _safe_num(base), "Mitigación aplicada (0–1)": _safe_num(mitig)
    }
    return score, metricas



def _compute_autonomica(df: pd.DataFrame, alpha: float, lookback_days: int) -> Tuple[float, Dict[str,str]]:
    # EMA de estado
    est = _ema(_num_series(df, "estres"),         alpha).tail(lookback_days)
    act = _ema(_num_series(df, "activacion"),     alpha).tail(lookback_days)
    sue = _ema(_num_series(df, "sueno_calidad"),  alpha).tail(lookback_days)

    # Horas desde cafeína / alcohol (aliases típicos)
    # DESPUÉS (bien: faltante = NaN, no penaliza)
    cafe_ultima = _num_series(df, ["horas_desde_cafe","h_desde_cafe","horas_ult_cafe","cafe_ultima_hora"]).tail(lookback_days)
    alco_ultima = _num_series(df, ["horas_desde_alcohol","h_desde_alcohol","horas_ult_alcohol","alcohol_ultima_hora"]).tail(lookback_days)


    # Dolor (EMA) y sol
    dolor = _ema(_num_series(df, ["dolor_fisico", "dolor"]), alpha).tail(lookback_days)
    sol   = _num_series(df, ["exposicion_sol_min", "exposicion_sol_manana_min"]).tail(lookback_days)

    # Hora de despertar y despertares (aliases)
    wake_txt = _str_series(df, ["hora_despertar"])
    hora_despertar = _hhmm_to_minutes_series(wake_txt).tail(lookback_days)
    despertares = _num_series(df, ["despertares_nocturnos"]).tail(lookback_days)

    # Escalas (0–1)
    est  = _ema(_num_series(df, "estres"),        alpha).tail(lookback_days)
    act  = _ema(_num_series(df, "activacion"),    alpha).tail(lookback_days)
    sue  = _ema(_num_series(df, "sueno_calidad"), alpha).tail(lookback_days)
    
    cafe_ultima = _num0(df, ["horas_desde_cafe","h_desde_cafe","horas_ult_cafe","cafe_ultima_hora"]).tail(lookback_days)
    alco_ultima = _num0(df, ["horas_desde_alcohol","h_desde_alcohol","horas_ult_alcohol","alcohol_ultima_hora"]).tail(lookback_days)
    dolor = _ema(_num_series(df, ["dolor_fisico","dolor"]), alpha).tail(lookback_days)
    sol   = _num0(df, ["exposicion_sol_min","exposicion_sol_manana_min"]).tail(lookback_days)
    desp  = _num0(df, ["despertares_nocturnos","DSPTes_nocturnos"]).tail(lookback_days)

    cafe01 = _scale_01(cafe_ultima, 0, 20) # 0–20 h desde la última cafeína
    est01 = _scale_01(est, 0, 10) 
    act01 = _scale_01(act, 0, 10) 
    sue01 = _scale_01(sue, 0, 10) # más alto = mejor 
    cafe01 = _scale_01(cafe_ultima, 0, 20) # 0–20 h desde la última cafeína 
    alco01 = _scale_01(alco_ultima, 0, 20) # 0–20 h desde alcohol 
    dolor01 = _scale_01(dolor, 0, 10) if not dolor.dropna().empty else pd.Series([0]*len(est), index=est.index) 
    sol01 = _scale_01(sol, 0, 60) # 0–60 min de sol 
    desp01 = _scale_01(despertares, 0, 5) # 0–5 despertares # variabilidad de despertar: desviación estándar en min (0–120 -> 0–1)

    # Variabilidad de despertar (SD en minutos → 0–1 en 0–120)
    sd_wake = float(hora_despertar.dropna().std()) if not hora_despertar.dropna().empty else np.nan
    wake_sd01 = _scale_01(pd.Series([sd_wake]*len(est), index=est.index), 0, 120) if not np.isnan(sd_wake) else pd.Series([0]*len(est), index=est.index)

    # Disautonomía (alto = peor)
    disauto = (
        est01*0.28 + act01*0.18 + (1 - sue01)*0.20 +
        (1 - cafe01)*0.08 + (1 - alco01)*0.08 +
        dolor01*0.06 + (1 - sol01)*0.06 +
        desp01*0.04 + wake_sd01*0.02
    ).mean()

    disauto_score = float(max(0.0, min(1.0, disauto)) * 100.0)

    metricas = {
        "Estrés EMA (0–10)":            f"{_last_valid(est):.2f}" if _last_valid(est) is not None else "—",
        "Activación EMA (0–10)":        f"{_last_valid(act):.2f}" if _last_valid(act) is not None else "—",
        "Sueño EMA (0–10)":             f"{_last_valid(sue):.2f}" if _last_valid(sue) is not None else "—",
        "h desde café (media)":         f"{float(cafe_ultima.mean()):.1f}"  if len(cafe_ultima) else "—",
        "h desde alcohol (media)":      f"{float(alco_ultima.mean()):.1f}"  if len(alco_ultima) else "—",
        "Dolor EMA (0–10)":             f"{_last_valid(dolor):.2f}" if _last_valid(dolor) is not None else "—",
        "Sol (min, media)":             f"{float(sol.mean()):.0f}" if len(sol) else "—",
        "Despertares nocturnos (media)":f"{float(despertares.mean()):.1f}" if len(despertares) else "—",
        "Hora despertar SD (min)":      "—" if np.isnan(sd_wake) else f"{sd_wake:.0f}",
    }
    return disauto_score, metricas

# ------------------------------- orquestador ---------------------------------

# dado df ya filtrado por 'correo' y ordenado por fecha ascendente
def _find_row_idx_for_day(df, target_day: str) -> int:
    fechas = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce").dt.date
    td = pd.to_datetime(target_day, format="%d-%m-%Y", errors="coerce").date()
    exact = np.where(fechas.values == td)[0]
    if exact.size: return int(exact[-1])
    leq = np.where((pd.notna(fechas.values)) & (fechas.values <= td))[0]
    return int(leq[-1]) if leq.size else len(df)-1


def _estado_from_score(score: float) -> str:
    if score < 33: return "verde"
    if score < 66: return "ambar"
    return "rojo"

def _confiabilidad(df: pd.DataFrame, lookback_days: int) -> str:
    claves = ["animo","activacion","claridad","estres","ansiedad","sueno_calidad"]
    disponibles = sum(int(df.get(c, pd.Series(dtype=float)).dropna().shape[0] > 0) for c in claves)
    dias = int(df.shape[0])
    if dias < 7 or disponibles < 4: return "baja"
    if dias < 10 or disponibles < 5: return "media"
    return "alta"

def compute_mini_guia(
    df: pd.DataFrame,
    alpha: float = 0.30,
    lookback_days: int = 14,
    row_idx: int | None = None
) -> List[DianaCard]:
    """
    Calcula las 4 tarjetas (Amenaza, Anhedonia, Rumiación, Autonómica) sobre
    una ventana ANCLADA en 'row_idx' de largo 'lookback_days'.

    - Si 'row_idx' es None: usa el último día (comportamiento previo).
    - Si hay 'fecha' (y opcionalmente 'hora'), usa ventana por fecha; si no, por índice.
    - Todas las submétricas usan ese recorte, por lo que los EMA y medias se
      calculan sobre la historia relevante a ese corte.
    """
    # 0) Orden base para consistencia
    if "fecha" in df.columns:
        try:
            df = df.copy()
            df["__ts"] = pd.to_datetime(
                df["fecha"].astype(str) + " " + df.get("hora", "00:00").astype(str),
                dayfirst=True, errors="coerce"
            )
            df = df.sort_values("__ts").drop(columns=["__ts"])
        except Exception:
            df = df.copy()
    else:
        df = df.copy()

    # 1) Determina el índice de corte (último si no pasan row_idx)
    ridx = _clamp_row_idx(df, row_idx)

    # 2) Ventana anclada al día elegido
    df_win, _meta = _window_by_days(df, ridx, lookback_days)

    # 3) Confiabilidad calculada sobre la ventana efectiva
    conf = _confiabilidad(df_win, lookback_days)

    # 4) Cálculos de dianas sobre la ventana (cada función ya hace sus EMA y tail)
    s1, m1 = _compute_sesgo_amenaza(df_win, alpha, lookback_days)
    s2, m2 = _compute_anhedonia(df_win, alpha, lookback_days)
    s3, m3 = _compute_rumiacion(df_win, alpha, lookback_days)
    s4, m4 = _compute_autonomica(df_win, alpha, lookback_days)

    # 5) Construcción de tarjetas
    cards = [
        DianaCard(
            key="amenaza",
            titulo="Sesgo de amenaza (amígdala)",
            subtitulo="Reappraisal breve + exposición graduada",
            emoji="🛡️",
            protocolo="1) Nombra la amenaza (pensamiento ≠ peligro). 2) Reencuadre en 1–2 frases. "
                      "3) Exposición graduada 10–30 min con respiración suave. Registrar antes/después.",
            medir="Ansiedad anticipatoria (EMA), Estrés (EMA), días con exposición, Δ Ansiedad post-exposición (día+1 − día0).",
            score=s1, estado=_estado_from_score(s1), confiabilidad=conf, metricas=m1,
        ),
        DianaCard(
            key="anhedonia",
            titulo="Anhedonia (estriado/recompensa)",
            subtitulo="Activación conductual + micro-recompensas",
            emoji="✨",
            protocolo="1) 1–3 micro-tareas (≤5 min) con recompensa inmediata simbólica. "
                      "2) Cadena diaria (no romper racha). 3) Escalar a 15–20 min según energía.",
            medir="Placer (Ánimo+Autocuidado), Energía (Activación+Claridad), Adherencia micro-tareas (%).",
            score=s2, estado=_estado_from_score(s2), confiabilidad=conf, metricas=m2,
        ),
        DianaCard(
            key="rumiacion",
            titulo="Rumiación (DMN↑ / control top-down↓)",
            subtitulo="MBCT / atención al cuerpo",
            emoji="🧘",
            protocolo="1) 3×/día: 3 min respiración + escaneo. 2) Ancla sensorial en pies/manos. "
                      "3) Etiqueta 'pensando' y vuelve al ancla.",
            medir="Claridad (EMA), Ansiedad (EMA), Irritabilidad (EMA), Pantallas noche (min), Meditación (min).",
            score=s3, estado=_estado_from_score(s3), confiabilidad=conf, metricas=m3,
        ),
        DianaCard(
            key="autonomica",
            titulo="Disregulación autonómica (HRV baja)",
            subtitulo="Respiración 5–6 rpm + higiene del sueño",
            emoji="🌬️",
            protocolo="1) 2–3×/día: respiración 4–6/6–8. 2) Sol 5–20 min AM. 3) Cortar cafeína ≥8 h y alcohol ≥4–6 h antes de dormir. "
                      "4) Descarga corporal suave 10 min.",
            medir="Estrés/Activación/Sueño (EMA), h desde café/alcohol, dolor, minutos de sol, despertares y regularidad de despertar.",
            score=s4, estado=_estado_from_score(s4), confiabilidad=conf, metricas=m4,
        ),
    ]
    return cards


# ------------------------------- render HTML ---------------------------------

CSS = """
.mg-wrap { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0,1fr)); }
@media (max-width: 900px){ .mg-wrap { grid-template-columns: 1fr; } }
.mg-card { border-radius: 14px; padding: 14px 16px; border: 1px solid #e8e8ef; background: #0b1020; box-shadow: 0 2px 6px rgba(10,10,30,0.04); }
.mg-head { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.mg-emoji { font-size: 20px; }
.mg-title { font-weight: 700; font-size: 16px; }
.mg-sub { color: #4b5563; font-size: 12px; margin-bottom: 10px; }
.mg-pill { font-size: 11px; padding: 2px 8px; border-radius: 999px; display:inline-block; margin-right:6px; border:1px solid rgba(0,0,0,0.06); }
.mg-pill.verde { background: #ecfdf5; color: #065f46; }
.mg-pill.ambar { background: #fffbeb; color: #92400e; }
.mg-pill.rojo { background: #fef2f2; color: #991b1b; }
.mg-pill.gray { background:#f3f4f6; color:#374151; }
.mg-proto { font-size: 12px; line-height: 1.4; margin: 8px 0 10px; }
.mg-medir { font-size: 12px; color:#4b5563; margin-bottom: 8px; }
.mg-metrics { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:6px; font-size:12px; }
.mg-metric { background:#0b1020; border:1px solid #eef2f7; border-radius:10px; padding:6px 8px;font-size:10px; }
.mg-foot { font-size: 11px; color:#6b7280; margin-top:8px; }
.mg-score { font-weight:700; }
"""

CSS2 = """
.mg-wrap2 { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0,1fr)); }
@media (max-width: 900px){ .mg-wrap2 { grid-template-columns: 1fr; } }
.mg-card2 { border-radius: 14px; padding: 14px 16px; border: 1px solid #e8e8ef; background: #0b1020; box-shadow: 0 2px 6px rgba(10,10,30,0.04); }
.mg-head2 { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.mg-emoji2 { font-size: 20px; }
.mg-title2 { font-weight: 700; font-size: 16px; }
.mg-foot2 { font-size: 11px; color:#6b7280; margin-top:8px; }
.mg-score2 { font-weight:700; }
"""

def _card_to_html(c: DianaCard) -> str:
    metr_html = "".join([f'<div class="mg-metric"><strong class="mg-metric">{html.escape(k)}:</strong> {html.escape(v)}</div>' for k,v in c.metricas.items()])
    return f"""
    <article class="mg-card">
      <div class="mg-head"><span class="mg-emoji">{html.escape(c.emoji)}</span><span class="mg-title">{html.escape(c.titulo)}</span></div>
      <div class="mg-sub">{html.escape(c.subtitulo)}</div>
      <div>
        <span class="mg-pill {c.estado}">Estado: {c.estado.upper()}</span>
        <span class="mg-pill gray">Score: <span class="mg-score">{c.score:.0f}</span>/100</span>
        <span class="mg-pill gray">Confianza: {html.escape(c.confiabilidad)}</span>
      </div>
      <div class="mg-proto">{html.escape(c.protocolo)}</div>
      <div class="mg-medir"><em>Medir:</em> {html.escape(c.medir)}</div>
      <div class="mg-metrics">{metr_html}</div>
    </article>
    """

# --- 🎨 CSS bonito y con scope a #mini-guia-neuro ----------------------------
CSS_MG_BONITO = """
#mini-guia-neuro { font: inherit; color: var(--mg-txt, #111); }
#mini-guia-neuro .mg-wrap{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:14px; align-items:stretch
}
#mini-guia-neuro .mg-card{
  background:var(--mg-surface,#fff); border:1px solid rgba(0,0,0,.07);
  border-radius:16px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.06);
  display:flex; flex-direction:column; gap:12px
}
@media (prefers-color-scheme: dark){
  #mini-guia-neuro { --mg-surface:#141414; --mg-txt:#eee }
  #mini-guia-neuro .mg-card{ border-color: rgba(255,255,255,.08) }
}
#mini-guia-neuro .mg-head{ display:flex; gap:12px; align-items:center }
#mini-guia-neuro .mg-emoji{ font-size:26px; line-height:1 }
#mini-guia-neuro .mg-tt{ flex:1 }
#mini-guia-neuro .mg-title{ margin:0; font-size:1.05rem; font-weight:700 }
#mini-guia-neuro .mg-sub{ margin:1px 0 0; opacity:.8; font-size:.93rem }
#mini-guia-neuro .mg-badges{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px }
#mini-guia-neuro .chip{
  --c: 0,0,0; display:inline-flex; align-items:center; gap:6px;
  border:1px solid rgba(var(--c),.25); color:rgb(var(--c));
  background:rgba(var(--c), .06); padding:4px 8px; border-radius:999px;
  font-size:.82rem; font-weight:600
}
#mini-guia-neuro .chip.ok{   --c: 34, 160,  99 }
#mini-guia-neuro .chip.warn{ --c: 227,160,  62 }
#mini-guia-neuro .chip.bad{  --c: 225, 89,  92 }
#mini-guia-neuro .chip.conf{ --c: 102,119,136 }
#mini-guia-neuro .mg-core{ display:flex; align-items:center; gap:14px }
#mini-guia-neuro .ring{ width:88px; height:88px; display:inline-block }
#mini-guia-neuro .ring .lbl{ font-size:12px; opacity:.75 }
#mini-guia-neuro .ring .num{ font-weight:800; font-size:20px }
#mini-guia-neuro .mg-core-rt{ display:flex; flex-direction:column; gap:8px; flex:1 }
#mini-guia-neuro .mg-metrics{ display:grid; grid-template-columns:1fr auto; row-gap:6px; column-gap:12px }
#mini-guia-neuro .mg-metric{ font-size: 10px }
#mini-guia-neuro .metric-k{ opacity:.8 }
#mini-guia-neuro .metric-v{ font-weight:700 }
#mini-guia-neuro .mg-proto{ margin-top:4px; font-size:.92rem; opacity:.9 }
#mini-guia-neuro .mg-med{ font-size:.88rem; opacity:.85 }
#mini-guia-neuro .mg-empty{
  padding:14px; border:1px dashed rgba(0,0,0,.2); border-radius:12px; opacity:.8
}
/* SVG ring colors */
#mini-guia-neuro .ring.ok   .prog{ stroke: rgb(34,160,99) }
#mini-guia-neuro .ring.warn .prog{ stroke: rgb(227,160,62) }
#mini-guia-neuro .ring.bad  .prog{ stroke: rgb(225,89,92) }
#mini-guia-neuro .ring .bg{ stroke: rgba(0,0,0,.12) }
"""

# --- 🧩 helpers de vista -----------------------------------------------------
import math

def _estado_class(estado:str) -> str:
    e = (estado or "").lower()
    if "ver" in e:  return "ok"
    if "ám" in e or "amb" in e: return "warn"
    if "ama" in e: return "warn"
    if "roj" in e or "alto" in e or "crítico" in e: return "bad"
    return "warn"  # neutro→ámbar

def _score_ring_html(score: float, estado: str) -> str:
    # ring SVG 88x88, r=36 → circ = 226.19
    pct = max(0.0, min(100.0, float(score)))
    r = 36.0
    circ = 2*math.pi*r
    off = (1 - pct/100.0) * circ
    tone = _estado_class(estado)
    return f"""
<svg viewBox="0 0 100 100" class="ring {tone}" role="img" aria-label="Score {pct:.0f}">
  <circle class="bg"   cx="50" cy="50" r="{r}" fill="none" stroke-width="10"/>
  <circle class="prog" cx="50" cy="50" r="{r}" fill="none" stroke-width="10"
          stroke-dasharray="{circ:.2f}" stroke-dashoffset="{off:.2f}" stroke-linecap="round"
          transform="rotate(-90 50 50)"/>
  <foreignObject x="20" y="28" width="60" height="44">
    <div xmlns="http://www.w3.org/1999/xhtml" style="display:flex;flex-direction:column;align-items:center;justify-content:center">
      <div class="lbl">score</div>
      <div class="num">{pct:.0f}</div>
    </div>
  </foreignObject>
</svg>""".strip()

def _chip(texto:str, cls:str) -> str:
    return f'<span class="chip {cls}">{texto}</span>'

# --- 🔖 tarjeta bonita -------------------------------------------------------
def _card_to_html(c) -> str:
    """
    Espera atributos:
      c.titulo, c.subtitulo, c.emoji, c.score (0-100), c.estado ("verde/ámbar/rojo"),
      c.confiabilidad ("baja/media/alta"), c.metricas (dict), c.protocolo, c.medir
    """
    estado_cls = _estado_class(getattr(c, "estado", ""))
    ring = _score_ring_html(getattr(c, "score", 0), getattr(c, "estado", ""))

    # chips: estado y confiabilidad
    chip_estado = _chip(f"Estado: {getattr(c,'estado','—').capitalize()}", estado_cls)
    chip_conf   = _chip(f"Confianza: {getattr(c,'confiabilidad','—')}", "conf")

    # métricas (pares clave-valor)
    mets = getattr(c, "metricas", {}) or {}
    # respeta el orden de inserción (Py3.7+)
    metrics_rows = []
    for k,v in mets.items():
        metrics_rows.append(f'<div class="metric-k">{k}</div><div class="metric-v">{v}</div>')
    metrics_html = "".join(metrics_rows) if metrics_rows else '<div class="metric-k">Sin métricas</div><div class="metric-v">—</div>'

    # protocolo y qué medir
    proto = getattr(c, "protocolo", "") or ""
    medir = getattr(c, "medir", "") or ""

    return f"""
<article class="mg-card" data-key="{getattr(c,'key','')}">
  <header class="mg-head">
    <div class="mg-emoji">{getattr(c, 'emoji', '•')}</div>
    <div class="mg-tt">
      <h3 class="mg-title">{getattr(c, 'titulo', '—')}</h3>
      <div class="mg-sub">{getattr(c, 'subtitulo', '')}</div>
      <div class="mg-badges">{chip_estado} {chip_conf}</div>
    </div>
  </header>

  <div class="mg-core">
    {ring}
    <div class="mg-core-rt">
      <div class="mg-metrics">{metrics_html}</div>
      <div class="mg-med"><strong>Medir:</strong> {medir}</div>
    </div>
  </div>

  <div class="mg-proto"><strong>Micro-protocolo:</strong> {proto}</div>
</article>
""".strip()



def render_mini_guia_section(
    df: pd.DataFrame,
    alpha: float = 0.30,
    lookback_days: int = 14,
    row_idx: int = -1,
    add_style: bool = True
) -> str:
    """
    Renderiza la mini-guía usando una ventana de 'lookback_days' días ANCLADA en row_idx.
    - df: se asume ya filtrado por 'correo' y ordenado por fecha ascendente.
    - row_idx: índice (0-based) del día del reporte.
    - alpha: EMA α para el cálculo interno de señales.
    """
    ridx = _clamp_row_idx(df, row_idx)
    if ridx < 0:
        return ""  # sin datos

    # 1) Seleccionar ventana anclada al día elegido
    df_win, meta = _window_by_days(df, row_idx, lookback_days)

    # 2) Tarjetas sobre ESA ventana (usa el último de df_win como “día de corte”)
    cards = compute_mini_guia(
        df_win,
        alpha=alpha,
        lookback_days=lookback_days,
        row_idx=None    #  que use el último de df_win
    )
    cards_html = "".join(_card_to_html(c) for c in cards)
    

    style_parts = []
    if add_style and 'CSS' in globals() and CSS:
        style_parts.append(str(CSS))  
    style_parts.append(CSS_MG_BONITO)  
    style_block = f"<style>{'\n'.join(style_parts)}</style>"
    style_block = f"<style>{'\n'.join(style_parts)}</style>"
    
    # 4) Pie
    corte_txt  = f" • corte: {meta['end'].strftime('%d-%m-%Y')}" if meta.get("end") else ""
    warmup_txt = " • confianza baja (historia <7 días)" if meta.get("warmup") else ""
    foot = f"Notas: EMA α={alpha:.2f}, ventana={lookback_days} días{corte_txt}. Puntajes altos = mayor prioridad{warmup_txt}."

    # 5) HTML
    return f"""
    <section id="mini-guia-neuro" aria-labelledby="mini-guia-title">
    {style_block}
    <h2 id="mini-guia-title">Mini-guía de intervención por diana neuro-conductual</h2>
    <div class="mg-wrap">
        {cards_html if cards_html else '<div class="mg-empty">Sin señal suficiente en la ventana seleccionada.</div>'}
    </div>
    <div class="mg-foot">{foot}</div>
    </section>
    """.strip()