
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from reporte_python.bdp_schema import DEFAULT_COLUMNS
from reporte_python.bdp_messages import stack_human_messages, kind_for_category

# Variables numéricas adicionales que queremos promediar por día (si existen en DEFAULT_COLUMNS y en el DF)
EXTRA_NUM_VARS = [
    "autocuidado", "alimentacion", "movimiento", "dolor_fisico",
    "ansiedad", "irritabilidad", "meditacion_min", "exposicion_sol_min",
    "agua_litros", "cafe_cucharaditas", "alcohol_ud", "glicemia", "sueno_horas", "siesta_min"
]

# ----------------- Helpers -----------------
def _tidy_note_text(s: str) -> str:
    import re as _re
    if s is None:
        return ''
    s = str(s)
    # colapsa espacios múltiples
    s = _re.sub(r'\s+', ' ', s).strip()
    # normaliza comas (espacio después) y elimina comas repetidas
    s = _re.sub(r'\s*,\s*', ', ', s)
    s = _re.sub(r',\s*,+', ', ', s)
    # elimina coma/punto/; final sobrantes
    s = _re.sub(r'[\s,;:.]+$', '', s)
    return s

def _is_blank_or_nan(x):
    if x is None: return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan" or s.lower() == "none"

def _clean_notes_list(items):
    # Remove empties/NaN and deduplicate while preserving order
    seen = set()
    out = []
    for it in items:
        # it is expected like "[hh:mm] text"
        # extract the text part after the bracket to validate
        try:
            txt = it.split("]", 1)[1].strip()
        except Exception:
            txt = it
        if _is_blank_or_nan(txt):
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _sleep_score_from_hours(h):
    """Mapea horas de sueño a escala 0–10 (óptimo ~8h)."""
    try:
        x = float(h)
    except Exception:
        return float("nan")
    if x < 0: x = 0.0
    if x > 14: x = 14.0
    # Triangular con pico en 8h: 8→10, 7/9→8, 6/10→6, 5/11→4, 4/12→2, <=3/>=13→0
    score = max(0.0, 10.0 - abs(x - 8.0) * 2.0)
    return score

def _derive_sleep_series(day_df: pd.DataFrame, columns: Dict[str,str]) -> pd.Series:
    """Devuelve una Serie 0–10 para 'sueno' usando prioridad:
    1) sueno (si existe y tiene datos)
    2) sueno_calidad (0–10)
    3) horas_sueno (mapeado a 0–10)
    """
    # 1) sueno directo
    col_su = columns.get("sueno")
    if col_su and col_su in day_df.columns:
        s = pd.to_numeric(day_df[col_su], errors="coerce")
        if s.notna().any():
            return s

    # 2) sueno_calidad
    col_sc = columns.get("sueno_calidad")
    if col_sc and col_sc in day_df.columns:
        s2 = pd.to_numeric(day_df[col_sc], errors="coerce")
        if s2.notna().any():
            return s2

    # 3) horas_sueno (o sueno_horas por compatibilidad)
    col_h = columns.get("horas_sueno") or columns.get("sueno_horas")
    if col_h and col_h in day_df.columns:
        s3 = pd.to_numeric(day_df[col_h], errors="coerce")
        if s3.notna().any():
            return s3.apply(_sleep_score_from_hours)

    # Nada disponible
    return pd.Series([float("nan")] * len(day_df), index=day_df.index)


def ensure_str_list(x: Any) -> List[str]:
    """Convierte una celda en lista de strings (split por comas/; | saltos de línea)."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    # Normaliza separadores
    for sep in ["\n", "|", ";"]:
        s = s.replace(sep, ",")
    parts = [p.strip(" [](){}\"'").strip() for p in s.split(",")]
    return [p for p in parts if p]

def _parse_hora_to_hhmm(raw: object) -> str:
    """Convierte distintos formatos de hora a 'HH:MM'. Devuelve '--:--' si no se puede."""
    import re
    if raw is None:
        return "--:--"
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return "--:--"
    # HH:MM[:SS]
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::\d{1,2})?$", s)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2))
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"
    # HH.MM
    m = re.match(r"^(\d{1,2})[.,](\d{1,2})$", s)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2))
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"
    # 0930 o 930
    m = re.match(r"^(\d{3,4})$", s)
    if m:
        num = int(m.group(1)); h = num // 100; mnt = num % 100
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"
    # 9h05
    m = re.match(r"^(\d{1,2})\s*h\s*(\d{1,2})$", s, flags=re.IGNORECASE)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2))
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"
    # Pandas fallback
    try:
        t = pd.to_datetime(s, errors="coerce", format="%H:%M").time()
        if t:
            return f"{t.hour:02d}:{t.minute:02d}"
    except Exception:
        pass
    try:
        t = pd.to_datetime(s, errors="coerce").time()
        if t:
            return f"{t.hour:02d}:{t.minute:02d}"
    except Exception:
        pass
    return "--:--"

def _list_notes_and_stressors(day_df: pd.DataFrame, columns: Dict[str,str]) -> Tuple[List[str], List[str]]:
    notas_col   = columns.get("notas", "notas")
    estres_col  = columns.get("estresores", "estresores")
    ev_col      = columns.get("eventos_estresores", None)
    hora_col    = columns.get("hora", None)
    fecha_col   = columns.get("fecha", "fecha")

    # --- Notas ---
    notas: List[str] = []
    if not day_df.empty and notas_col in day_df.columns:
        for _, row in day_df.iterrows():
            raw = row.get(notas_col, "")
            # descarta vacíos/NaN
            if _is_blank_or_nan(raw):
                continue
            txt = _tidy_note_text(str(raw).strip())

            # hora a [HH:MM]
            hhmm = "--:--"
            if hora_col and hora_col in day_df.columns:
                hhmm = _parse_hora_to_hhmm(row.get(hora_col))
            if hhmm == "--:--" and fecha_col in day_df.columns:
                ts = pd.to_datetime(row.get(fecha_col), errors="coerce", dayfirst=True)
                if pd.notna(ts) and not (ts.hour == 0 and ts.minute == 0):
                    hhmm = f"{ts.hour:02d}:{ts.minute:02d}"

            notas.append(f"[{hhmm}] {txt}")

    # limpia vacías/NaN y deduplica preservando orden
    notas = _clean_notes_list(notas)

    # --- Estresores (incluye eventos_estresores si está) ---
    estresores_raw: List[str] = []
    for col in [estres_col, ev_col]:
        if col and col in day_df.columns:
            for x in day_df[col].tolist():
                estresores_raw.extend(ensure_str_list(x))

    # dedupe estresores, preservando orden y filtrando vacíos/NaN
    seen = set()
    estresores: List[str] = []
    for e in estresores_raw:
        if _is_blank_or_nan(e):
            continue
        if e in seen:
            continue
        seen.add(e)
        estresores.append(e)

    return notas, estresores


def _normalize01(v):
    """Lleva métricas a rango 0..1 si vienen en 0..10 o 0..100; clampa extremos."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan
    try:
        x = float(v)
    except Exception:
        return np.nan
    if x > 1.5 and x <= 10.5:
        x = x / 10.0
    elif x > 10.5:
        x = x / 100.0
    return float(max(0.0, min(1.0, x)))

def _categorize_day(resumen: Dict[str, float]) -> int:
    """Devuelve 0,1,2,3 para amabilidad del día (muy difícil -> muy positivo). Heurística suave."""
    an = _normalize01(resumen.get("animo"))
    es = _normalize01(resumen.get("estres"))
    su = _normalize01(resumen.get("sueno"))
    if np.isnan(an) and np.isnan(es):
        return 2
    # bueno
    if (not np.isnan(an) and an >= 0.7) and (np.isnan(es) or es <= 0.4):
        return 3
    # difícil
    if (not np.isnan(an) and an <= 0.35) or (not np.isnan(es) and es >= 0.7) or (not np.isnan(su) and su <= 0.35):
        return 1
    return 2

# ----------------- Core -----------------

def build_daily_cards(df: pd.DataFrame, columns: Dict[str, str], fill_missing_days: bool = True) -> List[Dict[str, Any]]:
    """Agrupa el DF por día (fecha), calcula promedios y construye cards."""
    if columns.get("fecha") not in df.columns:
        raise ValueError("La columna de fecha especificada en DEFAULT_COLUMNS no está en el DataFrame.")

    # Asegura tipo datetime con dayfirst
    if not np.issubdtype(df[columns["fecha"]].dtype, np.datetime64):
        df = df.copy()
        df[columns["fecha"]] = pd.to_datetime(df[columns["fecha"]], errors="coerce", dayfirst=True)

    # Agrupa por fecha normalizada (día)
    df["_day"] = df[columns["fecha"]].dt.normalize()

    # Genera lista de días incluyendo faltantes
    if fill_missing_days and df["_day"].notna().any():
        day_index = pd.date_range(start=df["_day"].min(), end=df["_day"].max(), freq="D")
    else:
        day_index = pd.DatetimeIndex(sorted(df["_day"].dropna().unique()))

    cards: List[Dict[str, Any]] = []
    prev_real_summary: Dict[str, float] | None = None

    # Columnas mapeadas
    area_keys = ["animo","activacion","conexion","proposito","claridad","estres","sueno"]
    mapped_cols = {k: columns.get(k) for k in area_keys if columns.get(k) in df.columns}

    for day in day_index:
        # Slice del día
        day_df = df[df["_day"] == day].copy()
        registros = int(len(day_df))

        # Resumen de áreas: promedios (numéricos)
        resumen: Dict[str, float] = {}
        for k, col in mapped_cols.items():
            if col in day_df.columns:
                s = pd.to_numeric(day_df[col], errors="coerce")
                if s.notna().any():
                    if k == "estres":
                        resumen[k] = float(s.median(skipna=True))
                    else:
                        resumen[k] = float(s.mean(skipna=True))

        # Asegurar que "sueno" exista (derivado si falta o NaN)
        if "sueno" not in resumen or pd.isna(resumen.get("sueno")):
            su_series = _derive_sleep_series(day_df, columns)
            if su_series.notna().any():
                resumen["sueno"] = float(su_series.mean(skipna=True))

        # Extras
        for key in EXTRA_NUM_VARS:
            col = columns.get(key)
            if col and col in day_df.columns:
                s = pd.to_numeric(day_df[col], errors="coerce")
                if s.notna().any():
                    resumen[key] = float(s.mean(skipna=True))

        # Notas y estresores
        notas, estresores = _list_notes_and_stressors(day_df, columns)

        # Tendencias (vs día anterior real)
        area_trends: Dict[str, str] = {}
        eps = 0.02
        for k in ["animo","activacion","sueno","conexion","proposito","claridad","estres"]:
            v = resumen.get(k, np.nan)
            pv = prev_real_summary.get(k, np.nan) if prev_real_summary else np.nan
            if np.isnan(v) or np.isnan(pv):
                t = "flat"
            else:
                dv = v - pv
                if abs(dv) < eps:
                    t = "flat"
                else:
                    t = "up" if dv > 0 else "down"
            area_trends[k] = t

        mensajes_hum = stack_human_messages(area_trends, resumen)

        # Interpretación del día (amable)
        cat = _categorize_day(resumen)
        interpretacion = kind_for_category(cat)

        card = {
            "fecha": day,
            "registros": registros,
            "resumen_areas": resumen,
            "mensajes_humanos": mensajes_hum,
            "notas": notas,
            "estresores": estresores if registros >= 2 else [],
            "interpretacion_dia": interpretacion,
            "faltante": registros == 0
        }
        cards.append(card)

        # Actualiza prev_real_summary sólo si el día tiene registros
        if registros > 0:
            prev_real_summary = resumen

    return cards
