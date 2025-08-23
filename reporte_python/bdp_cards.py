
from __future__ import annotations
from typing import Dict, List
import numpy as np

from reporte_python.bdp_schema import DayCard, DEFAULT_COLUMNS, AREAS, ensure_str_list, coerce_date, EXTRA_NUM_VARS
from reporte_python.bdp_trends import compute_daily_areas, compute_trends, compute_composite_and_category
from reporte_python.bdp_messages import stack_human_messages, kind_for_category

import re
import pandas as pd

def _parse_hora_to_hhmm(raw: object) -> str:
    """Convierte distintos formatos de hora a 'HH:MM'. Devuelve '--:--' si no se puede."""
    if raw is None:
        return "--:--"
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return "--:--"

    # Formatos comunes: "9", "09", "9:5", "09:05", "09:05:00", "9.5", "0930", "9h05"
    # 1) HH:MM[:SS]
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::\d{1,2})?$", s)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2))
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"

    # 2) HH.MM
    m = re.match(r"^(\d{1,2})[.,](\d{1,2})$", s)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2))
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"

    # 3) "0930" o "930" => HHMM
    m = re.match(r"^(\d{3,4})$", s)
    if m:
        num = int(m.group(1))
        h = num // 100
        mnt = num % 100
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"

    # 4) "9h05" o "9 h 05"
    m = re.match(r"^(\d{1,2})\s*h\s*(\d{1,2})$", s, flags=re.IGNORECASE)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2))
        if 0 <= h < 24 and 0 <= mnt < 60:
            return f"{h:02d}:{mnt:02d}"

    # 5) Dejar que pandas intente interpretar (hora suelta)
    try:
        t = pd.to_datetime(s, errors="coerce", format="%H:%M").time()
        if t:
            return f"{t.hour:02d}:{t.minute:02d}"
    except Exception:
        pass
    try:
        # sin format fijo, por si viene "HH:MM:SS"
        t = pd.to_datetime(s, errors="coerce").time()
        if t:
            return f"{t.hour:02d}:{t.minute:02d}"
    except Exception:
        pass

    return "--:--"

def _list_notes_and_stressors(day_df: pd.DataFrame, columns: Dict[str, str]) -> (List[str], List[str]):
    notas_col = columns.get("notas", "notas")
    estres_col = columns.get("estresores", "estresores")
    fecha_col = columns.get("fecha", "fecha")
    hora_col = columns.get("hora", None)
    sig_col = columns.get("interacciones_significativas", None)
    ev_col = columns.get("eventos_estresores", None)

    notas: List[str] = []
    estresores: List[str] = []

    if not day_df.empty and notas_col in day_df.columns:
        for _, row in day_df.iterrows():
            note = str(row.get(notas_col, "")).strip()
            if not note:
                continue

            hhmm = "--:--"
            # 1) si hay columna hora, úsala con parser robusto
            if hora_col and hora_col in day_df.columns:
                hhmm = _parse_hora_to_hhmm(row.get(hora_col))

            # 2) si sigue sin hora, intenta extraer de 'fecha' (por si trae timestamp con hora)
            if hhmm == "--:--" and fecha_col in day_df.columns:
                ts = pd.to_datetime(row.get(fecha_col), errors="coerce", dayfirst=True)
                if pd.notna(ts) and not (ts.hour == 0 and ts.minute == 0):
                    hhmm = f"{ts.hour:02d}:{ts.minute:02d}"

            notas.append(f"[{hhmm}] {note}")

    # Interacciones significativas como notas (si existen)
    if sig_col and sig_col in day_df.columns:
        for _, row in day_df.iterrows():
            txt = str(row.get(sig_col, "")).strip()
            if txt:
                hhmm = "--:--"
                if hora_col and hora_col in day_df.columns:
                    hhmm = _parse_hora_to_hhmm(row.get(hora_col))
                if hhmm == "--:--" and fecha_col in day_df.columns:
                    ts = pd.to_datetime(row.get(fecha_col), errors="coerce", dayfirst=True)
                    if pd.notna(ts) and not (ts.hour == 0 and ts.minute == 0):
                        hhmm = f"{ts.hour:02d}:{ts.minute:02d}"
                notas.append(f"[{hhmm}] {txt}")


    # Limpieza: remover "nan"/vacías y deduplicar preservando orden
    clean_notas = []
    seen = set()
    for n in notas:
        s = str(n).strip()
        if not s or s.lower().endswith(" nan") or s.lower() == "nan":
            continue
        if s not in seen:
            seen.add(s)
            clean_notas.append(s)
    notas = clean_notas

    # Estresores: normaliza separadores y dedup
    norm_est = []
    seen_e = set()
    for e in estresores:
        if not e: continue
        parts = re.split(r"[;,|]", str(e))
        for p in parts:
            t = p.strip()
            if not t: continue
            low = t.lower()
            if low not in seen_e:
                seen_e.add(low)
                # capitaliza una vez
                norm_est.append(t[0].upper()+t[1:] if len(t)>1 else t.upper())
    estresores = norm_est

    # Estresores clásicos + eventos_estresores
    if estres_col in day_df.columns:
        for x in day_df[estres_col].tolist():
            estresores.extend(ensure_str_list(x))
    if ev_col and ev_col in day_df.columns:
        for x in day_df[ev_col].tolist():
            estresores.extend(ensure_str_list(x))

    return notas, estresores


def _area_trend_labels(row: pd.Series, area_cols: List[str]) -> Dict[str, str]:
    out = {}
    for c in area_cols:
        tcol = c + "_trend"
        if tcol in row and isinstance(row[tcol], str):
            if row[tcol] in ("up","down","flat"):
                # Use original area key names for labels
                out[c] = row[tcol]
    return out

def _compare_prev(row: pd.Series, prev: pd.Series, area_cols: List[str]) -> str:
    if prev is None:
        return "Sin referencia previa."
    diffs = []
    for c in area_cols:
        if c in row and c in prev and pd.notna(row[c]) and pd.notna(prev[c]):
            delta = row[c] - prev[c]
            if abs(delta) < 0.05:
                continue
            arrow = "↑" if delta > 0 else "↓"
            diffs.append(f"{c}: {arrow}{abs(delta):.2f}")
    return "Cambios vs. día anterior: " + (", ".join(diffs) if diffs else "sin cambios relevantes.")

def build_daily_cards(df: pd.DataFrame, columns: Dict[str,str] = None, fill_missing_days: bool = True) -> List[DayCard]:
    columns = columns or DEFAULT_COLUMNS
    dcol = columns.get("fecha","fecha")

    keep = set(columns.values())
    keep.update([dcol])
    df = df[[c for c in df.columns if c in keep]].copy()

    df[dcol] = coerce_date(df[dcol])
    df["__day__"] = df[dcol].dt.normalize()

    daily = compute_daily_areas(df, columns)
    daily = compute_trends(daily, columns)

    day_groups = dict(tuple(df.groupby("__day__")))

    all_days = list(daily["fecha"].sort_values())
    if fill_missing_days and len(all_days) >= 2:
        full_range = pd.date_range(start=all_days[0], end=all_days[-1], freq="D")
        missing = [d for d in full_range if d not in set(all_days)]
        if missing:
            add = pd.DataFrame({"fecha": missing})
            daily = pd.concat([daily, add], ignore_index=True)
            daily = daily.sort_values("fecha").reset_index(drop=True)

    area_cols = [columns.get(a,a) for a in AREAS if columns.get(a,a) in daily.columns]

    cards: List[DayCard] = []
    prev_row = None
    for _, row in daily.sort_values("fecha").iterrows():
        fecha = row["fecha"]
        is_missing = row.isna().all() or pd.isna(row.get("registros", np.nan))

        if not is_missing:
            resumen = {c: float(row[c]) if c in row and pd.notna(row[c]) else float("nan") for c in area_cols}
            comp, cat = compute_composite_and_category(row, columns)
            amable = kind_for_category(cat)
            trend_labels = _area_trend_labels(row, area_cols)
            mensajes = stack_human_messages(trend_labels)
            cmp_prev = _compare_prev(row, prev_row, area_cols)
            day_df = day_groups.get(pd.Timestamp(fecha), pd.DataFrame(columns=df.columns))
            notas, estresores = _list_notes_and_stressors(day_df, columns)
            registros = int(row.get("registros", 1))
        else:
            resumen = {}
            amable = "Sin registro. ¿Te das permiso para escucharte y anotar algo breve mañana?"
            mensajes = []
            cmp_prev = "Sin datos del día, se usa la última referencia disponible."
            notas = []
            estresores = []
            registros = 0

        cards.append(DayCard(
            fecha=str(pd.to_datetime(fecha).date()),
            registros=registros,
            resumen_areas=resumen,
            mensajes_humanos=mensajes,
            notas=notas,
            estresores=estresores if registros >= 2 else [],
            comparacion_prev=cmp_prev,
            interpretacion_dia=amable,
            faltante=is_missing
        ))
        if not is_missing:
            prev_row = row

    return cards
