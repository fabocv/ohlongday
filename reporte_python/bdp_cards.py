
from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np

from reporte_python.bdp_schema import DayCard, DEFAULT_COLUMNS, AREAS, ensure_str_list, coerce_date
from reporte_python.bdp_trends import compute_daily_areas, compute_trends, compute_composite_and_category
from reporte_python.bdp_messages import stack_human_messages, kind_for_category

def _list_notes_and_stressors(day_df: pd.DataFrame, columns: Dict[str,str]) -> (List[str], List[str]):
    notas_col = columns.get("notas", "notas")
    estres_col = columns.get("estresores", "estresores")
    fecha_col = columns.get("fecha", "fecha")
    notas = []
    estresores = []
    if not day_df.empty and notas_col in day_df.columns:
        for _, row in day_df.iterrows():
            note = str(row.get(notas_col, "")).strip()
            if note:
                ts = row.get(fecha_col, None)
                try:
                    hhmm = pd.to_datetime(ts).strftime("%H:%M") if pd.notna(ts) else "--:--"
                except Exception:
                    hhmm = "--:--"
                notas.append(f"[{hhmm}] {note}")
    if estres_col in day_df.columns:
        for x in day_df[estres_col].tolist():
            estresores.extend(ensure_str_list(x))
    return notas, estresores

def _area_trend_labels(row: pd.Series, area_cols: List[str]) -> Dict[str, str]:
    out = {}
    for c in area_cols:
        tcol = c + "_trend"
        if tcol in row and isinstance(row[tcol], str):
            # keep only up/down/flat
            if row[tcol] in ("up","down","flat"):
                out[c] = row[tcol]
    # Present areas with nicer keys
    return {k: out[k] for k in out}

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

    # Keep only relevant columns
    keep = set(columns.values())
    keep.update([dcol])
    df = df[[c for c in df.columns if c in keep]].copy()

    # Normalize date
    df[dcol] = coerce_date(df[dcol])
    df["__day__"] = df[dcol].dt.normalize()

    # Per-day averages for areas + registros count
    daily = compute_daily_areas(df, columns)
    # Compute trends (adds *_trend)
    daily = compute_trends(daily, columns)

    # Build map date -> slice of original rows for notes/stressors
    day_groups = dict(tuple(df.groupby("__day__")))

    # Fill missing days if requested
    all_days = list(daily["fecha"].sort_values())
    if fill_missing_days and len(all_days) >= 2:
        full_range = pd.date_range(start=all_days[0], end=all_days[-1], freq="D")
        # append any days missing to daily
        missing = [d for d in full_range if d not in set(all_days)]
        if missing:
            add = pd.DataFrame({"fecha": missing})
            daily = pd.concat([daily, add], ignore_index=True)
            daily = daily.sort_values("fecha").reset_index(drop=True)

    # Identify area cols present
    area_cols = [columns.get(a,a) for a in AREAS if columns.get(a,a) in daily.columns]

    # Iterate chronologically and build cards
    cards: List[DayCard] = []
    prev_row = None
    for _, row in daily.sort_values("fecha").iterrows():
        fecha = row["fecha"]
        is_missing = row.isna().all() or pd.isna(row.get("registros", np.nan))

        # Gather summaries
        if not is_missing:
            resumen = {c: float(row[c]) if c in row and pd.notna(row[c]) else float("nan") for c in area_cols}
            # Per-day category & kind message
            comp, cat = compute_composite_and_category(row, columns)
            amable = kind_for_category(cat)

            # Trend labels & stacked human messages
            trend_labels = _area_trend_labels(row, area_cols)
            mensajes = stack_human_messages(trend_labels)

            # Compare with previous day
            cmp_prev = _compare_prev(row, prev_row, area_cols)

            # Notes / Stressors
            day_df = day_groups.get(pd.Timestamp(fecha), pd.DataFrame(columns=df.columns))

            notas, estresores = _list_notes_and_stressors(day_df, columns)

            # If >1 registro, list notes & stressors in the card body
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

def cards_to_markdown(cards: List[DayCard]) -> str:
    lines = []
    for c in cards:
        lines.append(f"### {c.fecha}  {'(sin registro)' if c.faltante else ''}")
        lines.append(f"- Registros: {c.registros}")
        if c.resumen_areas:
            pretty = ', '.join([f"{k}: {v:.2f}" for k, v in c.resumen_areas.items() if not pd.isna(v)])
            if pretty:
                lines.append(f"- Áreas (promedio del día): {pretty}")
        # Only when multiple entries: list notes & stressors
        if c.registros >= 2:
            if c.notas:
                lines.append("- Notas:")
                for n in c.notas:
                    lines.append(f"  - {n}")
            if c.estresores:
                lines.append("- Estresores:")
                for e in c.estresores:
                    lines.append(f"  - {e}")
        # Stacked human messages (trend-based)
        if c.mensajes_humanos:
            lines.append("- Mensaje humano (tendencias):")
            for m in c.mensajes_humanos:
                lines.append(f"  - {m}")
        # Footer: interpretation (I_dia) + comparison to previous day
        lines.append(f"- I_dia: {c.interpretacion_dia}")
        if c.comparacion_prev:
            lines.append(f"- {c.comparacion_prev}")
        lines.append("")
    return "\n".join(lines)
