# --- Preprocesador para normalizar CSV antes del reporte ----------------------
import os, tempfile, re
import pandas as pd
import numpy as np
import unicodedata
from typing import List, Dict, Any

def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.strip().lower()


def _resolve_columns(df: pd.DataFrame, expected: Dict[str,str]) -> pd.DataFrame:
    """
    Intenta mapear/renombrar columnas reales a los nombres esperados en DEFAULT_COLUMNS,
    usando comparación insensible a mayúsculas/acentos/espacios.
    No elimina columnas; sólo renombra si encuentra equivalentes.
    """
    if df is None or df.empty:
        return df
    # Normaliza encabezados reales
    real_cols = list(df.columns)
    norm_to_real = {_norm(c): c for c in real_cols}
    rename_map = {}
    for k, exp in expected.items():
        if not isinstance(exp, str): 
            continue
        if exp in df.columns:
            continue  # ya está
        # buscar por forma normalizada
        real = norm_to_real.get(_norm(exp))
        if real:
            rename_map[real] = exp
        else:
            # sinónimo básico para fecha
            if k == "fecha":
                for alt in ["fecha", "fechas", "dia", "día", "date", "fecha_registro"]:
                    real = norm_to_real.get(_norm(alt))
                    if real and real not in rename_map:
                        rename_map[real] = exp
                        break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# Soporta '10:00:00 a. m.' / '9:30 p. m.' / '09:30' / '930' / '7.5' → 'HH:MM'
def _coerce_hhmm_latam_ampm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\xa0", " ", regex=False).str.strip().str.lower()
    s = s.str.replace(r"\s*a\.?\s*m\.?\s*", " am ", regex=True)
    s = s.str.replace(r"\s*p\.?\s*m\.?\s*", " pm ", regex=True)
    s = s.str.replace(".", ":", regex=False).str.replace(",", ":", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    out = pd.Series("", index=s.index, dtype=object)

    m = s.str.match(r"^\d{1,2}:\d{2}:\d{2}\s*(am|pm)$")
    if m.any():
        t = pd.to_datetime(s[m], format="%I:%M:%S %p", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    m = s.str.match(r"^\d{1,2}:\d{2}\s*(am|pm)$")
    if m.any():
        t = pd.to_datetime(s[m], format="%I:%M %p", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    m = s.str.match(r"^\d{1,2}:\d{2}:\d{2}$")
    if m.any():
        t = pd.to_datetime(s[m], format="%H:%M:%S", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    m = s.str.match(r"^\d{1,2}:\d{2}$")
    if m.any():
        t = pd.to_datetime(s[m], format="%H:%M", errors="coerce")
        out.loc[m & t.notna()] = t.dt.strftime("%H:%M")

    m = s.str.match(r"^\d{3,4}$")
    if m.any():
        d = s[m]
        h = d.str[:-2].astype(int)
        mi = d.str[-2:].astype(int)
        ok = (h.between(0,23) & mi.between(0,59))
        out.loc[m & ok] = h[ok].map("{:02d}".format) + ":" + mi[ok].map("{:02d}".format)

    m = s.str.match(r"^\d+([\.:]\d+)?$")
    if m.any():
        dec = s[m].str.replace(":", ".", regex=False)
        f = pd.to_numeric(dec, errors="coerce")
        frac = (f >= 0) & (f <= 1)
        if frac.any():
            total_min = (f[frac] * 24 * 60).round().astype(int)
            hh = (total_min // 60).clip(0,23)
            mm = (total_min % 60).clip(0,59)
            out.loc[m[frac].index] = hh.map("{:02d}".format) + ":" + mm.map("{:02d}".format)
        hrs = (f > 0) & (f < 24)
        if hrs.any():
            h = f[hrs].astype(int)
            mm = ((f[hrs] - h) * 60).round().astype(int).clip(0,59)
            out.loc[m[hrs].index] = h.map("{:02d}".format) + ":" + mm.map("{:02d}".format)

    return out

def _preprocess_csv_for_coach(input_csv: str, email: str | None) -> str:
    # 1) Lee y normaliza columnas (si tienes normalizador propio, úsalo aquí)
    try:
        from bdp_calcs.normalize import normalize_bdp_df
        raw = pd.read_csv(input_csv, encoding="utf-8")
        df, _ = normalize_bdp_df(raw)
    except Exception:
        # fallback mínimo: lee y limpia encabezados
        df = pd.read_csv(input_csv, encoding="utf-8")
        df.columns = [str(c).strip() for c in df.columns]

    # 2) Filtra por correo si llega
    if email and "correo" in df.columns:
        df = df[df["correo"].astype(str).str.strip().str.lower() == email.strip().lower()].copy()

    # 3) Asegura 'fecha' y 'hora'
    #    - fecha → dd-MM-YYYY
    #    - hora  → HH:MM (LATAM am/pm soportado)
    # Intenta detectar alias si fuera necesario
    if "fecha" not in df.columns:
        for alt in ["fechas","dia","día","date","fecha_registro"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "fecha"})
                break
    if "hora" not in df.columns:
        for alt in ["hora_registro","hora_medicion","time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "hora"})
                break

    # Normaliza fecha
    if "fecha" in df.columns:
        f = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
        df["fecha"] = f.dt.strftime("%d-%m-%Y")

    # Normaliza hora (si no existe, crea vacía)
    if "hora" in df.columns:
        df["hora"] = _coerce_hhmm_latam_ampm(df["hora"].astype(str)).replace("", np.nan)
        df["hora"] = df["hora"].fillna("00:00")
    else:
        df["hora"] = "00:00"

    # 4) Ordena por timestamp para estabilidad
    ts = pd.to_datetime(df["fecha"] + " " + df["hora"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.assign(__ts=ts).sort_values("__ts").drop(columns="__ts")

    # 5) Escribe CSV temporal y devuelve la ruta
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, "_bdp_coach_pre.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
