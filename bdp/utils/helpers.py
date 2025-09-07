# --- Preprocesador para normalizar CSV antes del reporte ----------------------
import os, tempfile, re
import pandas as pd
import numpy as np
import unicodedata
from typing import List, Dict, Any

pd.set_option('future.no_silent_downcasting', True)

def _series(s): return s if isinstance(s, pd.Series) else pd.Series([s])

def join_unique(s, sep=" | ", limit=2000):
    s = _series(s)
    vals = [str(x).strip() for x in s.dropna().astype(str) if str(x).strip()]
    uniq = list(dict.fromkeys(vals))  # preserva orden
    out = sep.join(uniq)
    return out[:limit] if limit else out

def max_01(s):
    s = _series(s)
    num = pd.to_numeric(s, errors="coerce").fillna(0)
    return int((num > 0).any())

def mean_num(s):
    s = _series(s)
    num = pd.to_numeric(s, errors="coerce")
    return float(num.mean()) if num.notna().any() else np.nan

def smart_mean_or_join(s):
    s = _series(s)
    num = pd.to_numeric(s, errors="coerce")
    ratio = num.notna().mean()
    return mean_num(s) if ratio >= 0.60 else join_unique(s)

def _strip_accents(txt):
    if pd.isna(txt): return ""
    return "".join(c for c in unicodedata.normalize("NFD", str(txt)) if unicodedata.category(c) != "Mn").lower().strip()

def _norm_periodo(v):
    s = _strip_accents(v)
    if "día" in s: return "DIA"
    if "manana" in s: return "AM"
    if "tarde" in s: return "PM"
    if "noche" in s: return "NOCHE"
    return "UNK"


def _safe_agg_for(df_cols, agg):
    return {k: v for k, v in agg.items() if k in df_cols}

# ===================== agregados =====================
def aggregate_by_block(df, agg):
    """
    Agrega por fecha + registro_periodo(normalizado).
    Devuelve largo: columnas agregadas + 'periodo'.
    """
    if "fecha" not in df.columns:
        raise ValueError("'fecha' no está en df.columns")
    if "registro_periodo" not in df.columns:
        raise ValueError("'registro_periodo' no está en df.columns")

    tmp = df.copy()
    tmp["periodo"] = tmp["registro_periodo"].map(_norm_periodo)
    final_agg = _safe_agg_for(tmp.columns, agg)

    out = (tmp
           .groupby(["fecha","periodo"], as_index=False)
           .agg(final_agg))

    # Para métricas de conteo diario que luego sumarás (acto_verdad, limites_practicados, etc.)
    return out

def aggregate_daily(df, agg):
    """
    Consolida a un registro por día:
    - Si existe algún registro con periodo 'DIA', usa solo esos.
    - Si no, combina AM/PM/NOCHE.
    - Para flags tipo acto_verdad, cuenta (suma) a nivel día.
    """
    tmp = df.copy()
    tmp["periodo"] = tmp["registro_periodo"].map(_norm_periodo)

    cols = tmp.columns
    final_agg = _safe_agg_for(cols, agg)

    # 1) Agregado por bloque (para después consolidar reglas por día)
    by_block = (tmp
        .groupby(["fecha","periodo"], as_index=False)
        .agg(final_agg))

    # 2) Elegir fuente por día:
    #    si hay DIA → quedarnos con esos; si no, sumar/mezclar AM/PM/NOCHE.
    def _pick_rows(g):
        if (g["periodo"] == "DIA").any():
            return g[g["periodo"] == "DIA"]
        return g  # se combinan abajo

    
    lista = [a for a in final_agg]

    picked = by_block.groupby("fecha", group_keys=False).apply(_pick_rows, include_groups=True) \
        .reset_index(drop=True)

    print(picked.head(20))
 
    # 3) Re-agregar a nivel día:
    #    - numéricos: mean o sum según el mapeo original
    #    - textos: join_unique
    # Detectamos cuáles funciones son sum/mean/func
    sum_cols = [k for k, v in agg.items() if v == "sum" and k in picked.columns]
    mean_cols = [k for k, v in agg.items() if v == "mean" and k in picked.columns]
    func_cols = {k: v for k, v in agg.items() if k in picked.columns and v not in {"sum","mean"}}

    parts = []
    if sum_cols:
        parts.append(picked.groupby("fecha", as_index=False)[sum_cols].sum())
    if mean_cols:
        parts.append(picked.groupby("fecha", as_index=False)[mean_cols].mean())

    # columnas con funciones personalizadas (join_unique, mean_num, max_01...)
    if func_cols:
        fdf = picked.groupby("fecha").agg(func_cols).reset_index()
        parts.append(fdf)

    # merge progresivo por 'fecha'
    if not parts:
        return picked.groupby("fecha", as_index=False).first()

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="fecha", how="outer")

    # ejemplo de métricas derivadas útiles a nivel día:
    # - total_verdades = conteo de actos de verdad en el día (si hubo múltiples bloques)
    if "acto_verdad" in picked.columns:
        verd = (picked
                .assign(acto_verdad_num=pd.to_numeric(picked["acto_verdad"], errors="coerce").fillna(0))
                .groupby("fecha", as_index=False)["acto_verdad_num"].sum()
                .rename(columns={"acto_verdad_num":"acto_verdad_total"}))
        out = out.merge(verd, on="fecha", how="left")

    if "limites_practicados" in picked.columns and "limites_practicados" not in sum_cols:
        lim = (picked.groupby("fecha", as_index=False)["limites_practicados"].sum()
               .rename(columns={"limites_practicados":"limites_practicados_total"}))
        out = out.merge(lim, on="fecha", how="left")

    return out

def aggregate_blocks_wide(df, suffix_map=None, aggregate=None):
    """
    Igual que aggregate_by_block, pero pivotea a columnas por bloque.
    Sufijos por defecto: AM→_am, PM→_pm, NOCHE→_noche, DIA→_dia
    """
    if suffix_map is None:
        suffix_map = {"AM":"_am","PM":"_pm","NOCHE":"_noche","DIA":"_dia","UNK":"_unk"}
    by_block = aggregate_by_block(df, aggregate)
    value_cols = [c for c in by_block.columns if c not in {"fecha","periodo"}]

    wide = (by_block
            .pivot(index="fecha", columns="periodo", values=value_cols))

    # flatea MultiIndex columnas → colname + sufijo
    wide.columns = [f"{col}{suffix_map.get(per, f'_{per.lower()}')}" for (col, per) in wide.columns]
    wide = wide.reset_index()
    return wide


# Exceso sobre un límite: (x - L)+ * k, saturado a 10
def s_exceso(x, limite, k=1.0, top=10):
    return ((x - limite).clip(lower=0) * k).clip(upper=top)

# Déficit respecto a un objetivo: (T - x)+ * k, saturado a 10
def s_deficit(x, objetivo, k=1.0, top=10):
    return ((objetivo - x).clip(lower=0) * k).clip(upper=top)

# U-shape respecto a [A, B]: distancia fuera del rango * k, saturado a 10
def s_u_shape(x, A, B, k=1.0, top=10):
    dist = (A - x).clip(lower=0) + (x - B).clip(lower=0)
    return (dist * k).clip(upper=top)

# Escalado robusto por cuantiles a 0–10 (opcional para outliers)
def robust_scale_010(s, ql=0.10, qh=0.90):
    a, b = s.quantile(ql), s.quantile(qh)
    scaled = 10 * (s - a) / max(b - a, 1e-9)
    return scaled.clip(0, 10)

def to_minutes(t):
    # t en "HH:MM"
    h, m = map(int, str(t).split(":"))
    return h*60 + m

def sleep_hours(gr):
    # usa último dormir y último despertar del día; si despertar < dormir, cruza medianoche
    try:
        d = to_minutes(gr["hora_dormir"].dropna().iloc[-1])
        w = to_minutes(gr["hora_despertar"].dropna().iloc[-1])
        if w < d: w += 24*60
        siesta = gr.get("siesta_min", pd.Series([0])).fillna(0).sum()
        return (w - d)/60 + siesta/60
    except Exception:
        return np.nan

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

def _preprocess_csv_for_coach(input_csv: str, email: str | None) -> pd.DataFrame :
    # 1) Lee y normaliza columnas (si tienes normalizador propio, úsalo aquí)
    print("archivo: %s   |   correo: %s... " % (input_csv, email[0:3]) )
    try:
        with open(input_csv, "r") as file:
            data = file.read()
    except FileNotFoundError:
        print("Archivo no encontrado")
        return None
    try:
        from bdp.utils.normalize import normalize_bdp_df
        raw = pd.read_csv(input_csv, encoding="utf-8")
        df, _ = normalize_bdp_df(raw)
    except Exception:
        # fallback mínimo: lee y limpia encabezados
        df = pd.read_csv(input_csv, encoding="utf-8")
        df.columns = [str(c).strip() for c in df.columns]

    # 2) Filtra por correo si llega
    if email and "correo" in df.columns:
        df = df[df["correo"].astype(str).str.strip().str.lower() == email.strip().lower()].copy()
        if (len(df) == 0 ):
            print("El dataset no contiene al correo ingresado")
            return None

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
    if "hora" not in df.columns:
        df["hora"] = "00:00"

    col_times = [x  for x in df.columns if "hora" in x]

    for columna in col_times:
        df[columna] = _coerce_hhmm_latam_ampm(df[columna].astype(str)).replace("", np.nan)
        ### df[columna] = df[columna].fillna("00:00") ## ojo con esto

    # 4) Ordena por timestamp para estabilidad
    ts = pd.to_datetime(df["fecha"] + " " + df["hora"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.assign(__ts=ts).sort_values("__ts").drop(columns="__ts")

    # 5) Escribe CSV temporal y devuelve la ruta
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, "_bdp_coach_pre.csv")
    #df.to_csv(out_path, index=False, encoding="utf-8")
    #return out_path
    return df

def prepare_for_merge_on_fecha(left: pd.DataFrame, right: pd.DataFrame,
                               left_col="fecha", right_col="fecha",
                               dayfirst=True):
    L = left.copy()
    R = right.copy()

    # 1) Parseo a datetime
    for df, col in ((L, left_col), (R, right_col)):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
        # si viene con tz, lo volvemos naive
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_convert(None)

        # 2) Normalizar a fecha (00:00:00)
        df[col] = df[col].dt.normalize()

    # 3) Asegurar 1 fila por fecha en el 'right' (por si acaso)
    R = R.sort_values(right_col).drop_duplicates(subset=[right_col], keep="last")

    # 4) Merge m:1 (crudo -> diario)
    merged = L.merge(R, left_on=left_col, right_on=right_col, how="left", validate="m:1")
    if right_col != left_col:
        merged = merged.drop(columns=[right_col])
    return merged
