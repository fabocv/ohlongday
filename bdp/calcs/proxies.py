import numpy as np
import pandas as pd
from datetime import time

# ---------------- utils robustas ----------------
def parse_hhmm_to_min(x):
    """Convierte 'HH:MM' o datetime.time a minutos; si no, NaN."""
    if pd.isna(x): return np.nan
    if isinstance(x, time): return x.hour*60 + x.minute
    s = str(x).strip()
    if not s: return np.nan
    try:
        hh, mm = s.split(":")[:2]
        return int(hh)*60 + int(mm)
    except Exception:
        # si viene un número tipo 1830 -> 18:30
        try:
            v = int(float(s))
            if v > 2400: return np.nan
            return (v // 100)*60 + (v % 100)
        except Exception:
            return np.nan

def robust_z(x):
    """z-score robusto por mediana/MAD; fallback a std. Centra y escala por persona."""
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() < 3: 
        return pd.Series(np.zeros(len(x)), index=x.index)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad and mad > 0:
        z = 0.6745 * (x - med) / mad
    else:
        mu, sd = np.nanmean(x), np.nanstd(x)
        z = (x - mu) / (sd if sd and sd > 0 else 1.0)
    return z.fillna(0.0)

def ema_series(x, alpha=0.30):
    """EMA simple; ignora NaN al inicio."""
    s = pd.to_numeric(x, errors="coerce")
    return s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

def ushape_penalty(x, low=2, high=8, mid_low=4, mid_high=7, k=0.5):
    """
    Penaliza intensidades muy bajas (<2) o muy altas (>8).
    0 en zona óptima [4,7], sube lineal hacia 1 en extremos; escala k.
    """
    x = pd.to_numeric(x, errors="coerce")
    pen = pd.Series(0.0, index=x.index)
    pen = np.where(x < mid_low, np.clip((mid_low - x) / (mid_low - low + 1e-9), 0, 1),
          np.where(x > mid_high, np.clip((x - mid_high) / (high - mid_high + 1e-9), 0, 1), 0))
    return k * pd.Series(pen, index=x.index).fillna(0.0)

def count_tokens(text, seps=(",","|",";")):
    if pd.isna(text): return 0
    s = str(text)
    for sp in seps:
        s = s.replace(sp, ",")
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return len(toks)

def qcut_labels(x, low_q=0.15, medlow_q=0.30, high_q=0.85, medhigh_q=0.70, invert=False):
    """
    Etiquetas por cuantiles de HISTÓRICO PERSONAL.
    - Para índices "mal = alto": Alert si x >= p85; Caution p70–p85.
    - Para índices "mal = bajo": invierte (Alert si x <= p15; Caution p15–p30).
    """
    s = pd.to_numeric(x, errors="coerce")
    p15 = np.nanquantile(s, low_q) if s.notna().sum() >= 5 else s.min()
    p30 = np.nanquantile(s, medlow_q) if s.notna().sum() >= 5 else (s.min()+s.max())/2
    p70 = np.nanquantile(s, medhigh_q) if s.notna().sum() >= 5 else (s.min()+s.max())/2
    p85 = np.nanquantile(s, high_q) if s.notna().sum() >= 5 else s.max()
    labels = []
    for v in s:
        if np.isnan(v):
            labels.append("NA")
            continue
        if not invert:  # mal = alto
            if v >= p85: labels.append("Alert")
            elif v >= p70: labels.append("Caution")
            else: labels.append("OK")
        else:           # mal = bajo
            if v <= p15: labels.append("Alert")
            elif v <= p30: labels.append("Caution")
            else: labels.append("OK")
    return pd.Series(labels, index=x.index), {"p15":p15,"p30":p30,"p70":p70,"p85":p85}

# ---------- helpers robustos ----------
def parse_hhmm_to_min(x):
    if pd.isna(x): return np.nan
    if isinstance(x, time): return x.hour*60 + x.minute
    s = str(x).strip()
    if not s: return np.nan
    try:
        hh, mm = s.split(":")[:2]
        return int(hh)*60 + int(mm)
    except Exception:
        try:
            v = int(float(s))
            if v > 2400: return np.nan
            return (v // 100)*60 + (v % 100)
        except Exception:
            return np.nan

def _series_zeros_like(df, val=0.0, dtype=float):
    return pd.Series(val, index=df.index, dtype=dtype)

def get_num(df, *names, default=0.0):
    """
    Devuelve SIEMPRE una Series numérica alineada al índice de df.
    Busca la 1ª columna existente en 'names'; si no hay, devuelve zeros.
    """
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return _series_zeros_like(df, default)

def get_timeflag(df, name, hour_cut=18):
    """Flag 0/1 si hora >= hour_cut (HH:MM). Devuelve Series alineada."""
    if name in df.columns:
        mins = df[name].apply(parse_hhmm_to_min)
        return (mins >= hour_cut*60).astype(float).fillna(0.0)
    return _series_zeros_like(df, 0.0)

def count_tokens(text, seps=(",","|",";")):
    if pd.isna(text): return 0
    s = str(text)
    for sp in seps:
        s = s.replace(sp, ",")
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return len(toks)

# ---------- función parcheada ----------
def derive_features_for_proxies(daily):
    d = daily.copy()

    # Pantallas noche (si tienes una específica, úsala; si no, 0)
    d["pantalla_noche_min"] = get_num(d, "tiempo_pantalla_noche_min", default=0.0).fillna(0.0)

    # Pantallas totales
    d["pantallas_total_min"] = get_num(d, "tiempo_pantallas", default=0.0).fillna(0.0)

    # Café/Alcohol (flags por horario + totales)
    d["cafe_tarde_flag"]    = get_timeflag(d, "cafe_ultima_hora", hour_cut=18)
    d["alcohol_tarde_flag"] = get_timeflag(d, "alcohol_ultima_hora", hour_cut=18)
    d["cafe_total"]         = get_num(d, "cafe_cucharaditas", default=0.0).fillna(0.0)
    d["alcohol_total"]      = get_num(d, "alcohol_ud", default=0.0).fillna(0.0)

    # Luz de mañana (maneja alias con/ sin tildes)
    d["luz_manana_min"] = (
        get_num(d,
                "exposicion_sol_manana_min",
                "exposición_sol_mañana_min",
                "exposicion_sol_mañana_min",
                "luz_manana_min",   # por si ya venía con ese nombre
                default=0.0)
        .fillna(0.0)
    )

    # Interacciones significativas: hoy es TEXTO → cuenta tokens
    # (Si ya agregaste por día con join_unique, igual funciona)
    if "interacciones_significativas" in d.columns:
        d["interacciones_count"] = d["interacciones_significativas"].apply(count_tokens)
    else:
        d["interacciones_count"] = _series_zeros_like(d, 0)

    # Meditación, sueño, ejercicio, movimiento
    d["meditacion_min"]   = get_num(d, "meditacion_min", default=0.0).fillna(0.0)
    d["sueno_calidad"]    = get_num(d, "sueno_calidad", "sueño_calidad").astype(float)
    d["tiempo_ejercicio"] = get_num(d, "tiempo_ejercicio", default=0.0).fillna(0.0)
    d["mov_intensidad"]   = get_num(d, "mov_intensidad").astype(float)

    # Afecto/estrés y otros numéricos comunes (si existen)
    for col in ["estres","irritabilidad","ansiedad","animo","claridad","conexion","proposito","glicemia","alimentacion","agua_litros"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Actos de verdad (total/día). Si no existe el total, intento usar el crudo.
    if "acto_verdad_total" in d.columns:
        d["acto_verdad_total"] = pd.to_numeric(d["acto_verdad_total"], errors="coerce").fillna(0.0)
    elif "acto_verdad" in d.columns:
        # Fallback: 0/1 en el día (si no agregaste por bloque).
        d["acto_verdad_total"] = pd.to_numeric(d["acto_verdad"], errors="coerce").fillna(0.0)
    else:
        d["acto_verdad_total"] = _series_zeros_like(d, 0.0)

    # Juego en dispositivo (para DBI; maneja dos posibles nombres)
    d["juego_en_dispositivo_min"] = get_num(
        d, "juego_en_dispositivo_min", "juego_en_dispositivo", default=0.0
    ).fillna(0.0)

    return d

# ---------------- índices proxy ----------------
def compute_proxy_indices(daily, alpha=0.30):
    d = derive_features_for_proxies(daily)

    # z-scores (robustos) de las variables usadas
    z_estres        = robust_z(d.get("estres"))
    z_irritabilidad = robust_z(d.get("irritabilidad"))
    z_ansiedad      = robust_z(d.get("ansiedad"))
    z_sueno         = robust_z(d.get("sueno_calidad"))
    z_meditacion    = robust_z(d.get("meditacion_min"))
    z_pantalla_noc  = robust_z(d.get("pantalla_noche_min"))
    z_pantallas_tot = robust_z(d.get("pantallas_total_min"))
    z_cafe_total    = robust_z(d.get("cafe_total"))
    z_alcohol_total = robust_z(d.get("alcohol_total"))
    z_mov_int       = robust_z(d.get("mov_intensidad"))
    z_interacciones = robust_z(d.get("interacciones_count"))
    z_conexion      = robust_z(d.get("conexion"))
    z_claridad      = robust_z(d.get("claridad"))
    z_animo         = robust_z(d.get("animo"))
    z_glicemia      = robust_z(d.get("glicemia"))
    z_alimentacion  = robust_z(d.get("alimentacion"))
    z_ejercicio     = robust_z(d.get("tiempo_ejercicio"))
    z_luz_am        = robust_z(d.get("luz_manana_min"))

    # U-shape penalization por mov_intensidad
    u_pen = ushape_penalty(d.get("mov_intensidad"))

    # CLI (Cortisol Load Index) -> mal = ALTO
    CLI_raw = (
        1.00*z_estres
        + 0.30*z_irritabilidad
        + 0.20*z_ansiedad
        - 0.35*z_sueno
        - 0.25*z_meditacion
        + 0.25*z_pantalla_noc
        + 0.10*z_cafe_total + 0.10*z_alcohol_total
        + 0.15*d.get("cafe_tarde_flag") + 0.15*d.get("alcohol_tarde_flag")
        + u_pen
    )

    # MRS (Melatonin Rhythm Score) -> mal = BAJO
    MRS_raw = (
        + 0.40*z_sueno
        - 0.30*z_pantalla_noc
        - 0.20*d.get("cafe_tarde_flag")
        - 0.20*d.get("alcohol_tarde_flag")
        + 0.25*z_luz_am
    )

    # SSI (Serotonin Support Index) -> mal = BAJO
    SSI_raw = (
        0.35*z_conexion
        + 0.25*z_claridad
        + 0.20*z_interacciones
        + 0.15*(0.5*z_ejercicio + 0.5*z_mov_int)  # mezcla volumen + intensidad
        + 0.10*robust_z(d.get("agua_litros"))     # si existe, suma; si no es 0 por robust_z
    )

    # DBI (Dopamine Balance Index) -> mal = BAJO (equilibrio y logro)
    DBI_raw = (
        0.35*robust_z(d.get("micro_reparaciones"))
        + 0.25*(0.5*z_ejercicio + 0.5*z_mov_int)
        - 0.30*z_pantallas_tot
        - 0.15*robust_z(d.get("juego_en_dispositivo_min", d.get("juego_en_dispositivo", 0)))
        + 0.10*z_cafe_total
    )

    # MSI (Metabolic Strain Index) -> mal = ALTO
    MSI_raw = (
        0.45*z_glicemia
        + 0.20*z_estres
        - 0.25*(0.6*z_ejercicio + 0.4*z_mov_int)
        - 0.20*z_sueno
        + 0.10*robust_z(d.get("imc", d.get("peso", np.nan)))  # opcional si existe
    )

    # Suavizado EMA
    out = pd.DataFrame({
        "fecha": d["fecha"].values,
        "CLI": ema_series(CLI_raw, alpha=alpha),
        "MRS": ema_series(MRS_raw, alpha=alpha),
        "SSI": ema_series(SSI_raw, alpha=alpha),
        "DBI": ema_series(DBI_raw, alpha=alpha),
        "MSI": ema_series(MSI_raw, alpha=alpha),
    })

    # Bandas (Alert/Caution/OK)
    bands = {}
    for col, invert in [("CLI", False), ("MSI", False), ("MRS", True), ("SSI", True), ("DBI", True)]:
        labels, qs = qcut_labels(out[col], invert=invert)
        out[f"{col}_band"] = labels
        bands[col] = qs
        
    return out, bands

# ---------------- uso típico ----------------
# 1) daily = aggregate_daily(df)  # ya lo tienes hecho en el paso anterior
# 2) proxies = compute_proxy_indices(daily, alpha=0.30)
# 3) daily_with_proxies = daily.merge(proxies, on="fecha", how="left")

import matplotlib.pyplot as plt
from PIL import Image

out_files = []
def plot_index(plot_df, base, series_name, title):
    if series_name not in plot_df.columns or plot_df[series_name].dropna().empty:
        return None
    s = plot_df[["fecha", series_name]].dropna()
    fig = plt.figure(figsize=(10.6, 2.0))
    ax = plt.gca()
    ax.plot(s["fecha"], s[series_name], linewidth=1.5)
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("fecha")
    ax.set_ylabel(series_name)
    x_last = s["fecha"].iloc[-1]
    y_last = s[series_name].iloc[-1]
    ax.scatter([x_last], [y_last], s=18)
    ax.text(x_last, y_last, f" {y_last:.2f}", va="center", fontsize=7)
    fig.autofmt_xdate(rotation=30)
    fname = f"{base}{series_name}_proxy.png"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    out_files.append(fname)
    return fname

# Interpretations
def last_and_trend(s, window=7):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan, "NA"
    last = float(s.iloc[-1])
    use = s.iloc[-window:] if len(s) >= window else s
    x = np.arange(len(use))
    try:
        slope = np.polyfit(x, use.values, 1)[0]
    except Exception:
        slope = 0.0
    if abs(slope) < 0.01:
        t = "estable"
    elif slope > 0:
        t = "al alza"
    else:
        t = "a la baja"
    return last, t

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import time
from pathlib import Path

# ----------------- Gráfico apilado (5 paneles) -----------------
def plot_proxies_stacked_seaborn(proxies_df, bands, out_path="BDP_proxies_stacked_seaborn.png", 
                                 width_in=10.6, height_in=10.5):
    fig, axes = plt.subplots(5, 1, figsize=(width_in, height_in), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    cfg = [
        ("CLI", "CLI (Cortisol Load) — alto = peor"),
        ("MRS", "MRS (Melatonin Rhythm) — alto = mejor"),
        ("SSI", "SSI (Serotonin Support) — alto = mejor"),
        ("DBI", "DBI (Dopamine Balance) — alto = mejor"),
        ("MSI", "MSI (Metabolic Strain) — alto = peor"),
    ]
    proxies_df = proxies_df.copy()
    proxies_df["fecha"] = pd.to_datetime(proxies_df["fecha"])

    for ax, (col, title) in zip(axes, cfg):
        if col not in proxies_df.columns: 
            ax.set_visible(False); continue
        s = proxies_df[["fecha", col]].dropna()
        sns.lineplot(ax=ax, data=s, x="fecha", y=col, linewidth=1.5)
        qs = bands[col]  # del compute_proxy_indices
        ymin, ymax = ax.get_ylim()
        # Caution (p70–p85) para índices “mal=alto”
        if col in ("CLI","MSI"):
            ax.axhspan(qs["p70"], qs["p85"], alpha=.07)
            ax.axhspan(qs["p85"], ymax, alpha=.10)
        # Para “mal=bajo” (MRS/SSI/DBI) invierte:
        else:
            ax.axhspan(ymin, qs["p15"], alpha=.10)
            ax.axhspan(qs["p15"], qs["p30"], alpha=.07)
        # línea base 0
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.3")
        # último punto marcado
        if not s.empty:
            x_last = s["fecha"].iloc[-1]
            y_last = float(s[col].iloc[-1])
            ax.scatter([x_last], [y_last], s=20, zorder=3)
            ax.text(x_last, y_last, f" {y_last:.2f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("fecha")
        ax.set_ylabel(col)
        ax.grid(True, linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)

# ----------------- Interpretación del último día + tendencia -----------------
def last_and_trend(series, window=7):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan, "NA"
    last = float(s.iloc[-1])
    use = s.iloc[-window:] if len(s) >= window else s
    x = np.arange(len(use))
    slope = np.polyfit(x, use.values, 1)[0] if len(use) >= 2 else 0.0
    if abs(slope) < 0.01: trend = "estable"
    elif slope > 0: trend = "al alza"
    else: trend = "a la baja"
    return last, trend

def describe_proxies(proxies_df):
    mapping = [
        ("CLI", False, "Carga de estrés (HPA/cortisol): alto = peor."),
        ("MRS", True,  "Ritmo circadiano/melatonina: alto = mejor."),
        ("SSI", True,  "Soporte serotoninérgico (vínculo/claridad/mov): alto = mejor."),
        ("DBI", True,  "Equilibrio dopaminérgico (logro vs pantallas): alto = mejor."),
        ("MSI", False, "Carga metabólica (glicemia/estrés/sueño/IMC): alto = peor."),
    ]
    lines = []
    for col, invert, desc in mapping:
        if col not in proxies_df.columns: 
            continue
        last, trend = last_and_trend(proxies_df[col])
        labels, _ = qcut_labels(proxies_df[col], invert=invert)
        band = labels.iloc[-1] if len(labels) else "NA"
        lines.append(f"{col} ({desc}) Último={last:.2f} | banda={band} | tendencia {trend}.")
    return lines if lines else "No hay datos suficientes para interpretar."
