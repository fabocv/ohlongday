# ================== ACWR (Seaborn, altura 250 px) ==================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import time

sns.set_theme(style="whitegrid", context="paper", font_scale=0.8)

def _parse_hhmm_to_min(x):
    if pd.isna(x): return np.nan
    if isinstance(x, time): return x.hour*60 + x.minute
    s = str(x).strip()
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

def compute_acwr(daily: pd.DataFrame,
                 minutes_col="tiempo_ejercicio",
                 intensity_col="mov_intensidad",
                 acute_win=7, chronic_win=28):
    """
    Carga interna diaria:
      load = minutos * (1 + intensidad/10)  # si no hay intensidad -> solo minutos
    ACWR = media móvil (7d) / media móvil (28d)
    Requiere 'fecha' (datetime-like).
    """
    d = daily.copy().sort_values("fecha").reset_index(drop=True)
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")

    mins = pd.to_numeric(d.get(minutes_col, 0), errors="coerce").fillna(0.0)
    if intensity_col in d.columns:
        inten = pd.to_numeric(d[intensity_col], errors="coerce").fillna(0.0).clip(0, 10)
        load = mins * (1.0 + inten/10.0)   # TRIMP-lite
    else:
        load = mins.copy()

    acute   = load.rolling(window=acute_win,  min_periods=max(3, acute_win//2)).mean()
    chronic = load.rolling(window=chronic_win, min_periods=max(7, chronic_win//4)).mean()
    acwr = (acute / chronic).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({
        "fecha": d["fecha"],
        "load":  load,
        "acute": acute,
        "chronic": chronic,
        "ACWR": acwr
    })

def plot_acwr_seaborn(acwr_df: pd.DataFrame,
                      out_path="ACWR_250px.png",
                      width_px=1000, height_px=250, dpi=100):
    """
    Dibuja ACWR con altura EXACTA en píxeles (por defecto 250 px).
    Bandas guía:
      0.8–1.3 → “zona dulce”
      >1.5    → mayor riesgo de sobrecarga
    """
    df = acwr_df.dropna(subset=["fecha","ACWR"]).copy()
    if df.empty:
        raise ValueError("No hay datos suficientes para ACWR.")

    width_in  = width_px  / dpi
    height_in = height_px / dpi

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    sns.lineplot(ax=ax, data=df, x="fecha", y="ACWR", linewidth=1.4)

    # Bandas de referencia (neutras)
    ymin, ymax = ax.get_ylim()
    ax.axhspan(0.8, 1.3, alpha=0.08)
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.axhline(1.5, linestyle=":",  linewidth=0.8)

    # Último punto
    x_last = df["fecha"].iloc[-1]
    y_last = float(df["ACWR"].iloc[-1])
    ax.scatter([x_last], [y_last], s=18, zorder=3)
    ax.text(x_last, y_last, f" {y_last:.2f}", va="center", fontsize=8)

    ax.set_title("ACWR (carga aguda:crónica [de ejercicio]) — 0.8–1.3 ideal; >1.5 riesgo")
    ax.set_xlabel("fecha"); ax.set_ylabel("ACWR")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path), y_last

# ================== USO ==================
# acwr_df = compute_acwr(daily)             # 'daily' = salida de aggregate_daily(df)
# png_path, last_val = plot_acwr_seaborn(acwr_df, height_px=250)
# print(f"ACWR último día: {last_val:.2f} | Gráfico: {png_path}")


def _get_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def compute_monotony_strain(
    daily: pd.DataFrame,
    minutes_col="tiempo_ejercicio",
    intensity_col="mov_intensidad",
    win=7,
):
    """
    Carga interna diaria:
      load = minutos * (1 + intensidad/10)   # si no hay intensidad -> solo minutos
    Monotony = mean(load_7d) / std(load_7d)
    Strain = Monotony * sum(load_7d)
    Requiere 'fecha' (datetime-like).
    """
    d = daily.copy().sort_values("fecha").reset_index(drop=True)
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")

    mins = _get_num(d.get(minutes_col, 0.0), 0.0)
    if intensity_col in d.columns:
        inten = _get_num(d[intensity_col], 0.0).clip(0, 10)
        load = mins * (1.0 + inten/10.0)  # TRIMP-lite estable
    else:
        load = mins.copy()

    mean7 = load.rolling(win, min_periods=max(3, win//2)).mean()
    std7  = load.rolling(win, min_periods=max(3, win//2)).std(ddof=0)  # ddof=0 más estable
    sum7  = load.rolling(win, min_periods=max(3, win//2)).sum()
    eps = 1e-6
    monotony = (mean7 / (std7 + eps)).replace([np.inf, -np.inf], np.nan)
    strain   = monotony * sum7

    out = pd.DataFrame({
        "fecha": d["fecha"],
        "load": load,
        "mean7": mean7,
        "std7": std7,
        "sum7": sum7,
        "Monotony": monotony,
        "Strain": strain
    })
    return out

def _last_and_trend(s, window=7):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return np.nan, "NA"
    last = float(s.iloc[-1])
    use = s.iloc[-window:] if len(s) >= window else s
    x = np.arange(len(use))
    slope = np.polyfit(x, use.values, 1)[0] if len(use) >= 2 else 0.0
    trend = "estable" if abs(slope) < 0.01 else ("al alza" if slope > 0 else "a la baja")
    return last, trend

def _quant_bands(s, low=0.70, high=0.85):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 5:
        return None
    return {"p70": float(np.nanquantile(s, low)), "p85": float(np.nanquantile(s, high))}

def plot_monotony_seaborn(ms_df: pd.DataFrame, out_path="Monotony_250px.png",
                          width_px=1000, height_px=250, dpi=100):
    df = ms_df.dropna(subset=["fecha","Monotony"]).copy()
    if df.empty: raise ValueError("No hay datos suficientes para Monotony.")
    w_in, h_in = width_px/dpi, height_px/dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    sns.lineplot(ax=ax, data=df, x="fecha", y="Monotony", linewidth=1.4)
    # Guías típicas
    ax.axhline(1.5, linestyle="--", linewidth=0.8)   # aviso suave
    ax.axhline(2.0, linestyle=":",  linewidth=0.8)   # precaución
    # Último punto
    x_last = df["fecha"].iloc[-1]; y_last = float(df["Monotony"].iloc[-1])
    ax.scatter([x_last], [y_last], s=18, zorder=3)
    ax.text(x_last, y_last, f" {y_last:.2f}", va="center", fontsize=8)
    ax.set_title("Foster Monotony — variación semanal (más alto = menos variación)")
    ax.set_xlabel("fecha"); ax.set_ylabel("Monotony")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return Path(out_path), y_last

def plot_strain_seaborn(ms_df: pd.DataFrame, out_path="Strain_250px.png",
                        width_px=1000, height_px=250, dpi=100):
    df = ms_df.dropna(subset=["fecha","Strain"]).copy()
    if df.empty: raise ValueError("No hay datos suficientes para Strain.")
    w_in, h_in = width_px/dpi, height_px/dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    sns.lineplot(ax=ax, data=df, x="fecha", y="Strain", linewidth=1.4)
    # Bandas personales (p70/p85) para “ojo/alerta”
    qs = _quant_bands(df["Strain"])
    if qs:
        ymin, ymax = ax.get_ylim()
        ax.axhspan(qs["p70"], qs["p85"], alpha=0.08)
        ax.axhspan(qs["p85"], ymax,    alpha=0.10)
    # Último punto
    x_last = df["fecha"].iloc[-1]; y_last = float(df["Strain"].iloc[-1])
    ax.scatter([x_last], [y_last], s=18, zorder=3)
    ax.text(x_last, y_last, f" {y_last:.0f}", va="center", fontsize=8)
    ax.set_title("Foster Strain — estrés semanal total (personalizado por cuantiles)")
    ax.set_xlabel("fecha"); ax.set_ylabel("Strain")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return Path(out_path), y_last

# ================== USO ==================
# 1) 'daily' = salida de aggregate_daily(df) con al menos: fecha, tiempo_ejercicio (y ojalá mov_intensidad)
# ms = compute_monotony_strain(daily)
# mono_png, mono_last = plot_monotony_seaborn(ms, height_px=250)
# strain_png, strain_last = plot_strain_seaborn(ms, height_px=250)
# print(f"Monotony último día: {mono_last:.2f} | Gráfico: {mono_png}")
# print(f"Strain   último día: {strain_last:.0f} | Gráfico: {strain_png}")