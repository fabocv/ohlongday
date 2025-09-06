from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

def generar_grafico_bienestar_seaborn(
    df: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str] = None,
    out_html: Optional[str] = None,
    bienestar_neto = "WBN_ex",
    title: str = "Bienestar: Autopercibido vs. Neto",
    y_domain: Optional[Tuple[float, float]] = (0, 10),
    smooth_neto_window: int = 0,  # 0 = sin suavizado; si pones 3–7 hace rolling mean
) -> Dict[str, Optional[str]]:
    """
    Grafica:
      - bienestar_ema  -> área + línea sólida
      - bienestar_neto -> línea discontinua (opcional suavizado rolling)

    Requisitos columnas: ['fecha', 'bienestar_ema', bienestar_neto].
    Escala esperada: 0–10 (puedes desactivar y_domain para autoescalar).
    """

    dfi = df.copy()

    # --- Parseo robusto ---
    if "fecha" not in dfi.columns:
        raise ValueError("Falta columna 'fecha'")
    if not pd.api.types.is_datetime64_any_dtype(dfi["fecha"]):
        dfi["fecha"] = pd.to_datetime(dfi["fecha"], dayfirst=True, errors="coerce")

    # Ordenar y tipar numérico
    dfi = dfi.sort_values("fecha").reset_index(drop=True)
    for col in ("bienestar_ema", bienestar_neto):
        if col not in dfi.columns:
            dfi[col] = np.nan
        dfi[col] = pd.to_numeric(dfi[col], errors="coerce").astype(float)

    # Suavizado opcional para NETO (rolling centrado)
    if smooth_neto_window and smooth_neto_window >= 3:
        dfi["bienestar_neto_smooth"] = (
            dfi[bienestar_neto]
            .rolling(window=smooth_neto_window, center=True, min_periods=1)
            .mean()
        )
        neto_col = "bienestar_neto_smooth"
    else:
        neto_col = bienestar_neto

    # --- Estilo seaborn ---
    sns.set_theme(style="whitegrid",
                  rc={"axes.spines.right": False, "axes.spines.top": False})

    # Paleta
    ema_line = "#2aa574"     # verde
    ema_fill = "#e7f6f0"     # verde pálido
    neto_line = "#55677a"    # gris azulado

    fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=160)

    # --- EMA: área + línea ---
    if dfi["bienestar_ema"].notna().any():
        ax.fill_between(
            dfi["fecha"], 0, dfi["bienestar_ema"],
            where=~dfi["bienestar_ema"].isna(),
            color=ema_fill, alpha=0.90, linewidth=0
        )
        sns.lineplot(
            data=dfi, x="fecha", y="bienestar_ema",
            ax=ax, linewidth=2.2, color=ema_line, label="Autopercibido (EMA)"
        )

    # --- Neto: línea discontinua ---
    if dfi[neto_col].notna().any():
        sns.lineplot(
            data=dfi, x="fecha", y=neto_col,
            ax=ax, linewidth=1.8, color=neto_line,
            dashes=True, label="Neto"
        )

    # --- Anotaciones último punto por serie ---
    def _anota_ultimo(col, color, dy):
        s = dfi[col].dropna()
        if s.empty:
            return
        idx = s.index[-1]
        x = dfi.loc[idx, "fecha"]
        y = dfi.loc[idx, col]
        ax.scatter([x], [y], s=46, zorder=5, color=color, edgecolor="white", linewidth=1)
        label = f"{'Autoperc.' if col=='bienestar_ema' else 'Neto'} {y:.2f}"
        ax.annotate(label, (x, y), xytext=(8, dy), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold")

    _anota_ultimo("bienestar_ema", ema_line, dy=-8)
    _anota_ultimo(neto_col, neto_line, dy=10)

    # --- Formato ejes ---
    ax.set_title(title, fontsize=12, weight="bold", color="#364554")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if y_domain is not None:
        ax.set_ylim(*y_domain)

    # Fechas legibles
    locator = AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    # Leyenda limpia
    leg = ax.legend(frameon=False, loc="upper left")
    if leg:  # proteger por si no hay líneas
        for t in leg.get_texts():
            t.set_color("#546471")

    plt.tight_layout()

    # --- Guardados ---
    out_png_path = Path(out_png).resolve()
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight")

    out_svg_path = None
    if out_svg:
        out_svg_path = Path(out_svg).resolve()
        fig.savefig(out_svg_path, bbox_inches="tight")

    # HTML simple (img embebida)
    out_html_path = None
    if out_html:
        out_html_path = Path(out_html).resolve()
        out_html_path.parent.mkdir(parents=True, exist_ok=True)
        html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{background:#f6f8fb;margin:0;padding:24px;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto}}
.card{{max-width:900px;margin:0 auto;background:#fff;border-radius:14px;border:1px solid #e9eef5;
box-shadow:0 6px 22px rgba(16,24,40,0.08);padding:10px}}
.card img{{width:100%;display:block;border-radius:10px}}
.hint{{color:#6b7a86;font-size:12px;margin:8px 2px}}
</style></head>
<body>
  <div class="card">
    <img src="{out_png_path.as_posix()}" alt="Tendencia de bienestar">
    <div class="hint">Autopercibido (área verde) vs. Neto (línea discontinua).</div>
  </div>
</body></html>"""
        out_html_path.write_text(html, encoding="utf-8")

    plt.close(fig)

    return {
        "png_path": out_png_path.as_posix(),
        "svg_path": out_svg_path.as_posix() if out_svg else None,
        "html_path": out_html_path.as_posix() if out_html else None,
    }

from matplotlib.dates import DateFormatter
'''
    Gráfico de sueño, bienestar percibido y malestar
'''
def grafico_sueno_y_bienestar_seaborn(
    df: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str] = None,
    *,
    date_col: str = "fecha",
    night_col: str = "sueno_noche_h",              # horas de sueño nocturno
    siesta_cols: Tuple[str, ...] = ("siesta_manana_h","siesta_tarde_h","siesta_otros_h"),
    siesta_total_col: str = "siesta_total_h",      # si la tienes ya calculada
    siesta_min_col: str = "siesta_min",            # fallback en minutos (se dividirá por 60)
    bienestar_col: str = "bienestar_ema",
    malestar_col: str = "malestar_ema",
    # criterios de riesgo
    target_h: float = 7.5,               # objetivo de horas totales
    noct_min_risk: float = 4.5,          # nocturno < 4.5h => riesgo
    total_dev_risk: float = 2.0,         # |total - objetivo| > 2h => riesgo
    frag_col: str = "sleep_episodes",    # si existe y >=3 => riesgo
    frag_risk_threshold: int = 3,
    # tamaño/estética
    width_px: int = 1100,
    height_px: int = 700,
    dpi: int = 200,
    font_scale=0.8
):
    dfi = df.copy()

    # --- fechas ordenadas y seguras ---
    if date_col not in dfi.columns:
        raise ValueError(f"Falta columna '{date_col}'")
    if not pd.api.types.is_datetime64_any_dtype(dfi[date_col]):
        dfi[date_col] = pd.to_datetime(dfi[date_col], dayfirst=True, errors="coerce")
    dfi = dfi.sort_values(date_col).reset_index(drop=True)

    # --- columnas de sueño ---
    # nocturno
    noct = pd.to_numeric(dfi.get(night_col), errors="coerce")
    # siestas: intenta total, luego suma de columnas, luego minutos
    if siesta_total_col in dfi.columns:
        siestas = pd.to_numeric(dfi[siesta_total_col], errors="coerce")
    else:
        have_parts = [c for c in siesta_cols if c in dfi.columns]
        if have_parts:
            siestas = sum(pd.to_numeric(dfi[c], errors="coerce").fillna(0.0) for c in have_parts)
        elif siesta_min_col in dfi.columns:
            siestas = pd.to_numeric(dfi[siesta_min_col], errors="coerce")/60.0
        else:
            siestas = pd.Series(0.0, index=dfi.index)
    #noct = noct.fillna(0)
    siestas = siestas.fillna(0.0)
    total_sleep = noct + siestas

    # --- riesgo ---
    risky = (
        (noct < noct_min_risk) |
        (total_sleep.sub(target_h).abs() > total_dev_risk) |
        ((frag_col in dfi.columns) & (pd.to_numeric(dfi[frag_col], errors="coerce").fillna(0) >= frag_risk_threshold))
    )

    # --- bienestar/malestar ---
    bien = pd.to_numeric(dfi.get(bienestar_col), errors="coerce")
    mal  = pd.to_numeric(dfi.get(malestar_col), errors="coerce")

    # --- plotting ---
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    fig_w = max(600, int(width_px)) / dpi
    fig_h = max(320, int(height_px)) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    x = dfi[date_col]
    bar_width = 0.8  # en días (matplotlib maneja fechas en días)

    # colores
    col_noct = "#6aaed6"   # azul medio
    col_sies = "#c7e9c0"   # verde suave
    col_risk = "#e74c3c"   # rojo

    # barras apiladas
    ax.bar(x, noct, width=bar_width, color=col_noct, label="Sueño nocturno (h)")
    ax.bar(x, siestas, bottom=noct, width=bar_width, color=col_sies, label="Siestas (h)")

    # resaltar días de riesgo: contorno rojo alrededor del total
    if risky.any():
        ax.bar(x[risky], total_sleep[risky], width=bar_width, bottom=0,
               facecolor="none", edgecolor=col_risk, linewidth=1.8, label="Día de riesgo")

    # eje y primario (horas)
    ymax = float(np.nanmax(total_sleep)) if np.isfinite(np.nanmax(total_sleep)) else 0
    ymax = max(8.0, min(14.0, ymax * 1.2))
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Horas de sueño (nocturno + siestas)")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.set_xlabel("Fecha")

    # eje secundario (0–10)
    ax2 = ax.twinx()
    if bien.notna().any():
        sns.lineplot(x=x, y=bien, ax=ax2, linewidth=2.0, label="Bienestar percibido", color="#3aa882")
    if mal.notna().any():
        sns.lineplot(x=x, y=mal,  ax=ax2, linewidth=2.0, label="Malestar del día",   color="#c0392b", alpha=0.9, linestyle="--")
    ax2.set_ylim(0, 10)
    ax2.set_ylabel("Escala 0–10")

    # leyenda combinada
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, ncol=2, fontsize=9)

    # título
    #ax.set_title("Sueño nocturno y siestas · Bienestar/Malestar (EMA)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # guardar
    out_png_path = Path(out_png).resolve()
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    out_svg_path = None
    if out_svg:
        out_svg_path = Path(out_svg).resolve()
        fig.savefig(out_svg_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {
        "png_path": out_png_path.as_posix(),
        "svg_path": out_svg_path.as_posix() if out_svg else None,
        "risk_days": dfi.loc[risky, date_col].dt.strftime("%Y-%m-%d").tolist()
    }


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

def _to_minutes(dt):
    return dt.dt.hour*60 + dt.dt.minute + dt.dt.second/60

def _as_series(x, index, dtype=float):
    import pandas as pd, numpy as np
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").astype(dtype)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.Series(np.nan, index=index, dtype=dtype)
    # escalar, array o similar: conviértelo a Series alineada al índice
    try:
        xv = pd.to_numeric(x, errors="coerce")
    except Exception:
        xv = np.nan
    return pd.Series(xv, index=index, dtype=dtype)


def _parse_horas_simple(s, date_col):
    """Parsea horas comunes (23:15, 11:40 pm, 23h15, 23.15, '11 PM') y las combina con la fecha."""
    st = (s.astype(str).str.strip().str.lower()
            .str.replace(r'@', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'hs\b|hrs\b|hr\b', '', regex=True)
            .str.replace(r'(\d{1,2})h(\d{2})', r'\1:\2', regex=True)
            .str.replace(r'\.', ':', regex=True)
            .str.replace(r'\ba\.?\s*m\.?\b', 'am', regex=True)
            .str.replace(r'\bp\.?\s*m\.?\b', 'pm', regex=True))
    res = pd.Series(pd.NaT, index=st.index, dtype='datetime64[ns]')
    for fmt in ("%H:%M:%S","%H:%M","%H%M","%H","%I:%M:%S %p","%I:%M %p","%I%M %p","%I %p"):
        m = res.isna()
        if not m.any(): break
        parsed = pd.to_datetime(st[m], format=fmt, errors="coerce")
        res.loc[m] = parsed
    d = pd.to_datetime(date_col, errors="coerce").dt.normalize()
    td = (pd.to_timedelta(res.dt.hour.fillna(0).astype(int), unit="h") +
          pd.to_timedelta(res.dt.minute.fillna(0).astype(int), unit="m") +
          pd.to_timedelta(res.dt.second.fillna(0).astype(int), unit="s"))
    return (d + td)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

def _as_series(x, index, dtype=float):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").astype(dtype)
    if x is None:
        return pd.Series(np.nan, index=index, dtype=dtype)
    try:
        xv = pd.to_numeric(x, errors="coerce")
    except Exception:
        xv = np.nan
    return pd.Series(xv, index=index, dtype=dtype)

def _parse_horas_simple(s, date_col):
    st = (s.astype(str).str.strip().str.lower()
            .str.replace(r'@', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'hs\b|hrs\b|hr\b', '', regex=True)
            .str.replace(r'(\d{1,2})h(\d{2})', r'\1:\2', regex=True)
            .str.replace(r'\.', ':', regex=True)
            .str.replace(r'\ba\.?\s*m\.?\b', 'am', regex=True)
            .str.replace(r'\bp\.?\s*m\.?\b', 'pm', regex=True))
    res = pd.Series(pd.NaT, index=st.index, dtype='datetime64[ns]')
    for fmt in ("%H:%M:%S","%H:%M","%H%M","%H","%I:%M:%S %p","%I:%M %p","%I%M %p","%I %p"):
        m = res.isna()
        if not m.any(): break
        parsed = pd.to_datetime(st[m], format=fmt, errors="coerce")
        res.loc[m] = parsed
    d = pd.to_datetime(date_col, errors="coerce").dt.normalize()
    td = (pd.to_timedelta(res.dt.hour.fillna(0).astype(int), unit="h") +
          pd.to_timedelta(res.dt.minute.fillna(0).astype(int), unit="m") +
          pd.to_timedelta(res.dt.second.fillna(0).astype(int), unit="s"))
    return (d + td)

def grafico_actograma_sueno(
    df: pd.DataFrame,
    out_png: str,
    out_svg: str | None = None,
    *,
    date_col="fecha",
    main_start_col="hora_dormir_dt",
    main_end_col="hora_levantar_dt",
    nap1_start_col=None, nap1_end_col=None,
    nap2_start_col=None, nap2_end_col=None,
    siesta_am_col="siesta_manana_h",
    siesta_pm_col="siesta_tarde_h",
    siesta_ot_col="siesta_otros_h",
    anchor_am=10.5*60, anchor_pm=15.5*60, anchor_ot=19.0*60,
    fallback_night_duration_col="sueno_noche_h",
    fallback_wakeup_anchor_min=7.5*60,
    noct_min_risk=4.5, reh_min_ok=6.5,
    font_scale=0.85, width=1100, height=520, dpi=200
):
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)

    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=font_scale)

    # --- asegurar columnas de sueño principal ---
    have_dt_cols = (main_start_col in d.columns) and (main_end_col in d.columns)
    if not have_dt_cols:
        # ¿tenemos texto?
        has_text = ("hora_dormir" in d.columns) and ("hora_levantar" in d.columns)
        if has_text:
            d["__hora_dormir_dt__"] = _parse_horas_simple(d["hora_dormir"], d[date_col])
            d["__hora_levantar_dt__"] = _parse_horas_simple(d["hora_levantar"], d[date_col])
            main_start_col = "__hora_dormir_dt__"
            main_end_col   = "__hora_levantar_dt__"
        else:
            # fallback: solo duración nocturna → intervalo que termina a 07:30
            dur = _as_series(d.get(fallback_night_duration_col), d.index)
            wake = pd.to_datetime(d[date_col].dt.normalize(), errors="coerce") + \
                   pd.to_timedelta(int(fallback_wakeup_anchor_min)//60, unit="h") + \
                   pd.to_timedelta(int(fallback_wakeup_anchor_min)%60, unit="m")
            d["__hora_levantar_dt__"] = wake
            d["__hora_dormir_dt__"] = (wake - pd.to_timedelta(dur.fillna(0), unit="h"))
            main_start_col = "__hora_dormir_dt__"
            main_end_col   = "__hora_levantar_dt__"

    # --- segmentos nocturnos ---
    def _intervals_from_main(row, s_col, e_col):
        s = row.get(s_col, pd.NaT)
        e = row.get(e_col, pd.NaT)
        if pd.isna(s) or pd.isna(e):
            return []
        s_h = s.hour + s.minute/60
        e_h = e.hour + e.minute/60
        if e < s:  # cruza medianoche
            return [(s_h*60, 24*60), (0, e_h*60)]
        return [(s_h*60, e_h*60)]

    d["_main_segments"] = d.apply(lambda r: _intervals_from_main(r, main_start_col, main_end_col), axis=1)

    # --- siestas con intervalos reales (si existen) ---
    nap_segments_cols = []
    for a,b in [(nap1_start_col, nap1_end_col), (nap2_start_col, nap2_end_col)]:
        if a and b and a in d.columns and b in d.columns:
            A = pd.to_datetime(d[a], errors="coerce"); B = pd.to_datetime(d[b], errors="coerce")
            segs = []
            for i in range(len(d)):
                if pd.notna(A.iloc[i]) and pd.notna(B.iloc[i]):
                    ah, bh = A.iloc[i].hour + A.iloc[i].minute/60, B.iloc[i].hour + B.iloc[i].minute/60
                    segs.append([(ah*60, 24*60), (0, bh*60)] if B.iloc[i] < A.iloc[i] else [(ah*60, bh*60)])
                else:
                    segs.append([])
            colname = f"_nap_segments_{a}"
            d[colname] = segs
            nap_segments_cols.append(colname)

    # --- siestas estimadas por bloques (si no hay intervalos) ---
    am_h = _as_series(d.get(siesta_am_col), d.index)
    pm_h = _as_series(d.get(siesta_pm_col), d.index)
    ot_h = _as_series(d.get(siesta_ot_col), d.index)

    def _centered(anchor, dur_h):
        dur_m = max(0.0, float(dur_h))*60.0
        return (anchor - dur_m/2, anchor + dur_m/2)

    d["_nap_est_segments"] = [[] for _ in range(len(d))]
    for i in range(len(d)):
        segs = []
        if (siesta_am_col in d.columns) and pd.notna(am_h.iloc[i]) and am_h.iloc[i] > 0:
            a,b = _centered(anchor_am, am_h.iloc[i]); segs.append((max(a,0), min(b,24*60)))
        if (siesta_pm_col in d.columns) and pd.notna(pm_h.iloc[i]) and pm_h.iloc[i] > 0:
            a,b = _centered(anchor_pm, pm_h.iloc[i]); segs.append((max(a,0), min(b,24*60)))
        if (siesta_ot_col in d.columns) and pd.notna(ot_h.iloc[i]) and ot_h.iloc[i] > 0:
            a,b = _centered(anchor_ot, ot_h.iloc[i]); segs.append((max(a,0), min(b,24*60)))
        d.at[i, "_nap_est_segments"] = segs

    # --- RIESGO robusto ---
    # --- RIESGO robusto (con wrap a medianoche) ---
    end_dt   = pd.to_datetime(d.get(main_end_col),   errors="coerce")
    start_dt = pd.to_datetime(d.get(main_start_col), errors="coerce")

    # Si el fin es <= inicio (cruza medianoche), suma 1 día al fin
    wrap = end_dt.notna() & start_dt.notna() & (end_dt <= start_dt)
    end_fix = end_dt.where(~wrap, end_dt + pd.Timedelta(days=1))

    # Duración nocturna (h)
    noct_h = (end_fix - start_dt).dt.total_seconds() / 3600.0

    # Fallback: usa 'sueno_noche_h' donde falte
    if "sueno_noche_h" in d.columns:
        fallback_noct = pd.to_numeric(d["sueno_noche_h"], errors="coerce")
        noct_h = noct_h.where(noct_h.notna(), fallback_noct)

    # Sumas de siestas (ya son Series por _as_series)
    si_sum = am_h.fillna(0) + pm_h.fillna(0) + ot_h.fillna(0)

    # Total efectivo con pesos circadianos
    total_h_eff = noct_h.fillna(0) + am_h.fillna(0)*0.6 + pm_h.fillna(0)*0.85 + ot_h.fillna(0)*0.75

    # Marca riesgo SOLO si hay señal: (noct válido >0) o (hay siestas)
    valid_day = ((noct_h.notna() & (noct_h > 0)) | (si_sum > 0))

    risky_raw = ((noct_h.notna() & (noct_h < noct_min_risk)) | (total_h_eff < reh_min_ok))
    risky = valid_day & risky_raw


    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(max(800,width)/dpi, max(400,height)/dpi), dpi=dpi)
    ax.set_facecolor("white")
    y_positions = np.arange(len(d))
    y_height = 0.75
    col_night = "#6aaed6"; col_nap = "#c7e9c0"; col_risk = "#e74c3c"

    for i, y in enumerate(y_positions):
        if bool(risky.iloc[i]):
            ax.add_patch(Rectangle((0, y - y_height/2), 24*60, y_height, facecolor="#fdecea", edgecolor="none", zorder=0))
        for (a,b) in d["_main_segments"].iloc[i]:
            ax.broken_barh([(a, b-a)], (y - y_height/2, y_height), facecolors=col_night, edgecolors="none", zorder=2)
        any_nap_real = False
        for colname in nap_segments_cols:
            for (a,b) in d[colname].iloc[i]:
                any_nap_real = True
                ax.broken_barh([(a, b-a)], (y - y_height/2, y_height), facecolors=col_nap, edgecolors="none", zorder=3)
        if not any_nap_real:
            for (a,b) in d["_nap_est_segments"].iloc[i]:
                ax.broken_barh([(a, b-a)], (y - y_height/2, y_height), facecolors=col_nap, edgecolors="none", zorder=3)

    ax.set_xlim(0, 24*60)
    ax.set_ylim(-0.5, len(d)-0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(d[date_col].dt.strftime("%d %b"))
    ax.set_xlabel("Hora del día")
    ax.set_title("Actograma simplificado (noche + siestas)")

    xticks = np.arange(0, 25, 3)*60
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=col_night, lw=8, label="Sueño nocturno"),
        Line2D([0],[0], color=col_nap,   lw=8, label="Siestas"),
        Line2D([0],[0], color=col_risk,  lw=3, label="Día riesgoso", linestyle="-")
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return {"png_path": out_png, "svg_path": out_svg,
            "risky_days": d.loc[risky, date_col].dt.strftime("%Y-%m-%d").tolist()}



def grafico_ptn_vs_sueno_siguiente(
    df: pd.DataFrame,
    out_png: str,
    out_svg: str | None = None,
    *,
    date_col="fecha",
    ptn_col="PTN_today_adj",           # o "PTN_today"
    sleep_col_pref=("sleep_reh_adj","sueno_noche_h"),
    font_scale=0.9,
    sleep_low_h=6.0,
    width=900, height=520, dpi=200,
    lowess=False
):
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)

    # elegir columna de sueño
    ycol = None
    for c in sleep_col_pref:
        if c in d.columns:
            ycol = c; break
    if ycol is None:
        raise ValueError("No se encontró ninguna columna de sueño para Y.")

    # construir pares (PTN hoy -> Sueño mañana)
    d["_y_next"] = d[ycol].shift(-1)
    d = d[[date_col, ptn_col, "_y_next"]].dropna()

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    fig, ax = plt.subplots(figsize=(max(700,width)/dpi, max(400,height)/dpi), dpi=dpi)

    risky = d["_y_next"] < sleep_low_h

    # dispersión con color por riesgo
    sns.scatterplot(data=d, x=ptn_col, y="_y_next", hue=risky, palette={True:"#e74c3c", False:"#2c3e50"},
                    ax=ax, s=40, edgecolor="white", linewidth=0.6, legend=False)

    # regresión (lineal o lowess)
    sns.regplot(data=d, x=ptn_col, y="_y_next", ax=ax, scatter=False,
                lowess=lowess, line_kws={"linewidth":2.0, "alpha":0.9})

    ax.axhline(sleep_low_h, color="#e74c3c", linestyle="--", linewidth=1, label=f"Umbral {sleep_low_h:.1f} h")
    ax.set_xlabel("Pantallas nocturnas (PTN hoy)")
    ax.set_ylabel("Sueño del día siguiente (h)")
    ax.set_title("Pantallas (hoy) vs Sueño (mañana)")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return {"png_path": out_png, "svg_path": out_svg}


from pathlib import Path
from typing import Sequence, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def grafico_cm_last7(
    df: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str] = None,
    *,
    date_col: str = "fecha",
    comp_cols: Sequence[str] = ("s_sleep","s_gly","s_alc","s_caf","s_mov","s_nic","s_thc","s_alim","s_hyd","s_stim"),
    kind: str = "stacked",         # "stacked" | "heatmap" | "multiples" | "area_suave"
    mode: str = "raw",             # "raw" | "percent" | "weighted"
    weights: Optional[dict] = None,# usado solo si mode="weighted"
    font_scale: float = 0.9,
    width_px: int = 1000,
    height_px: int = 460,
    dpi: int = 200
):
    """
    Representa CM por componentes para los últimos 7 días.
    - kind="stacked": barras apiladas por día (ideal para pocos días).
    - kind="heatmap": mapa de calor componentes×día (0–10).
    - kind="multiples": small multiples (mini-series por componente).
    - kind="area_suave": áreas apiladas suavizadas (rolling=3).
    - mode:
        - "raw": apila/visualiza s_* tal cual (0–10).
        - "percent": composición % por día (suma=100).
        - "weighted": contribución ponderada (renormalizada por faltantes).
    Evita fillna(0) global: sólo rellena dentro de días válidos.
    """
    d = df.copy()
    if date_col not in d.columns:
        raise ValueError(f"Falta columna '{date_col}'")
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")

    # Orden y últimos 7 días (1 registro por día)
    d = (d.sort_values(date_col)
           .drop_duplicates(subset=[date_col], keep="last"))
    d_last7 = d.tail(7).reset_index(drop=True)

    # Componentes disponibles (numéricos)
    cols_present = [c for c in comp_cols if c in d_last7.columns]
    if not cols_present:
        raise ValueError("No hay columnas s_* presentes en el DataFrame para graficar.")

    comps = pd.DataFrame({c: pd.to_numeric(d_last7[c], errors="coerce") for c in cols_present})
    # Días válidos: al menos un componente con dato
    valid = comps.notna().any(axis=1)
    d_plot = d_last7.loc[valid].reset_index(drop=True)
    comps = comps.loc[valid].reset_index(drop=True)

    if len(d_plot) == 0:
        return {"png_path": None, "svg_path": None, "message": "No hay datos en los últimos 7 días."}

    # Modo de agregación
    if mode == "raw":
        M = comps.fillna(0.0)
        y_label = "Score por componente (0–10)"
        ylim = (0, max(10.0, float(M.sum(axis=1).max()) * 1.05))
    elif mode == "percent":
        row_sum = comps.sum(axis=1, skipna=True).replace(0, np.nan)
        M = comps.div(row_sum, axis=0).fillna(0.0) * 100.0
        y_label = "Composición (%)"
        ylim = (0, 100)
    elif mode == "weighted":
        if weights is None:
            weights = {"s_sleep":0.26,"s_gly":0.14,"s_alc":0.12,"s_caf":0.10,
                       "s_mov":0.12,"s_nic":0.07,"s_thc":0.05,"s_alim":0.07,"s_hyd":0.05,"s_stim":0.02}
        W = pd.Series(weights, dtype=float).reindex(M.columns).fillna(0.0)
        present = comps.notna() & (W > 0)
        sumw = present.mul(W, axis=1).sum(axis=1).replace(0, np.nan)
        M = comps.mul(W, axis=1).div(sumw, axis=0).fillna(0.0)
        y_label = "Contribución ponderada (0–10)"
        ylim = (0, max(10.0, float(M.sum(axis=1).max()) * 1.05))
    else:
        raise ValueError("mode debe ser 'raw', 'percent' o 'weighted'.")

    # Filtra componentes completamente silenciosos (todo 0) para no “pintar verde”
    nonzero_cols = [c for c in M.columns if (M[c].abs() > 1e-6).any()]
    if not nonzero_cols:
        # todo cero → nada que apilar
        nonzero_cols = list(M.columns)

    M = M[nonzero_cols]
    labels = [c.replace("s_", "") for c in nonzero_cols]
    x = d_plot[date_col]
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    fig_w = max(700, width_px) / dpi
    fig_h = max(380, height_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if kind == "stacked":
        palette = sns.color_palette("tab20", n_colors=len(nonzero_cols))
        bottom = np.zeros(len(M), dtype=float)
        for i, c in enumerate(nonzero_cols):
            vals = M[c].to_numpy()
            ax.bar(x, vals, bottom=bottom, color=palette[i], edgecolor="white", linewidth=0.5, label=labels[i])
            bottom = bottom + vals
        ax.set_ylim(*ylim)
        ax.set_ylabel(y_label)
        ax.set_title("CM por componentes (últimos 7 días) · Barras apiladas")

    elif kind == "heatmap":
        # matriz componentes × fecha
        mat = M.copy()
        mat.index = x.dt.strftime("%d %b")  # fechas como filas temporales si prefieres transpón
        mat = mat.set_index(mat.index)
        mat_hm = mat.T  # comp × día
        cmap = sns.color_palette("light:#5A9", as_cmap=True)  # paleta legible
        vmin, vmax = (0, 10) if mode in ("raw","weighted") else (0, 100)
        sns.heatmap(mat_hm, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"shrink":0.8})
        ax.set_xlabel("Día")
        ax.set_ylabel("Componente")
        ax.set_title("CM por componentes (últimos 7 días) · Heatmap")
        ax.set_yticklabels([lab.replace("s_", "") for lab in mat_hm.index])

    elif kind == "multiples":
        n = len(nonzero_cols)
        ncols = 3 if n >= 3 else n
        nrows = int(math.ceil(n / ncols))
        fig.clear()
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi, sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        palette = sns.color_palette("tab10", n_colors=n)
        for i, c in enumerate(nonzero_cols):
            ax_i = axes[i]
            sns.lineplot(x=x, y=M[c], ax=ax_i, color=palette[i], marker="o", linewidth=1.5)
            ax_i.set_title(c.replace("s_", ""))
            ax_i.set_ylim(*((0, 10) if mode in ("raw","weighted") else (0, 100)))
            ax_i.grid(True, alpha=0.3)
        # apaga subplots vacíos
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        fig.supylabel(y_label)
        fig.suptitle("CM por componentes (últimos 7 días) · Small multiples", y=1.02, fontsize=11, fontweight="bold")
        plt.tight_layout()

    elif kind == "area_suave":
        palette = sns.color_palette("tab20", n_colors=len(nonzero_cols))
        # suaviza con media móvil corta (win=3) para evitar dependencia de statsmodels
        Ms = M.rolling(window=3, min_periods=1, center=True).mean()
        Y = [Ms[c].to_numpy() for c in nonzero_cols]
        ax.stackplot(x, *Y, labels=labels, colors=palette, alpha=0.9, edgecolor="white", linewidth=0.5)
        # bordes entre capas
        cum = np.cumsum(np.row_stack(Y), axis=0)
        for yb in cum:
            ax.plot(x, yb, lw=0.5, color="white", alpha=0.9)
        ax.set_ylim(*((0, 10) if mode in ("raw","weighted") else (0, 100)))
        ax.set_ylabel(y_label)
        ax.set_title("CM por componentes (últimos 7 días) · Área apilada suavizada")
    else:
        raise ValueError("kind debe ser 'stacked', 'heatmap', 'multiples' o 'area_suave'.")

    if kind in ("stacked","area_suave"):
        ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)

    plt.tight_layout()
    out_png_path = Path(out_png).resolve()
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    out_svg_path = None
    if out_svg:
        out_svg_path = Path(out_svg).resolve()
        fig.savefig(out_svg_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    comp_candidates = ["s_sleep","s_gly","s_alc","s_caf","s_mov","s_nic","s_thc","s_alim","s_hyd","s_stim"]

    d7 = (df.sort_values("fecha")
            .drop_duplicates(subset=["fecha"], keep="last")
            .tail(7)
            .reset_index(drop=True))

    present = [c for c in comp_candidates if c in d7.columns]
    nz_per_comp = {c: int((pd.to_numeric(d7[c], errors="coerce").fillna(0).abs()>1e-6).sum()) for c in present}
    print("present:", present)
    print("no-cero (últimos 7d):", nz_per_comp)

    return {
        "png_path": out_png_path.as_posix(),
        "svg_path": out_svg_path.as_posix() if out_svg else None,
        "kind": kind,
        "mode": mode,
        "used_components": labels,
        "days": d_plot[date_col].dt.strftime("%Y-%m-%d").tolist()
    }


def grafico_cm_waterfall(
    df: pd.DataFrame,
    out_png: str,
    out_svg: str | None = None,
    *,
    date_col="fecha",
    cm_col="carga_metabolica",
    comp_cols=("s_sleep","s_gly","s_alc","s_caf","s_mov","s_nic","s_thc","s_alim","s_hyd","s_stim"),
    weights=None,
    ref_date=None,         # si None, usa el último día con datos
    font_scale=0.95,
    width=880, height=520, dpi=200
):
    if weights is None:
        weights = {"s_sleep":0.26,"s_gly":0.14,"s_alc":0.12,"s_caf":0.10,
                   "s_mov":0.12,"s_nic":0.07,"s_thc":0.05,"s_alim":0.07,"s_hyd":0.05,"s_stim":0.02}

    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)

    # elegir índice del día de referencia
    if ref_date is None:
        i = d[cm_col].last_valid_index()
    else:
        ref_date = pd.to_datetime(ref_date, errors="coerce")
        i = d.index[d[date_col] == ref_date]
        i = int(i[0]) if len(i) else d[cm_col].last_valid_index()
    if i is None or i == 0:
        raise ValueError("No hay suficiente historial para calcular Δ (se necesita día previo).")

    # util renormalización por fila
    W = pd.Series(weights)
    def row_contrib(row):
        s = pd.to_numeric(row[comp_cols], errors="coerce")
        present = s.notna()
        wsum = (W[present.index] * present).sum()
        if wsum == 0: return pd.Series({c:0.0 for c in comp_cols})
        ww = (W / wsum)
        return (s.fillna(0) * ww).clip(lower=0, upper=10)

    c_prev = row_contrib(d.loc[i-1])
    c_curr = row_contrib(d.loc[i])

    deltas = (c_curr - c_prev).dropna()
    deltas = deltas[deltas != 0]
    # ordenar por impacto
    deltas = deltas.reindex(deltas.abs().sort_values(ascending=False).index)

    cm_prev = float(pd.to_numeric(d.loc[i-1, cm_col], errors="coerce")) if cm_col in d.columns else float(c_prev.sum())
    cm_curr = float(pd.to_numeric(d.loc[i, cm_col], errors="coerce")) if cm_col in d.columns else float(c_curr.sum())
    delta_total = cm_curr - cm_prev

    # construir waterfall
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    fig, ax = plt.subplots(figsize=(max(700,width)/dpi, max(400,height)/dpi), dpi=dpi)

    labels = ["CM (t-1)"] + [k.replace("s_","") for k in deltas.index] + ["CM (t)"]
    values = [cm_prev] + deltas.tolist() + [cm_curr]

    # posiciones y acumulado
    cum = [cm_prev]
    for v in deltas:
        cum.append(cum[-1] + v)
    # barras
    x = np.arange(len(labels))
    colors = []
    bars = []

    # barra inicial
    bars.append(ax.bar(x[0], cm_prev, color="#95a5a6"))
    colors.append("#95a5a6")

    running = cm_prev
    for j, comp in enumerate(deltas.index, start=1):
        v = deltas.loc[comp]
        color = "#27ae60" if v < 0 else "#e74c3c"
        colors.append(color)
        bars.append(ax.bar(x[j], v, bottom=running if v>=0 else running+v, color=color))
        running += v

    # barra final
    bars.append(ax.bar(x[-1], cm_curr, color="#34495e"))
    colors.append("#34495e")

    ax.axhline(0, color="#bdc3c7", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("CM (0–10)")
    t0 = d.loc[i-1, date_col].strftime("%d %b")
    t1 = d.loc[i, date_col].strftime("%d %b")
    ax.set_title(f"Waterfall de cambio de CM · {t0} → {t1} (Δ={delta_total:+.2f})")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return {"png_path": out_png, "svg_path": out_svg, "delta_total": round(delta_total,2), "top_changes": deltas.head(5).round(2).to_dict()}
