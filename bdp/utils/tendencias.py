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
