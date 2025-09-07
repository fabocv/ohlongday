from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def grafico_cm_sparkline_seaborn(
    df: pd.DataFrame,
    out_png: str,
    out_svg: Optional[str] = None,
    date_col: str = "fecha",
    cm_col: str = "carga_metabolica",
    width_px: int = 600,
    height_px: int = 100,             # ¡no excede 50px!
    dpi: int = 300,                  # 50px = 0.25" a 200 dpi
    y_domain: Tuple[float, float] = (0, 10),
    font_scale = 0.8,
    show_band: bool = True           # banda “óptimo” opcional (CM bajo)
) -> Dict[str, Optional[str]]:
    """
    Dibuja un sparkline ultra-compacto de CM (0–10).
    - Altura en píxeles limitada a <=50.
    - Sin ejes, sin ticks, solo línea + área + último valor.
    """
    if date_col not in df.columns or cm_col not in df.columns:
        raise ValueError(f"Faltan columnas '{date_col}' o '{cm_col}'")

    # Parseo y orden
    dfi = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dfi[date_col]):
        dfi[date_col] = pd.to_datetime(dfi[date_col], dayfirst=True, errors="coerce")
    dfi[cm_col] = pd.to_numeric(dfi[cm_col], errors="coerce").astype(float)
    dfi = dfi.sort_values(date_col).reset_index(drop=True)

    sns.set_context("paper", font_scale=font_scale)

    # Filtrar NaN completos
    if dfi[cm_col].notna().sum() == 0:
        # Crear una imagen vacía mínima que diga “Sin datos”
        fig, ax = plt.subplots(figsize=(max(1, width_px)/dpi, min(height_px, 100)/dpi), dpi=dpi)
        ax.text(0.5, 0.5, "CM sin datos", ha="center", va="center", fontsize=8, color="#6b7a86")
        ax.axis("off")
    else:
        # Estilo
        sns.set_theme(style="white", rc={
            "axes.spines.right": False, "axes.spines.top": False,
            "axes.spines.left": False, "axes.spines.bottom": False
        })

        # Tamaño físico según píxeles (altura limitada)
        height_px = min(int(height_px), 100)
        fig_w = max(1, int(width_px)) / dpi
        fig_h = height_px / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        # Banda opcional (CM bajo “mejor”): 0–3 verde pálido
        if show_band:
            ax.axhspan(0, 3, color="#e9f7f2", zorder=0)

        # Área + línea
        x = dfi[date_col]
        y = dfi[cm_col].clip(*y_domain)

        # área
        ax.fill_between(x, 0, y, where=y.notna(), alpha=0.25, color="#c0392b", linewidth=0)  # rojo suave
        # línea
        sns.lineplot(data=dfi, x=date_col, y=cm_col, ax=ax, linewidth=1.6, color="#c0392b")

        # Último punto + etiqueta compacta
        last_valid = dfi[cm_col].last_valid_index()
        if last_valid is not None:
            x_last = dfi.loc[last_valid, date_col]
            y_last = float(np.clip(dfi.loc[last_valid, cm_col], *y_domain))
            ax.scatter([x_last], [y_last], s=14, color="#c0392b", edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(
                f"{y_last:.1f}",
                xy=(x_last, y_last),
                xytext=(2, 2),
                textcoords="offset points",
                color="#c0392b",
                fontsize=4,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        # Limpieza total de ejes
        ax.set_ylim(*y_domain)
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.margins(x=0)

        plt.tight_layout(pad=0)

    # Guardados
    out_png_path = Path(out_png).resolve()
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight", pad_inches=0)

    out_svg_path = None
    if out_svg:
        out_svg_path = Path(out_svg).resolve()
        fig.savefig(out_svg_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    return {"png_path": out_png_path.as_posix(),
            "svg_path": out_svg_path.as_posix() if out_svg else None}
