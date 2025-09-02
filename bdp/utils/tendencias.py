import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MaxNLocator

def generar_grafico_tendencias(df, out_path, out_path_rounded):
    dates = df["fecha"]
    x = np.arange(len(dates))
    y = pd.to_numeric(df["bienestar"], errors="coerce").to_numpy(dtype=float)
    y_ema = pd.to_numeric(df["bienestar_ema"], errors="coerce").to_numpy(dtype=float)

    # Upsample y suavizado ligero para líneas más suaves (interpolación + convolution)
    if len(x) >= 3:
        xi = np.linspace(x.min(), x.max(), max(200, len(x)*40))
        yi = np.interp(xi, x, np.nan_to_num(y, nan=np.nanmean(y)))
        yi_ema = np.interp(xi, x, np.nan_to_num(y_ema, nan=np.nanmean(y_ema)))
        # aplicar suavizado con ventana gaussiana (convolution)
        def smooth(arr, window_len=9):
            if window_len < 3:
                return arr
            win = np.exp(-0.5 * (np.linspace(-2,2,window_len))**2)
            win = win / win.sum()
            return np.convolve(arr, win, mode='same')
        yi_s = smooth(yi, window_len=9)
        yi_ema_s = smooth(yi_ema, window_len=7)
    else:
        xi = x
        yi_s = y
        yi_ema_s = y_ema

    # Palette "semi-Airbnb": acento cálido+suave verde/azul
    accent_line = "#3aa882"    # verde menta/teal
    muted_line = "#56687a"     # gris azulado
    accent_fill = "#e9f7f2"    # muy suave green background for area
    muted_grid = "#f0f4f8"

    # Crear figura
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9,3.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Área suave bajo EMA
    ax.fill_between(xi, yi_ema_s, yi_ema_s.min()-0.5, alpha=0.9, color=accent_fill, linewidth=0)

    # Líneas
    ax.plot(xi, yi_s, linewidth=1.25, color=muted_line, label="Bienestar (diario)")
    ax.plot(xi, yi_ema_s, linewidth=2.2, color=accent_line, label="Tendencia (EMA)")

    # Marcar últimos puntos
    if len(x) >= 1:
        ax.scatter([x[-1]], [y[-1]], s=28, color=muted_line, zorder=5)
        ax.scatter([x[-1]], [y_ema[-1]], s=44, color=accent_line, zorder=6, edgecolor='white', linewidth=1)

    # Etiqueta del último valor EMA
    try:
        last_val = float(y_ema[-1])
        label = f"EMA {last_val:.2f}"
        ax.annotate(label, xy=(x[-1], y_ema[-1]), xytext=(10, 8),
                    textcoords="offset points", fontsize=9, fontweight=700, color=accent_line)
    except Exception:
        pass

    # Ejes y estética minimalista
    ax.set_xlim(xi.min(), xi.max())
    ax.set_ylim(min(np.nanmin(yi_s)-0.8, np.nanmin(yi_ema_s)-0.8), max(np.nanmax(yi_s)+0.8, np.nanmax(yi_ema_s)+0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color("#e6eef6")
    ax.tick_params(axis='y', which='both', left=False, labelleft=True)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("")

    # X ticks labels as short dates
    xticks_idx = np.linspace(0, len(dates)-1, min(6, len(dates))).astype(int)
    xticks_labels = [dates.iloc[i].strftime("%d %b") for i in xticks_idx]
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xticks_labels, fontsize=9, color="#6b7a86")

    # Y ticks
    ax.set_yticks(np.linspace(0,10,6))
    ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(0,10,6)], fontsize=9, color="#6b7a86")

    # Thin horizontal grid
    for ygrid in np.linspace(0,10,6):
        ax.hlines(y=ygrid, xmin=xi.min(), xmax=xi.max(), colors=muted_grid, linewidth=0.8, zorder=0)

    # Legend minimal
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    plt.tight_layout(pad=8)
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Intentar crear versión con esquinas redondeadas (PIL)
    try:
        from PIL import Image, ImageDraw, ImageFilter
        im = Image.open(out_path).convert("RGBA")
        w,h = im.size
        radius = int(min(w,h) * 0.035)  # ~3.5% del tamaño
        # Crear máscara redondeada
        mask = Image.new("L", (w,h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0,0),(w,h)], radius=radius, fill=255)
        # Aplicar máscara
        rounded = Image.new("RGBA", (w,h), (255,255,255,0))
        rounded.paste(im, (0,0), mask=mask)
        # Añadir ligera sombra/outline
        shadow = Image.new("RGBA", (w+8, h+8), (0,0,0,0))
        shadow_draw = ImageDraw.Draw(shadow)
        # soft shadow by blur
        shadow_draw.rounded_rectangle([(4,4),(w+4,h+4)], radius=radius, fill=(0,0,0,50))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
        shadow.paste(rounded, (0,0), rounded)
        shadow.save(out_path_rounded, format="PNG")
        rounded_created = True
    except Exception as e:
        # PIL no disponible o fallo; ignoramos la versión redondeada
        rounded_created = False
