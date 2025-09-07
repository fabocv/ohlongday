# ==================== BDP Dash Pro (seaborn) ====================
from pathlib import Path
import math, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ---------- Tema estético (tipo "postdoc de Harvard") ----------
def bdp_theme(font_scale: float = 0.95):
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette("tab20")  # variada y accesible
    plt.rcParams.update({
        "axes.facecolor": "white",
        "axes.edgecolor": "#e9eef3",
        "axes.linewidth": 0.8,
        "grid.color": "#eef3f7",
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    })

# ---------------------- Panel A: Actograma ----------------------
def panel_actograma_minimo(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    noct_col="sueno_noche_h",
    am_col="siesta_manana_h",
    pm_col="siesta_tarde_h",
    ot_col="siesta_otros_h",
    wake_anchor_min=7.5*60,    # 07:30
    anchor_am=10.5*60, anchor_pm=15.5*60, anchor_ot=19.0*60,
    noct_min_risk=4.5, reh_min_ok=6.5,   # umbrales
    title="Actograma (noche + siestas)"
):
    d = daily.copy()
    if date_col not in d: raise ValueError(f"Falta {date_col}")
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = (d.sort_values(date_col)
           .drop_duplicates(subset=[date_col], keep="last")
           .tail(7).reset_index(drop=True))

    def _s(col, default=np.nan):
        if col in d: return pd.to_numeric(d[col], errors="coerce")
        return pd.Series(default, index=d.index, dtype=float)

    hN = _s(noct_col).fillna(0.0)
    hA = _s(am_col).fillna(0.0); hP = _s(pm_col).fillna(0.0); hO = _s(ot_col).fillna(0.0)

    def segs_noct(h):
        if pd.isna(h) or h <= 0: return []
        dur = float(h)*60.0; end = float(wake_anchor_min); start = end - dur
        if start >= 0: return [(start, end)]
        return [(0, end), (1440+start, 1440)]  # cruza medianoche

    def seg_center(anchor_min, h):
        if pd.isna(h) or h <= 0: return []
        dur = float(h)*60.0; a = max(0.0, anchor_min - dur/2); b = min(1440.0, anchor_min + dur/2)
        if b <= a: return []
        return [(a, b)]

    night_segments = [segs_noct(hN.iloc[i]) for i in range(len(d))]
    nap_segments   = [seg_center(anchor_am, hA.iloc[i]) + seg_center(anchor_pm, hP.iloc[i]) +
                      seg_center(anchor_ot, hO.iloc[i]) for i in range(len(d))]

    # riesgo sólo si hay señal
    total_eff = hN.fillna(0) + hA*0.6 + hP*0.85 + hO*0.75
    si_sum = (hA+hP+hO)
    valid = (hN.notna() & (hN > 0)) | (si_sum > 0)
    risky = valid & ( (hN.notna() & (hN < noct_min_risk)) | (total_eff < reh_min_ok) )

    y_positions = np.arange(len(d)); y_h = 0.7
    col_night = "#6aaed6"; col_nap = "#c7e9c0"; col_risk = "#fdecea"

    ax.clear(); ax.set_facecolor("white")
    for i, y in enumerate(y_positions):
        if bool(risky.iloc[i]):
            ax.add_patch(Rectangle((0, y - y_h/2), 1440, y_h, facecolor=col_risk, edgecolor="none", zorder=0))
        for (a,b) in night_segments[i]:
            ax.broken_barh([(a, b-a)], (y - y_h/2, y_h), facecolors=col_night, edgecolors="none", zorder=2)
        for (a,b) in nap_segments[i]:
            ax.broken_barh([(a, b-a)], (y - y_h/2, y_h), facecolors=col_nap, edgecolors="none", zorder=3)

    ax.set_xlim(0, 1440); ax.set_ylim(-0.5, len(d)-0.5)
    ax.set_yticks(y_positions); ax.set_yticklabels(d[date_col].dt.strftime("%d %b"))
    xticks = np.arange(0, 25, 3)*60
    ax.set_xticks(xticks); ax.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])
    ax.set_xlabel("Hora del día"); ax.set_title(title)
    sns.despine(ax=ax, left=False, bottom=False)
    # leyenda compacta
    handles = [Line2D([0],[0], color="#6aaed6", lw=6, label="Nocturno"),
               Line2D([0],[0], color="#c7e9c0", lw=6, label="Siestas")]
    ax.legend(handles=handles, loc="upper right", frameon=True, ncol=2)

# --------------- Panel B: PTN → sueño siguiente ----------------

def panel_ptn_vs_sleep_next_v2(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    ptn_candidates=("PTN_today_adj","PTN_d1","tiempo_pantalla_noche_min","pantallas_noche_min"),
    sleep_candidates=("sleep_reh_adj","sueno_noche_h","sueno"),
    prefer_last_days=7, max_lookback=30,
    sleep_low_h=6.0,
    title="Pantallas (hoy) vs Sueño (mañana)",
    lowess=False,
    # NUEVO:
    y_signal_col="sleep_signal_flag",   # si existe, lo usamos
    drop_x_zeros_for_scores=True,       # si PTN es score, x==0 suele ser “sin dato” → filtrar
    convert_min_to_hours=True,
    winsorize_pct=None
):
    d = daily.copy()
    if not pd.api.types.is_datetime64_any_dtype(d.get(date_col)):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)

    ptn_col = next((c for c in ptn_candidates if c in d.columns), None)
    ycol    = next((c for c in sleep_candidates if c in d.columns), None)

    ax.clear()
    if ptn_col is None or ycol is None:
        ax.text(0.5,0.5,"Faltan columnas PTN/sueño para correlación", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return

    def _pairs(df, mode="next"):
        tmp = df[[ptn_col, ycol]].copy()
        tmp[ptn_col] = pd.to_numeric(tmp[ptn_col], errors="coerce")
        yy = pd.to_numeric(tmp[ycol], errors="coerce")
        tmp["_y"] = yy.shift(-1) if mode == "next" else yy

        # FILTRO por señal real de sueño del día siguiente (si existe la columna)
        if y_signal_col in df.columns:
            sig = pd.to_numeric(df[y_signal_col], errors="coerce").shift(-1) if mode=="next" else pd.to_numeric(df[y_signal_col], errors="coerce")
            tmp = tmp.loc[sig == 1]

        tmp = tmp[[ptn_col, "_y"]].dropna()

        # Convertir min→h si corresponde
        if convert_min_to_hours and "min" in ptn_col.lower():
            tmp[ptn_col] = tmp[ptn_col] / 60.0

        # Opcional: quitar x==0 si PTN es score (no minutos)
        if drop_x_zeros_for_scores and ("PTN" in ptn_col) and ("min" not in ptn_col.lower()):
            tmp = tmp.loc[tmp[ptn_col] > 0]

        # Winsorización opcional
        if winsorize_pct and 0 < winsorize_pct < 0.5 and len(tmp) >= 5:
            p = winsorize_pct
            xl, xu = tmp[ptn_col].quantile([p, 1-p]).to_numpy()
            yl, yu = tmp["_y"].quantile([p, 1-p]).to_numpy()
            tmp[ptn_col] = tmp[ptn_col].clip(xl, xu)
            tmp["_y"]    = tmp["_y"].clip(yl, yu)
        return tmp

    used_days = prefer_last_days; note = ""; pairing = "PTN(hoy) → Sueño(mañana)"
    dd = _pairs(d.tail(prefer_last_days), mode="next")

    if len(dd) < 3 and len(d) > prefer_last_days:
        dd2 = _pairs(d.tail(max_lookback), mode="next")
        if len(dd2) >= 3:
            dd = dd2; used_days = min(max_lookback, len(d)); note = f" · ventana ampliada a {used_days} días"

    if len(dd) < 3:
        pairing = "PTN(día) ↔ Sueño(mismo día)"
        dd3 = _pairs(d.tail(prefer_last_days), mode="same")
        if len(dd3) >= 3:
            dd = dd3; used_days = prefer_last_days; note = " · mismo día"
        elif len(d) > prefer_last_days:
            dd4 = _pairs(d.tail(max_lookback), mode="same")
            if len(dd4) >= 3:
                dd = dd4; used_days = min(max_lookback, len(d)); note = f" · mismo día · {used_days}d"

    if dd.empty:
        ax.text(0.5,0.5,"Sin pares PTN↔Sueño en la ventana", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return

    risky = dd["_y"] < sleep_low_h
    sns.scatterplot(data=dd, x=ptn_col, y="_y", hue=risky,
                    palette={True:"#e74c3c", False:"#2c3e50"},
                    ax=ax, s=42, edgecolor="white", linewidth=0.6, legend=False)
    sns.regplot(data=dd, x=ptn_col, y="_y", ax=ax, scatter=False,
                lowess=lowess, line_kws={"linewidth":2, "alpha":0.95})
    ax.axhline(sleep_low_h, color="#e74c3c", linestyle="--", linewidth=1)

    x_is_minutes = ("min" in ptn_col.lower())
    ax.set_xlabel("Pantallas noche (h)" if (convert_min_to_hours and x_is_minutes) else ("Pantallas noche (min)" if x_is_minutes else "PTN (score)"))
    ax.set_ylabel("Sueño (h)")

    x = dd[ptn_col].to_numpy(); y = dd["_y"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[m], y[m])[0,1] if m.sum() >= 2 else np.nan
    slope = np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 2 else np.nan

    info = f"{pairing}{note} · n={len(dd)}"
    if np.isfinite(r):     info += f" · r={r:+.2f}"
    if np.isfinite(slope): info += f" · m={slope:+.2f} h/{'h' if (convert_min_to_hours and x_is_minutes) else ('min' if x_is_minutes else 'ptn')}"
    ax.set_title(f"{title} — {info}")
    sns.despine(ax=ax)


# -------- Panel C: Heatmap CM componentes (últimos 7d) ---------
def panel_cm_heatmap_last7(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    comp_cols=("s_sleep","s_gly","s_alc","s_caf","s_mov","s_nic","s_thc","s_alim","s_hyd","s_stim"),
    mode="raw",  # "raw" | "percent"
    title="CM por componentes (últimos 7 días)"
):
    d = daily.copy()
    if date_col not in d: raise ValueError(f"Falta {date_col}")
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = (d.sort_values(date_col)
           .drop_duplicates(subset=[date_col], keep="last")
           .tail(7).reset_index(drop=True))

    cols = [c for c in comp_cols if c in d.columns]
    if not cols:
        ax.text(0.5,0.5,"No hay s_* para mostrar", ha="center", va="center", transform=ax.transAxes); return

    M = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce") for c in cols})
    # días válidos
    valid = M.notna().any(axis=1)
    M = M.loc[valid].fillna(0.0)
    labels_days = d.loc[valid, date_col].dt.strftime("%d %b")

    if mode == "percent":
        row_sum = M.sum(axis=1).replace(0, np.nan)
        M = M.div(row_sum, axis=0).fillna(0.0) * 100.0
        vmin, vmax = 0, 100
        cbar_label = "Composición (%)"
    else:
        vmin, vmax = 0, 10
        cbar_label = "Score (0–10)"

    mat = M.T  # comp × día
    sns.heatmap(mat, ax=ax, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar_kws={"shrink":0.8, "label":cbar_label})
    ax.set_xlabel("Día"); ax.set_ylabel("Componente"); ax.set_title(title)
    ax.set_xticklabels(labels_days, rotation=0)
    ax.set_yticklabels([lab.replace("s_","") for lab in mat.index])

# ------------- Panel D: Waterfall ΔCM (t-1 → t) ----------------
def panel_cm_waterfall(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    cm_col="carga_metabolica",
    comp_cols=("s_sleep","s_gly","s_alc","s_caf","s_mov","s_nic","s_thc","s_alim","s_hyd","s_stim"),
    weights=None,
    title="Cambio diario de CM (waterfall)"
):
    if weights is None:
        weights = {"s_sleep":0.26,"s_gly":0.14,"s_alc":0.12,"s_caf":0.10,
                   "s_mov":0.12,"s_nic":0.07,"s_thc":0.05,"s_alim":0.07,"s_hyd":0.05,"s_stim":0.02}

    d = daily.copy()
    if date_col not in d: raise ValueError(f"Falta {date_col}")
    if not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)

    # índices últimos dos días válidos de CM
    cm = daily[cm_col]
    idx_valid = cm.dropna().index
    if len(idx_valid) < 2:
        ax.text(0.5,0.5,"No hay 2 días válidos de CM", ha="center", va="center", transform=ax.transAxes); return
    i = idx_valid[-1]; j = idx_valid[-2]

    # contribución renormalizada por fila
    W = pd.Series(weights, dtype=float)
    def contrib_row(row):
        s = pd.Series({c: pd.to_numeric(row.get(c), errors="coerce") for c in comp_cols})
        pres = s.notna() & (W.reindex(s.index).fillna(0) > 0)
        sumw = (W.reindex(s.index).fillna(0)[pres]).sum()
        if sumw == 0: return pd.Series({c:0.0 for c in comp_cols})
        ww = (W / sumw).reindex(s.index).fillna(0)
        return (s.fillna(0) * ww).clip(0, 10)

    c_prev = contrib_row(d.loc[j]); c_curr = contrib_row(d.loc[i])
    deltas = (c_curr - c_prev)
    deltas = deltas[deltas != 0].sort_values(key=lambda s: s.abs(), ascending=False)

    cm_prev = float(cm.loc[j]); cm_curr = float(cm.loc[i]); dlt = cm_curr - cm_prev
    labels = ["CM (t-1)"] + [k.replace("s_","") for k in deltas.index] + ["CM (t)"]
    x = np.arange(len(labels))
    ax.clear()

    # barras
    running = cm_prev
    ax.bar(x[0], cm_prev, color="#95a5a6")
    k = 1
    for comp, v in deltas.items():
        color = "#27ae60" if v < 0 else "#e74c3c"
        bottom = running if v >= 0 else running + v
        ax.bar(x[k], v, bottom=bottom, color=color)
        running += v; k += 1
    ax.bar(x[-1], cm_curr, color="#34495e")

    ax.axhline(0, color="#bdc3c7", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    t0 = d.loc[j, date_col].strftime("%d %b"); t1 = d.loc[i, date_col].strftime("%d %b")
    ax.set_title(f"{title} · {t0} → {t1} (Δ={dlt:+.2f})")
    ax.set_ylabel("CM (0–10)"); sns.despine(ax=ax)

# ---------------- Dashboard: 2 por fila, N filas ----------------
def dashboard_semana_bdp(
    daily: pd.DataFrame,
    out_png: str,
    *,
    panels=("actograma","ptn","cm_heat","waterfall"),
    font_scale=0.95,
    suptitle="Resumen semanal · BDP"
):
    bdp_theme(font_scale=font_scale)
    n = len(panels); cols = 2; rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2*rows), dpi=200, constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(-1)

    k = 0
    for p in panels:
        ax = axes[k]; 
        if p == "actograma":
            #panel_actograma_minimo(daily, ax)
            panel_actograma_minimo_v2(daily, axes[0])
        elif p == "ptn":
            panel_ptn_vs_sleep_next_v2(daily, axes[1])
        elif p == "cm_heat":
            panel_cm_heatmap_last7(daily, ax, mode="raw")
        elif p == "waterfall":
            panel_cm_waterfall(daily, ax)
        else:
            ax.text(0.5,0.5,f"Panel desconocido: {p}", ha="center", va="center", transform=ax.transAxes)
        k += 1

    # apaga ejes sobrantes si n impar
    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=14, fontweight="bold")
    out = Path(out_png).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out.as_posix()
# ================== /BDP Dash Pro (seaborn) ==================


import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def panel_actograma_minimo_v2(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    noct_candidates=("sueno_noche_h","sueno","sueno_horas","sleep_hours"),
    siesta_am="siesta_manana_h",
    siesta_pm="siesta_tarde_h",
    siesta_ot="siesta_otros_h",
    siesta_total_min_candidates=("siesta_min","siestas_min"),
    wake_anchor_min=7.5*60,    # 07:30
    anchor_am=10.5*60, anchor_pm=15.5*60, anchor_ot=19.0*60,
    noct_min_risk=4.5, reh_min_ok=6.5,
    lookback_days=7,
    title="Actograma (noche + siestas)"
):
    col_night = "#6aaed6"
    col_nap   = "#c7e9c0"
    col_shade = "#fdecea"

    d = daily.copy()
    if not pd.api.types.is_datetime64_any_dtype(d.get(date_col)):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = (d.sort_values(date_col)
           .drop_duplicates(subset=[date_col], keep="last"))

    # últimos 7 días
    d7 = d.tail(lookback_days).reset_index(drop=True)

    # elegir columna de nocturno
    noct_col = next((c for c in noct_candidates if c in d7.columns), None)
    noct = pd.to_numeric(d7[noct_col], errors="coerce") if noct_col else pd.Series(np.nan, index=d7.index)

    # siestas por bloque (h)
    am = pd.to_numeric(d7.get(siesta_am), errors="coerce") if siesta_am in d7 else pd.Series(0.0, index=d7.index)
    pm = pd.to_numeric(d7.get(siesta_pm), errors="coerce") if siesta_pm in d7 else pd.Series(0.0, index=d7.index)
    ot = pd.to_numeric(d7.get(siesta_ot), errors="coerce") if siesta_ot in d7 else pd.Series(0.0, index=d7.index)

    # si solo hay total en minutos, úsalo como siesta_tarde por defecto (mejor visible)
    if (am.fillna(0).eq(0) & pm.fillna(0).eq(0) & ot.fillna(0).eq(0)).all():
        tot_col = next((c for c in siesta_total_min_candidates if c in d7.columns), None)
        if tot_col:
            tot_h = pd.to_numeric(d7[tot_col], errors="coerce")/60.0
            pm = tot_h  # centraremos en anchor_pm

    # helpers para segmentos
    def segs_noct(h):
        if pd.isna(h) or h <= 0: return []
        dur = float(h)*60.0; end = float(wake_anchor_min); start = end - dur
        if start >= 0: return [(start, end)]
        return [(0, end), (1440+start, 1440)]

    def seg_center(anchor_min, h):
        if pd.isna(h) or h <= 0: return []
        dur = float(h)*60.0; a = max(0.0, anchor_min - dur/2); b = min(1440.0, anchor_min + dur/2)
        if b <= a: return []
        return [(a, b)]

    night_segments = [segs_noct(noct.iloc[i]) for i in range(len(d7))]
    nap_segments   = [seg_center(anchor_am, am.fillna(0).iloc[i]) +
                      seg_center(anchor_pm, pm.fillna(0).iloc[i]) +
                      seg_center(anchor_ot, ot.fillna(0).iloc[i]) for i in range(len(d7))]

    # riesgo (solo si hay señal)
    total_eff = noct.fillna(0) + am.fillna(0)*0.6 + pm.fillna(0)*0.85 + ot.fillna(0)*0.75
    si_sum = (am.fillna(0)+pm.fillna(0)+ot.fillna(0))
    valid = (noct.notna() & (noct > 0)) | (si_sum > 0)
    risky = valid & ((noct.notna() & (noct < noct_min_risk)) | (total_eff < reh_min_ok))

    # si no hay nada que mostrar, dilo claramente
    ax.clear()
    if not valid.any():
        ax.text(0.5, 0.5, "Sin registros de sueño en los últimos 7 días",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # plot
    handles = [
        Line2D([0],[0], color=col_night, lw=6),
        Line2D([0],[0], color=col_nap,   lw=6),
    ]
    labels = ["Nocturno", "Siestas"]

    
    y_positions = np.arange(len(d7)); y_h = 0.7
    col_night = "#6aaed6"; col_nap = "#c7e9c0"; col_shade = "#fdecea"
    ax.set_facecolor("white")
    for i, y in enumerate(y_positions):
        if bool(risky.iloc[i]):
            ax.add_patch(Rectangle((0, y - y_h/2), 1440, y_h, facecolor=col_shade, edgecolor="none", zorder=0))
        for (a,b) in night_segments[i]:
            ax.broken_barh([(a, b-a)], (y - y_h/2, y_h), facecolors=col_night, edgecolors="none", zorder=2)
        for (a,b) in nap_segments[i]:
            ax.broken_barh([(a, b-a)], (y - y_h/2, y_h), facecolors=col_nap, edgecolors="none", zorder=3)

    ax.set_xlim(0, 1440); ax.set_ylim(-0.5, len(d7)-0.5)
    ax.set_yticks(y_positions); ax.set_yticklabels(d7[date_col].dt.strftime("%d %b"))
    xticks = np.arange(0, 25, 3)*60
    ax.set_xticks(xticks); ax.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])
    ax.set_xlabel("Hora del día"); ax.set_title(title)
    sns.despine(ax=ax, left=False, bottom=False)
    ax.legend(handles=handles, labels=labels,
            loc="upper right", frameon=True, ncol=2)
    


def panel_ptn_vs_sleep_next_v2(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    # Buscamos PTN en este orden: score → minutos noche
    ptn_candidates=("PTN_today_adj","PTN_d1","tiempo_pantalla_noche_min","pantallas_noche_min"),
    # Buscamos sueño para "mañana": reh ajustado → nocturno → sueno total
    sleep_candidates=("sleep_reh_adj","sueno_noche_h","sueno"),
    prefer_last_days=7,    # intenta 7d; si no alcanza, expande hasta 30
    max_lookback=30,
    sleep_low_h=6.0,
    title="Pantallas (hoy) vs Sueño (mañana)",
    lowess=False
):
    d = daily.copy()
    if not pd.api.types.is_datetime64_any_dtype(d.get(date_col)):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).reset_index(drop=True)

    # elige columnas
    ptn_col = next((c for c in ptn_candidates if c in d.columns), None)
    ycol    = next((c for c in sleep_candidates if c in d.columns), None)

    ax.clear()
    if ptn_col is None or ycol is None:
        ax.text(0.5,0.5,"Faltan columnas PTN/sueño para correlación", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return

    # ventana: intenta 7d; si hay <3 pares, expande hasta 30d
    def build_pairs(df):
        tmp = df[[ptn_col, ycol]].copy()
        tmp[ptn_col] = pd.to_numeric(tmp[ptn_col], errors="coerce")
        tmp["_y_next"] = pd.to_numeric(tmp[ycol], errors="coerce").shift(-1)
        return tmp[[ptn_col, "_y_next"]].dropna()

    d7  = d.tail(prefer_last_days)
    dd  = build_pairs(d7)
    used_days = prefer_last_days
    note = ""
    if len(dd) < 3 and len(d) > prefer_last_days:
        d30 = d.tail(max_lookback)
        dd2 = build_pairs(d30)
        if len(dd2) >= 3:
            dd = dd2; used_days = min(max_lookback, len(d))
            note = f" · ventana ampliada a {used_days} días por baja muestra"

    if dd.empty:
        ax.text(0.5,0.5,"Sin pares PTN→Sueño en la ventana", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return

    # plot
    risky = dd["_y_next"] < sleep_low_h
    sns.scatterplot(data=dd, x=ptn_col, y="_y_next", hue=risky,
                    palette={True:"#e74c3c", False:"#2c3e50"},
                    ax=ax, s=42, edgecolor="white", linewidth=0.6, legend=False)
    sns.regplot(data=dd, x=ptn_col, y="_y_next", ax=ax, scatter=False,
                lowess=lowess, line_kws={"linewidth":2, "alpha":0.95})
    ax.axhline(sleep_low_h, color="#e74c3c", linestyle="--", linewidth=1)
    # ejes
    ax.set_xlabel(("PTN (score)" if "PTN" in ptn_col else "Pantallas noche (min)"))
    ax.set_ylabel("Sueño (mañana, h)")
    ax.set_title(title + note)
    sns.despine(ax=ax)



########### HIDRATACION ################

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

def panel_hidratacion_vs_bienestar(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    # x: intentamos en este orden (convierte a litros si hace falta)
    agua_candidates=("agua_litros","hidratacion_litros","agua_ml","agua_cc"),
    # y: bienestar (0–10)
    y_candidates=("WBN_ex","bienestar_neto","bienestar"),
    lookback_days=30,
    lowess=True,             # usa LOESS (suave); pon False para OLS
    winsorize_pct=None,      # e.g. 0.01 para recortar outliers
    palette_points=("#2c3e50","#e74c3c"),  # normal, bajo
    font_scale=0.95,
    title="Hidratación vs Bienestar",
    agua_min_l=2.0,
    agua_caution_l=4.5,
    # ↓↓↓ cap visual superior
    x_max_cap=6.0,
):
    d = daily.copy()
    # ordenar por fecha y recortar ventana
    if date_col in d and not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).tail(lookback_days)

    # elegir columnas
    xcol = next((c for c in agua_candidates if c in d.columns), None)
    ycol = next((c for c in y_candidates    if c in d.columns), None)

    ax.clear()
    if xcol is None or ycol is None:
        ax.text(0.5,0.5,"Faltan columnas de agua/bienestar", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return {"used": None, "n": 0}

    # construir pares (x en litros)
    x = pd.to_numeric(d[xcol], errors="coerce")
    if xcol.endswith("_ml") or xcol.endswith("_cc"):
        x = x/1000.0
    y = pd.to_numeric(d[ycol], errors="coerce")

    dd = pd.DataFrame({"x": x, "y": y}).dropna()
    if dd.empty:
        ax.text(0.5,0.5,"Sin datos válidos en la ventana", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return {"used": {"x":xcol,"y":ycol}, "n": 0}

    # winsorización opcional
    if winsorize_pct and 0 < winsorize_pct < 0.5 and len(dd) >= 5:
        p = winsorize_pct
        xl, xu = dd["x"].quantile([p, 1-p]).to_numpy()
        yl, yu = dd["y"].quantile([p, 1-p]).to_numpy()
        dd["x"] = dd["x"].clip(xl, xu)
        dd["y"] = dd["y"].clip(yl, yu)

    # puntos y ajuste
    risky = dd["y"] < 3.0
    sns.scatterplot(data=dd, x="x", y="y", hue=risky,
                    palette={False:palette_points[0], True:palette_points[1]},
                    ax=ax, s=44, edgecolor="white", linewidth=0.6, legend=False)
    sns.regplot(data=dd, x="x", y="y", ax=ax, scatter=False, lowess=lowess,
                line_kws={"linewidth":2, "alpha":0.95})

    # ===== líneas guía y límites X =====
    # escala X dinámica (redondeo a múltiplos de 0.5) con tope en x_max_cap
    x_obs_max = float(dd["x"].max())
    x_upper = min(x_max_cap, max(3.5, np.ceil((x_obs_max+0.1)*2)/2))  # p.ej. 3.5, 4.0, 4.5, ...
    ax.set_xlim(left=0, right=x_upper)

    # 2.0 L: ideal mínima (punteada verde)
    ax.axvline(agua_min_l, color="#2d6a4f", linestyle="--", linewidth=1.2)
    ax.text(agua_min_l, 9.6, "≥ 2.0 L ideal", color="#2d6a4f",
            fontsize=9, ha="left", va="top")

    # 4.5 L: precaución (punteo rojo + sombreado a la derecha)
    if agua_caution_l < x_upper:
        ax.axvline(agua_caution_l, color="#c0392b", linestyle=":", linewidth=1)
        ax.axvspan(agua_caution_l, x_upper, color="#fdecea", zorder=0)

    # límites Y y resto
    ax.set_ylim(0, 6)

    # métricas
    xv, yv = dd["x"].to_numpy(), dd["y"].to_numpy()
    m = np.isfinite(xv) & np.isfinite(yv)
    r = np.corrcoef(xv[m], yv[m])[0,1] if m.sum() >= 2 else np.nan
    ax.set_title(f"{title} — n={len(dd)} · r={r:+.2f}" if np.isfinite(r) else f"{title} — n={len(dd)}")

    ax.set_xlabel("Hidratación (L/día)")
    ax.set_ylabel("Bienestar (0–10)")
    sns.despine(ax=ax)
    return {"used": {"x": xcol, "y": ycol}, "n": int(len(dd))}


############ALIMENTACION Y BIENESTAR #################

def panel_alimentacion_vs_bienestar(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    # x: preferimos score de carga (0–10, ↑=peor). Si no, convertimos calidad→carga.
    alim_score_candidates=("s_alim","alimentacion"),
    alim_quality_candidates=("alimentacion_calidad","calidad_alimentacion"),
    raw_is_quality_positive=True,   # si True: carga = 10 - calidad
    y_candidates=("WBN_ex","bienestar_neto","bienestar"),
    lookback_days=30,
    lowess=True,
    winsorize_pct=None,
    x_bands=(3.3, 6.7),             # límites bajo/medio/alto
    palette_points=("#2c3e50","#e74c3c"),
    title="Alimentación (carga) vs Bienestar"
):
    d = daily.copy()
    if date_col in d and not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).tail(lookback_days)

    # elegir x como carga 0–10
    xcol = next((c for c in alim_score_candidates if c in d.columns), None)
    if xcol is not None:
        x = pd.to_numeric(d[xcol], errors="coerce")
    else:
        qcol = next((c for c in alim_quality_candidates if c in d.columns), None)
        if qcol is None:
            ax.clear(); ax.text(0.5,0.5,"Faltan columnas de alimentación", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off(); return {"used": None, "n": 0}
        q = pd.to_numeric(d[qcol], errors="coerce")
        x = (10 - q) if raw_is_quality_positive else q  # convertir a carga

    ycol = next((c for c in y_candidates if c in d.columns), None)
    if ycol is None:
        ax.clear(); ax.text(0.5,0.5,"Falta columna de bienestar", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return {"used": None, "n": 0}

    dd = pd.DataFrame({"x": x, "y": pd.to_numeric(d[ycol], errors="coerce")}).dropna()
    ax.clear()
    if dd.empty:
        ax.text(0.5,0.5,"Sin datos válidos en la ventana", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return {"used": {"x": xcol or qcol, "y": ycol}, "n": 0}

    # winsorización opcional
    if winsorize_pct and 0 < winsorize_pct < 0.5 and len(dd) >= 5:
        p = winsorize_pct
        xl, xu = dd["x"].quantile([p, 1-p]).to_numpy()
        yl, yu = dd["y"].quantile([p, 1-p]).to_numpy()
        dd["x"] = dd["x"].clip(xl, xu)
        dd["y"] = dd["y"].clip(yl, yu)

    # bandas bajo/medio/alto
    if x_bands and len(x_bands) == 2:
        a, b = x_bands
        ax.axvspan(0, a, color="#ecf9f1", zorder=0)
        ax.axvspan(a, b, color="#fdf7e3", zorder=0)
        ax.axvspan(b, 10, color="#fdecea", zorder=0)

    # puntos y ajuste
    risky = dd["y"] < 3.0
    sns.scatterplot(data=dd, x="x", y="y", hue=risky, palette={False:palette_points[0], True:palette_points[1]},
                    ax=ax, s=44, edgecolor="white", linewidth=0.6, legend=False)
    sns.regplot(data=dd, x="x", y="y", ax=ax, scatter=False, lowess=lowess,
                line_kws={"linewidth":2, "alpha":0.95})

    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel("Carga alimentaria (0–10, ↑=más carga)")
    ax.set_ylabel("Bienestar (0–10)")

    xv, yv = dd["x"].to_numpy(), dd["y"].to_numpy()
    m = np.isfinite(xv) & np.isfinite(yv)
    r = np.corrcoef(xv[m], yv[m])[0,1] if m.sum() >= 2 else np.nan
    ax.set_title(f"{title} — n={len(dd)} · r={r:+.2f}" if np.isfinite(r) else f"{title} — n={len(dd)}")
    sns.despine(ax=ax)
    return {"used": {"x": xcol or qcol, "y": ycol}, "n": int(len(dd))}


from pathlib import Path

def figura_hidratacion_y_alimentacion(
    daily: pd.DataFrame,
    *,
    out_png: str | None = None,
    lookback_days=30,
    theme="whitegrid",       # "white", "ticks", etc.
    font_scale=0.95,
    dpi=160,
):
    sns.set_theme(style=theme)
    sns.set_context("paper", font_scale=font_scale)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    m1 = panel_hidratacion_vs_bienestar(daily, axes[0], lookback_days=lookback_days)
    m2 = panel_alimentacion_vs_bienestar(daily, axes[1], lookback_days=lookback_days)

    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    return {"png_path": out_png, "metrics": {"hidratacion": m1, "alimentacion": m2}}
