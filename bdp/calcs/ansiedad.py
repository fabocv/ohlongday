import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def _safe_num(s):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    if s is None:
        return np.nan  # lo manejará _renorm_weighted_sum
    return pd.to_numeric(s, errors="coerce")


def _scale_minmax(x, lo, hi, cap_hi=True):
    z = (_safe_num(x) - lo) / max(1e-9, (hi - lo))
    if cap_hi:
        z = z.clip(lower=0, upper=1)
    return 10.0 * z

def _ptn_score_from_daily(d):
    # Preferencia: scores ya calculados
    for c in ("PTN_today_adj","PTN_d1"):
        if c in d.columns:
            v = _safe_num(d[c])
            # si vienen en 0–10, úsalo tal cual (cap)
            if v.max(skipna=True) <= 10.5:
                return v.clip(0,10)
    # Fallback a minutos de pantallas nocturnas → 0..10 en 0..120 min
    for c in ("tiempo_pantalla_noche_min","pantallas_noche_min"):
        if c in d.columns:
            return _scale_minmax(d[c], 0, 120)
    return pd.Series(np.nan, index=d.index)

def _sleep_deficit_score(d, baseline_days=28, sleep_col="sleep_reh_adj", max_def_h=3.5):
    """Déficit relativo vs mediana 28d: (mediana - sueño), solo si positivo; map a 0..10."""
    if sleep_col not in d.columns:
        return pd.Series(np.nan, index=d.index)
    s = _safe_num(d[sleep_col])
    # mediana móvil (ventana amplia); si hay pocos datos, usa mediana global
    med_rolling = s.rolling(window=baseline_days, min_periods=max(3, baseline_days//4)).median()
    med_global  = s.median(skipna=True)
    baseline = med_rolling.fillna(med_global)
    deficit_h = (baseline - s).clip(lower=0)  # solo déficit, nunca superávit
    return (deficit_h / max(1e-9, max_def_h) * 10.0).clip(0, 10)

def _activation_score(d, minutos_col="tiempo_ejercicio", s_mov_col="s_mov"):
    """Proxy de activación física (0–10): si hay minutos, escalar 0..90 → 0..10; si no, usar 10 - s_mov."""
    if minutos_col in d.columns:
        return _scale_minmax(d[minutos_col], 0, 90)  # cap a 90 min
    if s_mov_col in d.columns:
        sm = _safe_num(d[s_mov_col])
        return (10.0 - sm).clip(0,10)  # s_mov alto = peor (poca act.); invertimos a activación
    return pd.Series(np.nan, index=d.index)

def _renorm_weighted_sum(values: dict, weights: dict, index=None) -> pd.Series:
    """
    Suma ponderada ignorando NaN y re-normalizando pesos presentes.
    Acepta Series, arrays o escalares; todo se alinea al mismo índice.
    """
    # 0) índice de referencia
    idx = None
    if index is not None:
        idx = pd.Index(index)
    else:
        for v in values.values():
            if isinstance(v, pd.Series):
                idx = v.index
                break
    if idx is None:
        # Sin Series disponibles: intenta longitud de algún array; si no, devuelve NaN vacío
        for v in values.values():
            try:
                n = len(v)  # arrays/listas
                idx = pd.RangeIndex(n)
                break
            except Exception:
                continue
    if idx is None:
        return pd.Series(np.nan)

    # 1) coerción a Series alineadas
    S = {}
    for k, v in values.items():
        if isinstance(v, pd.Series):
            S[k] = pd.to_numeric(v, errors="coerce").reindex(idx)
            continue
        # array-like
        try:
            arr = pd.to_numeric(v, errors="coerce")
        except Exception:
            arr = np.nan
        if np.isscalar(arr) or (isinstance(arr, float) and np.isnan(arr)):
            # escalar → broadcast
            S[k] = pd.Series(arr, index=idx, dtype="float64")
        else:
            s = pd.Series(arr)
            if len(s) != len(idx):
                # longitudes distintas: mejor NaN alineado
                S[k] = pd.Series(np.nan, index=idx, dtype="float64")
            else:
                s.index = idx
                S[k] = pd.to_numeric(s, errors="coerce")

    # 2) claves presentes (alguna señal y peso > 0)
    present = [k for k in S if weights.get(k, 0.0) > 0 and S[k].notna().any()]
    if not present:
        return pd.Series(np.nan, index=idx)

    # 3) renormaliza pesos y suma
    W = sum(weights[k] for k in present)
    if W <= 0:
        W = 1.0
    out = pd.Series(0.0, index=idx, dtype="float64")
    for k in present:
        out = out.add(S[k] * (weights[k] / W), fill_value=0.0)

    return out


def compute_ansiedad_proxy(daily: pd.DataFrame, weights=None) -> pd.Series:
    d = daily
    ptn = _ptn_score_from_daily(d)                     # 0..10
    vals = {
        "sleep": _safe_num(d.get("s_sleep")),          # 0..10 (carga)
        "ptn": ptn,                                    # 0..10
        "caf": _safe_num(d.get("s_caf")),              # 0..10
        "stim": _safe_num(d.get("s_stim")),            # 0..10
        "alc": _safe_num(d.get("s_alc")),              # 0..10
        "nic": _safe_num(d.get("s_nic")),              # 0..10
    }
    w = weights or {"sleep":0.35,"ptn":0.25,"caf":0.15,"stim":0.10,"alc":0.10,"nic":0.05}
    proxy = _renorm_weighted_sum(vals, w).clip(0,10)
    return proxy

def compute_hipomania_proxy(daily: pd.DataFrame, weights=None) -> pd.Series:
    d = daily
    sleep_def = _sleep_deficit_score(d)                # 0..10
    ptn = _ptn_score_from_daily(d)                     # 0..10
    act = _activation_score(d)                         # 0..10
    vals = {
        "sleep_def": sleep_def,                        # ↓ sueño → ↑ proxy
        "stim": _safe_num(d.get("s_stim")),            # 0..10
        "nic": _safe_num(d.get("s_nic")),              # 0..10
        "ptn": ptn,                                    # 0..10
        "act": act,                                    # 0..10
    }
    w = weights or {"sleep_def":0.45,"stim":0.20,"nic":0.10,"ptn":0.10,"act":0.15}
    proxy = _renorm_weighted_sum(vals, w).clip(0,10)
    return proxy

def add_affect_proxies(daily: pd.DataFrame, ema_days=7) -> pd.DataFrame:
    """Añade ansiedad_proxy / hipomania_proxy y sus EMA a 'daily'."""
    out = daily.copy()
    out["ansiedad_proxy"]  = compute_ansiedad_proxy(out)
    out["hipomania_proxy"] = compute_hipomania_proxy(out)
    alpha = 2.0/(ema_days+1.0)
    out["ansiedad_proxy_ema"]  = out["ansiedad_proxy"].ewm(alpha=alpha, adjust=False).mean()
    out["hipomania_proxy_ema"] = out["hipomania_proxy"].ewm(alpha=alpha, adjust=False).mean()
    return out

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def panel_ansiedad_hipomania_ema(
    daily: pd.DataFrame,
    ax: plt.Axes,
    *,
    date_col="fecha",
    ansiedad_col="ansiedad_proxy",
    hipomania_col="hipomania_proxy",
    lookback_days=60,
    ema_days=7,
    show_raw=True, raw_alpha=0.25, raw_lw=1.0, ema_lw=2.2,
    palette_anx="#e67e22", palette_hyp="#8e44ad",
    title="Ansiedad vs Hipomanía — EMA 7d",
    # ↓↓↓ nuevo: pie explicativo
    explain_footer=True,
    win_days=7,                     # ventana corta para pendiente/contribuciones
    max_drivers=3,                  # cuantos factores listar
):
    d = daily.copy()

    # --- fecha y recorte ---
    if date_col in d and not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.sort_values(date_col).tail(lookback_days).reset_index(drop=True)

    # --- columnas presentes ---
    has_anx = ansiedad_col in d.columns
    has_hyp = hipomania_col in d.columns
    ax.clear()
    if not (has_anx or has_hyp):
        ax.text(0.5,0.5,"Faltan columnas ansiedad/hipomanía", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return {"used": None, "n": 0}

    # --- numéricos ---
    x = d[date_col] if date_col in d else pd.Series(range(len(d)))
    anx = pd.to_numeric(d[ansiedad_col], errors="coerce") if has_anx else pd.Series(np.nan, index=d.index)
    hyp = pd.to_numeric(d[hipomania_col], errors="coerce") if has_hyp else pd.Series(np.nan, index=d.index)

    # --- EMAs ---
    alpha = 2.0 / (ema_days + 1.0)
    anx_ema = anx.ewm(alpha=alpha, adjust=False).mean() if has_anx else pd.Series(np.nan, index=d.index)
    hyp_ema = hyp.ewm(alpha=alpha, adjust=False).mean() if has_hyp else pd.Series(np.nan, index=d.index)

    # --- bandas ---
    ax.axhspan(0, 3,  color="#ecf9f1", zorder=0)
    ax.axhspan(3, 7,  color="#fdf7e3", zorder=0)
    ax.axhspan(7, 10, color="#fdecea", zorder=0)
    ax.axhline(3, color="#bdc3c7", ls="--", lw=1, zorder=1)
    ax.axhline(7, color="#bdc3c7", ls="--", lw=1, zorder=1)

    # --- crudas (opcionales) ---
    if show_raw:
        if has_anx and anx.notna().sum() >= 2:
            sns.lineplot(x=x, y=anx, ax=ax, color=palette_anx, alpha=raw_alpha, lw=raw_lw, label=None)
        if has_hyp and hyp.notna().sum() >= 2:
            sns.lineplot(x=x, y=hyp, ax=ax, color=palette_hyp, alpha=raw_alpha, lw=raw_lw, label=None)

    # --- EMAs principales + últimos labels ---
    handles, labels = [], []
    if has_anx and anx_ema.notna().sum() >= 2:
        sns.lineplot(x=x, y=anx_ema, ax=ax, color=palette_anx, lw=ema_lw, label=None)
        idx_last_anx = int(anx_ema.last_valid_index()) if anx_ema.last_valid_index() is not None else None
        if idx_last_anx is not None:
            ax.scatter([x.iloc[idx_last_anx]], [anx_ema.iloc[idx_last_anx]], color=palette_anx, s=28, zorder=5, edgecolor="white", lw=0.7)
            ax.text(x.iloc[idx_last_anx], anx_ema.iloc[idx_last_anx], f" {anx_ema.iloc[idx_last_anx]:.1f}", color=palette_anx,
                    va="center", ha="left", fontsize=9, fontweight="bold")
        handles.append(Line2D([0],[0], color=palette_anx, lw=ema_lw)); labels.append("Ansiedad (EMA)")

    if has_hyp and hyp_ema.notna().sum() >= 2:
        sns.lineplot(x=x, y=hyp_ema, ax=ax, color=palette_hyp, lw=ema_lw, label=None)
        idx_last_hyp = int(hyp_ema.last_valid_index()) if hyp_ema.last_valid_index() is not None else None
        if idx_last_hyp is not None:
            ax.scatter([x.iloc[idx_last_hyp]], [hyp_ema.iloc[idx_last_hyp]], color=palette_hyp, s=28, zorder=5, edgecolor="white", lw=0.7)
            ax.text(x.iloc[idx_last_hyp], hyp_ema.iloc[idx_last_hyp], f" {hyp_ema.iloc[idx_last_hyp]:.1f}", color=palette_hyp,
                    va="center", ha="left", fontsize=9, fontweight="bold")
        handles.append(Line2D([0],[0], color=palette_hyp, lw=ema_lw)); labels.append("Hipomanía (EMA)")

    ax.set_ylim(0, 10)
    ax.set_ylabel("Intensidad (0–10)")
    ax.set_xlabel("")
    ax.set_title(f"{title} · α={alpha:.2f} (≈{ema_days}d) · ventana={lookback_days}d")
    if handles:
        ax.legend(handles=handles, labels=labels, loc="upper left", frameon=True, fontsize=9)
    sns.despine(ax=ax)

    # === EXPLICACIÓN EN EL PIE (drivers) ===
    if explain_footer:
        # helpers locales (evita dependencias)
        def _num(s): return pd.to_numeric(d.get(s), errors="coerce") if s in d.columns else pd.Series(np.nan, index=d.index)
        # PTN (score 0–10 o desde minutos)
        def _ptn_score(df):
            if "PTN_today_adj" in df.columns: 
                v = pd.to_numeric(df["PTN_today_adj"], errors="coerce"); 
                return v.clip(0,10)
            if "PTN_d1" in df.columns: 
                v = pd.to_numeric(df["PTN_d1"], errors="coerce"); 
                return v.clip(0,10)
            for c in ("tiempo_pantalla_noche_min","pantallas_noche_min"):
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors="coerce")
                    return (v/120.0*10.0).clip(0,10)  # 0..120 min → 0..10
            return pd.Series(np.nan, index=df.index)
        # déficit de sueño relativo (0..10)
        def _sleep_def(df, col="sleep_reh_adj", baseline_days=28, max_def_h=3.5):
            if col not in df.columns: return pd.Series(np.nan, index=df.index)
            s = pd.to_numeric(df[col], errors="coerce")
            med_roll = s.rolling(window=baseline_days, min_periods=max(3, baseline_days//4)).median()
            base = med_roll.fillna(s.median(skipna=True))
            def_h = (base - s).clip(lower=0)
            return (def_h / max(1e-9, max_def_h) * 10.0).clip(0,10)

        # ventana corta
        k = min(win_days, len(d))
        sel = slice(len(d)-k, len(d))

        # drivers 0..10
        drv = {
            "carga de sueño": _num("s_sleep"),
            "pantallas": _ptn_score(d),
            "cafeína": _num("s_caf"),
            "alcohol": _num("s_alc"),
            "estimulantes": _num("s_stim"),
            "nicotina": _num("s_nic"),
            "déficit de sueño": _sleep_def(d)
        }
        # pesos por proxy
        w_anx = {"carga de sueño":0.35,"pantallas":0.25,"cafeína":0.15,"estimulantes":0.10,"alcohol":0.10,"nicotina":0.05,"déficit de sueño":0.0}
        w_hyp = {"déficit de sueño":0.45,"estimulantes":0.20,"nicotina":0.10,"pantallas":0.10,"carga de sueño":0.00,"cafeína":0.00,"alcohol":0.00}

        def _slope(y):
            y = pd.to_numeric(y, errors="coerce").iloc[sel].dropna()
            if len(y) < 3: return np.nan
            xi = np.arange(len(y))
            return np.polyfit(xi, y.to_numpy(), 1)[0]

        # pendientes EMA
        s_anx = _slope(anx_ema) if has_anx else np.nan
        s_hyp = _slope(hyp_ema) if has_hyp else np.nan

        # contribuciones medias en la ventana
        def _top_drivers(weights):
            contrib = {}
            for name, w in weights.items():
                if w <= 0: continue
                v = drv[name].iloc[sel]
                if v.notna().any():
                    contrib[name] = float((v.mean(skipna=True)/10.0) * w)
            # ordena
            top = sorted(contrib.items(), key=lambda t: t[1], reverse=True)[:max_drivers]
            return top

        top_anx = _top_drivers(w_anx)
        top_hyp = _top_drivers(w_hyp)

        def _fmt_top(top):
            lab = []
            for kname, _vv in top:
                if kname == "pantallas":
                    # intenta horas si hay minutos crudos
                    mins = None
                    for c in ("tiempo_pantalla_noche_min","pantallas_noche_min"):
                        if c in d.columns:
                            mins = pd.to_numeric(d[c], errors="coerce").iloc[sel]
                            break
                    if mins is not None and mins.notna().any():
                        lab.append(f"{kname} (~{mins.mean(skipna=True)/60:.1f} h)")
                    else:
                        lab.append(kname)
                elif kname in ("cafeína","alcohol","estimulantes","nicotina","carga de sueño","déficit de sueño"):
                    lab.append(kname)
            return ", ".join(lab) if lab else "—"

        txt_parts = []
        eps = 0.03  # umbral suave de pendiente por día

        if has_anx and np.isfinite(s_anx):
            if s_anx > eps:
                txt_parts.append(f"↑ Ansiedad: contribuyen { _fmt_top(top_anx) }.")
            elif s_anx < -eps:
                txt_parts.append(f"↓ Ansiedad: descenso reciente.")
            else:
                txt_parts.append(f"Ansiedad estable; dominante: { _fmt_top(top_anx) }.")
        if has_hyp and np.isfinite(s_hyp):
            if s_hyp > eps:
                txt_parts.append(f"↑ Hipomanía: contribuyen { _fmt_top(top_hyp) }.")
            elif s_hyp < -eps:
                txt_parts.append(f"↓ Hipomanía: descenso reciente.")
            else:
                txt_parts.append(f"Hipomanía estable; dominante: { _fmt_top(top_hyp) }.")

        footer = "  ".join(txt_parts) if txt_parts else "—"
        # pie: debajo del eje
        ax.text(0.0, -0.18, footer,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="#6b7a86", wrap=True)

    npoints = int((has_anx and anx.notna().sum() or 0) + (has_hyp and hyp.notna().sum() or 0))
    return {"used": {"ansiedad": has_anx, "hipomania": has_hyp}, "n": npoints, "ema_days": ema_days}


def figura_ansiedad_hipomania_png(
    daily,
    out_png: str,
    *,
    out_svg: str | None = None,
    lookback_days: int = 60,
    ema_days: int = 7,
    theme: str = "white",
    font_scale: float = 0.9,
    figsize=(10, 3.6),
    dpi: int = 140,
    title: str | None = None,  # si quieres sobreescribir el título
):
    # Estilo
    sns.set_theme(style=theme)
    sns.set_context("paper", font_scale=font_scale)

    sns.set_theme(style="white"); sns.set_context("paper", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10,3.8), dpi=140)
    meta = panel_ansiedad_hipomania_ema(daily, ax, lookback_days=60, ema_days=7, explain_footer=True)
    plt.tight_layout(rect=[0,0.12,1,1])  # deja ~12% de margen para el pie

    # Guardar archivos
    out_png_path = Path(out_png).resolve()
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    #plt.tight_layout()

    
    #fig.savefig(out_png_path.as_posix(), bbox_inches="tight", pad_inches=0.3)

    out_svg_path = None
    if out_svg:
        out_svg_path = Path(out_svg).resolve()
        out_svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg_path.as_posix(), bbox_inches="tight", pad_inches=0.3)

    plt.close(fig)
    return {"png_path": out_png_path.as_posix(), "svg_path": out_svg_path.as_posix() if out_svg else None, "meta": meta}
