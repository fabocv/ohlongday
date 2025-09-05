import numpy as np
import pandas as pd
from bdp.utils.helpers import s_u_shape

def compute_s_gly(daily: pd.DataFrame, A=70, B=180, k=0.5, plot_impute=False, ema_alpha=0.3):
    """
    Retorna:
      - s_gly: Serie con NaN donde no hay glicemia (apta para score)
      - s_gly_plot: Serie con imputación EMA opcional (solo para gráficos)
      - meta: dict con cobertura e info útil
    """
    g = pd.to_numeric(daily.get("glicemia"), errors="coerce")
    s_raw = s_u_shape(g, A=A, B=B, k=k)  # tu función

    # Para score: NaN donde no hay dato (no premiar castillos en el aire)
    s_gly = s_raw.copy()

    # Para gráfico: opcional imputar suave
    if plot_impute and g.notna().sum() >= 1:
        g_hat = g.copy()
        # EMA simple para rellenar huecos (solo estética)
        est = g_hat.ewm(alpha=ema_alpha, adjust=False).mean()
        g_hat = g_hat.fillna(est)  # sigue dejando NaN si nunca hubo dato
        s_gly_plot = s_u_shape(g_hat, A=A, B=B, k=k)
    else:
        s_gly_plot = s_gly.copy()

    meta = {
        "coverage": float(g.notna().mean()),
        "has_any_data": bool(g.notna().any()),
        "imputed_for_plot": bool(plot_impute),
    }
    return s_gly, s_gly_plot, meta


def weighted_mean_renorm(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """
    values: dict de {nombre: valor o None/NaN}
    weights: dict de {nombre: peso_original}
    Ignora None/NaN y renormaliza pesos. Retorna None si todo falta.
    """
    vs = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}
    if not vs:
        return None
    ws = {k: weights[k] for k in vs.keys()}
    s = sum(ws.values())
    if s <= 0:
        return None
    ws = {k: w / s for k, w in ws.items()}
    return sum(vs[k] * ws[k] for k in vs.keys())
