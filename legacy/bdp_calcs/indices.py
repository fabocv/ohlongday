
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .utils import clip01

def calc_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Índices sencillos compuestos (0–10):
    - arousal_index: tono de activación útil
    - clarity_calc: claridad estimada
    - regulacion_calc: regulación emocional/circadiana
    """
    out = df.copy()

    sleep_debt = pd.to_numeric(out.get("sleep_debt"), errors="coerce")
    move = pd.to_numeric(out.get("movement_score"), errors="coerce")
    light = pd.to_numeric(out.get("morning_light_score"), errors="coerce")
    relax = pd.to_numeric(out.get("relaxation_score"), errors="coerce")
    screen = pd.to_numeric(out.get("screen_night_score"), errors="coerce")
    stim = pd.to_numeric(out.get("stimulant_load"), errors="coerce")

    # arousal: + movimiento + luz (mañana) + pequeño café (implícito en stim) - sleep_debt - pantalla noche
    arousal = 0.35*move + 0.20*light + 0.10*clip01(1 - clip01((stim - 1.5)/2.5)) - 0.25*clip01(sleep_debt/2.0) + 0.10*relax - 0.10*(1-screen)
    out["arousal_index"] = 10 * clip01(arousal)

    # clarity: + sueño calculado + relax + movimiento moderado - pantalla noche - deudas
    scalc = pd.to_numeric(out.get("sueno_calidad_calc"), errors="coerce")
    clarity = 0.45*clip01(scalc/10.0) + 0.20*relax + 0.20*move - 0.10*(1-screen) - 0.15*clip01(sleep_debt/2.0)
    out["clarity_calc"] = 10 * clip01(clarity)

    # regulación: + relax + luz mañana - sleep_debt - stim alto
    regulacion = 0.40*relax + 0.25*light - 0.20*clip01(sleep_debt/2.0) - 0.15*clip01((stim-1.0)/3.0)
    out["regulacion_calc"] = 10 * clip01(regulacion)

    return out
