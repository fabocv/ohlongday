
# -*- coding: utf-8 -*-
"""Constantes y pesos por defecto."""
EMA_ALPHA = 0.30
EMA_LOOKBACK = 14
EMA_MIN_PERIODS = 7

# Pesos para sueno_calidad_calc (si están disponibles se re-normalizan)
SLEEP_WEIGHTS = {
    "duration": 0.25,
    "continuity": 0.20,
    "efficiency": 0.20,
    "circadian_alignment": 0.15,
    "caffeine_timing": 0.07,
    "alcohol_timing": 0.07,
    "morning_light": 0.03,
    "screen_night": 0.03,
}
