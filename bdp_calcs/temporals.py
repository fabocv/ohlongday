
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .utils import ema

def add_lags_rolls(df: pd.DataFrame, cols, lags=(1,3), roll_win=3) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        for L in lags:
            out[f"{c}_lag{L}"] = s.shift(L)
        out[f"{c}_roll{roll_win}"] = s.rolling(roll_win, min_periods=1).mean()
    return out

def add_ema_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_ema"] = ema(pd.to_numeric(out[c], errors="coerce"))
    return out

def calc_temporales(df: pd.DataFrame, ema_cols=(), lagcols=(), lags=(1,3), roll_win=3) -> pd.DataFrame:
    out = add_ema_cols(df, ema_cols)
    out = add_lags_rolls(out, lagcols, lags=lags, roll_win=roll_win)
    return out
