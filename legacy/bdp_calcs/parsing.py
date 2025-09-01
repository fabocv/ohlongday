
# -*- coding: utf-8 -*-
import pandas as pd
from .utils import to_datetime

def parsear_tiempos(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columna 'dt' a partir de fecha/hora y ordena cronológicamente."""
    df = df.copy()
    df["dt"] = to_datetime(df.get("fecha"), df.get("hora"))
    df = df.sort_values("dt").drop_duplicates(subset=["dt"], keep="last").reset_index(drop=True)
    return df
