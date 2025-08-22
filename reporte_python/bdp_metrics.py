import pandas as pd
import numpy as np
from datetime import datetime
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt

class BDPMetrics:
    @staticmethod
    def zscore(series: pd.Series):
        s = series.astype(float)
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series([0.0]*len(s), index=s.index)
        return (s - mean) / std
    
    @staticmethod
    def compute_indices(df: pd.DataFrame, ponderar_sueno=True, ponderar_autocuidado=True):
        # Z-scores básicos
        for col in ["animo","activacion","conexion","proposito","claridad","estres","sueno_calidad"]:
            if col in df.columns:
                df[f"z_{col}"] = BDPMetrics.zscore(df[col])
        
        # Ajustes/ponderaciones solicitadas por Fab
        v_term_activacion = df.get("z_activacion", pd.Series([0]*len(df)))
        v_term_sueno = df.get("z_sueno_calidad", pd.Series([0]*len(df)))
        if ponderar_sueno:
            v_term_sueno = 1.2 * v_term_sueno  # mayor peso al sueño
        
        p_term_proposito = df.get("z_proposito", pd.Series([0]*len(df)))
        p_term_claridad = df.get("z_claridad", pd.Series([0]*len(df)))
        
        # Índices teóricos
        df["H_t"] = df.get("z_animo", 0)  # Humor
        df["V_t"] = v_term_activacion + v_term_sueno  # Vitalidad
        df["C_t"] = df.get("z_conexion", 0)  # Conexión
        df["P_t"] = p_term_proposito + p_term_claridad  # Propósito/Claridad
        df["S_t_neg"] = - df.get("z_estres", 0)  # Estrés invertido
        
        # Índice compuesto (simple promedio con penalización por estrés)
        df["BDP_score"] = (df["H_t"] + df["V_t"] + df["C_t"] + df["P_t"] + df["S_t_neg"]) / 5.0
        
        # Bandas cualitativas (Escala fenomenológica 0–3)
        bins = [-np.inf, -0.5, 0.0, 0.5, np.inf]
        labels = [0, 1, 2, 3]
        df["BDP_feno_0_3"] = pd.cut(df["BDP_score"], bins=bins, labels=labels).astype(int)
        
        return df