# bdp_schema.py
import pandas as pd
import numpy as np

class BDPSchema:
    REQUIRED_COLUMNS = [
        "entry_id", "fecha", "hora",
        "animo", "activacion", "conexion", "proposito", "claridad", "estres",
        "sueno_calidad", "horas_sueno", "siesta_min",
        "autocuidado", "alimentacion", "movimiento", "dolor_fisico",
        "ansiedad", "irritabilidad",
        "meditacion_min", "exposicion_sol_min", "agua_litros",
        "cafeina_mg", "alcohol_ud",
        "medicacion_tomada", "medicacion_tipo", "otras_sustancias",
        "interacciones_significativas", "eventos_estresores", "tags", "notas"
    ]
    
    SCALE_0_10 = [
        "animo","activacion","conexion","proposito","claridad","estres",
        "sueno_calidad","autocuidado","alimentacion","movimiento","dolor_fisico",
        "ansiedad","irritabilidad"
    ]

    @staticmethod
    def validate_columns(df: pd.DataFrame):
        missing = [c for c in BDPSchema.REQUIRED_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in BDPSchema.REQUIRED_COLUMNS]
        return missing, extra

    @staticmethod
    def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        # Parsear fecha en formato dd-MM-YYYY
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce").dt.date
        # Hora simple mm:ss (guardamos como string para no complicar)
        if "hora" in df.columns:
            df["hora"] = df["hora"].astype(str)

        # Numéricos
        numeric_cols = [
            "horas_sueno","siesta_min","meditacion_min","exposicion_sol_min",
            "agua_litros","cafeina_mg","alcohol_ud"
        ]
        numeric_cols += BDPSchema.SCALE_0_10
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def clip_scales(df: pd.DataFrame) -> pd.DataFrame:
        for c in BDPSchema.SCALE_0_10:
            if c in df.columns:
                df[c] = df[c].clip(lower=0, upper=10)
        if "horas_sueno" in df.columns:
            df["horas_sueno"] = df["horas_sueno"].clip(lower=0, upper=24)
        if "agua_litros" in df.columns:
            df["agua_litros"] = df["agua_litros"].clip(lower=0, upper=10)
        if "cafeina_mg" in df.columns:
            df["cafeina_mg"] = df["cafeina_mg"].clip(lower=0, upper=1000)
        if "alcohol_ud" in df.columns:
            df["alcohol_ud"] = df["alcohol_ud"].clip(lower=0, upper=20)
        if "siesta_min" in df.columns:
            df["siesta_min"] = df["siesta_min"].clip(lower=0, upper=600)
        if "meditacion_min" in df.columns:
            df["meditacion_min"] = df["meditacion_min"].clip(lower=0, upper=600)
        if "exposicion_sol_min" in df.columns:
            df["exposicion_sol_min"] = df["exposicion_sol_min"].clip(lower=0, upper=600)
        return df
