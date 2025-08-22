import pandas as pd
from reporte_python.bdp_schema import *

class BDPEtl:
    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = BDPSchema.coerce_types(df)
        df = BDPSchema.clip_scales(df)
        # Orden por fecha/hora si existe
        if "timestamp_iso" in df.columns:
            df = df.sort_values("timestamp_iso")
        elif "date" in df.columns:
            df = df.sort_values("date")
        df = df.reset_index(drop=True)
        return df