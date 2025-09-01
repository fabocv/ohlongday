from bdp_calcs.normalize import *
from bdp_calcs.normalize import normalize_bdp_df
from bdp_calcs.bdp_intradia import aggregate_intraday, _timestamp_from_fecha_hora
from bdp_calcs.mini_guia_neuroconductual import _hhmm_to_minutes_series, _coerce_hhmm_latam_ampm
from bdp_calcs.helpers import interacciones_tags_features

def worked_df(input_csv,output,target_email,target_day, targets,lookback_days):
    # -------- LECTURA + NORMALIZACIÓN --------
    df_forms = pd.read_csv(input_csv, encoding="utf-8", decimal=",")
    df_clean, colmap = normalize_bdp_df(df_forms)
    
    # -------- FILTRO POR CORREO --------
    df_clean = df_clean[df_clean["correo"] == target_email].copy()
    if df_clean.empty:
        raise SystemExit("Correo no encontrado en el CSV normalizado")
    
    # -------- ESTRATEGIAS DE AGREGADO --------
    # Estado (promedio robusto diario)
    C_MEAN = [c for c in [
        "animo","autocuidado","claridad","estres","activacion",
        "ansiedad","irritabilidad","sueno_calidad","horas_sueno",
        "glicemia","mov_intensidad","dolor","dolor_fisico",
        "despertares_nocturnos"
    ] if c in df_clean.columns]
    
    # Dosis / duración (suma diaria)
    C_SUM = [c for c in [
        "cafe_cucharaditas","alcohol_ud",
        "tiempo_ejercicio_min","meditacion_min",
        "exposicion_sol_min","exposicion_sol_manana_min",
        "pantallas_min","pantallas_tarde_min","pantallas_min_tarde",
        "tiempo_pantalla_noche_min","agua_litros"
    ] if c in df_clean.columns]
    
    value_cols = list(dict.fromkeys(C_MEAN + C_SUM))
    strategy   = {c: "sum" for c in C_SUM}        # por defecto: sum para dosis/duración
    
    # Señales que sólo existen en el PRIMER registro del día → usar 'first'
    strategy.update({
        "despertares_nocturnos": "first",
        "tiempo_pantalla_noche_min": "first",
        "pantallas_tarde_min": "first",
        "pantallas_min_tarde": "first",
        "exposicion_sol_manana_min": "first",
        # si sueno_calidad está sólo en el 1º registro, 'first' es coherente:
        "sueno_calidad": "first",
    })
    
    # Horas desde cafe/alcohol (estado que crece durante el día) → usa 'max' (o 'last' si prefieres)
    for c in ["horas_desde_cafe","h_desde_cafe","horas_ult_cafe","cafe_ultima_hora",
              "horas_desde_alcohol","h_desde_alcohol","horas_ult_alcohol","alcohol_ultima_hora"]:
        if c in df_clean.columns:
            value_cols.append(c)
            strategy[c] = "max"
    
    # -------- AGREGADO INTRADÍA → DIARIO --------
    daily_df, daily_meta = aggregate_intraday(
        df_clean,
        value_cols=value_cols,
        date_col="fecha",
        time_col="hora",
        strategy_map=strategy,
        closing_weight=0.30,
        alpha_shrink=0.30,
    )

    feat = interacciones_tags_features(daily_df, col="interacciones_significativas")
    daily_df = pd.concat([daily_df, feat], axis=1)
    print(feat)

    # Para anhedonia (opción B por defecto):
    daily_df["interacciones_ok"] = (daily_df["inter_pos"] > 0).astype(int)
    
    # -------- CAMPOS DE TEXTO: PRIMER REGISTRO DEL DÍA (HH:MM) --------
        
    # 2) Traer texto del PRIMER registro del día desde el DF crudo (df_clean)
    ts_raw = _timestamp_from_fecha_hora(df_clean, date_col="fecha", time_col="hora")
    raw = df_clean.copy()
    raw["__day"] = pd.to_datetime(ts_raw, errors="coerce").dt.normalize()
    
    def _first_text_by_day(raw: pd.DataFrame, col: str) -> pd.Series:
        if col not in raw.columns:
            return pd.Series(dtype=object)
        tmp = raw.dropna(subset=["__day"])[["__day", col]].copy()
        tmp[col] = tmp[col].astype(str).str.strip()
        return tmp.groupby("__day")[col].apply(
            lambda s: next((x for x in s if x and x.lower() not in ("nan","nat","none","null")), "")
        )
    
    # ← Normalizacion de horas
    for col in ["hora_despertar", "hora_dormir","cafe_ultima_hora"]:
        s = _first_text_by_day(raw, col)
        if not s.empty:
            s = s.reindex(daily_df.index).fillna("")
            s = _coerce_hhmm_latam_ampm(s)           # 👈 ahora soporta muchos formatos
            daily_df[col] = s
    
    
    def _first_text_by_day(raw_df: pd.DataFrame, col: str) -> pd.Series:
        if col not in raw_df.columns:
            return pd.Series(dtype=object)
        tmp = raw_df.dropna(subset=["__day"])[["__day", col]].copy()
        tmp[col] = tmp[col].astype(str).str.strip()
        # primer no vacío del día
        return tmp.groupby("__day")[col].apply(
            lambda s: next((x for x in s if x and x.lower() not in ("nan","nat")), "")
        )
    
    
    # Otros textos “primer del día” que quieras tener a mano
    for col in ["medicacion_tipo", "medicacion_tomada"]:
        s = _first_text_by_day(raw, col)
        if not s.empty:
            daily_df[col] = s.reindex(daily_df.index).fillna("")

    df_day = daily_df.copy()
    s = _first_text_by_day(raw, 'hora_despertar')                   # primer valor del día (texto)
    s = s.reindex(daily_df.index).fillna("")
    daily_df["hora_despertar"] = _coerce_hhmm_latam_ampm(s)   
    df_day["fecha"] = df_day.index.strftime("%d-%m-%Y")
    df_day["hora"]  = "23:59"
    df_day = df_day.reset_index(drop=True)
    
    return daily_df, df_day