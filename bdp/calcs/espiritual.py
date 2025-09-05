import pandas as pd

import pandas as pd, unicodedata

# --- Config ---
BLOQUE_COL = 'registro_periodo'   # ahora usamos tu columna
CUTOFF_SHIFT_H = 16               # si es "noche" y registró <=16:00, se asigna al día anterior
HORA_COL = 'hora_registro'
FLAG_NOCHE_AYER_COL = 'noche_dia_anterior'  # opcional (bool)

def _norm_es(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)): return ''
    s = str(x).strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.replace('-', ' ').replace('_', ' ')
    return ' '.join(s.split())

def _std_periodo(x: str) -> str:
    n = _norm_es(x)
    # mapeos tolerantes
    if 'noche' in n: return 'noche'
    if 'tarde' in n: return 'tarde'
    if 'manana' in n or 'mañana' in n: return 'mañana'
    if any(k in n for k in ['dia completo','todo el dia','jornada completa','24h','full day']):
        return 'día completo'
    return 'desconocido'

def _es_noche(x) -> bool:
    return _std_periodo(x) == 'noche'

def _parse_fecha_chile(x):
    return pd.to_datetime(x, dayfirst=True, errors='coerce')

def _minutos_desde_hora(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ''):
        return 12*60
    s = str(x).strip().lower().replace('.', ':')
    if s.isdigit():
        if len(s) == 4: s = f"{s[:2]}:{s[2:]}"
        elif len(s) <= 2: s = f"{s}:00"
    if s.endswith('am') and ':' not in s: s = s[:-2] + ':00am'
    if s.endswith('pm') and ':' not in s: s = s[:-2] + ':00pm'
    for fmt in ('%H:%M','%I:%M%p','%H','%I%p'):
        try:
            dt = pd.to_datetime(s, format=fmt)
            return int(dt.hour)*60 + int(dt.minute)
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s)
        return int(dt.hour)*60 + int(dt.minute)
    except Exception:
        return 12*60

def aplicar_fecha_logica(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['fecha_parsed'] = df['fecha'].apply(_parse_fecha_chile)
    df['min_registro'] = df[HORA_COL].apply(_minutos_desde_hora) if HORA_COL in df.columns else 12*60
    # estándar de periodo
    if BLOQUE_COL in df.columns:
        df['periodo_std'] = df[BLOQUE_COL].apply(_std_periodo)
        cond_noche = df['periodo_std'].eq('noche')
    else:
        cond_noche = pd.Series(False, index=df.index)
    cond_cutoff = df['min_registro'] <= CUTOFF_SHIFT_H*60
    cond_flag = (df[FLAG_NOCHE_AYER_COL].astype(bool)) if FLAG_NOCHE_AYER_COL in df.columns else False

    desplazar = (cond_noche & cond_cutoff) | cond_flag
    df['fecha_logica'] = (df['fecha_parsed'] - pd.to_timedelta(desplazar.astype(int), unit='D')).dt.normalize()
    return df

def media_semanal_ultimos7(df: pd.DataFrame, col, modo='fair'):
    if 'fecha_logica' not in df.columns:
        raise ValueError("Ejecuta primero aplicar_fecha_logica(df).")
    last_day = df['fecha_logica'].max()
    if pd.isna(last_day): return None, None, None, None
    idx7 = pd.date_range(last_day - pd.Timedelta(days=6), last_day, freq='D')
    serie = (df[['fecha_logica', col]].groupby('fecha_logica', as_index=True)[col].mean())
    print(serie)
    if modo == 'fair':
        val = serie[serie.index.isin(idx7)].mean()
    elif modo == 'disciplina':
        val = serie.reindex(idx7).fillna(0).mean()
    else:
        raise ValueError("modo debe ser 'fair' o 'disciplina'")
    return round(float(val), 2) if pd.notna(val) else None, idx7.min().date(), idx7.max().date(), list(idx7.date)


def espirit_semana(df):

    df = df.copy()
    #df['fecha'] = pd.to_datetime(df['fecha'])

    # tags únicas por día
    def count_tags(s):
        s = (s or '')
        tags = {t.strip().lower() for t in str(s).replace(';',',').split(',') if t.strip()}
        return len(tags)

    df['esp_tags_unique'] = df['espiritual'].apply(count_tags)
    df['norm_tags'] = df['esp_tags_unique'].clip(0,3) / 3.0
    df['norm_minutes'] = df['meditacion_min'].fillna(0).clip(0,40) / 40.0

    # flag de práctica
    has_practice = (df['norm_minutes'] > 0) | (df['esp_tags_unique'] > 0)

    # streak_factor (0–1)
    streak = 0
    streak_list = []
    for hp in has_practice.tolist():
        streak = streak + 1 if hp else 0
        streak_list.append(min(streak,7)/7.0)
    df['streak_factor'] = streak_list

    α, β, γ = 0.6, 0.3, 0.1
    df['E_d'] = (10*(α*df['norm_minutes'] + β*df['norm_tags'] + γ*df['streak_factor'])).clip(0,10).round(2)


    df = aplicar_fecha_logica(df)
    fair = media_semanal_ultimos7(df, col="E_d", modo="fair")
    disciplina = media_semanal_ultimos7(df, col="E_d", modo="disciplina")

    prom, desde, hasta, dias = media_semanal_ultimos7(df, col='E_d', modo='fair')

    daily2 = df.groupby('fecha_logica')['E_d'].mean()
    idx7 = pd.to_datetime(dias)  # los 7 días de la ventana
    cal = daily2.reindex(idx7)

    dias_con_dato = [d.date() for d, v in cal.items() if pd.notna(v)]
    dias_sin_dato = [d.date() for d, v in cal.items() if pd.isna(v)]
    consistencia = round(len(dias_con_dato) / 7, 2)  # 0.00–1.00

    resumen = {
        "espiritualidad_semana_0a10": prom,
        "score_text": "%.2f/10" % prom,
        "ventana": {"desde": desde, "hasta": hasta},
        "dias_con_dato": dias_con_dato,
        "dias_sin_dato": dias_sin_dato,
        "consistencia_7d": consistencia
    }


    return df, fair[0], disciplina[0], resumen
