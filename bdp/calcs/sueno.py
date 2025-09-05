import pandas as pd
from bdp.utils.helpers import *

def sleep_hours_rowwise(hora_dormir, hora_despertar, *, max_hours=None):
    """
    Calcula horas de sueño por fila a partir de 'hora_dormir' y 'hora_despertar'.
    - Acepta Series/arrays con strings ('HH:MM' o datetimes), datetime64, o numéricos.
    - Si el despertar ocurre "al día siguiente", se resuelve con envoltura (mod 24h).
    - Devuelve horas crudas (float). No aplica clip/saturación.
    - max_hours (opcional): si se entrega (p.ej. 16), cualquier valor > max_hours se marca NaN como salvaguarda.

    Nota: Si pasas datetimes con fecha, se ignora la fecha y se toma solo la hora del día.
    """

    # 1) Parseo principal a datetime (para extraer hora:min:seg)
    hd = pd.to_datetime(hora_dormir, errors="coerce")
    he = pd.to_datetime(hora_despertar, errors="coerce")

    # 2) Minutos desde medianoche (ignorando la fecha)
    m_hd = (hd.dt.hour.astype("float") * 60 +
            hd.dt.minute.astype("float") +
            hd.dt.second.fillna(0).astype("float"))
    m_he = (he.dt.hour.astype("float") * 60 +
            he.dt.minute.astype("float") +
            he.dt.second.fillna(0).astype("float"))

    # 3) Backups: si no se pudo parsear, intenta interpretar numéricos
    def _coerce_numeric(col, backup):
        # Si el parseo falló (NaN), intenta numérico:
        numeric = pd.to_numeric(col, errors="coerce")
        # 0–24 -> interpreta como HORAS; >24 -> interpreta como MINUTOS
        m_from_hours = (numeric * 60).where((numeric >= 0) & (numeric <= 24))
        m_from_mins  = numeric.where(numeric > 24)
        m_numeric = m_from_hours.combine_first(m_from_mins)
        # Rellena solo donde backup es NaN
        out = backup.copy()
        mask = out.isna()
        out[mask] = m_numeric[mask]
        return out

    if m_hd.isna().any():
        m_hd = _coerce_numeric(hora_dormir, m_hd)
    if m_he.isna().any():
        m_he = _coerce_numeric(hora_despertar, m_he)

    # 4) Diferencia con envoltura a 24h (maneja cruces de medianoche)
    #    Si alguno es NaN, el resultado queda NaN automáticamente.
    diff_min = (m_he - m_hd) % (24 * 60)
    horas = diff_min / 60.0

    # 5) Salvaguarda opcional (no es "clip": solo invalida outliers evidentes)
    if max_hours is not None:
        horas = horas.mask(horas > float(max_hours))

    return horas

# 1) Fragmentación (despertares) escalonada (vectorizada)
def s_frag_despertares_escalonada(n):
    # bins: (-inf,0.5]=0; (0.5,1.5]=2; (1.5,2.5]=5; (2.5,3.5]=8; >3.5=10
    bins   = [-1, 0.5, 1.5, 2.5, 3.5, 999]
    labels = [0,   2,   5,   8,  10]
    return pd.cut(n.fillna(0), bins=bins, labels=labels, right=True).astype(int)

# 2) Déficit de duración (horas ideales 7.5; 4 pts por hora faltante, tope 10)
def s_dur_deficit_duracion_de_sueno(horas):
    return ((7.5 - horas).clip(lower=0) * 4).clip(upper=10)

# 3) Calidad invertida (0–10 donde 10 = peor)
def s_qual_sueno_invertido(sueno_calidad):
    return (10 - sueno_calidad).clip(lower=0, upper=10)

# 4) Residuo de fragmentación (evita doble conteo con calidad)
def s_frag_res_excedente_sueno(sueno_calidad, despertares):
    s_frag = s_frag_despertares_escalonada(despertares)
    return (s_frag * (sueno_calidad / 10)).clip(lower=0, upper=10)

# 5) Carga total de sueño (0–10) con pesos que suman 1
def s_sleep_carga_de_sueno_total(daily: pd.DataFrame):
    wd, wq, wf = 0.45, 0.30, 0.25   # ws·s_sleep
    s_dur  = s_dur_deficit_duracion_de_sueno(daily["sueno"])
    s_qual = s_qual_sueno_invertido(daily["sueno_calidad"])
    s_frag = s_frag_res_excedente_sueno(daily["sueno_calidad"], daily["despertares_nocturnos"])
    return (wd*s_dur + wq*s_qual + wf*s_frag).clip(0, 10)