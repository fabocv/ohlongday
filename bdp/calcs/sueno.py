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


import unicodedata

# ---------- helpers ----------
def _norm_txt(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s

def _period_from_registro(txt: str) -> str:
    """
    Map: 'manana','tarde','noche','dia completo','desconocido'
    (tolerante a acentos/variantes).
    """
    t = _norm_txt(txt)
    if "dia completo" in t or "todo el dia" in t:
        return "dia_completo"
    if "manana" in t:
        return "manana"
    if "tarde" in t:
        return "tarde"
    if "noche" in t:
        return "noche"
    return "desconocido"

# ---------- principal ----------
def siestas_daily_from_raw_with_full(
    df_raw: pd.DataFrame,
    *,
    fecha_col: str = "fecha",
    periodo_col: str = "registro_periodo",
    siesta_min_col: str = "siesta_min",
    include_night_as_siesta: bool = False,   # si False, la noche no suma a siestas (va a "otros")
    alloc_full_ratio: tuple[float, float, float] = (0.40, 0.60, 0.00),  # MAN, TAR, NOCHE
    tol_min: int = 20
) -> pd.DataFrame:
    d = df_raw.copy()

    # Fecha normalizada (día)
    if not pd.api.types.is_datetime64_any_dtype(d.get(fecha_col)):
        d[fecha_col] = pd.to_datetime(d[fecha_col], dayfirst=True, errors="coerce")
    d["fecha_d"] = d[fecha_col].dt.normalize()

    # Minutos seguros y episodios
    d["_min"] = pd.to_numeric(d.get(siesta_min_col), errors="coerce").fillna(0).clip(lower=0)
    d["_ep"]  = (d["_min"] > 0).astype(int)

    # Periodo normalizado
    d["_periodo"] = d[periodo_col].apply(_period_from_registro)

    # Agregación pivot (sólida)
    mins = d.pivot_table(values="_min", index="fecha_d", columns="_periodo",
                         aggfunc="sum", fill_value=0)
    eps  = d.pivot_table(values="_ep",  index="fecha_d", columns="_periodo",
                         aggfunc="sum", fill_value=0)

    # Asegura columnas y alinea índices
    for c in ["manana","tarde","noche","desconocido","dia_completo"]:
        if c not in mins.columns: mins[c] = 0.0
        if c not in eps.columns:  eps[c]  = 0
    idx = mins.index.union(eps.index)
    mins = mins.reindex(idx, fill_value=0.0)
    eps  = eps.reindex(idx,  fill_value=0)

    # Construye base diaria
    raw = pd.DataFrame({"fecha": idx})
    raw["siesta_manana_min"] = mins["manana"].astype(float)
    raw["siesta_tarde_min"]  = mins["tarde"].astype(float)
    raw["siesta_noche_min"]  = mins["noche"].astype(float)
    raw["siesta_otros_min"]  = mins["desconocido"].astype(float)
    raw["siesta_full_min"]   = mins["dia_completo"].astype(float)

    # Episodios blindados
    raw["eps_raw"]  = (eps["manana"] + eps["tarde"] + eps["noche"] + eps["desconocido"]).fillna(0).astype("Int64")
    raw["eps_full"] = eps["dia_completo"].fillna(0).astype("Int64")

    # Reconciliación con "día completo"
    raw["raw_sum"]    = raw["siesta_manana_min"] + raw["siesta_tarde_min"] + raw["siesta_noche_min"] + raw["siesta_otros_min"]
    raw["delta_full"] = raw["siesta_full_min"] - raw["raw_sum"]

    # Normaliza ratios (robusto)
    r = np.array(alloc_full_ratio, dtype=float)
    r = np.nan_to_num(r, nan=0.0); r = np.clip(r, 0, None)
    if r.sum() <= 0: r = np.array([0.40, 0.60, 0.00])
    r = r / r.sum()
    r_man, r_tar, r_noc = r.tolist()

    only_full = (raw["siesta_full_min"] > 0) & (raw["raw_sum"] <= 0)
    both      = (raw["siesta_full_min"] > 0) & (raw["raw_sum"] >  0)
    need_add  = both & (raw["delta_full"] > tol_min)

    # Solo "día completo": distribuir todo por ratios y marcar ≥1 episodio
    raw.loc[only_full, "siesta_manana_min"] = raw.loc[only_full, "siesta_full_min"] * r_man
    raw.loc[only_full, "siesta_tarde_min"]  = raw.loc[only_full, "siesta_full_min"] * r_tar
    raw.loc[only_full, "siesta_noche_min"]  = raw.loc[only_full, "siesta_full_min"] * r_noc
    raw.loc[only_full, "eps_raw"] = np.where(raw.loc[only_full, "siesta_full_min"] > 0, 1, 0)

    # Ambos presentes y falta por sumar: agrega delta proporcional
    raw.loc[need_add, "siesta_manana_min"] += raw.loc[need_add, "delta_full"] * r_man
    raw.loc[need_add, "siesta_tarde_min"]  += raw.loc[need_add, "delta_full"] * r_tar
    raw.loc[need_add, "siesta_noche_min"]  += raw.loc[need_add, "delta_full"] * r_noc

    # Fuente y delta (auditoría)
    raw["siesta_source"] = np.select(
        [only_full, need_add, both, raw["raw_sum"] > 0],
        ["summary", "raw+summary", "raw (>=full)", "raw"],
        default="none"
    )
    raw["siesta_delta_min"] = np.where(only_full, raw["siesta_full_min"],
                                np.where(need_add, raw["delta_full"], 0.0))

    # Totales y episodios
    raw["siesta_min"] = (raw["siesta_manana_min"] + raw["siesta_tarde_min"] +
                         raw["siesta_noche_min"]  + raw["siesta_otros_min"])

    raw["siesta_episodios_total"] = (raw["eps_raw"].fillna(0).astype(int) +
                                     raw["eps_full"].fillna(0).astype(int))
    added = only_full | need_add
    raw.loc[added, "siesta_episodios_total"] = raw.loc[added, "siesta_episodios_total"].clip(lower=1).astype(int)

    # Noche→otros si no cuenta como siesta
    if not include_night_as_siesta:
        raw["siesta_otros_min"] += raw["siesta_noche_min"]
        raw["siesta_noche_min"]  = 0.0

    out = raw[[
        "fecha","siesta_min","siesta_manana_min","siesta_tarde_min","siesta_noche_min","siesta_otros_min",
        "siesta_episodios_total","siesta_full_min","siesta_delta_min","siesta_source"
    ]].sort_values("fecha").reset_index(drop=True)

    return out


# OBSOLETO --- 3) Calcula s_sleep con pesos circadianos y fragmentación -OBSOLETO
def sleep_effective_hours(row,
                          h_night_col="sueno_noche_h",
                          h_nap_am_col="siesta_manana_h",
                          h_nap_pm_col="siesta_tarde_h",
                          h_nap_other_col="siesta_otros_h",
                          w_night=1.0, w_am=0.6, w_pm=0.85, w_other=0.75,
                          frag_penalty_per_ep=0.4,
                          target_h=7.5, pts_per_hour_def=2.5):
    import numpy as np, pandas as pd
    hN = float(pd.to_numeric(row.get(h_night_col), errors="coerce")) if h_night_col in row else np.nan
    hA = float(pd.to_numeric(row.get(h_nap_am_col), errors="coerce")) if h_nap_am_col in row else 0.0
    hP = float(pd.to_numeric(row.get(h_nap_pm_col), errors="coerce")) if h_nap_pm_col in row else 0.0
    hO = float(pd.to_numeric(row.get(h_nap_other_col), errors="coerce")) if h_nap_other_col in row else 0.0
    n_eps = (1 if (pd.notna(hN) and hN>0) else 0) + (1 if hA>0 else 0) + (1 if hP>0 else 0) + (1 if hO>0 else 0)
    reh = (0 if pd.isna(hN) else w_night*hN) + w_am*hA + w_pm*hP + w_other*hO
    reh_adj = max(0.0, reh - max(0, n_eps-1)*frag_penalty_per_ep)
    deficit_h = max(0.0, target_h - reh_adj)
    s_sleep = min(10.0, deficit_h * pts_per_hour_def)
    return pd.Series({"sleep_reh":reh, "sleep_reh_adj":reh_adj, "sleep_deficit_h":deficit_h,
                      "s_sleep":s_sleep, "sleep_episodes":n_eps})

import numpy as np
import pandas as pd

# Mapea x (>=0) a 0..10 con logística suave (para episodios macro, etc.)
def _score_logistic_0a10(x: pd.Series, m50=1.0, k=1.2, cap=None) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if cap is not None:
        s = s.clip(lower=0, upper=cap)
    y = 1.0 / (1.0 + np.exp(-k * (s - m50)))  # 0..1
    y[s.isna()] = np.nan
    return y * 10.0

def s_sleep_carga_de_sueno_total(
    daily: pd.DataFrame,
    *,
    # pesos dentro de s_sleep
    wd: float = 0.45, wq: float = 0.30, wf: float = 0.25,
    # parámetros circadianos / fragmentación
    w_night: float = 1.00, w_am: float = 0.60, w_pm: float = 0.85, w_other: float = 0.75,
    frag_penalty_per_ep: float = 0.40,     # resta (h) por episodio adicional
    target_h: float = 7.5, pts_per_hour_def: float = 2.5,
    # mezcla de fragmentación micro (despertares) y macro (episodios de sueño en el día)
    w_frag_micro: float = 0.60, w_frag_macro: float = 0.40,
    renormalizar_si_faltan: bool = True,
):
    """
    Devuelve un Series 0..10 (mayor = más carga por sueño).
    Usa siestas si existen columnas: siesta_manana_h, siesta_tarde_h, siesta_otros_h, sueno_noche_h, sleep_episodes.
    Si no existen, cae al esquema clásico con 'sueno' (horas), 'sueno_calidad', 'despertares_nocturnos'.
    Requiere tus funciones:
      - s_dur_deficit_duracion_de_sueno(horas)
      - s_qual_sueno_invertido(sueno_calidad)
      - s_frag_res_excedente_sueno(sueno_calidad, despertares_nocturnos)  # micro-fragmentación
    """
    d = daily.copy()

    # -------- DURACIÓN EFECTIVA (noche+siestas) --------
    have_siesta = any(c in d.columns for c in ["siesta_manana_h","siesta_tarde_h","siesta_otros_h","sueno_noche_h","sleep_episodes"])
    if have_siesta:
        hN = pd.to_numeric(d.get("sueno_noche_h", d.get("sueno", np.nan)), errors="coerce")
        hA = pd.to_numeric(d.get("siesta_manana_h", 0.0), errors="coerce").fillna(0.0)
        hP = pd.to_numeric(d.get("siesta_tarde_h",  0.0), errors="coerce").fillna(0.0)
        hO = pd.to_numeric(d.get("siesta_otros_h",  0.0), errors="coerce").fillna(0.0)
        # episodios macro (noche + siestas >0)
        n_eps = (
            (hN.fillna(0) > 0).astype(int)
            + (hA > 0).astype(int)
            + (hP > 0).astype(int)
            + (hO > 0).astype(int)
        )
        # horas efectivas con pesos circadianos
        reh = (hN.fillna(0)*w_night) + (hA*w_am) + (hP*w_pm) + (hO*w_other)
        reh_adj = (reh - (n_eps.clip(lower=0) - 1) * frag_penalty_per_ep).clip(lower=0)

        # pasa horas efectivas al score de duración (tu función existente)
        s_dur = s_dur_deficit_duracion_de_sueno(reh_adj)
    else:
        # fallback: usa tu columna 'sueno' como antes
        s_dur = s_dur_deficit_duracion_de_sueno(pd.to_numeric(d.get("sueno"), errors="coerce"))

    # -------- CALIDAD (igual que siempre) --------
    s_qual = s_qual_sueno_invertido(pd.to_numeric(d.get("sueno_calidad"), errors="coerce"))

    # -------- FRAGMENTACIÓN (micro + macro) --------
    # micro: tu función clásica con despertares
    s_frag_micro = s_frag_res_excedente_sueno(
        pd.to_numeric(d.get("sueno_calidad"), errors="coerce"),
        pd.to_numeric(d.get("despertares_nocturnos"), errors="coerce"),
    )

    # macro: episodios de sueño (si hay siestas); si no, 0
    if have_siesta:
        # si no trajiste 'sleep_episodes', lo derivamos como arriba
        sleep_eps = d.get("sleep_episodes")
        if sleep_eps is None:
            # re-usa el cálculo n_eps si no existe la columna
            hN = pd.to_numeric(d.get("sueno_noche_h", d.get("sueno", np.nan)), errors="coerce")
            hA = pd.to_numeric(d.get("siesta_manana_h", 0.0), errors="coerce").fillna(0.0)
            hP = pd.to_numeric(d.get("siesta_tarde_h",  0.0), errors="coerce").fillna(0.0)
            hO = pd.to_numeric(d.get("siesta_otros_h",  0.0), errors="coerce").fillna(0.0)
            sleep_eps = (
                (hN.fillna(0) > 0).astype(int)
                + (hA > 0).astype(int)
                + (hP > 0).astype(int)
                + (hO > 0).astype(int)
            )
        else:
            sleep_eps = pd.to_numeric(sleep_eps, errors="coerce").fillna(0).astype(int)

        macro_extras = (sleep_eps - 1).clip(lower=0)  # episodios adicionales a “uno ideal”
        s_frag_macro = _score_logistic_0a10(macro_extras, m50=1.0, k=1.2, cap=5)
    else:
        s_frag_macro = pd.Series(0.0, index=d.index, dtype=float)

    # mezcla micro/macro
    s_frag = (w_frag_micro * s_frag_micro) + (w_frag_macro * s_frag_macro)

    # -------- ENSAMBLADO (con renormalización si faltan señales) --------
    comps = pd.concat([
        s_dur.rename("s_dur"), s_qual.rename("s_qual"), s_frag.rename("s_frag")
    ], axis=1)

    if renormalizar_si_faltan:
        w = {"s_dur": wd, "s_qual": wq, "s_frag": wf}
        present_w = comps.notna().astype(float).mul(pd.Series(w))
        sumw = present_w.sum(axis=1).replace(0, np.nan)
        s_sleep = (comps.mul(pd.Series(w), axis=1).sum(axis=1) / sumw).clip(0, 10)
    else:
        s_sleep = (wd*comps["s_dur"] + wq*comps["s_qual"] + wf*comps["s_frag"]).clip(0, 10)

    return s_sleep

import numpy as np
import pandas as pd

# Mapea x (>=0) a 0..10 con logística suave (para episodios macro, etc.)
def _score_logistic_0a10(x: pd.Series, m50=1.0, k=1.2, cap=None) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if cap is not None:
        s = s.clip(lower=0, upper=cap)
    y = 1.0 / (1.0 + np.exp(-k * (s - m50)))  # 0..1
    y[s.isna()] = np.nan
    return y * 10.0

def s_sleep_carga_de_sueno_total_con_siestas(
    daily: pd.DataFrame,
    *,
    # pesos dentro de s_sleep
    wd: float = 0.45, wq: float = 0.30, wf: float = 0.25,
    # parámetros circadianos / fragmentación
    w_night: float = 1.00, w_am: float = 0.60, w_pm: float = 0.85, w_other: float = 0.75,
    frag_penalty_per_ep: float = 0.40,     # resta (h) por episodio adicional
    target_h: float = 7.5, pts_per_hour_def: float = 2.5,
    # mezcla de fragmentación micro (despertares) y macro (episodios de sueño en el día)
    w_frag_micro: float = 0.60, w_frag_macro: float = 0.40,
    renormalizar_si_faltan: bool = True,
):
    """
    Devuelve un Series 0..10 (mayor = más carga por sueño).
    Usa siestas si existen columnas: siesta_manana_h, siesta_tarde_h, siesta_otros_h, sueno_noche_h, sleep_episodes.
    Si no existen, cae al esquema clásico con 'sueno' (horas), 'sueno_calidad', 'despertares_nocturnos'.
    Requiere tus funciones:
      - s_dur_deficit_duracion_de_sueno(horas)
      - s_qual_sueno_invertido(sueno_calidad)
      - s_frag_res_excedente_sueno(sueno_calidad, despertares_nocturnos)  # micro-fragmentación
    """
    d = daily.copy()

    # -------- DURACIÓN EFECTIVA (noche+siestas) --------
    have_siesta = any(c in d.columns for c in ["siesta_manana_h","siesta_tarde_h","siesta_otros_h","sueno_noche_h","sleep_episodes"])
    if have_siesta:
        hN = pd.to_numeric(d.get("sueno_noche_h", d.get("sueno", np.nan)), errors="coerce")
        hA = pd.to_numeric(d.get("siesta_manana_h", 0.0), errors="coerce").fillna(0.0)
        hP = pd.to_numeric(d.get("siesta_tarde_h",  0.0), errors="coerce").fillna(0.0)
        hO = pd.to_numeric(d.get("siesta_otros_h",  0.0), errors="coerce").fillna(0.0)
        # episodios macro (noche + siestas >0)
        n_eps = (
            (hN.fillna(0) > 0).astype(int)
            + (hA > 0).astype(int)
            + (hP > 0).astype(int)
            + (hO > 0).astype(int)
        )
        # horas efectivas con pesos circadianos
        reh = (hN.fillna(0)*w_night) + (hA*w_am) + (hP*w_pm) + (hO*w_other)
        reh_adj = (reh - (n_eps.clip(lower=0) - 1) * frag_penalty_per_ep).clip(lower=0)

        # pasa horas efectivas al score de duración (tu función existente)
        s_dur = s_dur_deficit_duracion_de_sueno(reh_adj)
    else:
        # fallback: usa tu columna 'sueno' como antes
        s_dur = s_dur_deficit_duracion_de_sueno(pd.to_numeric(d.get("sueno"), errors="coerce"))

    # -------- CALIDAD (igual que siempre) --------
    s_qual = s_qual_sueno_invertido(pd.to_numeric(d.get("sueno_calidad"), errors="coerce"))

    # -------- FRAGMENTACIÓN (micro + macro) --------
    # micro: tu función clásica con despertares
    s_frag_micro = s_frag_res_excedente_sueno(
        pd.to_numeric(d.get("sueno_calidad"), errors="coerce"),
        pd.to_numeric(d.get("despertares_nocturnos"), errors="coerce"),
    )

    # macro: episodios de sueño (si hay siestas); si no, 0
    if have_siesta:
        # si no trajiste 'sleep_episodes', lo derivamos como arriba
        sleep_eps = d.get("sleep_episodes")
        if sleep_eps is None:
            # re-usa el cálculo n_eps si no existe la columna
            hN = pd.to_numeric(d.get("sueno_noche_h", d.get("sueno", np.nan)), errors="coerce")
            hA = pd.to_numeric(d.get("siesta_manana_h", 0.0), errors="coerce").fillna(0.0)
            hP = pd.to_numeric(d.get("siesta_tarde_h",  0.0), errors="coerce").fillna(0.0)
            hO = pd.to_numeric(d.get("siesta_otros_h",  0.0), errors="coerce").fillna(0.0)
            sleep_eps = (
                (hN.fillna(0) > 0).astype(int)
                + (hA > 0).astype(int)
                + (hP > 0).astype(int)
                + (hO > 0).astype(int)
            )
        else:
            sleep_eps = pd.to_numeric(sleep_eps, errors="coerce").fillna(0).astype(int)

        macro_extras = (sleep_eps - 1).clip(lower=0)  # episodios adicionales a “uno ideal”
        s_frag_macro = _score_logistic_0a10(macro_extras, m50=1.0, k=1.2, cap=5)
    else:
        s_frag_macro = pd.Series(0.0, index=d.index, dtype=float)

    # mezcla micro/macro
    s_frag = (w_frag_micro * s_frag_micro) + (w_frag_macro * s_frag_macro)

    # -------- ENSAMBLADO (con renormalización si faltan señales) --------
    comps = pd.concat([
        s_dur.rename("s_dur"), s_qual.rename("s_qual"), s_frag.rename("s_frag")
    ], axis=1)

    if renormalizar_si_faltan:
        w = {"s_dur": wd, "s_qual": wq, "s_frag": wf}
        present_w = comps.notna().astype(float).mul(pd.Series(w))
        sumw = present_w.sum(axis=1).replace(0, np.nan)
        s_sleep = (comps.mul(pd.Series(w), axis=1).sum(axis=1) / sumw).clip(0, 10)
    else:
        s_sleep = (wd*comps["s_dur"] + wq*comps["s_qual"] + wf*comps["s_frag"]).clip(0, 10)

    return s_sleep

def sleep_user(df_raw: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquecer 'daily' con siestas por bloque + resumen y derivar columnas de sueño:
      - siesta_manana_min/siesta_tarde_min/siesta_otros_min/siesta_full_min/siesta_delta_min/siesta_episodios_total/siesta_source
      - siesta_manana_h/siesta_tarde_h/siesta_otros_h/siesta_total_min/siesta_total_h
      - sueno_noche_h (si falta, se toma de 'sueno')
      - sleep_total_h
      - sleep_reh_adj (nocturno + siestas con pesos circadianos)
    """
    # A) Siestas diarias desde el crudo
    siestas_daily = siestas_daily_from_raw_with_full(
        df_raw,
        fecha_col="fecha",
        periodo_col="registro_periodo",
        siesta_min_col="siesta_min",
        include_night_as_siesta=False,      # la noche NO cuenta como siesta; ya se suma a 'otros' dentro de la función
        alloc_full_ratio=(0.40, 0.60, 0.00),
        tol_min=20
    )

    out = daily.copy()

    # Normaliza fechas para merge
    out["fecha"] = pd.to_datetime(out["fecha"], dayfirst=True, errors="coerce").dt.normalize()
    siestas_daily["fecha"] = pd.to_datetime(siestas_daily["fecha"], errors="coerce").dt.normalize()

    # B) Une todo lo que sirve para el dashboard/analítica
    cols_merge = [
        "fecha",
        "siesta_manana_min","siesta_tarde_min","siesta_otros_min",
        "siesta_full_min","siesta_delta_min",
        "siesta_episodios_total","siesta_source"
    ]
    out = out.merge(siestas_daily[cols_merge], on="fecha", how="left")

    # C) Minutos → horas + totales
    m2h = lambda s: pd.to_numeric(out.get(s), errors="coerce")/60.0
    out["siesta_manana_h"] = m2h("siesta_manana_min")
    out["siesta_tarde_h"]  = m2h("siesta_tarde_min")
    out["siesta_otros_h"]  = m2h("siesta_otros_min")

    out["siesta_total_min"] = (
        pd.to_numeric(out.get("siesta_manana_min"), errors="coerce").fillna(0) +
        pd.to_numeric(out.get("siesta_tarde_min"),  errors="coerce").fillna(0) +
        pd.to_numeric(out.get("siesta_otros_min"),  errors="coerce").fillna(0)
    )
    out["siesta_total_h"] = out["siesta_total_min"] / 60.0

    # D) Asegura 'sueno_noche_h' (ya lo tienes arriba)
    sueno_noche_h = pd.to_numeric(out.get("sueno_noche_h"), errors="coerce")
    am_h = pd.to_numeric(out.get("siesta_manana_h"), errors="coerce")
    pm_h = pd.to_numeric(out.get("siesta_tarde_h"), errors="coerce")
    ot_h = pd.to_numeric(out.get("siesta_otros_h"), errors="coerce")

    # NUEVO: flag de señal real de sueño (algún componente presente)
    sleep_signal = sueno_noche_h.notna() | am_h.notna() | pm_h.notna() | ot_h.notna()
    out["sleep_signal_flag"] = sleep_signal.astype(int)

    # E) Derivados SOLO cuando hay señal; si no, NaN (no 0)
    out["sleep_total_h"] = (
        sueno_noche_h.fillna(0) + (out["siesta_total_h"]).fillna(0)
    ).where(sleep_signal, np.nan)

    out["sleep_reh_adj"] = (
        sueno_noche_h.fillna(0) +
        am_h.fillna(0)*0.60 + pm_h.fillna(0)*0.85 + ot_h.fillna(0)*0.75
    ).where(sleep_signal, np.nan)

    # F) Episodios si falta
    if "sleep_episodes" not in out.columns:
        out["sleep_episodes"] = (
            (sueno_noche_h.fillna(0) > 0).astype(int) +
            (am_h.fillna(0) > 0).astype(int) +
            (pm_h.fillna(0) > 0).astype(int) +
            (ot_h.fillna(0) > 0).astype(int)
        ).astype(int)

    return out




##ULTIMA VERSION 5 SEPT
import numpy as np
import pandas as pd

def _norm_periodo(x: str) -> str:
    s = (str(x) or "").strip().lower()
    s = (s.replace("í","i").replace("á","a").replace("é","e")
           .replace("ó","o").replace("ú","u"))
    if "manana" in s or "morning" in s: return "manana"
    if "tarde"  in s or "afternoon" in s: return "tarde"
    if "noche"  in s or "night" in s: return "noche"
    if "dia completo" in s or "diacompleto" in s or "full" in s: return "dia_completo"
    return "otro"

def _parse_hora_on_date(hstr, base_date):
    """
    Parsea '23:15', '11pm', '7', '07:30', etc. y los ancla al día base (sin tz).
    Devuelve pd.Timestamp o pd.NaT.
    """
    if pd.isna(hstr): return pd.NaT
    s = str(hstr).strip().lower()
    # normaliza formatos básicos
    s = (s.replace("hs","").replace("hrs","").replace("hr","")
           .replace(" a. m.","am").replace(" p. m.","pm")
           .replace(" a.m.","am").replace(" p.m.","pm")
           .replace(".",":").replace(" ", ""))
    # intenta varios formatos
    for fmt in ("%H:%M:%S","%H:%M","%H","%I:%M:%S%p","%I:%M%p","%I%p"):
        try:
            t = pd.to_datetime(s, format=fmt, errors="raise").time()
            return pd.to_datetime(base_date) + pd.to_timedelta(t.hour, unit="h") + pd.to_timedelta(t.minute, unit="m") + pd.to_timedelta(t.second, unit="s")
        except Exception:
            continue
    # último recurso: to_datetime libre (puede ser ambiguo)
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt): return pd.NaT
        # si no trae fecha, ancla en base_date
        if dt.tzinfo is not None: dt = dt.tz_localize(None)
        return pd.to_datetime(base_date.normalize()) + pd.to_timedelta(dt.hour, unit="h") + pd.to_timedelta(dt.minute, unit="m") + pd.to_timedelta(dt.second, unit="s")
    except Exception:
        return pd.NaT

def build_main_sleep_from_raw(
    df_raw: pd.DataFrame,
    *,
    date_col="fecha",
    periodo_col="registro_periodo",
    start_candidates=("hora_dormir","hora_dormir_txt"),
    end_candidates=("hora_despertar","hora_levantar","hora_levantar_txt"),
    order_time_candidates=("hora_registro","created_at","timestamp"),
    fallback_duration_candidates=("sueno_noche_h","sueno"),
    fallback_wakeup_anchor_min=7.5*60  # 07:30
) -> pd.DataFrame:
    """Devuelve un DataFrame por día con:
       fecha, hora_dormir_dt, hora_levantar_dt, sueno_noche_h, sleep_source, wrap_midnight
    """
    d = df_raw.copy()

    # Normaliza fecha a día (sin hora)
    if not pd.api.types.is_datetime64_any_dtype(d.get(date_col)):
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d["fecha_d"] = d[date_col].dt.normalize()

    # Normaliza periodo
    d["_periodo"] = d[periodo_col].map(_norm_periodo) if periodo_col in d.columns else "otro"

    # Columna para ordenar registros de la mañana si existe tiempo; si no, usa índice
    ord_col = None
    for c in order_time_candidates:
        if c in d.columns:
            ord_col = c; break

    recs = []
    for fecha, sub in d.groupby("fecha_d", sort=True):
        sub = sub.copy()
        if ord_col is not None:
            try:
                sub = sub.sort_values(pd.to_datetime(sub[ord_col], errors="coerce"))
            except Exception:
                sub = sub.sort_index()
        else:
            sub = sub.sort_index()

        cand = None
        # 1) día completo único
        full = sub[sub["_periodo"]=="dia_completo"]
        if len(full) == 1:
            cand = full.iloc[0]
            source = "dia_completo"
        # 2) si no, primer 'mañana'
        if cand is None:
            man = sub[sub["_periodo"]=="manana"]
            if len(man) >= 1:
                cand = man.iloc[0]
                source = "manana_first"

        # 3) si no hay candidato, intenta solo por duración fallback (si existe en cualquier registro del día)
        if cand is None:
            # toma cualquier fila con duración válida
            dur = None
            for dc in fallback_duration_candidates:
                if dc in sub.columns:
                    dur = pd.to_numeric(sub[dc], errors="coerce")
                    if dur.notna().any():
                        dur = float(dur.dropna().iloc[-1])
                        break
            if dur is not None and np.isfinite(dur):
                end = pd.to_datetime(fecha) + pd.to_timedelta(int(fallback_wakeup_anchor_min)//60, unit="h") + pd.to_timedelta(int(fallback_wakeup_anchor_min)%60, unit="m")
                start = end - pd.to_timedelta(dur, unit="h")
                recs.append({
                    "fecha": fecha, "hora_dormir_dt": start, "hora_levantar_dt": end,
                    "sueno_noche_h": float(dur), "sleep_source": "fallback_duration", "wrap_midnight": bool(end <= start)
                })
                continue
            else:
                # sin datos
                recs.append({
                    "fecha": fecha, "hora_dormir_dt": pd.NaT, "hora_levantar_dt": pd.NaT,
                    "sueno_noche_h": np.nan, "sleep_source": "none", "wrap_midnight": False
                })
                continue

        # Si hay candidato (mañana o día completo), intenta parsear horas
        base = pd.to_datetime(fecha)
        # elige columnas reales presentes
        def _pick(colnames):
            for c in colnames:
                if c in sub.columns and pd.notna(cand.get(c)): return c
            return None

        sc = _pick(start_candidates)
        ec = _pick(end_candidates)

        sdt = _parse_hora_on_date(cand.get(sc) if sc else np.nan, base)
        edt = _parse_hora_on_date(cand.get(ec) if ec else np.nan, base)

        # si falta alguno, usa fallback por duración si existe
        dur = None
        for dc in fallback_duration_candidates:
            if dc in sub.columns:
                v = pd.to_numeric(sub[dc], errors="coerce")
                if v.notna().any():
                    dur = float(v.dropna().iloc[-1]); break

        if pd.isna(edt) and (dur is not None and np.isfinite(dur)):
            # sin hora de despertar: asumimos despierta a 07:30
            edt = base + pd.to_timedelta(int(fallback_wakeup_anchor_min)//60, unit="h") + pd.to_timedelta(int(fallback_wakeup_anchor_min)%60, unit="m")
        if pd.isna(sdt) and (pd.notna(edt) and dur is not None and np.isfinite(dur)):
            sdt = edt - pd.to_timedelta(dur, unit="h")

        # si aún faltan, no hay datos suficientes
        if pd.isna(sdt) or pd.isna(edt):
            recs.append({
                "fecha": fecha, "hora_dormir_dt": pd.NaT, "hora_levantar_dt": pd.NaT,
                "sueno_noche_h": np.nan, "sleep_source": source + "+incomplete", "wrap_midnight": False
            })
            continue

        # wrap medianoche: si end <= start, suma un día a end
        wrap = bool(edt <= sdt)
        if wrap:
            edt = edt + pd.Timedelta(days=1)

        dur_h = float((edt - sdt).total_seconds()/3600.0)
        recs.append({
            "fecha": fecha, "hora_dormir_dt": sdt, "hora_levantar_dt": edt,
            "sueno_noche_h": dur_h, "sleep_source": source, "wrap_midnight": wrap
        })

    out = pd.DataFrame(recs).sort_values("fecha").reset_index(drop=True)
    return out

## ULTIMA VERSION 5 SEPTR
def attach_main_sleep_to_daily(daily: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    main = build_main_sleep_from_raw(df_raw)
    out = daily.copy()
    out["fecha"] = pd.to_datetime(out["fecha"], dayfirst=True, errors="coerce").dt.normalize()
    main["fecha"] = pd.to_datetime(main["fecha"], errors="coerce").dt.normalize()
    out = out.merge(main, on="fecha", how="left")  # agrega hora_dormir_dt, hora_levantar_dt, sueno_noche_h, etc.
    # por compatibilidad: si no tienes 'sueno', úsalo desde 'sueno_noche_h'
    if "sueno" not in out or out["sueno"].isna().all():
        out["sueno"] = pd.to_numeric(out.get("sueno_noche_h"), errors="coerce")
    return out
