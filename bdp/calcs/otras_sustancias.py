import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

def _logistic_01(x: pd.Series, m50: float, k: float, free_min: float = 0.0, cap: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if cap is not None:
        s = s.clip(lower=0, upper=cap)
    y = 1.0 / (1.0 + np.exp(-k * (s - m50)))
    y = y.where(s > free_min, 0.0)
    y[s.isna()] = np.nan
    return y

def _renorm_weighted_sum(values: Dict[str, float], weights: Dict[str, float]) -> Optional[float]:
    present = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}
    if not present:
        return None
    w = {k: weights[k] for k in present}
    s = sum(w.values()) or 1.0
    w = {k: wv/s for k, wv in w.items()}
    return sum(present[k] * w[k] for k in present)

# === scorings (0..10 = mayor carga) ===
def score_glicemia(g: pd.Series, A=70, B=180) -> pd.Series:
    g = pd.to_numeric(g, errors="coerce").astype(float)
    lo = _logistic_01(A - g, m50=0, k=0.25)   # bajo A
    hi = _logistic_01(g - B, m50=0, k=0.25)   # sobre B
    s = (lo + hi).clip(0, 1) * 10.0
    s[g.isna()] = np.nan
    return s

def score_alcohol(units: pd.Series) -> pd.Series:
    return _logistic_01(units, m50=2.0, k=1.0, cap=6) * 10.0

def score_cafe(cucharaditas: pd.Series) -> pd.Series:
    return _logistic_01(cucharaditas, m50=3.0, k=0.9, cap=8) * 10.0

def score_mov(in_tiempo_min: pd.Series = None, in_intensidad_0a10: pd.Series = None) -> pd.Series:
    if in_tiempo_min is not None:
        ben = _logistic_01(in_tiempo_min, m50=40, k=0.12, free_min=8, cap=120)  # beneficio 0..1
        return (1.0 - ben).clip(0, 1) * 10.0
    x = pd.to_numeric(in_intensidad_0a10, errors="coerce").astype(float)
    return (10.0 - x).clip(0, 10)

def score_nicotina(cigs: pd.Series = None, mg: pd.Series = None, puffs: pd.Series = None) -> pd.Series:
    idx = (cigs.index if cigs is not None else mg.index if mg is not None else puffs.index)
    cigs_equiv = pd.Series(np.nan, index=idx, dtype=float)
    if cigs is not None:
        cigs_equiv = pd.to_numeric(cigs, errors="coerce").astype(float)
    if mg is not None:
        cigs_equiv = cigs_equiv.fillna(pd.to_numeric(mg, errors="coerce")/1.5)   # ~1.5 mg ≈ 1 cig
    if puffs is not None:
        cigs_equiv = cigs_equiv.fillna(pd.to_numeric(puffs, errors="coerce")/10) # ~10 puffs ≈ 1 cig
    return _logistic_01(cigs_equiv, m50=5.0, k=0.9, free_min=0.2, cap=20) * 10.0

def score_thc(joints: pd.Series = None, mg: pd.Series = None) -> pd.Series:
    idx = (joints.index if joints is not None else mg.index)
    eq = pd.Series(np.nan, index=idx, dtype=float)
    if joints is not None:
        eq = pd.to_numeric(joints, errors="coerce").astype(float)
    if mg is not None:
        eq = eq.fillna(pd.to_numeric(mg, errors="coerce")/10.0)  # ~10 mg ≈ 1 joint inhalado
    return _logistic_01(eq, m50=1.25, k=1.2, free_min=0.1, cap=4.0) * 10.0

def score_hidratacion(agua_litros: pd.Series,
                      opt_low: float = 1.8, opt_high: float = 3.0,
                      k_def: float = 2.5, k_exc: float = 1.6,
                      cap: float = 6.0) -> pd.Series:
    """
    Penaliza déficit (<1.8 L) y exceso (>3.0 L). Óptimo ≈ 1.8–3.0 L.
    U-shape: s = (pen_def + pen_exc) * 10, truncado a 10.
    """
    L = pd.to_numeric(agua_litros, errors="coerce").astype(float)
    L = L.clip(lower=0, upper=cap)
    pen_def = _logistic_01(opt_low - L, m50=0, k=k_def)     # por debajo del óptimo bajo
    pen_exc = _logistic_01(L - opt_high, m50=0, k=k_exc)    # por encima del óptimo alto
    s = (pen_def + pen_exc).clip(0, 1) * 10.0
    s[L.isna()] = np.nan
    return s

# Regex utilitarios
_NUM = r'(?P<num>\d+(?:[.,]\d+)?)'
RX_MG    = re.compile(_NUM + r'\s*mg\b', re.I)
RX_PUFFS = re.compile(_NUM + r'\s*(?:puff|puffs|calada|caladas)\b', re.I)

# Diccionarios de sinónimos (sin café)
NIC_TOK = re.compile(r'\b(?:nicotina|cig(?:arro|arros|s)?|pucho(?:s)?|vape(?:o|r)?|vaper|pod(?:s)?|puff(?:s)?)\b', re.I)
THC_TOK = re.compile(r'\b(?:thc|porro(?:s)?|joint(?:s)?|weed|cannabis|marihuana)\b', re.I)
STIM_TOK= re.compile(r'\b(?:modafinil(?:o)?|moda\b|metilfenidato|ritalin|adderall|anfet(?:amina|as)?|coca(?:ina)?|mdma)\b', re.I)

def _to_float(s: str) -> float:
    return float(s.replace(',', '.'))

def parse_row_otras_sustancias(text: str) -> dict:
    """
    Devuelve conteos por fila: eventos y cantidades si están (mg/puffs/cigs/joints).
    """
    txt = (text or "").strip().lower()
    if not txt:
        return dict(nic_ev=0, thc_ev=0, stim_ev=0, nic_cigs=0.0, nic_mg=0.0, nic_puffs=0.0, thc_joints=0.0, thc_mg=0.0)

    parts = [p.strip() for p in re.split(r'[,\|;]+', txt) if p.strip()]
    nic_ev = thc_ev = stim_ev = 0
    nic_cigs = nic_mg = nic_puffs = 0.0
    thc_joints = thc_mg = 0.0

    for tok in parts:
        # NICOTINA
        if NIC_TOK.search(tok):
            # Cantidades (si hay)
            m = RX_MG.search(tok)
            if m: nic_mg += _to_float(m.group('num'))
            p = RX_PUFFS.search(tok)
            if p: nic_puffs += _to_float(p.group('num'))
            # Cigarros explícitos
            if re.search(r'\bcig|pucho', tok):
                n = re.search(_NUM, tok)
                nic_cigs += _to_float(n.group('num')) if n else 1.0
            else:
                # Si no hay números pero hay mención inequívoca → cuenta evento
                if not (m or p):
                    nic_ev += 1

        # THC
        if THC_TOK.search(tok):
            m = RX_MG.search(tok)
            if m: thc_mg += _to_float(m.group('num'))
            if re.search(r'\bporro|joint', tok):
                n = re.search(_NUM, tok)
                thc_joints += _to_float(n.group('num')) if n else 1.0
            else:
                if not m:
                    thc_ev += 1

        # PSICOESTIMULANTES (no café)
        if STIM_TOK.search(tok):
            stim_ev += 1

    return dict(
        nic_ev=nic_ev, thc_ev=thc_ev, stim_ev=stim_ev,
        nic_cigs=nic_cigs, nic_mg=nic_mg, nic_puffs=nic_puffs,
        thc_joints=thc_joints, thc_mg=thc_mg
    )

def sustancias_to_daily(
    df_raw: pd.DataFrame,
    *,
    fecha_col="fecha",
    bloque_col="bloque",      # valores esperados: 'manana'/'tarde'/'noche' (normaliza a minúsculas)
    otras_col="otras_sustancias"
) -> pd.DataFrame:
    """
    Convierte registros por bloque → métricas diarias:
      - nic_ev_total / noche, thc_ev_total / noche, stim_ev_total / noche
      - nic_cigs, nic_mg, nic_puffs, thc_joints, thc_mg (sumas diarias)
    """
    d = df_raw.copy()
    if not pd.api.types.is_datetime64_any_dtype(d[fecha_col]):
        d[fecha_col] = pd.to_datetime(d[fecha_col], dayfirst=True, errors="coerce")
    if bloque_col in d.columns:
        d[bloque_col] = d[bloque_col].astype(str).str.lower().str.replace("mañana","manana",regex=False)

    parsed = d[otras_col].apply(parse_row_otras_sustancias)
    sub = pd.DataFrame(list(parsed), index=d.index)

    d = pd.concat([d, sub], axis=1)

    # Flags de noche
    is_night = (d[bloque_col].fillna("").str.lower() == "noche") if (bloque_col in d.columns) else pd.Series(False, index=d.index)

    grp = d.groupby(d[fecha_col].dt.normalize())
    out = pd.DataFrame({
        "nic_ev_total": grp["nic_ev"].sum(),
        "nic_ev_noche": grp.apply(lambda g: g.loc[is_night.loc[g.index], "nic_ev"].sum()),
        "thc_ev_total": grp["thc_ev"].sum(),
        "thc_ev_noche": grp.apply(lambda g: g.loc[is_night.loc[g.index], "thc_ev"].sum()),
        "stim_ev_total": grp["stim_ev"].sum(),
        "stim_ev_noche": grp.apply(lambda g: g.loc[is_night.loc[g.index], "stim_ev"].sum()),
        "nic_cigs": grp["nic_cigs"].sum(),
        "nic_mg": grp["nic_mg"].sum(),
        "nic_puffs": grp["nic_puffs"].sum(),
        "thc_joints": grp["thc_joints"].sum(),
        "thc_mg": grp["thc_mg"].sum(),
    }).reset_index().rename(columns={"index":"fecha"})
    return out.sort_values("fecha").reset_index(drop=True)

def eventos_a_equivalentes(
    daily_sub: pd.DataFrame,
    *,
    event_to_cig: float = 1.0,     # 1 evento ≈ 1 cig
    event_to_joint: float = 1.0,   # 1 evento ≈ 1 porro
    night_bonus: float = 0.25      # +25% por evento nocturno (prox sueño)
) -> pd.DataFrame:
    d = daily_sub.copy()

    # Equivalentes nicotina (sumar todo lo que haya)
    nic_eq_cigs = (
        pd.to_numeric(d.get("nic_cigs"), errors="coerce").fillna(0.0) +
        pd.to_numeric(d.get("nic_mg"), errors="coerce").fillna(0.0) / 1.5 +      # ~1.5 mg ≈ 1 cig
        pd.to_numeric(d.get("nic_puffs"), errors="coerce").fillna(0.0) / 10.0 +  # ~10 puffs ≈ 1 cig
        event_to_cig * pd.to_numeric(d.get("nic_ev_total"), errors="coerce").fillna(0.0) +
        night_bonus * event_to_cig * pd.to_numeric(d.get("nic_ev_noche"), errors="coerce").fillna(0.0)
    )

    # Equivalentes THC
    thc_eq_joints = (
        pd.to_numeric(d.get("thc_joints"), errors="coerce").fillna(0.0) +
        pd.to_numeric(d.get("thc_mg"), errors="coerce").fillna(0.0) / 10.0 +     # ~10 mg inhalado ≈ 1 joint
        event_to_joint * pd.to_numeric(d.get("thc_ev_total"), errors="coerce").fillna(0.0) +
        night_bonus * event_to_joint * pd.to_numeric(d.get("thc_ev_noche"), errors="coerce").fillna(0.0)
    )

    d["nic_cigs_equiv"] = nic_eq_cigs
    d["thc_joints_equiv"] = thc_eq_joints
    return d

def score_stims_from_events(events: pd.Series, m50=1.0, k=1.2, cap=4) -> pd.Series:
    """
    0..10 (más = mayor carga). 1 evento/d ≈ m50; satura ~4.
    """
    x = pd.to_numeric(events, errors="coerce").clip(lower=0, upper=cap)
    y = 1.0 / (1.0 + np.exp(-k * (x - m50)))  # 0..1
    y[events.isna()] = np.nan
    return (y * 10.0).astype(float)

def _renorm_weighted_sum(values: Dict[str, float], weights: Dict[str, float]) -> Optional[float]:
    present = {k: v for k, v in values.items() if v is not None and not pd.isna(v)}
    if not present:
        return None
    w = {k: weights[k] for k in present}
    s = sum(w.values()) or 1.0
    w = {k: wv/s for k, wv in w.items()}
    return sum(present[k] * w[k] for k in present)

def compute_CM(
    daily: pd.DataFrame,
    *,
    col_sleep_score: str = "s_sleep",          # 0..10 (más = peor sueño)
    col_glicemia: str = "glicemia",            # mg/dL (si no existe, se renormaliza sin gly)
    col_alcohol_units: str = "alcohol_ud",
    col_cafe_cucharaditas: str = "cafe_cucharaditas",
    col_mov_min: Optional[str] = "tiempo_ejercicio",
    col_mov_intensidad: Optional[str] = "mov_intensidad",
    # Alimentación
    col_alim_score: Optional[str] = "alimentacion",  # ya 0..10 (más = más carga)
    col_alim_raw: Optional[str] = None,                # usa si tienes calidad 0..10 (más = mejor)
    alim_raw_is_positive: bool = True,                 # si True: s_alim = 10 - calidad
    # Hidratación
    col_agua_litros: Optional[str] = "agua_litros",
    # Nicotina/THC (opcional)
    col_nic_cigs: Optional[str] = "nicotina",
    col_nic_mg: Optional[str] = None,
    col_nic_puffs: Optional[str] = None,
    col_thc_joints: Optional[str] = "thc",
    col_thc_mg: Optional[str] = None,
    col_stim_events: Optional[str] = "stim_ev_total",
    # Pesos (suman 1)
    weights: Dict[str, float] = None,
    return_components: bool=False
) -> Tuple[pd.Series, dict]:

    idx = daily.index
    if weights is None:
        weights = {"sleep": 0.26, "gly": 0.14, "alc": 0.12, "caf": 0.10,
            "mov": 0.12, "nic": 0.07, "thc": 0.05, "alim": 0.07,
            "hyd": 0.05, "stim": 0.02 }

    df = daily.copy()

    s_sleep = pd.to_numeric(df.get(col_sleep_score), errors="coerce").astype(float)

    s_gly = score_glicemia(df.get(col_glicemia)) if col_glicemia in df.columns else pd.Series(np.nan, index=df.index)

    s_alc = score_alcohol(df.get(col_alcohol_units)) if col_alcohol_units in df.columns else pd.Series(np.nan, index=df.index)
    s_caf = score_cafe(df.get(col_cafe_cucharaditas)) if col_cafe_cucharaditas in df.columns else pd.Series(np.nan, index=df.index)

    if col_mov_min and col_mov_min in df.columns:
        s_mov = score_mov(in_tiempo_min=df[col_mov_min])
    else:
        s_mov = score_mov(in_intensidad_0a10=df.get(col_mov_intensidad))

    # Alimentación
    if col_alim_score and col_alim_score in df.columns:
        s_alim = pd.to_numeric(df[col_alim_score], errors="coerce").astype(float).clip(0, 10)
    elif col_alim_raw and col_alim_raw in df.columns:
        raw = pd.to_numeric(df[col_alim_raw], errors="coerce").astype(float)
        s_alim = (10.0 - raw).clip(0, 10) if alim_raw_is_positive else raw.clip(0, 10)
    else:
        s_alim = pd.Series(np.nan, index=df.index, dtype=float)

    # Hidratación
    s_hyd = score_hidratacion(df.get(col_agua_litros)) if col_agua_litros and col_agua_litros in df.columns \
            else pd.Series(np.nan, index=df.index, dtype=float)

    s_nic = score_nicotina(df.get(col_nic_cigs) if col_nic_cigs else None,
                           df.get(col_nic_mg) if col_nic_mg else None,
                           df.get(col_nic_puffs) if col_nic_puffs else None)
    s_thc = score_thc(df.get(col_thc_joints) if col_thc_joints else None,
                      df.get(col_thc_mg) if col_thc_mg else None)


    df = daily.copy()
    # --- reutiliza tus scorings previos (sleep/gly/alc/caf/mov/alim/hyd) ---
    # (omito por brevedad; usa los que ya tienes)

    # Nic/THC (usa equivalentes si existen)
    s_nic = score_nicotina(
        cigs=df.get(col_nic_cigs) if col_nic_cigs and col_nic_cigs in df.columns else None,
        mg=df.get(col_nic_mg) if col_nic_mg and col_nic_mg in df.columns else None,
        puffs=df.get(col_nic_puffs) if col_nic_puffs and col_nic_puffs in df.columns else None
    )

    s_thc = score_thc(
        joints=df.get(col_thc_joints) if col_thc_joints and col_thc_joints in df.columns else None,
        mg=df.get(col_thc_mg) if col_thc_mg and col_thc_mg in df.columns else None
    )

    # Stims por eventos
    s_stim = score_stims_from_events(
        df.get(col_stim_events)
        if col_stim_events and col_stim_events in df.columns else pd.Series(np.nan, index=df.index)
    )

    metabolicos = {
        "s_sleep": s_sleep, "s_gly": s_gly, "s_alc": s_alc, "s_caf": s_caf,
        "s_mov": s_mov, "s_nic": s_nic, "s_thc": s_thc,
        "s_alim": s_alim, "s_hyd": s_hyd, "s_stim": s_stim
    }
    comps = pd.DataFrame(metabolicos, index=idx)  # ← fuerza índice igual a daily

    def _row_cm(r) -> float:
        vals = {
            "sleep": r["s_sleep"], "gly": r["s_gly"], "alc": r["s_alc"], "caf": r["s_caf"],
            "mov": r["s_mov"], "nic": r["s_nic"], "thc": r["s_thc"],
            "alim": r["s_alim"], "hyd": r["s_hyd"]
        }
        v = _renorm_weighted_sum(vals, weights)
        return np.nan if v is None else float(np.clip(v, 0, 10))

    CM = comps.apply(_row_cm, axis=1)
    CM.name = "CM"  # ← nómbralo para que el join sea directo

    meta = {
        "coverage_frac_mean": float(comps.notna().mean(axis=1).mean()),
        "components_present_per_day": comps.notna().sum(axis=1).to_dict(),
        "weights_used_base": weights,
    }
    return CM, meta, metabolicos


_NUM = r'(?P<num>\d+(?:[.,]\d+)?)'
# patrones base
RX_MG    = re.compile(_NUM + r'\s*mg\b')
RX_PUFFS = re.compile(_NUM + r'\s*(?:puff|puffs|calada|caladas)\b')
RX_TIME  = re.compile(r'@?\b(?P<h>\d{1,2}):(?P<m>\d{2})\b')

# diccionarios de sinónimos
NIC_TOKENS  = re.compile(r'\b(?:nicotina|cig(?:arro|arros|s)?|pucho(?:s)?|vape(?:o|r)?|vaper|pod(?:s)?|puff(?:s)?|mg)\b', re.I)
THC_TOKENS  = re.compile(r'\b(?:thc|porro(?:s)?|joint(?:s)?|weed|cannabis|marihuana|mg)\b', re.I)
CIG_TOK     = re.compile(r'\b(?:cig(?:arro|arros|s)?|pucho(?:s)?)\b', re.I)
VAPE_TOK    = re.compile(r'\b(?:vape(?:o|r)?|vaper|pod(?:s)?)\b', re.I)
PUFF_TOK    = re.compile(r'\b(?:puff(?:s)?|calada(?:s)?)\b', re.I)
JOINT_TOK   = re.compile(r'\b(?:porro(?:s)?|joint(?:s)?)\b', re.I)
THC_WORD    = re.compile(r'\bthc\b', re.I)

def _to_float(s: str) -> float:
    return float(s.replace(',', '.'))

def parse_otras_sustancias(serie: pd.Series) -> pd.DataFrame:
    """
    Entradas típicas por fila:
      - "2 cigarros, 1 porro"
      - "vape 12 puffs, thc 10 mg"
      - "pucho, @23:00" (sin número → 1 cigarro)
      - "vape 6 mg, 30 puffs"
    Devuelve DataFrame con columnas numéricas y horas opcionales:
      nicotina_cigarrillos, nicotina_mg, nicotina_puffs,
      thc_porros, thc_mg, hora_nicotina, hora_thc
    """
    out = {
        "nicotina_cigarrillos": [],
        "nicotina_mg": [],
        "nicotina_puffs": [],
        "thc_porros": [],
        "thc_mg": [],
        "hora_nicotina": [],
        "hora_thc": [],
    }

    for raw in serie.fillna("").astype(str):
        txt = raw.lower()
        # normaliza separadores
        txt = re.sub(r'[;|]', ',', txt)

        nic_cigs = nic_mg = nic_puffs = 0.0
        thc_joints = thc_mg = 0.0
        times = RX_TIME.findall(txt)  # lista de (h, m)
        nic_present = bool(NIC_TOKENS.search(txt))
        thc_present = bool(THC_TOKENS.search(txt))
        hora_nic = hora_thc = np.nan

        # split básico por coma
        for tok in [t.strip() for t in txt.split(',') if t.strip()]:
            # --- NICOTINA ---
            if NIC_TOKENS.search(tok):
                # mg (vape/nicotina)
                mg = RX_MG.search(tok)
                if mg:
                    nic_mg += _to_float(mg.group('num'))

                # puffs
                puf = RX_PUFFS.search(tok)
                if puf:
                    nic_puffs += _to_float(puf.group('num'))

                # cigarros
                if CIG_TOK.search(tok):
                    num = re.search(_NUM, tok)
                    if num:
                        nic_cigs += _to_float(num.group('num'))
                    else:
                        nic_cigs += 1.0  # "cigarro" sin número → 1

                # vape con número pero sin unidad explícita → ignoramos (ambigua)
                # si solo dice "nicotina" sin nada → ignoramos (no inventamos)

            # --- THC / cannabis ---
            if THC_TOKENS.search(tok):
                # mg THC
                mg = RX_MG.search(tok)
                if mg and THC_WORD.search(tok):
                    thc_mg += _to_float(mg.group('num'))

                # porros/joints
                if JOINT_TOK.search(tok):
                    num = re.search(_NUM, tok)
                    if num:
                        thc_joints += _to_float(num.group('num'))
                    else:
                        thc_joints += 1.0  # "porro" sin número → 1

                # "thc" con número pero sin mg/joint → se ignora por ambigüedad

        # asigna hora si hay UN solo tipo de sustancia en la fila
        if times:
            hhmm = f"{times[-1][0]}:{times[-1][1]}"
            if nic_present and not thc_present:
                hora_nic = hhmm
            elif thc_present and not nic_present:
                hora_thc = hhmm
            # si hay ambas, no asignamos (evitamos ambigüedad)

        out["nicotina_cigarrillos"].append(nic_cigs if nic_cigs > 0 else np.nan)
        out["nicotina_mg"].append(nic_mg if nic_mg > 0 else np.nan)
        out["nicotina_puffs"].append(nic_puffs if nic_puffs > 0 else np.nan)
        out["thc_porros"].append(thc_joints if thc_joints > 0 else np.nan)
        out["thc_mg"].append(thc_mg if thc_mg > 0 else np.nan)
        out["hora_nicotina"].append(hora_nic)
        out["hora_thc"].append(hora_thc)

    return pd.DataFrame(out, index=serie.index)
