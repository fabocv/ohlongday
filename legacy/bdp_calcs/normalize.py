import re, unicodedata, pandas as pd

def normalize_bdp_df(
    df: pd.DataFrame,
    *,
    tz: str = "America/Santiago",
    derive_fecha_hora: bool = True,
    overwrite_fecha_hora: bool = True,
    drop_google_aux: bool = False,
):
    """
    Normaliza nombres y valores del BDP:
      - Columnas → snake_case sin acentos; sinónimos a nombres canónicos.
      - `correo` ← direccion_de_correo_electronico / email / mail...
      - `medicacion_tipo` ∈ {psicofarmaco, general}
      - `otras_sustancias` ∈ {nicotina;thc;psicoestimulantes}
      - Convierte '1,2' → 1.2 en columnas numéricas típicas
      - Marca temporal (GMT-0) → `fecha` (dd-mm-YYYY), `hora` (HH:MM) en Chile
    Retorna: (df_limpio, mapping_columnas)
    """

    def strip_accents(text: str) -> str:
        if not isinstance(text, str): return text
        t = unicodedata.normalize("NFKD", text)
        return "".join(c for c in t if not unicodedata.combining(c))

    def norm_text(text: str) -> str:
        if text is None: return ""
        s = strip_accents(str(text)).lower().strip()
        return re.sub(r"\s+", " ", s)

    def snakeify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    # --- 1) Mapeo de columnas ---
    COL_SYNONYMS = {
        # core
        "estres":"estres","estres_":"estres","estres_score":"estres",
        "animo":"animo","animio":"animo","anim_o":"animo",
        "activacion":"activacion","claridad":"claridad",
        # sueño
        "sueno_calidad":"sueno_calidad","calidad_sueno":"sueno_calidad","suenocalidad":"sueno_calidad",
        "horas_sueno":"horas_sueno","horas_de_sueno":"horas_sueno",
        "siesta_min":"siesta_min","despertares_nocturnos":"despertares_nocturnos",
        "hora_dormir":"hora_dormir","hora_de_dormir":"hora_dormir","hora_despertar":"hora_despertar",
        # hábitos
        "meditacion_min":"meditacion_min","exposicion_sol_min":"exposicion_sol_min",
        "exposicion_sol_manana_min":"exposicion_sol_manana_min",
        "mov_intensidad":"mov_intensidad","movimiento":"movimiento",
        "tiempo_ejercicio_min":"tiempo_ejercicio_min",
        "tiempo_pantalla_noche_min":"tiempo_pantalla_noche_min",
        "agua_litros":"agua_litros","cafe_cucharaditas":"cafe_cucharaditas","alcohol_ud":"alcohol_ud",
        "cafe_ultima_hora":"cafe_ultima_hora","alcohol_ultima_hora":"alcohol_ultima_hora",
        # medicación / sustancias
        "medicacion_tomada":"medicacion_tomada","medicacion_tipo":"medicacion_tipo",
        "otras_sustancias":"otras_sustancias",
        # relaciones/eventos
        "interacciones_significativas":"interacciones_significativas",
        "interacciones_calidad":"interacciones_calidad",
        "eventos_estresores":"eventos_estresores",
        # otros
        "glicemia":"glicemia","fecha":"fecha","hora":"hora","tags":"tags","notas":"notas","entry_id":"entry_id",
        # correo (Google Forms)
        "direccion_de_correo_electronico":"correo","correo_electronico":"correo","email":"correo","mail":"correo","correo":"correo",
        # timestamp
        "marca_temporal":"marca_temporal","timestamp":"marca_temporal",
        "fecha_y_hora":"marca_temporal","fecha_hora":"marca_temporal",
    }

    mapping = {}
    for c in df.columns:
        base = snakeify(norm_text(c))
        new = COL_SYNONYMS.get(base, base)
        mapping[c] = new
    out = df.rename(columns=mapping).copy()

    # --- 2) Normalización de valores ---
    def normalize_email(val: str) -> str:
        s = norm_text(val).replace(" ", "")
        return s if "@" in s else s

    if "correo" in out.columns:
        out["correo"] = out["correo"].map(normalize_email)

    def normalize_medicacion_tipo(val: str) -> str:
        s = norm_text(val)
        if "psicofarmaco" in s: return "psicofarmaco"
        if "general" in s: return "general"
        return "general"

    if "medicacion_tipo" in out.columns:
        out["medicacion_tipo"] = out["medicacion_tipo"].map(normalize_medicacion_tipo)

    def normalize_otras_sustancias(val: str) -> str:
        s = norm_text(val)
        if not s: return ""
        parts = re.split(r"[;,/|\s]+", s)
        tokens = []
        for p in parts:
            t = norm_text(p)
            if not t: continue
            if t in {"nicotina","cigarrillos","cigarro","cigarros"}: t = "nicotina"
            elif t in {"thc","marihuana","weed","cannabis"}:        t = "thc"
            elif t in {"psicoestimulantes","estimulantes","stim"}:  t = "psicoestimulantes"
            if t in {"nicotina","thc","psicoestimulantes"} and t not in tokens:
                tokens.append(t)
        return ";".join(tokens)

    if "otras_sustancias" in out.columns:
        out["otras_sustancias"] = out["otras_sustancias"].map(normalize_otras_sustancias)

    def coerce_decimal_commas(s: pd.Series) -> pd.Series:
        if s.dtype == "O" or pd.api.types.is_string_dtype(s):
            return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return pd.to_numeric(s, errors="coerce")

    numeric_cols = [
        "animo","claridad","estres","activacion",
        "sueno_calidad","horas_sueno","siesta_min",
        "meditacion_min","exposicion_sol_min","exposicion_sol_manana_min",
        "mov_intensidad","movimiento","agua_litros",
        "cafe_cucharaditas","alcohol_ud","glicemia",
        "cafe_ultima_hora","alcohol_ultima_hora",
        "tiempo_pantalla_noche_min","tiempo_ejercicio_min","despertares_nocturnos",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = coerce_decimal_commas(out[col])

    # booleans típicos
    def to_bool01(series: pd.Series) -> pd.Series:
        def _map(v):
            s = norm_text(v)
            if s in {"si","sí","true","1","y","yes"}: return 1
            if s in {"no","false","0",""}:           return 0
            try:
                return int(float(v))
            except Exception:
                return pd.NA
        return series.map(_map)

    if "medicacion_tomada" in out.columns:
        out["medicacion_tomada"] = to_bool01(out["medicacion_tomada"])

    # --- 3) Derivar fecha/hora desde “Marca temporal” UTC→Chile ---
    if derive_fecha_hora:
        ts_col = None
        for c in out.columns:
            if re.search(r"(marca.*temporal|timestamp|fecha.*hora)", c, flags=re.I):
                ts_col = c; break
        if ts_col and pd.api.types.is_object_dtype(out[ts_col]):
            try:
                ts = pd.to_datetime(out[ts_col], dayfirst=True, errors="coerce", utc=True)
                ts_cl = ts.dt.tz_convert(tz)
                if overwrite_fecha_hora or "fecha" not in out.columns:
                    out["fecha"] = ts_cl.dt.strftime("%d-%m-%Y")
                if overwrite_fecha_hora or "hora" not in out.columns:
                    out["hora"]  = ts_cl.dt.strftime("%H:%M")
            except Exception:
                pass  # no romper si algo viene raro

    # --- 4) (Opcional) Dropear columnas auxiliares de Google ---
    if drop_google_aux:
        aux_patterns = [
            r"ip", r"user.?agent", r"se ha respondido", r"editar respuesta",
            r"puntuaci[oó]n|score", r"ubicaci[oó]n", r"last modified",
            r"start time|end time|completion"
        ]
        keep = []
        for c in out.columns:
            if any(re.search(p, c, flags=re.I) for p in aux_patterns):
                continue
            keep.append(c)
        out = out[keep]

    return out, mapping
