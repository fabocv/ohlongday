import re, unicodedata
import pandas as pd
import numpy as np

_CANON_MAP = {
    # sinónimos → canónicos
    "novia": "pareja", "novio": "pareja", "polola": "pareja", "pololo": "pareja",
    "amigx": "amistad", "amigos": "amistad", "amigas": "amistad", "amigo": "amistad", "amiga": "amistad",
    "perro": "mascota", "gato": "mascota",
}

_SUPPORTIVE = {"pareja", "familia", "amistad", "comunidad", "mascota", "terapia"}
_DEMANDING  = {"trabajo", "cliente", "desconocidos"}
_NEUTRAL    = {"vecinos", "online", "presencial"}

def _strip_accents(x: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))

def _parse_tags_cell(text: str) -> list[str]:
    if not isinstance(text, str): return []
    t = _strip_accents(text).lower().replace("\xa0"," ").strip()
    if not t or t in ("nan","none","null"): return []
    # separadores: coma/; / |  + limpia espacios
    raw = re.split(r"[,\;\|/]+", t)
    out = []
    for w in raw:
        w = w.strip()
        if not w: continue
        w = _CANON_MAP.get(w, w)
        out.append(w)
    # dedup preservando orden
    seen = set(); tags = []
    for w in out:
        if w not in seen:
            seen.add(w); tags.append(w)
    return tags

def interacciones_tags_features(df: pd.DataFrame, col: str = "interacciones_significativas") -> pd.DataFrame:
    s = df.get(col, pd.Series([""]*len(df), index=df.index))
    lists = s.apply(_parse_tags_cell)

    any_flag   = lists.apply(lambda lst: int(len(lst) > 0))
    counts     = lists.apply(len)
    sup_count  = lists.apply(lambda lst: sum(t in _SUPPORTIVE for t in lst))
    dem_count  = lists.apply(lambda lst: sum(t in _DEMANDING  for t in lst))
    neu_count  = lists.apply(lambda lst: sum(t in _NEUTRAL    for t in lst))
    valence    = (sup_count - dem_count) / counts.replace(0, np.nan)
    valence    = valence.fillna(0.0).clip(-1.0, 1.0)

    # flags por etiqueta canónica (útiles para debug o filtros)
    all_tags = sorted(_SUPPORTIVE | _DEMANDING | _NEUTRAL | set(_CANON_MAP.values()))
    tag_cols = {f"tag_{t}": lists.apply(lambda lst, t=t: int(t in lst)) for t in all_tags}

    out = pd.DataFrame({
        "inter_any": any_flag.astype(int),
        "inter_count": counts.astype(int),
        "inter_pos": sup_count.astype(int),
        "inter_neg": dem_count.astype(int),
        "inter_neu": neu_count.astype(int),
        "inter_valence": valence.astype(float),
    }, index=df.index)
    if tag_cols:
        out = pd.concat([out, pd.DataFrame(tag_cols, index=df.index)], axis=1)
    return out

