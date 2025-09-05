import math
import pandas as pd
from bdp.calcs.indicadores import *

def bucket_bienestar(score):
    if pd.isna(score): return "—"
    if score >= 7.0: return "alto"
    if score >= 3.0: return "medio"
    return "bajo"

def tendencia_from_ema(ema_series):
    if len(ema_series) < 2:
        return "estable"
    now = float(ema_series.iloc[-1])
    prev = float(ema_series.iloc[-7]) if len(ema_series) >= 7 else float(ema_series.iloc[0])
    slope = now - prev
    # Umbrales ajustables
    if slope >= 0.7: return "fuerte ↑"
    if slope >= 0.2: return "suave ↑"
    if slope <= -0.7: return "fuerte ↓"
    if slope <= -0.2: return "suave ↓"
    return "estable"

def delta_fmt(series):
    if len(series) < 2: return "±0.00", "flat"
    last = float(series.iloc[-1]); prev = float(series.iloc[-2])
    d = last - prev
    fmt = f"{d:+.2f}"
    if d >= 0.30: cls="up"
    elif d <= -0.30: cls="down"
    else: cls="flat"
    return fmt, cls

def bienestar_delta(daily: pd.DataFrame):
    return indicadores(daily,"bienestar_ema")

def bienestar_neto_delta(daily: pd.DataFrame):
    return indicadores(daily,"bienestar_neto")

def malestar(daily: pd.DataFrame):
    return indicadores(daily,"malestar_ema",normal = -1.0)

def bienestar_neto(daily: pd.DataFrame):
    return indicadores(daily,"bienestar_neto")

def animo_delta(daily: pd.DataFrame):
    return indicadores(daily,"animo")

def estres_delta(daily: pd.DataFrame):
    return indicadores(daily,"estres", normal = -1.0)

def ans_delta(daily: pd.DataFrame):
    return indicadores(daily,"ansiedad", normal = -1.0)

def claridad_delta(daily: pd.DataFrame):
    return indicadores(daily,"claridad")

import math
import numpy as np
import pandas as pd

def calc_resumen_plus(daily: pd.DataFrame, col: str = "bienestar_neto", alpha: float = 0.30):
    """
    Informe semanal completo (robusto a faltantes).

    Parámetros
    ----------
    daily : DataFrame (1 fila = 1 fecha). Idealmente contiene:
        fecha (datetime o parseable), <col> (p.ej., 'bienestar_neto' o 'WBN'/'WBN_ex')
        opcionales: ansiedad_proxy, hipomania_proxy, sueno (horas),
                    tiempo_pantalla_noche_min, PTN_today(_adj), PTN_d1(_adj),
                    CM, agua_litros, tiempo_ejercicio, s_alimentacion (0..10)
    col : str
        Columna a usar como bienestar neto de referencia.
    alpha : float
        Suavizado EMA para tendencia.

    Retorna
    -------
    dict con:
      - estado {'positivo','estable','preocupante','—'}
      - frase_humana, accion_corta, confianza {'baja','media','alta'}, comentario_tecnico
      - bien_mean_7, bien_ema_now, tendencia, delta7_fmt/delta7_clase
      - chips: [..]
      - metricas: {...}
      - factores_clave: [ {nombre, nivel, evidencia, recomendacion, impacto_0a1}, ... ] (hasta 3)
      - acciones_priorizadas: [..] (hasta 5)
    """
    out = {}

    # --- Validaciones y fecha ---
    if "fecha" not in daily.columns:
        return {"estado":"—","frase_humana":"Falta columna 'fecha'.","accion_corta":"Agregar fecha.","confianza":"baja"}

    d = daily.copy()
    if not pd.api.types.is_datetime64_any_dtype(d["fecha"]):
        d["fecha"] = pd.to_datetime(d["fecha"], dayfirst=True, errors="coerce")
    d = d.sort_values("fecha").reset_index(drop=True)

    if col not in d.columns:
        return {"estado":"—","frase_humana":f"No hay dato de '{col}'.","accion_corta":"Registra tu día.","confianza":"baja"}

    d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)

    # --- Confianza: cantidad y cobertura reciente (21d) ---
    n_dias_total = d["fecha"].nunique()
    fecha_max = d["fecha"].max()
    ult21 = d[d["fecha"] >= (fecha_max - pd.Timedelta(days=21))]
    cov21 = float(ult21[col].notna().mean()) if len(ult21) else 0.0
    if n_dias_total < 10 or cov21 < 0.4:
        confianza = "baja"
    elif n_dias_total < 30 or cov21 < 0.7:
        confianza = "media"
    else:
        confianza = "alta"
    out["confianza"] = confianza

    # --- EMA y métricas base ---
    ema_col = f"{col}_ema"
    if ema_col not in d.columns:
        d[ema_col] = d[col].ewm(alpha=alpha, adjust=False).mean()

    last7 = d[col].tail(7)
    bien_mean_7 = float(last7.mean()) if len(last7) else float("nan")
    bien_ema_now = float(d[ema_col].iloc[-1]) if len(d) else float("nan")

    out["bien_mean_7"] = round(bien_mean_7, 2) if not math.isnan(bien_mean_7) else None
    out["bien_ema_now"] = round(bien_ema_now, 2) if not math.isnan(bien_ema_now) else None

    # Estado por buckets
    def bucket_bienestar(x: float) -> str:
        if x != x: return "—"
        if x >= 6.5: return "alto"
        if x >= 3.5: return "medio"
        return "bajo"
    estado_map = {"alto":"positivo","medio":"estable","bajo":"preocupante","—":"—"}
    estado = estado_map.get(bucket_bienestar(bien_mean_7), "estable")
    out["estado"] = estado

    # Δ7 y tendencia (sobre EMA si está)
    def delta7(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() < 8: return float("nan")
        return float(s.iloc[-1] - s.iloc[-8])

    d7 = delta7(d[ema_col] if d[ema_col].notna().sum() >= 8 else d[col])
    def tendencia_from_delta(val: float) -> str:
        if val != val: return "estable"
        if val >= 0.50: return "fuerte ↑"
        if val >= 0.15: return "suave ↑"
        if val <= -0.50: return "fuerte ↓"
        if val <= -0.15: return "suave ↓"
        return "estable"
    tendencia = tendencia_from_delta(d7)

    def delta_fmt(val: float) -> tuple[str, str]:
        if val != val: return ("—", "neutral")
        if val > 0:  return (f"↑ {abs(val):.2f}", "up")
        if val < 0:  return (f"↓ {abs(val):.2f}", "down")
        return ("0.00", "neutral")
    delta7_fmt, delta7_cls = delta_fmt(d7)
    out["tendencia"] = tendencia
    out["delta7_fmt"] = delta7_fmt
    out["delta7_clase"] = delta7_cls

    # --- Helpers de medias 7d ---
    def mean7(name):
        return float(pd.to_numeric(d[name], errors="coerce").tail(7).mean()) if name in d.columns else float("nan")

    ansiedad_7  = mean7("ansiedad_proxy")
    hipomania_7 = mean7("hipomania_proxy")
    sueno_h_7   = mean7("sueno")
    pantallas_7 = mean7("tiempo_pantalla_noche_min")
    agua_7      = mean7("agua_litros")
    ejer_min_7  = mean7("tiempo_ejercicio")
    alim_s_7    = mean7("s_alimentacion")   # 0..10 (más = más carga)

    # PTN total 7d (usa ajustados si existen)
    ptn0 = d["PTN_today_adj"] if "PTN_today_adj" in d.columns else d.get("PTN_today")
    ptn1 = d["PTN_d1_adj"]    if "PTN_d1_adj" in d.columns    else d.get("PTN_d1")
    PTN_7 = float(((pd.to_numeric(ptn0, errors="coerce").fillna(0) if ptn0 is not None else 0) +
                   (pd.to_numeric(ptn1, errors="coerce").fillna(0) if ptn1 is not None else 0)
                  ).tail(7).mean()) if (ptn0 is not None or ptn1 is not None) else float("nan")

    CM_7 = mean7("CM")

    # Niveles
    def lvl01to10(v):
        if pd.isna(v): return "—"
        if v < 3: return "bajo"
        if v < 7: return "medio"
        return "alto"
    cm_lvl = lvl01to10(CM_7)

    # --- Plantillas y reglas (más ricas) ---
    templates = {
        "cm_high":      "Alta carga metabólica reciente. Prioriza hidratación/actividad ligera y reduce estimulantes.",
        "sleep_low":    "Sueño reducido: aplica 90′ sin pantallas y rutina de cierre.",
        "screens_high": "Uso de pantallas elevado: baja exposición nocturna y evita brillo alto.",
        "ansiedad_high":"Ansiedad alta: suma 2×5′ respiración y pausas activas.",
        "hipo_high":    "Activación elevada: limita multitarea y cierra 1 micro-tarea.",
        "fuerte_up":    "Buen impulso: mejora sostenida. Mantén lo que funciona.",
        "suave_up":     "Suave mejora en tendencia: pasos consistentes.",
        "fuerte_down":  "Retroceso notable. Revisa sueño, pantallas y cargas.",
        "suave_down":   "Bajada leve: ajusta pantallas y rutina de descanso.",
        "estable_ok":   "Buenos cimientos; afina el descanso."
    }

    # Alarmas prioritarias
    frase = accion = None
    if not math.isnan(CM_7) and CM_7 >= 6.5:
        frase, accion = templates["cm_high"], "Agua 2–3 L · 20–30′ caminata"
    elif sueno_h_7 == sueno_h_7 and sueno_h_7 < 6.5:
        frase, accion = templates["sleep_low"], "90′ sin pantallas · luces cálidas"
    elif not math.isnan(PTN_7) and PTN_7 >= 0.45:
        frase, accion = templates["screens_high"], "Bajar brillo y cortar 60–90′ antes de dormir"
    elif ansiedad_7 == ansiedad_7 and ansiedad_7 >= 7:
        frase, accion = templates["ansiedad_high"], "2×5′ respiración 4-4 + pausa caminando"
    elif hipomania_7 == hipomania_7 and hipomania_7 >= 7:
        frase, accion = templates["hipo_high"], "Cerrar 1 micro-tarea · 30′ descanso"

    # Estado muy bajo
    if frase is None and estado == "preocupante":
        frase, accion = "Estado bajo: enfoquémonos en sueño e higiene de pantallas.", "Dormir ≥7h y 20′ movimiento suave"

    # Tendencia si no hubo alarma
    if frase is None:
        if tendencia == "fuerte ↑": frase, accion = templates["fuerte_up"], "Mantener rutinas clave"
        elif tendencia == "suave ↑": frase, accion = templates["suave_up"], "Reforzar hábitos que ayudan"
        elif tendencia == "fuerte ↓": frase, accion = templates["fuerte_down"], "Pausa activa y revisar sueño"
        elif tendencia == "suave ↓": frase, accion = templates["suave_down"], "Reducir pantallas noche"
        else: frase, accion = templates["estable_ok"], "Mantener hábitos + 1 micro-reparación"

    # --- Factores claves (ranking heurístico 0..1) ---
    factores = []

    def add_factor(nombre, nivel, evidencia, recomendacion, impacto):
        factores.append({
            "nombre": nombre,
            "nivel": nivel,
            "evidencia": evidencia,
            "recomendacion": recomendacion,
            "impacto_0a1": float(np.clip(impacto, 0, 1))
        })

    # Pantallas (impacto ~ PTN_7 mapeado a 0..1)
    if PTN_7 == PTN_7:
        lvl = "elevado" if PTN_7 >= 0.45 else ("medio" if PTN_7 >= 0.25 else "bajo")
        evid = f"PTN·7d={PTN_7:.2f}" + (f" · noche={pantallas_7:.0f}m" if pantallas_7==pantallas_7 else "")
        rec  = "Corte 60–90′ antes de dormir · brillo bajo"
        add_factor("Pantallas", lvl, evid, rec, impacto=min(1.0, PTN_7/0.6))

    # Sueño
    if sueno_h_7 == sueno_h_7:
        lvl = "bajo" if sueno_h_7 < 6.5 else ("medio" if sueno_h_7 < 7.5 else "ok")
        evid = f" sueño·7d={sueno_h_7:.1f}h"
        rec  = "Rutina de cierre · horario estable"
        add_factor("Sueño", lvl, evid, rec, impacto=0.7 if sueno_h_7 < 6.5 else (0.4 if sueno_h_7 < 7.5 else 0.15))

    # Carga metabólica
    if CM_7 == CM_7:
        lvl = "alto" if CM_7 >= 6.5 else ("medio" if CM_7 >= 3.5 else "bajo")
        evid = f" CM·7d={CM_7:.1f}"
        rec  = "Agua 2–3 L · 20–30′ caminata · límite cafeína tarde"
        add_factor("Carga metabólica", lvl, evid, rec, impacto=0.9 if CM_7 >= 6.5 else (0.5 if CM_7 >= 3.5 else 0.2))

    # Hidratación (si hay litros)
    if agua_7 == agua_7:
        lvl = "baja" if agua_7 < 1.8 else ("exceso" if agua_7 > 3.5 else "óptima")
        evid = f" agua·7d={agua_7:.1f} L/d"
        rec  = "Meta 1.8–3.0 L/d (ajusta por calor/ejercicio)"
        add_factor("Hidratación", lvl, evid, rec, impacto=0.35 if (agua_7 < 1.8 or agua_7 > 3.5) else 0.15)

    # Ejercicio
    if ejer_min_7 == ejer_min_7:
        lvl = "bajo" if ejer_min_7 < 25 else ("medio" if ejer_min_7 < 45 else "ok")
        evid = f" ej·7d={ejer_min_7:.0f} min/d"
        rec  = "Caminar 20–30′ diarios o 3×/sem alta"
        add_factor("Ejercicio", lvl, evid, rec, impacto=0.4 if ejer_min_7 < 25 else (0.25 if ejer_min_7 < 45 else 0.1))

    # Alimentación (si ya traes score 0..10)
    if alim_s_7 == alim_s_7:
        lvl = "mejorar" if alim_s_7 >= 5 else "ok"
        evid = f" alimentación·7d={alim_s_7:.1f}/10 (carga)"
        rec  = "Verduras/fibra en 2 comidas · planificar snacks"
        add_factor("Alimentación", lvl, evid, rec, impacto=0.3 if alim_s_7 >= 5 else 0.15)

    # Ansiedad / activación
    if ansiedad_7 == ansiedad_7:
        lvl = "alta" if ansiedad_7 >= 7 else ("media" if ansiedad_7 >= 4 else "baja")
        add_factor("Ansiedad", lvl, f" {ansiedad_7:.1f}/10", "2×5′ respiración + pausa activa", impacto=0.35 if ansiedad_7 >= 7 else (0.2 if ansiedad_7 >= 4 else 0.1))
    if hipomania_7 == hipomania_7:
        lvl = "alta" if hipomania_7 >= 7 else ("media" if hipomania_7 >= 4 else "baja")
        add_factor("Activación", lvl, f" {hipomania_7:.1f}/10", "Limitar multitarea · cerrar 1 micro-tarea", impacto=0.35 if hipomania_7 >= 7 else (0.2 if hipomania_7 >= 4 else 0.1))

    # Ordena factores por impacto y deja top-3
    factores = sorted(factores, key=lambda x: x["impacto_0a1"], reverse=True)[:3]

    # --- Acciones priorizadas (de factores + tendencia) ---
    acciones = []
    for f in factores:
        if f["nombre"] == "Pantallas":
            acciones.append("Cortar pantallas 60–90′ antes de dormir (brillo bajo).")
        if f["nombre"] == "Sueño":
            acciones.append("Mantener horario de sueño y rutina de cierre.")
        if f["nombre"] == "Carga metabólica":
            acciones.append("Hidratar 2–3 L y caminar 20–30′ hoy.")
        if f["nombre"] == "Hidratación" and ("baja" in f["nivel"] or "exceso" in f["nivel"]):
            acciones.append("Ajustar agua a 1.8–3.0 L/d según actividad.")
        if f["nombre"] == "Ejercicio" and "bajo" in f["nivel"]:
            acciones.append("Sumar 20–30′ caminata suave.")
        if f["nombre"] == "Alimentación" and "mejorar" in f["nivel"]:
            acciones.append("Añadir verduras/fibra en 2 comidas.")
        if f["nombre"] == "Ansiedad" and "alta" in f["nivel"]:
            acciones.append("Respiración 4-4 (2×5′) + pausa caminando.")
        if f["nombre"] == "Activación" and "alta" in f["nivel"]:
            acciones.append("Cerrar 1 micro-tarea y pausar 30′.")

    # refuerzos por tendencia
    if tendencia in ("fuerte ↑","suave ↑"):
        acciones.append("Mantener lo que funciona (revisión breve al final del día).")
    elif tendencia in ("fuerte ↓","suave ↓"):
        acciones.append("Pausa activa de 10′ y plan de sueño esta noche.")

    # dedup y top-5
    acciones_priorizadas = []
    seen = set()
    for a in acciones:
        if a not in seen:
            acciones_priorizadas.append(a); seen.add(a)
        if len(acciones_priorizadas) == 5: break

    # --- Texto humano principal (headline) ---
    # Ej.: "Estado general: preocupante · Uso de pantallas elevado: baja exposición nocturna..."
    if factores:
        f0 = factores[0]
        sub = f"{f0['nombre']} {f0['nivel']}: {f0['recomendacion']}"
    else:
        sub = "Panorama estable; afina el descanso."
    frase_humana = f"Estado general: {estado} · {sub}"

    # --- Chips y métricas ---
    chips = [f"Estado: {estado}", f"Tendencia: {tendencia}", f"Confianza: {confianza}"]
    metricas = {
        "mean7": out["bien_mean_7"],
        "EMA_now": out["bien_ema_now"],
        "Δ7": delta7_fmt,
        "CM_7": round(CM_7, 2) if CM_7 == CM_7 else None,
        "PTN_7": round(PTN_7, 2) if PTN_7 == PTN_7 else None,
        "Sueño_h_7": round(sueno_h_7, 2) if sueno_h_7 == sueno_h_7 else None,
        "Pantallas_noche_min_7": round(pantallas_7, 1) if pantallas_7 == pantallas_7 else None,
        "Agua_L_7": round(agua_7, 2) if agua_7 == agua_7 else None,
        "Ejercicio_min_7": round(ejer_min_7, 0) if ejer_min_7 == ejer_min_7 else None,
        "Alimentación_s_7": round(alim_s_7, 1) if alim_s_7 == alim_s_7 else None,
    }

    # --- Ensamblado final ---
    out.update({
        "frase_humana": frase_humana,
        "accion_corta": accion,
        "chips": chips,
        "metricas": metricas,
        "factores_clave": factores,
        "acciones_priorizadas": acciones_priorizadas,
        "comentario_tecnico": (
            f"Días={n_dias_total} · Cobertura21d={cov21:.0%} · "
            f"mean7={metricas['mean7']} · EMA_now={metricas['EMA_now']} · Δ7={delta7_fmt} · "
            f"CM7={metricas['CM_7']} · PTN7={metricas['PTN_7']}"
        ),
    })
    return out




def tendencia_bienestar(daily):
    ema_now = daily["bienestar_neto"].iloc[-1]
    ema_prev = daily["bienestar_neto"].iloc[-7] if len(daily) >= 7 else daily["bienestar_neto"].iloc[0]
    slope = ema_now - ema_prev

    if slope > 0.7:
        tendencia = "fuerte ↑"
    elif slope > 0.2:
        tendencia = "suave ↑"
    elif slope < -0.7:
        tendencia = "fuerte ↓"
    elif slope < -0.2:
        tendencia = "suave ↓"
    else:
        tendencia = "estable"


    return tendencia

def kpi_bienestar(daily, alpha = 0.30):
    daily = daily.sort_values("fecha")  # asegúrate que está ordenado

    bienestar_datos, bienestar_score, bienestar_nivel, bienestar_fmt, bienestar_clase    = bienestar_delta(daily)
    _, bienestar_neto_score, bienestar_neto_nivel, bienestar_neto_fmt, bienestar_neto_clase    = bienestar_neto_delta(daily)
    animo_datos, animo_score, animo_nivel, animo_fmt, animo_clase                    = animo_delta(daily)
    estres_datos, estres_score, estres_nivel, estres_fmt, estres_clase                = estres_delta(daily)
    estres_datos, ans_score, ans_nivel, ans_fmt, ans_clase                = ans_delta(daily)
    claridad_datos, claridad_score, claridad_nivel, claridad_fmt, claridad_clase        = claridad_delta(daily)
    _, mal_score, mal_nivel, mal_fmt, mal_clase        = malestar(daily)
    kpi = {
        "bienestar_nivel": bienestar_neto_nivel,
        "bienestar_score":  bienestar_score,
        "bienestar_delta_fmt": bienestar_fmt ,
        "bienestar_delta_clase": bienestar_clase,
        "bienestar_neto_nivel": bienestar_neto_nivel,
        "bienestar_neto_score":  bienestar_neto_score,
        "bienestar_neto_delta_fmt": bienestar_neto_fmt ,
        "bienestar_neto_delta_clase": bienestar_neto_clase,
        "tendencia": tendencia_bienestar(daily),
        "ema_label": "α=0.30 • L=14",
        "mal_nivel": mal_score, "mal_delta_fmt": mal_fmt, "mal_delta_clase": mal_clase,
        "animo_nivel": animo_score, "animo_delta_fmt": animo_fmt, "animo_delta_clase": animo_clase,
        "ans_nivel": ans_score, "ans_delta_fmt": ans_fmt, "ans_delta_clase":  ans_clase,
        "estres_nivel": estres_score, "estres_delta_fmt": estres_fmt, "estres_delta_clase":  estres_clase,
        "claridad_nivel": claridad_score, "claridad_delta_fmt": claridad_fmt, "claridad_delta_clase": claridad_clase,
    }

    return kpi