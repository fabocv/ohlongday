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
    return indicadores(daily,"bienestar")

def animo_delta(daily: pd.DataFrame):
    return indicadores(daily,"animo")

def estres_delta(daily: pd.DataFrame):
    print(daily["estres"])
    return indicadores(daily,"estres", normal = -1.0)

def claridad_delta(daily: pd.DataFrame):
    return indicadores(daily,"claridad")

def calc_resumen(daily: pd.DataFrame, col="bienestar", alpha=0.30):
    """
    daily: DataFrame diario ordenado por fecha asc (fecha, bienestar, bienestar_ema optional, ansiedad_proxy, hipomania_proxy, sueno, tiempo_pantalla_noche_min)
    Devuelve dict con: estado, frase_humana, confianza, accion_corta, comentario_tecnico (opcional)
    """
    out = {}
    n_dias = daily["fecha"].nunique() if "fecha" in daily.columns else len(daily)
    confianza = "baja" if n_dias < 10 else ("media" if n_dias < 30 else "alta")
    out["confianza"] = confianza

    # Asegura orden y EMA
    d = daily.sort_values("fecha").reset_index(drop=True)
    if col not in d.columns:
        out.update({"estado":"—","frase_humana":"No hay dato de bienestar.","accion_corta":"Registra tu día."})
        return out

    # Calcula EMA si no existe
    ema_col = f"{col}_ema"
    if ema_col not in d.columns:
        d[ema_col] = d[col].ewm(alpha=alpha, adjust=False).mean()

    # Promedio últimos 7 días
    last7 = d[col].tail(7)
    bienestar_mean_7 = float(last7.mean()) if len(last7)>0 else float("nan")
    out["bienestar_mean_7"] = round(bienestar_mean_7,2)

    # Estado por bucket
    bucket = bucket_bienestar(bienestar_mean_7)
    estado_map = {"alto":"positivo","medio":"estable","bajo":"preocupante","—":"—"}
    out["estado"] = estado_map.get(bucket, "estable")

    # Tendencia y delta
    out["tendencia"] = tendencia_from_ema(d[ema_col])
    out["bienestar_ema_now"] = round(float(d[ema_col].iloc[-1]),2)
    dlt_fmt, dlt_cls = delta_fmt(d[col])
    out["bienestar_delta_fmt"] = dlt_fmt
    out["bienestar_delta_clase"] = dlt_cls

    # Señales de apoyo
    ansiedad = float(d["ansiedad_proxy"].tail(7).mean()) if "ansiedad_proxy" in d.columns else math.nan
    hipomania = float(d["hipomania_proxy"].tail(7).mean()) if "hipomania_proxy" in d.columns else math.nan
    sleep_h = float(d["sueno"].tail(7).mean()) if "sueno" in d.columns else math.nan
    pantallas = float(d["tiempo_pantalla_noche_min"].tail(7).mean()) if "tiempo_pantalla_noche_min" in d.columns else math.nan

    # Etiquetas rápidas
    def lvl_from_val(v):
        if pd.isna(v): return "—"
        if v < 3: return "bajo"
        if v < 7: return "medio"
        return "alto"

    ansiedad_lvl = lvl_from_val(ansiedad)
    hipomania_lvl = lvl_from_val(hipomania)
    out.update({
        "ansiedad_prom_7": round(ansiedad,2) if not math.isnan(ansiedad) else None,
        "ansiedad_nivel": ansiedad_lvl,
        "hipomania_prom_7": round(hipomania,2) if not math.isnan(hipomania) else None,
        "hipomania_nivel": hipomania_lvl,
        "sueno_prom_7h": round(sleep_h,2) if not math.isnan(sleep_h) else None,
        "pantallas_prom_7m": round(pantallas,1) if not math.isnan(pantallas) else None,
    })

    # Generar frase humana corta (templates, ver más abajo)
    # Prioriza: 1) alarmas (ansiedad alta / sueño muy bajo / caída fuerte), 2) mejoras, 3) estabilidad
    templates = {
        "ansiedad_high": "Carga de ansiedad alta. Prioriza sueño y ventanas sin pantallas esta noche.",
        "hipo_high": "Elevada activación e impulsividad. Prioriza sueño y límites en tareas urgentes.",
        "sleep_low": "Sueño reducido: intenta ventana sin pantallas 90′ antes de dormir.",
        "fuerte_up": "Buen impulso: la tendencia muestra mejora sostenida. Mantén lo que funciona.",
        "suave_up": "Suave mejora en la tendencia — pasos constantes, buen trabajo.",
        "fuerte_down": "Retroceso notable. Revisa sueño y cargas del día; prioriza autocuidado.",
        "suave_down": "Ligera bajada en la tendencia. Observa pantallas y micro-reparaciones faltantes.",
        "estable_ok": "Buenos cimientos; afina el descanso para ganar claridad.",
        "low_alert": "Estado bajo — centrémonos en sueño, hidratación y límites con pantallas."
    }

    # reglas
    if ansiedad_lvl == "alto":
        frase = templates["ansiedad_high"]
        accion = "Ventana sin pantallas 90′ · 10′ respiración 4-4"
    elif hipomania_lvl == "alto":
        frase = templates["hipo_high"]
        accion = "Cerrar 1 micro-tarea · 30′ descanso"
    elif not math.isnan(out["bienestar_mean_7"]) and out["bienestar_mean_7"] < 3:
        frase = templates["low_alert"]
        accion = "Priorizar sueño y contacto social breve"
    else:
        # tendencia/combos
        t = out["tendencia"]
        if t == "fuerte ↑":
            frase = templates["fuerte_up"]; accion = "Mantener rutinas claves"
        elif t == "suave ↑":
            frase = templates["suave_up"]; accion = "Reforzar lo que te ayuda"
        elif t == "fuerte ↓":
            frase = templates["fuerte_down"]; accion = "Pausa y revisar sueño"
        elif t == "suave ↓":
            frase = templates["suave_down"]; accion = "Reducir pantallas noche"
        else:
            # estable/medio
            frase = templates["estable_ok"]; accion = "Mantener hábitos y 1 micro-reparación diaria"

    out["frase_humana"] = frase
    out["accion_corta"] = accion
    out["comentario_tecnico"] = f"Días={n_dias} · Confianza={confianza}"
    return out



def tendencia_bienestar(daily):
    ema_now = daily["bienestar_ema"].iloc[-1]
    ema_prev = daily["bienestar_ema"].iloc[-7] if len(daily) >= 7 else daily["bienestar_ema"].iloc[0]
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
    daily["bienestar_ema"] = (
        daily["bienestar"]
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )
    bienestar_datos, bienestar_score, bienestar_nivel, bienestar_fmt, bienestar_clase    = bienestar_delta(daily)
    animo_datos, animo_score, animo_nivel, animo_fmt, animo_clase                    = animo_delta(daily)
    estres_datos, estres_score, estres_nivel, estres_fmt, estres_clase                = estres_delta(daily)
    claridad_datos, claridad_score, claridad_nivel, claridad_fmt, claridad_clase        = claridad_delta(daily)

    kpi = {
        "bienestar_nivel": bienestar_nivel,
        "bienestar_score":  bienestar_score,
        "bienestar_delta_fmt": bienestar_fmt ,
        "bienestar_delta_clase": bienestar_clase,
        "tendencia": tendencia_bienestar(daily),
        "ema_label": "α=0.30 • L=14",
        "animo_nivel": animo_score, "animo_delta_fmt": animo_fmt, "animo_delta_clase": animo_clase,
        "estres_nivel": estres_score, "estres_delta_fmt": estres_fmt, "estres_delta_clase":  estres_clase,
        "claridad_nivel": claridad_score, "claridad_delta_fmt": claridad_fmt, "claridad_delta_clase": claridad_clase,
    }

    return kpi