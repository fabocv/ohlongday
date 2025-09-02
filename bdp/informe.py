import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from bdp.bienestar import calc_resumen
from bdp.utils.tendencias import generar_grafico_tendencias
from weasyprint import HTML
import os

env = Environment(
    loader=FileSystemLoader("bdp/templates"), autoescape=select_autoescape(["html", "xml"])
)


context = {
    "periodo": {"titulo": "Semana 35", "rango_humano": "25–31 Ago"},
    "resumen": {"estado": "estable", "frase_humana": "Buenos cimientos; afina el descanso."},
    "kpi": {
        "bienestar_nivel": "medio",
        "bienestar_score": "6.8/10",
        "bienestar_delta_fmt": "+0.42",
        "bienestar_delta_clase": "up",
        "tendencia": "suave ↑",
        "ema_label": "α=0.30 • L=14",
        "animo_nivel": "7/10", "animo_delta_fmt": "-0.04", "animo_delta_clase": "flat",
        "estres_nivel": "4/10", "estres_delta_fmt": "+0.37", "estres_delta_clase": "down",
        "claridad_nivel": "7.5/10", "claridad_delta_fmt": "+1.25", "claridad_delta_clase": "up",
    },
    "datos": {"validos": 13},
    "narrativa": {"intro":"…", "destacado":"…"},
    "charts": {"trend_src": None, "dias": 7},
    "bloques": {
        "activacion":{"nivel":"6/10","frase":"…","insight":"…"},
        "sueno":{"calidad":"7/10","frase":"…","insight":"…"},
        "conexion":{"nivel":"6/10","frase":"…","insight":"…"},
        "espiritualidad":{"nivel":"3/5","frase":"…","insight":"…"},
        "microrep":{"cerradas":"2","frase":"…","insight":"…"},
        "juego":{"minutos":"45","frase":"…","insight":"…"},
    },
    "causas":{"elevadores":["…"],"cargas":["…"]},
    "momentos":{"stoic":{"fecha":"2025-08-29","motivo":"…"},
                "hard":{"fecha":"2025-08-27","aprendizaje":"…"}},
    "retro":{"hoy":"↑","hace_7":"↔"},
    "sugerencias":{"mantener":["…"],"probar":["…"]},
    "notas":{"tags":["rutina","descanso","limites"]},
}



# ---- Helpers
def to_minutes(t):
    # t en "HH:MM"
    h, m = map(int, str(t).split(":"))
    return h*60 + m

def sleep_hours(gr):
    # usa último dormir y último despertar del día; si despertar < dormir, cruza medianoche
    try:
        d = to_minutes(gr["hora_dormir"].dropna().iloc[-1])
        w = to_minutes(gr["hora_despertar"].dropna().iloc[-1])
        if w < d: w += 24*60
        siesta = gr.get("siesta_min", pd.Series([0])).fillna(0).sum()
        return (w - d)/60 + siesta/60
    except Exception:
        return np.nan

agg = {
  # medias
  "animo":"mean","activacion":"mean","conexion":"mean","proposito":"mean","claridad":"mean","estres":"mean",
  "mov_intensidad":"mean","glicemia":"mean",
  # sumas
  "meditacion_min":"sum","siesta_min":"sum","tiempo_ejercicio":"sum",
  "juego_en_dispositivo_min":"sum","juego_en_persona_min":"sum",
  "micro_reparaciones":"sum","interacciones_significativas":"sum",
  "cafe_cucharaditas":"sum","alcohol_ud":"sum",
  # max/último
  "tiempo_pantalla_noche_min":"max","cafe_ultima_hora":"max","alcohol_ultima_hora":"max",
  "hora_dormir":"last","hora_despertar":"last",
}

def variable_nivel(score: float) -> str:
    if pd.isna(score):
        return "—"
    if score < 3:
        return "bajo"
    if score < 7:
        return "medio"
    return "alto"

def indicadores(daily, variable, normal = 1.0):
    variable_score_mean = daily[variable].tail(7).mean()
    nivel = variable_nivel(variable_score_mean)
    last = daily[variable].iloc[-1]
    prev = daily[variable].iloc[-2]

    variable_score = "%.2f/10" % variable_score_mean
     
    variable_delta_ayer_hoy = last - prev

    

    if normal * variable_delta_ayer_hoy > 0:
        signo = "+" if normal==1 else "-"
        return variable_score, nivel, ("%s%.2f" % (signo, (normal *variable_delta_ayer_hoy))), "up"
    elif normal * variable_delta_ayer_hoy < 0:
        signo = "-" if normal==1 else "+"
        return variable_score, nivel, ("%s%.2f" % (signo, (-1*normal *variable_delta_ayer_hoy))) , "down"
    else:
        return variable_score, nivel, "s/variar", "flat"

def bienestar_delta(daily: pd.DataFrame):
    return indicadores(daily,"bienestar")

def animo_delta(daily: pd.DataFrame):
    return indicadores(daily,"animo")

def estres_delta(daily: pd.DataFrame):
    return indicadores(daily,"estres", normal = -1.0)

def claridad_delta(daily: pd.DataFrame):
    return indicadores(daily,"claridad")

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

def bienestar(daily):
    alpha = 0.30  # el peso a lo más reciente
    daily = daily.sort_values("fecha")  # asegúrate que está ordenado
    daily["bienestar_ema"] = (
        daily["bienestar"]
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )
    bienestar_score, bienestar_nivel, bienestar_fmt, bienestar_clase = bienestar_delta(daily)
    animo_score, animo_nivel, animo_fmt, animo_clase = animo_delta(daily)
    estres_score, estres_nivel, estres_fmt, estres_clase = estres_delta(daily)
    claridad_score, claridad_nivel, claridad_fmt, claridad_clase = claridad_delta(daily)

    out = calc_resumen(daily=daily)

    out_path =  os.getcwd() + "/bdp/templates/tendencia.png"
    grafico = generar_grafico_tendencias(daily, out_path, out_path)

    context.update({
        "resumen": {"estado": out["estado"], "frase_humana": out["frase_humana"]},
    })

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

    context.update({ "kpi": kpi, "charts": {"main_src": out_path, "dias": 7},})


def datos_diarios(df):
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True).dt.date
    daily = df.groupby("fecha").agg(agg).reset_index()
    daily["sueno_duracion_h"] = df.groupby("fecha").apply(sleep_hours).values

    # Normalizaciones simples a 0–10 (ejemplo)
    for col in ["animo","activacion","conexion","proposito","claridad","estres"]:
        if col in daily:
            daily[col] = daily[col].clip(0,10)

    # Bienestar compuesto (ejemplo legible; ajusta pesos luego)
    daily["bienestar"] = (
        daily["animo"]*0.25 + daily["claridad"]*0.25 + daily["conexion"]*0.20 +
        daily["activacion"]*0.15 + (10 - daily["estres"])*0.15
    )

    bienestar(daily)

    # EMA (alpha≈0.30); ajusta lookback en la interpretación, no en ewm
    daily = daily.sort_values("fecha")
    daily["bienestar_ema"] = daily["bienestar"].ewm(alpha=0.30, adjust=False).mean()
    daily["estres_inv"]   = (10 - daily["estres"]).clip(0,10)
    daily["estres_inv_ema"]= daily["estres_inv"].ewm(alpha=0.30, adjust=False).mean()

    # Confianza
    n_dias = daily["fecha"].nunique()
    conf = "baja" if n_dias < 10 else ("media" if n_dias < 30 else "alta")
    print(f"Días={n_dias}, Entradas={len(df)}, Confianza={conf}")
    return daily

def preparar_datos(daily: pd.DataFrame):
    # fecha, bienestar_ema, estres_inv_ema, ansiedad_proxy, ansiedad_nivel, hipomania_proxy, hipomania_nivel
    last_n = 10
    rows = []
    for _, r in daily.tail(last_n).iterrows():
        rows.append({
            "fecha": str(r["fecha"]),
            "bienestar_ema": float(r.get("bienestar_ema", float("nan"))),
            "estres_inv_ema": float(r.get("estres_inv_ema", float("nan"))),
            "ansiedad_proxy": float(r.get("ansiedad_proxy", float("nan"))),
            "ansiedad_nivel":  str(r.get("ansiedad_nivel", "—")),
            "hipomania_proxy": float(r.get("hipomania_proxy", float("nan"))),
            "hipomania_nivel": str(r.get("hipomania_nivel", "—")),
        })

    context.update({
        "charts": {"main_src": "chart_bienestar.png", "dias": last_n},
        "tabla":  {"titulo": f"últimos {last_n} días", "rows": rows},
    })



def gen_informe(df: pd.DataFrame) :
    diarios = datos_diarios(df=df)
    tpl = env.get_template("main.html")
    html_rendered = tpl.render(**context)
    HTML(string=html_rendered, base_url=".").write_pdf("Mi BDP.pdf")
    print("reporte generado como 'Mi BDP.pdf")
    return diarios




