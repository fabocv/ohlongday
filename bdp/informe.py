import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from bdp.calcs.bienestar import *
from bdp.calcs.bloques import bloques, sleep_hours
from bdp.utils.tendencias import generar_grafico_tendencias_altair
from bdp.calcs.espiritual import espirit_semana
from weasyprint import HTML
import os

env = Environment(
    loader=FileSystemLoader("bdp/templates"), autoescape=select_autoescape(["html", "xml"])
)


context = {
    "periodo": {"titulo": "Semana 35", "rango_humano": "26 Ago – 2 Sept"},
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


agg = {
  # medias
  "animo":"mean","activacion":"mean","conexion":"mean","proposito":"mean","claridad":"mean","estres":"mean",
  "mov_intensidad":"mean","glicemia":"mean", "irritabilidad": "mean", "ansiedad": "mean", "dolor_fisico": "mean",

  # sumas
  "meditacion_min":"sum","siesta_min":"sum","tiempo_ejercicio":"sum",
  "juego_en_dispositivo_min":"sum","juego_en_persona_min":"sum",
  "micro_reparaciones":"sum","interacciones_significativas":"sum",
  "cafe_cucharaditas":"sum","alcohol_ud":"sum",
  # max/último
  "tiempo_pantalla_noche_min":"max","cafe_ultima_hora":"max","alcohol_ultima_hora":"max",
  "hora_dormir":"last","hora_despertar":"last", "espiritual":"last","sueno_calidad": "max",
  "despertares_nocturnos":"max"

}





def datos_diarios(df):
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True).dt.date
    daily = df.groupby("fecha").agg(agg).reset_index()

    
    #bloque sueño semanal usandoc columna calculada: sueno
    daily["sueno"] = daily.groupby("fecha").apply(sleep_hours).values

    daily["estres_inv"]   = (10 - daily["estres"]).clip(0,10)
    # Bienestar WB
    # WB=f(Aˊnimo,Activacioˊn,Conexioˊn,Propoˊsito,Claridad)−g(Estreˊs percibido)
    daily["bienestar"] = (
        daily["animo"]*0.25 + daily["claridad"]*0.25 + daily["conexion"]*0.20 +
        daily["activacion"]*0.15 + daily["proposito"]*0.15
    ) - daily["estres_inv"]*0.1

    # EMA (alpha≈0.30); ajusta lookback en la interpretación, no en ewm
    #daily = daily.sort_values("fecha")
    daily["bienestar_ema"] = daily["bienestar"].ewm(alpha=0.30, adjust=False).mean()
    print(daily["bienestar_ema"])
    daily["estres_inv_ema"]= daily["estres_inv"].ewm(alpha=0.30, adjust=False).mean()
    

    limite_cafe_diario = 6
    cafe_exceso = (daily["cafe_cucharaditas"] - limite_cafe_diario).clip(lower=0)
    # target 7.5h -> ajusta a tu realidad
    sleep_def = (7.5 - daily["sueno"]).clip(lower=0)  # más déficit => más carga

    A, B = 4.0, 7.0
    mov = daily["mov_intensidad"]
    mov_costo = (A - mov).clip(lower=0) + (mov - B).clip(lower=0)

    def despertares_escalonada(interrupciones):
        if(interrupciones == 0): return 0
        if(interrupciones == 1): return 2
        if(interrupciones == 2): return 5
        if(interrupciones == 3): return 8
        else: return 10

    #CMt​=w1​⋅(Suen˜o deficiente)+w2​⋅(Glicemia anoˊmala)+w3​⋅(Alcohol/cafeıˊna)+w4​⋅(Deˊficit/exceso de movimiento)
    if daily["glicemia"].isna:
        daily["carga_metabolica"] = (
            0.40*sleep_def.fillna(0) +
            0.15*daily["alcohol_ud"].fillna(0) +
            0.10*cafe_exceso.fillna(0) +
            0.35*mov_costo.fillna(0)
        )
    else:
        daily["carga_metabolica"] = (
            0.35*sleep_def.fillna(0) +
            0.10*daily["glicemia"] +
            0.15*daily["alcohol_ud"].fillna(0) +
            0.10*cafe_exceso.fillna(0) +
            0.30*mov_costo.fillna(0)
        )

    print(daily["carga_metabolica"])

    # Malestar MBt
    # MBt​=h(Estreˊs percibido,Irritabilidad,Dolor fıˊsico,Suen˜o deficiente,Carga metaboˊlica)

    #
    calidad_sueno = 10 - daily["sueno_calidad"]
    
    daily["malestar"] =  (
        daily["estres"]*0.18 + daily["irritabilidad"]*0.22 + daily["ansiedad"]*0.2 + daily["dolor_fisico"]*0.12 
        + daily["carga_metabolica"] *0.12 + (calidad_sueno) * 0.16
    )

    daily["malestar_ema"] = daily["malestar"].ewm(alpha=0.30, adjust=False).mean()

    print(daily["malestar_ema"])

    daily["bienestar_neto_ema"] = daily["bienestar_ema"] - daily["malestar_ema"]

    print(daily["bienestar_neto_ema"])
    

    # Normalizaciones simples a 0–10 (ejemplo)
    for col in ["animo","activacion","conexion","proposito","claridad","estres"]:
        if col in daily:
            daily[col] = daily[col].clip(0,10)

    

    out_path =  os.getcwd() + "/bdp/templates/tendencia.png"
    out = calc_resumen(daily=daily)

    context.update({
        "resumen": {"estado": out["estado"], "frase_humana": out["frase_humana"]},
    })

    kpi = kpi_bienestar(daily)

    context.update({ "kpi": kpi, "charts": {"main_src": out_path, "dias": 7},})

    data_bloques = bloques(daily, df)
    context.update({"bloques": data_bloques})


    # Confianza
    n_dias = daily["fecha"].nunique()
    conf = "baja" if n_dias < 10 else ("media" if n_dias < 30 else "alta")
    print(f"Días={n_dias}, Entradas={len(df)}, Confianza={conf}")


    res = generar_grafico_tendencias_altair(
        daily,
        out_html=out_path.replace(".png", ".html"),
        out_png=out_path,   # opcional
        smooth_raw=True,
        title="Tendencia de Bienestar (7–14d)"
    )

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
