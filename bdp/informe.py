import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from bdp.calcs.bienestar import *
from bdp.calcs.bloques import bloques, sleep_hours
from bdp.utils.helpers import *
from bdp.utils.tendencias import generar_grafico_bienestar_seaborn
from bdp.calcs.espiritual import espirit_semana
from bdp.calcs.glicemia import compute_s_gly
from bdp.calcs.sueno import *
from bdp.calcs.ejercicio import *
from bdp.calcs.pantallas import compute_WBN, build_ptn_from_screens
from bdp.calcs.otras_sustancias import *
from bdp.calcs.carga_metabolica import *
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
  "tiempo_pantalla_noche_min":"max","cafe_ultima_hora":"last","alcohol_ultima_hora":"max",
  "hora_dormir":"last","hora_despertar":"last", "espiritual":"last","sueno_calidad": "max",
  "despertares_nocturnos":"max", "otras_sustancias": "last"

}



'''
Funcion principal que calcula los datos del dataframe y 
exporta como json para ser rendereados.
'''
def datos_diarios(df):
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True).dt.date
    daily = df.groupby("fecha").agg(agg).reset_index()

    context.update({ "datos": {"validos": len(daily)}})

    #bloque sueño semanal usandoc columna calculada: sueno
    daily["sueno"] = sleep_hours_rowwise(daily["hora_dormir"], daily["hora_despertar"])
    s_sleep = s_sleep_carga_de_sueno_total(daily).fillna(0)
    s_gly, s_gly_plot, meta_gly = compute_s_gly(daily, plot_impute=True)
    s_caf = s_exceso(daily["cafe_cucharaditas"], limite=6, k=2).fillna(0)  # ~2 pts por cdta extra
    s_alc = s_exceso(daily["alcohol_ud"], limite=0, k=2).fillna(0)  # o limite=1 si aceptas 1 UD
    s_mov = s_u_shape(daily["mov_intensidad"], A=4.0, B=7.0, k=2).fillna(0)
   
    # Carga metabólica (CM) 0–10
    # 1) Deriva columnas desde 'otras_sustancias'
    deriv = parse_otras_sustancias(daily["otras_sustancias"])
    daily = pd.concat([daily, deriv], axis=1)

    sub_daily = sustancias_to_daily(df, fecha_col="fecha", bloque_col="bloque", otras_col="otras_sustancias")

    # B) Convierte eventos → equivalentes
    sub_daily = eventos_a_equivalentes(sub_daily, event_to_cig=1.0, event_to_joint=1.0, night_bonus=0.25)

    # C) Une con tu 'daily' principal (WB/CM/sueño/etc.)
    # A) sub_daily viene de sustancias_to_daily(df_raw, ...)
    # B) Tu df crudo “trabajado” (no daily) está en `df_raw_trabajado`
    daily = prepare_for_merge_on_fecha(df, sub_daily, left_col="fecha", right_col="fecha")

    daily["sueno"] = sleep_hours_rowwise(daily["hora_dormir"], daily["hora_despertar"])
    # 2) Llama a tu compute_CM usando las columnas derivadas
    CM, cm_meta = compute_CM(
        daily,
        col_sleep_score="s_sleep",
        col_glicemia="glicemia",                 # si no existe, renormaliza
        col_alcohol_units="alcohol_ud",
        col_cafe_cucharaditas="cafe_cucharaditas",
        col_mov_min="tiempo_ejercicio",          # o col_mov_intensidad="mov_intensidad"
        col_alim_score="alimentacion",         # ya en 0–10
        col_agua_litros="agua_litros",
        # ← aquí las derivadas:
        col_nic_mg="nicotina_mg",
        col_nic_puffs="nicotina_puffs",
        col_thc_mg="thc_mg",
        col_nic_cigs="nic_cigs_equiv",
        col_thc_joints="thc_joints_equiv",
    )


    daily["carga_metabolica"] = CM

     # Activación efectiva: penaliza fuera de [4,7]; 2.5 = 10/dmax con dmax=4
    daily["activacion_eff"] = daily["activacion"].astype(float).pipe(
        lambda s: (s * (1 - ((2.5 * ((4 - s).clip(lower=0) + (s - 7).clip(lower=0))).clip(upper=10) / 10)))).clip(lower=0)

    #Malestar
    MB = (0.22*daily["estres"] + 0.20*daily["irritabilidad"] + 0.20*daily["ansiedad"]
      + 0.13*daily["dolor_fisico"] + 0.25*CM).clip(0,10)

    # Bienestar (WB) y Neto (WBN)
    WB  = (0.3*daily["animo"] + 0.20*daily["claridad"] + 0.20*daily["conexion"]
        + 0.20*daily["activacion_eff"] + 0.10*daily["proposito"] - 0.10*daily["estres"])
    WB_ema = WB.ewm(alpha=0.30, adjust=False).mean()
    MB_ema = MB.ewm(alpha=0.30, adjust=False).mean()
    WBN    = (WB_ema - MB_ema)

    # 1) Construir PTN con bloques (suma por día)
    daily = build_ptn_from_screens(
        daily,
        col_fecha="fecha",
        col_noche_explicit="tiempo_pantalla_noche_min",   # si no la tienes, omite y usa col_noche_block
        col_manana="tiempo_pantallas",          # opcional
        col_tarde="tiempo_pantallas",            # opcional
        col_noche_block="tiempo_pantalla_noche_min",      # si no tienes la explícita, puedes reutilizar
        apply_shift_d1=True,                               # efecto al día siguiente
        gamma_luz_azul=0.6                                # ↑ si quieres más sensibilidad a últimas 2h
    )

    daily["bienestar"] = WB
    daily["bienestar_ema"] = WB_ema
    daily["malestar"] = MB
    daily["malestar_ema"] = MB_ema

    # 2) Calcular WBN = WB - MB - PTN_d1
    daily["bienestar_neto"] = compute_WBN(daily, col_WB="bienestar_ema", col_MB="malestar_ema", col_PTN_d1="PTN_d1", clip_0_10=True)


    daily = compute_exercise_effect(
        daily,
        col_min="tiempo_ejercicio",
        col_int="mov_intensidad",
        col_hora_ej="hora_ejercicio",      # si no la tienes, la función se adapta
        col_hora_dormir="hora_dormir",     # si no la tienes, factor horario=1.0
        alpha_ex=0.4,
        rho_today=0.35,
        rho_d1=0.25
    )

    daily = apply_exercise_to_WBN(
        daily,
        col_WB="bienestar_ema",
        col_MB="malestar_ema",
        col_PTN_today="PTN_today",
        col_PTN_d1="PTN_d1",
        relief_MB_mu=0.25,   # pon 0.0 si no quieres tocar MB
        clip_0_10=True
    ) # WBN_ex

    out_path_cm =  os.getcwd() + "/bdp/templates/cm_sparkline.png" 
    res = grafico_cm_sparkline_seaborn(
        daily,
        out_png= out_path_cm,
        out_svg= out_path_cm.replace(".png",".svg"),   # opcional
        width_px=800,                     # ajusta al contenedor
        height_px=70,                     # jamás excede 50 px
        y_domain=(0,10),
        show_band=True
    )
    

    # Normalizaciones simples a 0–10 (ejemplo)
    for col in ["animo","activacion","conexion","proposito","claridad","estres"]:
        if col in daily:
            daily[col] = daily[col].clip(0,10)

    

    out_path =  os.getcwd() + "/bdp/templates/tendencia.png"
    

    kpi = kpi_bienestar(daily)

    context.update({ 
        "kpi": kpi, 
        "charts": {
            "main_src": out_path,
            "cm_png": out_path_cm,
            "dias": 7},
        }
    )

    data_bloques = bloques(daily, df)
    context.update({"bloques": data_bloques})


    # Confianza
    n_dias = daily["fecha"].nunique()
    conf = "baja" if n_dias < 10 else ("media" if n_dias < 30 else "alta")
    print(f"Días={n_dias}, Entradas={len(df)}, Confianza={conf}")


    res = generar_grafico_bienestar_seaborn(
        daily,
        out_html=out_path.replace(".png", ".html"),
        out_png=out_path,   # opcional
        #title="Tendencia de Bienestar Neto (7–14d)"
    )

    out = calc_resumen_plus(daily=daily,col="WBN_ex")

    context.update({
        "resumen": {"estado": out["estado"], "frase_humana": out["frase_humana"]},
    })

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
