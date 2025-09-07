import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from bdp.calcs.bienestar import *
from bdp.calcs.bloques import bloques, sleep_hours
from bdp.utils.helpers import *
from bdp.utils.tendencias import *
from bdp.calcs.espiritual import espirit_semana
from bdp.calcs.glicemia import compute_s_gly
from bdp.calcs.sueno import *
from bdp.calcs.ejercicio import *
from bdp.calcs.pantallas import compute_WBN, build_ptn_from_screens
from bdp.calcs.otras_sustancias import *
from bdp.calcs.carga_metabolica import *
from bdp.calcs.graficos import *
from bdp.calcs.ansiedad import *
from bdp.calcs.proxies import *
from bdp.calcs.movimiento import * 
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
  "alimentacion": "mean",
  # sumas
  "meditacion_min":"sum","siesta_min":"sum","tiempo_ejercicio":"sum",
  "juego_en_dispositivo_min":"sum","juego_en_persona_min":"sum",
  "micro_reparaciones":"sum","interacciones_significativas":"sum",
  "cafe_cucharaditas":"sum","alcohol_ud":"sum", "agua_litros":"sum",
  # max/último
  "tiempo_pantalla_noche_min":"max","cafe_ultima_hora":"last","alcohol_ultima_hora":"max",
  "hora_dormir":"last","hora_despertar":"last", "espiritual":"last","sueno_calidad": "max",
  "despertares_nocturnos":"max", "otras_sustancias": "last", "imc": "last", "peso": "last"

}


# ---------- NUEVAS COLUMNAS (v1.6.1) ----------
# Nuevas columnas v1.6.1 (ajustadas a bloques)
agg_updates = {
    # textos / códigos personales
    "eventos_estresores": join_unique,
    "tags": join_unique,
    "notas": join_unique,
    "otras_medicinas": join_unique,

    # minutos / cargas
    "tiempo_pantallas": "sum",
    "naturaleza_min": "sum",

    # calidad numérica o texto libre
    "interacciones_significativas": smart_mean_or_join,

    # flags/modulador (por bloque): al consolidar por día los contaremos
    "mod_uso_hoy": max_01,

    # escalas 0–10 (efectos del modulador)
    "mod_efecto_foco_0_10": mean_num,
    "mod_efecto_relajo_0_10": mean_num,
    "mod_efectos_secundarios_0_10": mean_num,

    # bloque “verdad / límites”
    "amnistia": mean_num,
    "evitacion_verdad": mean_num,
    "acto_verdad": max_01,       # por bloque: ocurrió (1/0)
    "verdad_dificultad": mean_num,
    "verdad_resultado": mean_num,
    "limites_practicados": "sum",

    # movimiento / ejecución
    "movimiento_calidad": mean_num,
    "micro_reparaciones": mean_num,
}

# mezcla sin pisar funciones existentes incompatibles
aggregate = {**agg, **agg_updates}

'''
Funcion principal que calcula los datos del dataframe y 
exporta como json para ser rendereados.
'''
def datos_diarios(df):
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True).dt.date

    daily = aggregate_daily(df, agg=aggregate)           # 1 registro por día
    by_block = aggregate_by_block(df, agg=aggregate)     # largo
    wide_blocks = aggregate_blocks_wide(df, aggregate=aggregate)  # pivoteado (ancho)

    ### rutas de gráficos png  ###

    base = os.getcwd() + "/bdp/templates/out/"
    out_path =  base + "tendencia.png"
    out_path_sueno =  base + "tendencia_sueno.png"
    out_path_actograma_sueno = base + "actograma_sueno.png"
    out_path_screen_sueno = base + "ptn_sueno.png"
    out_path_cm_descomp = base + "cm_descomp.png"
    out_path_cm_waterfall = base + "cm_waterfall.png"
    out_path_dash = base + "dashboard_semana_bdp.png"
    out_png_hydra_alim = base + "hydra_alim.png"
    out_path_hipo_ans = base + "hipo_ans.png"
    out_path_proxies = base + "proxies.png"
    out_path_movimiento = base + "mov.png"
    out_path_mono = base + "ms_monotony.png"
    out_path_str = base+ "ms_strain.png"

    proxies, bands = compute_proxy_indices(daily, alpha=0.30)
    daily_with_proxies = daily.merge(proxies, on="fecha", how="left")
    out_png = plot_proxies_stacked_seaborn(proxies, bands, out_path = out_path_proxies)
    msje_proxies = describe_proxies(proxies)

    acwr_df = compute_acwr(daily)
    png_path, last_val = plot_acwr_seaborn(acwr_df, out_path=out_path_movimiento)

    ms = compute_monotony_strain(daily)
    mono_png, mono_last = plot_monotony_seaborn(ms, height_px=250, out_path=out_path_mono)
    strain_png, strain_last = plot_strain_seaborn(ms, height_px=250, out_path=out_path_str)

    context.update({ "datos": {"validos": len(daily)}})

    #bloque sueño semanal usandoc columna calculada: sueno
    daily["sueno"] = sleep_hours_rowwise(daily["hora_dormir"], daily["hora_despertar"])
    daily = attach_main_sleep_to_daily(daily, df)      # noche principal
    daily = sleep_user(df, daily)                      # siestas por bloques → daily
    daily["s_sleep"] = s_sleep_carga_de_sueno_total_con_siestas(daily).clip(0,10)

    

    #s_gly, s_gly_plot, meta_gly = compute_s_gly(daily, plot_impute=True)
    #s_caf = s_exceso(daily["cafe_cucharaditas"], limite=6, k=2).fillna(0)  # ~2 pts por cdta extra
    #s_alc = s_exceso(daily["alcohol_ud"], limite=0, k=2).fillna(0)  # o limite=1 si aceptas 1 UD
    #s_mov = s_u_shape(daily["mov_intensidad"], A=4.0, B=7.0, k=2).fillna(0)


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
    #daily = prepare_for_merge_on_fecha(df, sub_daily, left_col="fecha", right_col="fecha")

    #daily["sueno"] = sleep_hours_rowwise(daily["hora_dormir"], daily["hora_despertar"])
    
    #daily[["sleep_reh","sleep_reh_adj","sleep_deficit_h","s_sleep","sleep_episodes"]] = daily.apply(sleep_effective_hours, axis=1)

    # 2) Llama a tu compute_CM usando las columnas derivadas
    CM, cm_meta, metabolicos = compute_CM(
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

    

    daily["carga_metabolica"] = CM.to_numpy()

    daily = add_affect_proxies(daily, ema_days=7)
    
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

    PTN_d1 = daily["PTN_d1"]
    #print(daily[["PTN_d1","sueno_noche_h","siesta_full_min","siesta_total_h","sleep_total_h","sleep_reh_adj"]].tail(7))

    

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

    
    daily["bienestar"] = WB
    daily["bienestar_ema"] = WB_ema
    daily["malestar"] = MB
    daily["malestar_ema"] = MB_ema

    # 2) Calcular WBN = WB - MB - PTN_d1
    daily["bienestar_neto"] = compute_WBN(daily, col_WB="bienestar_ema", col_MB="malestar_ema", col_PTN_d1="PTN_d1", clip_0_10=True)

    res = figura_hidratacion_y_alimentacion(daily, out_png=out_png_hydra_alim, lookback_days=30)

    

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
    
    #daily["PTN_d1"] = PTN_d1.fillna(0)
    
    

    # Normalizaciones simples a 0–10 (ejemplo)
    for col in ["animo","activacion","conexion","proposito","claridad","estres"]:
        if col in daily:
            daily[col] = daily[col].clip(0,10)

    
    

    kpi = kpi_bienestar(daily)

    context.update({ 
        "kpi": kpi, 
        "charts": {
            "main_src": out_path,
            "cm_png": out_path_cm,
            'sueno_png': out_path_sueno,
            'actograma_sueno': out_path_actograma_sueno,
            'ptn_sueno': out_path_screen_sueno,
            'cm_descomp': out_path_cm_descomp,
            'cm_waterfall': out_path_cm_waterfall,
            'dash': out_path_dash,
            "hydra_alim": out_png_hydra_alim,
            "hipo_ans": out_path_hipo_ans,
            "proxies": out_path_proxies,
            "msje_proxies": msje_proxies,
            "movimiento": out_path_movimiento,
            "ms_mono": out_path_mono,
            "ms_strain":out_path_str,
            "dias": 7},
        }
    )

    data_bloques = bloques(daily, df)
    context.update({"bloques": data_bloques})


    # Confianza
    n_dias = daily["fecha"].nunique()
    conf = "baja" if n_dias < 10 else ("media" if n_dias < 30 else "alta")
    print(f"Días={n_dias}, Entradas={len(df)}, Confianza={conf}")

    for nombre_met in metabolicos.keys():
        daily[nombre_met] = metabolicos[nombre_met]

    png = dashboard_semana_bdp(
        daily,
        out_png=out_path_dash,
        panels=("actograma","ptn","cm_heat","waterfall"),  # orden y selección
        font_scale=0.9,
        suptitle="Resumen semanal · BDP"
    )
    # 14) Actograma
    #res14 = grafico_actograma_sueno(daily, out_png=out_path_actograma_sueno)

    # 5) PTN vs Sueño siguiente
    #res5  = grafico_ptn_vs_sueno_siguiente(daily, out_png=out_path_screen_sueno, out_svg=out_path_screen_sueno.replace("png", "svg"))

    
    # 12) Waterfall ΔCM (último día)
    #res12 = grafico_cm_waterfall(daily, out_png= out_path_cm_waterfall, out_svg=out_path_cm_waterfall.replace("png", "svg"))


    res = figura_ansiedad_hipomania_png(
        daily,
        out_png=out_path_hipo_ans,
        lookback_days=60,
        ema_days=7
    )

    res = grafico_cm_sparkline_seaborn(
        daily,
        out_png= out_path_cm,
        out_svg= out_path_cm.replace(".png",".svg"),   # opcional
        width_px=800,                     # ajusta al contenedor
        height_px=80,                     # jamás excede 50 px
        y_domain=(0,10),
        show_band=True
    )

    res = generar_grafico_bienestar_seaborn(
        daily,
        out_html=out_path.replace(".png", ".html"),
        out_png=out_path,   # opcional
        #title="Tendencia de Bienestar Neto (7–14d)"
    )

 # 4) Descomposición CM
    res = grafico_cm_last7(daily, out_png=out_path_cm_descomp,
                       kind="multiples", mode="raw", font_scale=0.85)
    
    res = grafico_sueno_y_bienestar_seaborn(
        daily,
        out_png= out_path_sueno,
        out_svg= out_path_sueno.replace("png","svg")  # opcional
    )

    dias_de_riesgo =  res["risk_days"]

    out = calc_resumen_plus(daily=daily,col="WBN_ex")

    context.update({
        "resumen": {"estado": out["estado"], "frase_humana": out["frase_humana"]},
    })

    return daily


def gen_informe(df: pd.DataFrame) :
    diarios = datos_diarios(df=df)
    tpl = env.get_template("main.html")
    html_rendered = tpl.render(**context)
    HTML(string=html_rendered, base_url=".").write_pdf("Mi BDP.pdf")
    print("reporte generado como 'Mi BDP.pdf")
    return diarios
