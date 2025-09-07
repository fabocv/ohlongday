import pandas as pd
from bdp.calcs.indicadores import indicadores
from bdp.utils.helpers import sleep_hours
from bdp.calcs.espiritual import espirit_semana
from bdp.calcs.heuristicas import render_bloque, frase_bloque, insight_bloque_tema, render_dashboard


def bloque(daily, columna):
    datos, score_txt, nivel_txt, fmt_txt, clase_txt = indicadores(daily=daily, variable=columna)
    frase_del_bloque = frase_bloque(columna if columna!="sueno" else "sueno", datos["score_semanal"])  
    return datos, score_txt, nivel_txt, fmt_txt, clase_txt, frase_del_bloque, "insight " + columna

def bloque_espiritual(df):
    df_esp, semanal_fair, semanal_disciplina, resumen = espirit_semana(df)
    score = semanal_fair[0]
    frase_del_bloque = frase_bloque("sueno", score)  
    score_txt = "%.2f/10" % semanal_fair[0]
    _, insight = insight_bloque_tema("sueno", score)

    return score_txt, frase_del_bloque, insight


def bloques(daily: pd.DataFrame, df: pd.DataFrame):

    df2, fair, disciplina, resumen = espirit_semana(df=daily)
    daily = df2.copy()
    
    THEME = "calido"
    # => bloques['espiritual'] -> {title, score_txt, frase, insight}
    info = render_dashboard(daily, theme=THEME, modo='fair', consistencias={"espiritual":0.8}) 

    actv = info["activacion"]
    sueno = info["sueno"]
    conex = info["conexion"]
    spirit = info["espiritual"]
    c_met = info["carga_metabolica"]

    

    new_bloques = {
        "activacion":   {"nivel":actv["score_txt"],"frase":actv["frase"],"insight":actv["insight"]},
        "sueno":        {"calidad":sueno["score_txt"],"frase":sueno["frase"],"insight":sueno["insight"]},
        "conexion":     {"nivel":conex["score_txt"],"frase":conex["frase"],"insight":conex["insight"]},
        "espiritualidad":{"nivel": spirit["score_txt"],"frase": spirit["frase"],"insight":spirit["insight"]},
        "carga_metab":  {"nivel": c_met["score_txt"],"frase": c_met["frase"],"insight":c_met["insight"]},
    }

    return new_bloques