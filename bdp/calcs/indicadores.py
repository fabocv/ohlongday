import pandas as pd

def variable_nivel(score: float) -> str:
    if pd.isna(score):
        return "—"
    if score < 3:
        return "bajo"
    if score < 7:
        return "medio"
    return "alto"

def indicadores(daily: pd.DataFrame, variable: str, normal = 1.0):
    variable_score_mean = daily[variable].tail(7).mean()
    nivel = variable_nivel(variable_score_mean)
    last = daily[variable].iloc[-1]
    prev = daily[variable].iloc[-2]

    variable_score = "%.2f/10" % variable_score_mean
     
    variable_delta_ayer_hoy = last - prev

    datos = {
        "score_semanal": variable_score_mean,
        "fmt_ayer_hoy": variable_delta_ayer_hoy
    }

    if normal * variable_delta_ayer_hoy > 0:
        signo = "Subió" if normal==1 else "Bajó"
        return datos, variable_score, nivel, ("%s% .1f%s " % (signo, porcentaje_indicador(normal *variable_delta_ayer_hoy, 10), "%" )), "up"
    elif normal * variable_delta_ayer_hoy < 0:
        signo = "Bajó" if normal==1 else "Subió"
        return datos, variable_score, nivel, ("%s %.1f%s " % (signo, porcentaje_indicador(-1*normal *variable_delta_ayer_hoy, 10), "%")) , "down"
    else:
        return datos, variable_score, nivel, "s/variar", "flat"

def porcentaje_indicador (valor, relacion):
    return 100*valor/relacion

