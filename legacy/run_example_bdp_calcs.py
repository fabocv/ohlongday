
# -*- coding: utf-8 -*-
import pandas as pd
from bdp_calcs import (
    parsear_tiempos, calc_sueno, calc_estimulantes, calc_actividad_luz_pantalla,
    calc_psicosocial, calc_medicacion_sustancias, calc_indices, calc_temporales
)

# Carga o arma un df mínimo (demo)
df = pd.DataFrame({
    "fecha": ["2025-08-20","2025-08-21","2025-08-22"],
    "hora": ["09:00","09:10","08:55"],
    "horas_sueno": [6.5, 7.8, 8.2],
    "sueno_calidad": [6, 7, 8],
    "hora_dormir": ["23:45","23:30","00:10"],
    "hora_despertar": ["07:00","07:10","07:15"],
    "despertares_nocturnos": [2,1,0],
    "cafe_cucharaditas": [2,1,2],
    "cafe_ultima_hora": ["14:30","16:00","13:00"],
    "alcohol_ud": [0,1,0],
    "alcohol_ultima_hora": ["", "21:30", ""],
    "agua_litros": [1.5, 2.0, 1.2],
    "mov_intensidad": [5,6,4],
    "meditacion_min": [10,15,0],
    "exposicion_sol_manana_min": [10,20,5],
    "tiempo_pantalla_noche_min": [60,30,90],
    "interacciones_significativas": ["2", "3", "1"],
    "interacciones_calidad": [7,8,6],
    "eventos_estresores": ["plazo, multitarea", "conflicto", ""],
    "otras_sustancias": ["nicotina", "", ""],
    "animo": [6,7,6],
    "claridad": [6,7,7],
    "estres": [5,4,6],
    "activacion": [6,6,7],
})

df = parsear_tiempos(df)
df = calc_sueno(df)
df = calc_estimulantes(df)
df = calc_actividad_luz_pantalla(df)
df = calc_psicosocial(df)
df = calc_medicacion_sustancias(df)
df = calc_indices(df)
df = calc_temporales(df, ema_cols=["animo","claridad","estres","activacion"], lagcols=["animo","claridad","estres","activacion"])

# Guarda resultado
df.to_csv("/mnt/data/demo_bdp_calcs_output.csv", index=False, encoding="utf-8")
print("Demo listo: /mnt/data/demo_bdp_calcs_output.csv")
