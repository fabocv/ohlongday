# main.py
# Ejecuta el reporte "coach" con motivaciones intercaladas.
# Requisitos: pandas, matplotlib, numpy
#   pip install pandas matplotlib numpy
#
# Ajusta las rutas de entrada/salida según tu entorno.
from reporte_python.bdp_report_coach import generate_report_coach

def generate_report(INPUT, OUTPUT):
    path = generate_report_coach(INPUT, OUTPUT)
    print(f"Reporte generado en: {path}")
