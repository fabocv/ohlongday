from bdp.utils.helpers import _preprocess_csv_for_coach
import pandas as pd
import sys, argparse

input_csv ="bdp_sample.csv"
targets = "animo,claridad,estres,activacion".split(",")
output ="./outputs"
target_email = "test@gmail.com"
lookback_days = 6
row_idx = -1
target_day = "26-08-2025"



def parse_known_args(argv=None):
    p = argparse.ArgumentParser(description="Genera el informe BDP (coaching).")
    p.add_argument("--input", "-i", default="data/bdp_sample.csv", help="Ruta CSV de entrada")
    p.add_argument("--output", "-o", default="output/informe.html", help="Ruta HTML de salida")
    p.add_argument("--config", "-c", default="reporte_python.bdp_config.json", help="Ruta JSON de configuración (opcional)")
    p.add_argument("--start", "-s", help="Fecha inicio (dd-MM-YYYY)")
    p.add_argument("--end", "-e", help="Fecha fin (dd-MM-YYYY)")
    p.add_argument("--tag", "-t", help="Filtrar por tag (subcadena)")
    p.add_argument("--email", "-m", default=target_email, help="Filtrar por correo (columna 'correo')")
    return p.parse_known_args(argv)

def dataset(input_csv = input_csv, target_email = target_email):
    # Preprocesa: normaliza hora 'a. m./p. m.', garantiza 'fecha'/'hora', filtra por correo
    return _preprocess_csv_for_coach(input_csv, target_email)
    
if __name__ == "__main__":
    args, _ = parse_known_args(sys.argv[1:])
    cfg = None
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = None

    if (args.input is not None):
        input_csv = args.input
    else:
        print("Leyendo archivo CSV '%s' (defecto)" % input_csv)
        
    if (args.email is not None):
        target_email = args.email
    else:
        print ("Asumiendo que todos los registros son suyos.")
        
    dataset(input_csv, target_email)