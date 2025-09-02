from bdp.utils.helpers import _preprocess_csv_for_coach
import pandas as pd
import sys, argparse, json

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
    p.add_argument("--email", "-m", help="Filtrar por correo (columna 'correo')")
    return p.parse_known_args(argv)

def init():
    try:
        with open('personal.json') as f:
            d = json.load(f)
            return d['email']
    except Exception:
        return None
        
def dataset(input_csv = input_csv, target_email = target_email):
    # Preprocesa: normaliza hora 'a. m./p. m.', garantiza 'fecha'/'hora', filtra por correo
    data = _preprocess_csv_for_coach(input_csv, target_email)
    print("%i registros, desde el %s HASTA el %s" % (
    len(data), data['fecha'].iloc[[0]].item(), data['fecha'].iloc[[-1]].item() )) 
    return data
    
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

    print("args-email: %s" % args.email)
    
    if (args.email is not None):
        print("Asignando email %s" % args.email)
        target_email = args.email
    else:
        load_mail_user = init()
        if ( load_mail_user ) :
            target_email = load_mail_user
        print ("Asumiendo que todos los registros son suyos.")
        
    dataset(input_csv, target_email)