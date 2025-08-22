import argparse, json
from reporte_python.bdp_report_coach import generate_report_coach

def parse_args():
    p = argparse.ArgumentParser(description="Genera el informe BDP (coaching)." )
    p.add_argument("--input", "-i", default="data/bdp_data_fake.csv", help="Ruta CSV de entrada")
    p.add_argument("--output", "-o", default="output/reporte.html", help="Ruta HTML de salida")
    p.add_argument("--config", "-c", default="reporte_python/bdp_config.json", help="Ruta JSON de configuración (opcional)")
    p.add_argument("--start", "-s", help="Fecha inicio (dd-MM-YYYY)")
    p.add_argument("--end", "-e", help="Fecha fin (dd-MM-YYYY)")
    p.add_argument("--tag", "-t", help="Filtrar por tag (expresión contiene)" )
    return p.parse_args()

def launch():
    args = parse_args()
    cfg = None
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = None
    path = generate_report_coach(
        input_csv=args.input,
        output_html=args.output,
        config=cfg,
        start_date=args.start,
        end_date=args.end,
        tag_filter=args.tag
    )
    print(f"Reporte generado en: {path}")