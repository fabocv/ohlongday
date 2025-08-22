
# bdp_main.py
import argparse, json, sys, os
from bdp_report_coach import generate_report_coach

def parse_known_args(argv=None):
    p = argparse.ArgumentParser(description="Genera el informe BDP (coaching).")
    p.add_argument("--input", "-i", default="data/bdp_data_fake.csv", help="Ruta CSV de entrada")
    p.add_argument("--output", "-o", default="output/informe.html", help="Ruta HTML de salida")
    p.add_argument("--config", "-c", default="bdp_compare_config.json", help="Ruta JSON de configuración (opcional)")
    p.add_argument("--start", "-s", help="Fecha inicio (dd-MM-YYYY). Si no se especifica, se usan todos los datos disponibles.")
    p.add_argument("--end", "-e", help="Fecha fin (dd-MM-YYYY). Si no se especifica, se usan todos los datos disponibles.")
    p.add_argument("--tag", "-t", help="Filtrar por tag (subcadena). Si no se especifica, NO se filtra por tag.")
    return p.parse_known_args(argv)

if __name__ == "__main__":
    args, _ = parse_known_args(sys.argv)
    cfg = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
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
