
# -*- coding: utf-8 -*-
"""
run_relations_and_wb.py
Analiza relaciones entre targets core y estima sinergias de bienestar a partir de drivers históricos.
"""
import os, sys, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "bdp_calcs")
if os.path.isdir(PKG) and PKG not in sys.path:
    sys.path.insert(0, HERE)

from bdp_calcs import (
    wellbeing_index, core_correlations, crosslag_influence,
    aggregate_drivers_history, drivers_del_dia, wellbeing_synergy
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV enriquecido (bdp_enriquecido.csv)")
    ap.add_argument("--out", default="./outputs", help="Carpeta de salida")
    ap.add_argument("--targets", default="animo,claridad,estres,activacion", help="Targets core separados por coma")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv, encoding="utf-8")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    # 1) Wellbeing index
    wb = wellbeing_index(df, targets=targets)
    wb_path = os.path.join(args.out, "wellbeing_index_series.csv")
    pd.DataFrame({"WB": wb}).to_csv(wb_path, index=False, encoding="utf-8")
    print("WB guardado en:", wb_path)

    # 2) Correlaciones (niveles y deltas)
    cors = core_correlations(df, targets=targets)
    cors["levels"].to_csv(os.path.join(args.out, "core_cor_levels.csv"), encoding="utf-8")
    cors["deltas"].to_csv(os.path.join(args.out, "core_cor_deltas.csv"), encoding="utf-8")
    print("Correlaciones listas: core_cor_levels.csv, core_cor_deltas.csv")

    # 3) Influencia cruzada (lag-1)
    infl = crosslag_influence(df, targets=targets)
    if isinstance(infl, pd.DataFrame) and not infl.empty:
        infl.to_csv(os.path.join(args.out, "core_crosslag_influence.csv"), encoding="utf-8")
        print("Influencia cruzada guardada en: core_crosslag_influence.csv")
    else:
        print("Influencia cruzada: insuficiente señal (necesita más filas).")

    # 4) Agregación de drivers del día a lo largo de la historia
    agg = aggregate_drivers_history(df, drivers_func=drivers_del_dia, targets=targets, top_k=3)
    # Exportar por target
    for t, ser in agg["avg_contrib"].items():
        if not ser.empty:
            ser.to_csv(os.path.join(args.out, f"{t}_avg_contrib.csv"), header=["avg_contrib"], encoding="utf-8")
    for t, ser in agg["abs_contrib"].items():
        if not ser.empty:
            ser.to_csv(os.path.join(args.out, f"{t}_abs_contrib.csv"), header=["mean_abs_contrib"], encoding="utf-8")
    for t, ser in agg["freq_sign"].items():
        if not ser.empty:
            ser.to_csv(os.path.join(args.out, f"{t}_freq_pos.csv"), header=["freq_positive"], encoding="utf-8")
    print("Agregación de drivers exportada (avg/abs/freq).")

    # 5) Sinergia de bienestar (qué drivers tienden a mejorar/empeorar el compuesto)
    syn = wellbeing_synergy(agg)
    if not syn.empty:
        syn.to_csv(os.path.join(args.out, "wellbeing_synergy.csv"), header=["wb_synergy"], encoding="utf-8")
        print("wellbeing_synergy.csv creado (features que más ayudan/perjudican WB).")
    else:
        print("Sinergia de bienestar: aún sin datos suficientes.")

if __name__ == "__main__":
    main()
