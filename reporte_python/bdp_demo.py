
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from reporte_python.bdp_schema import DEFAULT_COLUMNS
from reporte_python.bdp_cards import build_daily_cards, cards_to_markdown

# Synthesize a small demo dataset: up to 3 rows per day
base = datetime(2025, 8, 15, 10, 0, 0)
rows = []
rng = np.random.default_rng(0)
for d in range(0, 7):
    day = base + timedelta(days=d)
    n = int(rng.integers(1, 4))  # 1..3 registros
    for i in range(n):
        rows.append({
            "fecha": day + timedelta(hours=3*i),
            "animo": np.clip(rng.normal(0.6 + 0.05*d, 0.07), 0, 1),
            "activacion": np.clip(rng.normal(0.55 + 0.03*d, 0.05), 0, 1),
            "sueno": np.clip(rng.normal(0.5 + 0.02*d, 0.06), 0, 1),
            "conexion": np.clip(rng.normal(0.5, 0.08), 0, 1),
            "proposito": np.clip(rng.normal(0.52, 0.06), 0, 1),
            "claridad": np.clip(rng.normal(0.5 + 0.02*d, 0.07), 0, 1),
            "estres": np.clip(rng.normal(0.35 - 0.02*d, 0.05), 0, 1),
            "notas": f"nota {d}-{i}",
            "estresores": ["trabajo", "sueño corto"] if i == 1 else "tránsito"
        })
# Insert a missing day between day 2 and 3 to demo "sin registro"
df = pd.DataFrame(rows)
df = df[df["fecha"].dt.date != (base + timedelta(days=3)).date()].reset_index(drop=True)

cards = build_daily_cards(df, DEFAULT_COLUMNS, fill_missing_days=True)
md = cards_to_markdown(cards)

with open("output/BDP_daily_report.md", "w", encoding="utf-8") as f:
    f.write(md)

print("Generated /mnt/data/BDP_daily_report.md")
print("\n--- Preview ---\n")
print(md[:1200] + ("..." if len(md) > 1200 else ""))
