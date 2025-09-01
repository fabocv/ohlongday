import pandas as pd
import numpy as np
from datetime import datetime
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt

class BDPReport:
    @staticmethod
    def fig_to_base64(figure):
        buf = BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight")
        plt.close(figure)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    
    @staticmethod
    def simple_line_plot(dates, values, title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(dates, values)
        ax.set_title(title)
        ax.set_xlabel("Fecha")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        encoded = BDPReport.fig_to_base64(fig)
        return f'<img alt="{title}" src="data:image/png;base64,{encoded}"/>'
    
    @staticmethod
    def icon_for_level(x):
        # Mapea la Escala fenomenológica 0–3 a un icono
        mapping = {
            0: "⬛️ Bloqueo fuerte",
            1: "🟨 Débil/retroceso",
            2: "🟩 En camino",
            3: "🟦 Avance claro",
        }
        return mapping.get(int(x), "⬜️ N/A")
    
    @staticmethod
    def to_html(df: pd.DataFrame) -> str:
        # Resumen últimas N filas (o todo si pequeño)
        last_n = min(len(df), 14)
        tail = df.tail(last_n).copy()
        
        # Tabla compacta (fechas + índices clave)
        small = tail[["fecha","H_t","V_t","C_t","P_t","S_t_neg","BDP_score","BDP_feno_0_3"]].copy()
        small["BDP_feno_icon"] = small["BDP_feno_0_3"].apply(BDPReport.icon_for_level)
        
        # Gráficos inline
        dates = pd.to_datetime(tail["fecha"])
        imgs = []
        imgs.append(BDPReport.simple_line_plot(dates, tail["BDP_score"], "BDP Score", "z-compuesto"))
        imgs.append(BDPReport.simple_line_plot(dates, tail["H_t"], "H_t (Humor)", "z"))
        imgs.append(BDPReport.simple_line_plot(dates, tail["V_t"], "V_t (Vitalidad)", "z"))
        imgs.append(BDPReport.simple_line_plot(dates, tail["P_t"], "P_t (Prop/Claridad)", "z"))
        imgs.append(BDPReport.simple_line_plot(dates, tail["C_t"], "C_t (Conexión)", "z"))
        imgs.append(BDPReport.simple_line_plot(dates, tail["S_t_neg"], "S_t⁻ (Estrés invertido)", "z"))
        imgs_html = "\n".join(imgs)
        
        # Tabla HTML estilizada
        styled_table = (
            small.rename(columns={
                "fecha": "Fecha",
                "BDP_score": "BDP Score",
                "BDP_feno_0_3": "Fenomenología (0–3)",
                "BDP_feno_icon": "Estado"
            })
            .to_html(index=False, float_format=lambda x: f"{x:.2f}")
        )
        
        # Documento HTML
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BDP – Informe</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Noto Sans', Arial; margin: 24px; }}
  h1, h2 {{ margin: 0.2rem 0 0.6rem; }}
  .pill {{ display:inline-block;padding:4px 8px;border-radius:999px;background:#eee;margin-right:6px;font-size:12px }}
  .grid {{ display:grid; grid-template-columns: 1fr; gap: 16px; }}
  .card {{ border:1px solid #ddd; border-radius:12px; padding:16px; box-shadow: 0 2px 8px rgba(0,0,0,.04);}}
  .muted {{ color:#666; font-size: 12px; }}
  img {{ max-width: 100%; height:auto; border-radius:8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
  th {{ background: #fafafa; }}
</style>
</head>
<body>
  <h1>BDP – Informe diario</h1>
  <div class="muted">Generado: {now} · Zona horaria esperada: America/Santiago</div>
  
  <div class="grid">
    <div class="card">
      <h2>Resumen (últimos {last_n} días)</h2>
      <div>{styled_table}</div>
    </div>
    <div class="card">
      <h2>Tendencias</h2>
      {imgs_html}
    </div>
    <div class="card">
      <h2>Notas</h2>
      <p>Este informe aplica z-scores por la ventana actual y pondera el sueño en <code>V_t</code> (peso 1.2). 
      El estrés se invierte en <code>S_t⁻</code>. La escala fenomenológica 0–3 clasifica el estado global.</p>
      <p class="muted">Este informe no reemplaza a profesionales de salud mental.</p>
    </div>
  </div>
</body>
</html>
"""
        return html