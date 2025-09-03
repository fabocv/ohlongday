from pathlib import Path
import pandas as pd
import altair as alt

def generar_grafico_tendencias_altair(df, out_html, out_png=None, smooth_raw=True, title=None):
    # --- parseo y base (igual que antes, omitido por brevedad) ---
    dfi = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(dfi['fecha']):
        dfi['fecha'] = pd.to_datetime(dfi['fecha'], dayfirst=True, errors='coerce')
    dfi = dfi.sort_values('fecha').reset_index(drop=True)
    dfi['bienestar'] = pd.to_numeric(dfi['bienestar'], errors='coerce').astype(float)
    dfi['bienestar_ema'] = pd.to_numeric(dfi['bienestar_ema'], errors='coerce').astype(float)
    last_idx = len(dfi) - 1
    dfi['is_last'] = False
    if last_idx >= 0:
        dfi.loc[last_idx, 'is_last'] = True
        last_ema = dfi.loc[last_idx, 'bienestar_ema']
        ema_label = f"EMA {last_ema:.2f}" if pd.notna(last_ema) else ""

    accent_line = "#3aa882"; muted_line = "#56687a"; accent_fill = "#e9f7f2"; grid_color="#eef3f7"

    def _theme_bdp():
        return {"config":{
            "background":"white","view":{"stroke":"transparent"},
            "axis":{"labelColor":"#6b7a86","titleColor":"#6b7a86","gridColor":grid_color,
                    "domainColor":"#e6eef6","tickColor":"#e6eef6","labelFontSize":11,"titleFontSize":12},
            "legend":{"titleColor":"#6b7a86","labelColor":"#6b7a86","orient":"top-left","padding":4,"labelFontSize":11},
            "title":{"color":"#3a4955","fontSize":14,"fontWeight":700,"anchor":"start"}}}
    alt.themes.register('bdp_air', _theme_bdp); alt.themes.enable('bdp_air')

    base = alt.Chart(dfi).transform_calculate(ema_v="isValid(datum.bienestar_ema) ? datum.bienestar_ema : 0") \
                         .properties(width=820, height=220, title=title or "")

    area_ema = base.mark_area(opacity=0.9, color=accent_fill).encode(
        x=alt.X('fecha:T', axis=alt.Axis(title=None, tickCount=6, format='%d %b')),
        y=alt.Y('ema_v:Q', scale=alt.Scale(domain=[0,10]), axis=alt.Axis(title=None))
    )
    line_ema = alt.Chart(dfi).mark_line(stroke=accent_line, strokeWidth=2.2).encode(
        x='fecha:T',
        y=alt.Y('bienestar_ema:Q', scale=alt.Scale(domain=[0,10]), title=None),
        tooltip=[alt.Tooltip('fecha:T', title='Fecha', format='%d %b %Y'),
                 alt.Tooltip('bienestar_ema:Q', title='EMA', format='.2f')]
    )
    if smooth_raw and len(dfi)>=3 and dfi['bienestar'].notna().sum()>=3:
        line_raw = (alt.Chart(dfi).transform_loess('fecha','bienestar',bandwidth=0.35)
                    .mark_line(stroke=muted_line, strokeWidth=1.25)
                    .encode(x='fecha:T', y=alt.Y('loess:Q', scale=alt.Scale(domain=[0,10]), title=None),
                            tooltip=[alt.Tooltip('fecha:T', title='Fecha', format='%d %b %Y'),
                                     alt.Tooltip('loess:Q', title='Bienestar (suave)', format='.2f')]))
    else:
        line_raw = alt.Chart(dfi).mark_line(stroke=muted_line, strokeWidth=1.25).encode(
            x='fecha:T', y=alt.Y('bienestar:Q', scale=alt.Scale(domain=[0,10]), title=None),
            tooltip=[alt.Tooltip('fecha:T', title='Fecha', format='%d %b %Y'),
                     alt.Tooltip('bienestar:Q', title='Bienestar', format='.2f')]
        )
    pts_last = alt.Chart(dfi[dfi['is_last']==True]).mark_point(
        filled=True, size=85, color=accent_line, stroke='white', strokeWidth=1
    ).encode(x='fecha:T', y=alt.Y('bienestar_ema:Q', scale=alt.Scale(domain=[0,10])))
    txt_last = alt.Chart(dfi[dfi['is_last']==True]).mark_text(
        align='left', dx=8, dy=-6, color=accent_line, fontWeight='bold'
    ).encode(x='fecha:T', y='bienestar_ema:Q', text=alt.value(ema_label))

    chart = (area_ema + line_raw + line_ema + pts_last + txt_last) \
        .configure_axisX(tickSize=0, domain=False) \
        .configure_axisY(tickCount=6, values=[0,2,4,6,8,10], domain=False) \
        .interactive(bind_y=False)

    # HTML (siempre funciona)
    inner = chart.to_html(embed_options={'actions': False})
    card_css = """
    <style>
    .bdp-card{max-width:860px;margin:8px auto;padding:8px;background:#fff;border-radius:14px;
              box-shadow:0 6px 22px rgba(16,24,40,0.08);border:1px solid #eef3f7;}
    </style>"""
    out_html_path = Path(out_html).resolve(); out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(f"{card_css}\n<div class='bdp-card'>{inner}</div>", encoding='utf-8')

    # PNG opcional
    out_png_path = None
    if out_png:
        out_png_path = Path(out_png).resolve()
        try:
            # Ruta 1: vl-convert (recomendada)
            chart.save(out_png_path.as_posix(), method='vl-convert')
        except Exception as e1:
            try:
                # Ruta 2: forzar una versión compatible de vegalite si el backend se queja
                from altair_saver import Saver
                saver = Saver(chart, method='vl-convert', vegalite_version='5.17.0')  # ajusta si tu backend soporta otra
                saver.save(out_png_path.as_posix())
            except Exception as e2:
                try:
                    # Ruta 3: selenium (si tu entorno tiene driver)
                    chart.save(out_png_path.as_posix(), method='selenium')
                except Exception:
                    out_png_path = None  # sin PNG, pero HTML ya quedó OK

    return {"html_path": out_html_path.as_posix(),
            "png_path": out_png_path.as_posix() if out_png_path else None}
