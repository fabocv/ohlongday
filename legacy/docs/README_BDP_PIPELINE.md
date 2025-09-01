
# BDP Baseline Pipeline (EMA + Predicción + Importancias)

**EMA baseline**: α≈0.30, lookback≈14 días, warm‑up ≥7–10 datos, winsorización de deltas, imputación suave (last valid hasta 3 días).

## 1) Requisitos
- Python 3.9+
- `pip install pandas numpy scikit-learn`

Opcional: `matplotlib` para gráficos.

## 2) Estructura de datos
CSV con columnas en español (no es obligatorio tener todas):
```
fecha,hora,animo,activacion,conexion,proposito,claridad,estres,sueno_calidad,horas_sueno,siesta_min,autocuidado,alimentacion,movimiento,dolor_fisico,ansiedad,irritabilidad,meditacion_min,exposicion_sol_min,agua_litros,cafe_cucharaditas,alcohol_ud,medicacion_tomada,medicacion_tipo,otras_sustancias,interacciones_significativas,eventos_estresores,tags,notas,glicemia
```

- `eventos_estresores`/`tags` pueden ser listas separadas por `,` `;` `|` `/`.
- `medicacion_tomada` debe ser 0/1 (o numérico).

## 3) Cómo correr
Copia tu CSV (por ejemplo `bdp.csv`) en la misma carpeta del script y ejecuta:

```bash
python bdp_pipeline.py --csv ./bdp.csv --out ./outputs
```

Si el CSV no existe, el script generará un **template vacío** para que veas las columnas esperadas.

## 4) Qué produce
- `outputs/bdp_enriquecido.csv` → dataset con EMA, lags, features derivadas.
- `outputs/{target}_cv_metrics.csv` → métricas por modelo (MAE, RMSE) en validación temporal.
- `outputs/{target}_feature_importance.csv` → importancias por permutación.
- `outputs/{target}_drivers_del_dia.txt` y `.csv` → explicación heurística de los “impulsores del día”.
- `outputs/resumen_modelos.csv` → resumen de mejores modelos por target.

Targets por defecto: `estres`, `animo`, `claridad` (puedes editar `TARGETS` en el script).

## 5) Notas de modelado
- Validación temporal con ventanas crecientes (`min_train≈30`, `test≈7`). Edita en `time_series_splits` si deseas.
- **Ponderaciones** para el reporte coach: usa `*_feature_importance.csv` (normalizadas) para destacar “drivers del día”.
- Si tu `alimentacion`/`medicacion_tipo` tienen pocas categorías, se codifican con One‑Hot automáticamente.
- Si no tienes suficientes datos, el script seguirá, pero la validación será débil.

## 6) Siguientes pasos sugeridos
- Añadir variantes ordinales para `estres/ansiedad/irritabilidad` si están en escalas discretas (regresión ordinal).
- Explorar **GAM** o **árboles con restricciones monotónicas** cuando el entorno lo permita.
- Conectar el pipeline a tu **reporte coach** comparando `observado` vs `EMA14` y explicando Δ por “drivers”.
