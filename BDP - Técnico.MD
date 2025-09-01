# 📘 BDP — Documento Técnico (v1.0)
**Bienestar Data Project (BDP)** · Definiciones de variables, normalización, EMA, modelos (≥30 registros) e índices compuestos (ansiedad, hipomanía, etc.).  
> **Aviso importante:** Todo lo que sigue son **estimaciones proxy** para apoyo personal y terapéutico. **No son diagnósticos clínicos.**

---

## 0) Alcance e implementación actual
- **Captura de datos:** Google Forms → Google Sheets/CSV (por rapidez y menor fricción).
- **Procesamiento:** Python (pandas, numpy, scipy, scikit-learn opcional), scripts modulares.
- **Salida:** HTML/PDF (reportes “coach”), métricas y gráficos con interpretación técnica + humana.
- **Frecuencia:** 1–3 registros día (ideal: mañana/tarde/noche) o 1 consolidado diario.

---

## 1) Esquema de datos (columnas crudas)
**Mínimo recomendado** (nombres orientativos, ajusta a tu CSV real):

### 1.1 Núcleo emocional / cognitivo
- `animo` (0–10)
- `activacion` (0–10)
- `conexion` (0–10)
- `proposito` (0–10)
- `claridad` (0–10)
- `estres` (0–10)
- `ansiedad` (0–10, si existe)
- `irritabilidad` (0–10, si existe)

### 1.2 Sueño / ritmos
- `hora_dormir` (HH:MM)
- `hora_despertar` (`hora_DSPT`) (HH:MM)
- `sueno_calidad` (0–10)
- `despertares_nocturnos` (`DSPTes_nocturnos`) (0,1,2,3…)

### 1.3 Movimiento y fisiología
- `mov_intensidad` (0–10)
- `tiempo_ejercicio` (min)
- `siesta_min` (min)
- `glicemia` (mg/dL, si aplica)

### 1.4 Hábitos / consumo
- `cafe_cucharaditas` (≈ 1 cdita ~ 60–80 mg cafeína)
- `cafe_ultima_hora` (bool/0–1)
- `alcohol_ud` (unidades estándar)
- `alcohol_ultima_hora` (bool/0–1)
- `agua_litros` (L)
- `otras_sustancias` (texto/opcional)
- `medicacion_tipo` (texto)
- `medicacion_tomada` (bool/0–1)

### 1.5 Contexto y sociales
- `interacciones_calidad` (−10..+10 o 0–10; definir)
- `interacciones_significativas` (conteo 0,1,2…)
- `tiempo_pantalla_noche_min` (min)
- `juego_en_dispositivo_min` (min)
- `juego_en_persona_min` (min)
- `exposicion_sol_manana_min` (min)
- `espiritual` (lista de etiquetas)
- `tags` / `eventos_estresores` / `notas` (con **códigos personales**: ej. `ABL01`, `FJJ`)

### 1.6 Metadatos
- `fecha` (YYYY-MM-DD), `hora` (HH:MM), `bloque` (`mañana`/`tarde`/`noche`/`dia`)

---

## 2) Limpieza y tipado
1. **Parseo temporal:** convertir `fecha` y `hora` a `datetime`; derivar `bloque` si falta:
   - `mañana`: 05:00–11:59; `tarde`: 12:00–18:59; `noche`: 19:00–04:59.
2. **Casting numérico:** forzar columnas 0–10 y minutos a `float`.
3. **Winsorización suave:** recortar p1/p99 en métricas con outliers (p.ej. `tiempo_pantalla_noche_min`).
4. **Imputación mínima:** **LOCF** (last observation carried forward) con tope de 2 días; si falla, usar mediana personal.
5. **Flags de faltantes:** crear columnas `_isna` por variable crítica para trazabilidad.

---

## 3) Variables intermedias (derivadas)

### 3.1 Sueño y descanso
- **Duración de sueño (h):**  
  Si `hora_dormir` > `hora_despertar`, sumar 24h al despertar.  
  $$ \text{sueno\_duracion\_h} = \frac{\text{despertar} - \text{dormir}}{60}$$ 
- **Déficit de sueño (h):** respecto a objetivo personal (ej. 7.5 h).  
  $$ \text{sueno\_deficit_h} = 7.5 - \text{sueno\_duracion\_h}$$ 
- **Indicador de fragmentación:**  
  $$ \text{sueno\_fragmentacion} = \text{scale}( \text{despertares\_nocturnos} )$$ 
- **Índice de sueño compuesto (0–10):** pondera calidad, duración y fragmentación:  
  $$ I_{sueno} = 0.5\cdot \text{norm}(\text{sueno\_calidad}) + 0.35\cdot \text{norm}(\text{sueno\_duracion\_h}) - 0.15\cdot \text{norm}(\text{sueno\_fragmentacion})$$ 

### 3.2 Movimiento y energía
- **Carga de actividad (AU):**  
  $$ \text{actividad\_carga} = \text{mov\_intensidad}\times \sqrt{\text{tiempo\_ejercicio (min)}}$$ 
- **Recuperación rápida (proxy):** si `siesta_min`>0 y `actividad_carga` alta → +recuperación.
- **Índice metabólico/energía (0–10):**  
  $$ I_{energia} = 0.45\cdot \text{norm}(\text{actividad\_carga}) + 0.35\cdot \text{norm}(I_{sueno}) + 0.20\cdot \text{norm}(-|\text{glicemia}-\mu_{g}|),$$ 
  donde $$ \(\mu_{g}\)$$  es el **rango objetivo personal** (p.ej. 80–110 mg/dL en ayunas). Se penaliza desviación absoluta.

### 3.3 Estimulantes / depresores
- **Cafeína estimada (mg):**  
  $$ \text{cafeina\_mg} \approx 75 \times \text{cafe\_cucharaditas}$$ 
- **Alcohol reciente (flag):** `alcohol_ultima_hora` o `alcohol_ud`≥1.
- **Carga psicoactiva (proxy 0–10):**
  $$ \text{psicoactiva} = \text{scale}\big( \text{cafeina\_mg} + 120\cdot\text{alcohol\_ud} \big)$$ 

### 3.4 Sociales, pantalla y espiritualidad
- **Balance social (−1..+1):**  
  $$ \text{social\_balance} = \text{scale}(\text{interacciones\_calidad}) + 0.1\cdot \text{scale}(\text{interacciones\_significativas})$$ 
- **Carga nocturna de pantalla (0–10):** `scale(tiempo_pantalla_noche_min)`.
- **Juego digital / presencial:** dos canales (`juego_en_dispositivo_min`, `juego_en_persona_min`) para contrastar regulaciones opuestas.
- **Práctica espiritual (0–10):** frecuencia/etiquetas → conteo o minutos → `norm`.

---

## 4) Normalización
> Buscamos comparabilidad **intra-persona** (baseline personal).

### 4.1 Z-score clásico (si n≥20 válidos)
$$ z(x_t) = \frac{x_t - \mu_x}{\sigma\_x}$$ 
- \(\mu_x\), \(\sigma\_x\) = media y desvío **personales** (últimos 60–90 días si existen).

### 4.2 Z robusto (si n<20 o outliers)
$$ z_r(x_t) = \frac{x_t - \text{mediana}(x)}{1.4826\cdot \text{MAD}(x)}$$ 
- **Winsorizar** p05–p95 antes si hay extremos.

### 4.3 `norm(·)` y `scale(·)`
- `norm`: llevar a 0–10 post-z con clip (ej.: `norm = clip(5 + 2*z, 0, 10)`).
- `scale`: estandarizar a 0–10 con cuantiles personales (p10→2, p90→8).

---

## 5) Suavizado EMA (α≈0.30, lookback≈14)
$$ S_t = \alpha\cdot x_t + (1-\alpha)\cdot S_{t-1}$$ 
- **Warm-up:** mostrar EMA solo si n ≥ 7–10.  
- **Confianza EMA:**  
  $$ \text{conf}_{EMA}(t) = \min\left(1,\ \frac{n}{14}\right)\cdot \frac{1}{1 + \text{var}(x_{t-13..t})}$$ 
- **Bandas “Caution/Alert”:** cuantiles personales p20/p80 (o p15/p85).  
- **Autotune opcional:** grid‐search \(\alpha\in[0.2,0.35]\), \(L\in[10,21]\) minimizando error 1-paso (MAE).

---

## 6) Índices compuestos del BDP (día t)
> Vector **H-V-C-P-S⁻** (base teórica del BDP).

$$ \begin{aligned}
H_t &= z(\text{animo}_t) \\
V_t &= z(\text{activacion}_t) + z(I_{sueno,t}) \\
C_t &= z(\text{conexion}_t) \\
P_t &= z(\text{proposito}_t) + z(\text{claridad}_t) \\
S_t^- &= -\,z(\text{estres}_t)
\end{aligned}$$ 

- **Bienestar compuesto (WB):**  
  $$ WB_t = w_H H_t + w_V V_t + w_C C_t + w_P P_t + w_{S^-} S_t^-$$ 
  **Pesos sugeridos (hipótesis):** \(w_H=0.25,\ w_V=0.20,\ w_C=0.15,\ w_P=0.20,\ w_{S^-}=0.20\).  
  > Ajustables por validación personal o aprendizaje (sección 8).

- **Índice de Carga (Stress Load):**  
  $$ SL_t = z(\text{estres}_t) + 0.3\cdot \text{scale}(\text{eventos\_estresores}) + 0.2\cdot \text{scale}(\text{tiempo\_pantalla\_noche\_min})$$ 

- **Índice Energético (Energy):** usar \(I_{energia}\) (3.2) y su z.

---

## 7) Índices clínico-proxy (no diagnósticos)

### 7.1 Ansiedad-proxy (AX)
$$ AX_t = 0.40\,z(\text{estres}_t) + 0.25\,z(\text{ansiedad}_t) + 0.15\,z_r(\text{irritabilidad}_t) + 0.10\,\text{scale}(\text{cafeina\_mg}_t) + 0.10\,\text{scale}(\text{despertares\_nocturnos}_t)$$ 
- **Lectura:** ↑AX sugiere activación/hipervigilancia; revisar sueño, cafeína y regulación.

### 7.2 Hipomanía-proxy (HYPO)
$$ \begin{aligned}
HYPO_t =\ & 0.35\,z(\text{activacion}_t) + 0.20\,z(\text{animo}_t) \\
& - 0.15\,\text{norm}(\text{sueno\_duracion\_h}_t) + 0.10\,\text{scale}(\text{cafeina\_mg}_t) \\
& + 0.10\,\text{scale}(\text{interacciones\_significativas}_t) + 0.10\,\text{scale}(\text{impulso\_nocturno}_t)
\end{aligned}$$ 
- `impulso_nocturno` ~ combinación de `tiempo_pantalla_noche_min`, juego nocturno, compras/decisiones impulsivas (si etiqueta existe).
- **Lectura:** ↑HYPO sugiere patrón de **alta activación con menor descanso**.

### 7.3 Depresión-proxy (DEP)
$$ DEP_t = -0.35\,z(\text{animo}_t) - 0.20\,z(\text{activacion}_t) - 0.15\,z(\text{proposito}_t) - 0.10\,z(\text{claridad}_t) + 0.10\,\text{scale}(\text{sueno\_fragmentacion}_t) - 0.10\,\text{norm}(\text{autocuidado}_t)$$ 
- **Lectura:** ↑DEP sugiere **bajo tono emocional/energía** + trastornos de sueño.

> **Recordatorio:** Estos índices son **orientativos** y deben leerse con **contexto personal y clínico**.

---

## 8) Modelos cuando n ≥ 30 registros
**Objetivo típico:** predecir \(WB_{t+1}\), \(H_{t+1}\) o \(AX_{t+1}\) para alertas y planificación.

### 8.1 Ingeniería de variables
- **Features base (t):** z y EMA de: `animo, activacion, claridad, proposito, conexion, estres`, `I_sueno`, `actividad_carga`, `cafeina_mg`, `alcohol_ud`, `AX_t`, `HYPO_t`, `DEP_t`, `social_balance`, `pantalla_noche`, `espiritual_score`.
- **Deltas:** \(x_t - x_{t-1}\) en claves (ánimo, estrés, sueño, actividad).
- **Ventanas:** medias y varianzas móviles (3,7,14 días).
- **Estacionales:** dummies por día de semana y bloque.

> **Evitar leakage:** solo usar información **hasta t** para predecir **t+1**.

### 8.2 Candidatos
1. **Baseline-EMA:** \( \hat{y}_{t+1} = S_t^{(y)} \)  
2. **Regresión Ridge/Lasso:** lineal y parsimoniosa (explicable).
3. **Árboles (RandomForest/GBDT):** capturan no-linealidades; limitar profundidad (evitar overfit).
4. **Regresión robusta (Huber):** tolerante a outliers.

### 8.3 Selección y validación
- **CV** 5-fold temporal (bloques contiguos) o **walk-forward**.
- **Métricas:** MAE (principal), RMSE, MAPE (si procede).
- **Criterio de elección:** modelo con **MAE mínimo** y **estabilidad** (varianza baja entre folds).
- **Comparación base:** mejora relativa sobre EMA ≥ **10–15%** para justificar complejidad.

### 8.4 Importancias y explicabilidad
- **Lineales:** pesos estandarizados (coeficientes sobre z-features).
- **Árboles:** importancia por ganancia; usar **SHAP** para casos.

---

## 9) Reglas de negocio y focos
- **Bandas y alertas:** si \(AX_t\) > p85 personal → **Caution**; > p95 → **Alert**.  
- **Reglas si/entonces (ejemplos):**  
  - Si `activacion_pre_salida ≥ 7` y `sueno_deficit_h ≥ 1` → **evitar grupos grandes**.  
  - Si `pantalla_noche` p>90 dos días seguidos → **“higiene de sueño +30 min off-screen”**.  
  - Si `alcohol_ud ≥ 2` → sugerir **re-hidratación** y **suave actividad** al día siguiente.

---

## 10) Salidas y columnas finales (ejemplo)
- **Series z y EMA:** `animo_z`, `animo_ema`, … (para todas las principales).  
- **Derivadas:** `sueno_duracion_h`, `sueno_deficit_h`, `actividad_carga`, `cafeina_mg`, `social_balance`, `pantalla_noche_scaled`, `espiritual_score`.  
- **Índices:** `WB`, `SL` (Stress Load), `Energy`, `AX`, `HYPO`, `DEP`.  
- **Confianzas:** `conf_ema_animo`, `conf_ema_wb`.  
- **Alertas:** `flag_caution_ax`, `flag_alert_ax`, etc.  
- **Cards coach:** `stoic_day_card`, `hard_day_card`, `retro_now_card`, `week_trendly_card` (estructura JSON por card con `resumen_areas`, `drivers_pos/neg`, `mensajes`).

---

## 11) Pesos: hipótesis y ajuste
- **Iniciales** (sección 6) basados en literatura general y heurística de regulación.  
- **Ajuste personal:** optimizar \(w\) maximizando correlación entre `WB` y **auto-rating semanal** (“¿cómo evaluaste tu semana?” 0–10), con **regularización L1** para evitar sobreajuste.  
- **Revisión trimestral:** recalcular z-scores con ventana móvil y revalidar pesos.

---

## 12) Buenas prácticas y notas
- **Estabilidad primero:** preferir **z robusto** en etapas tempranas.  
- **Separar “proxies clínicos” del lenguaje clínico:** nomenclatura `*_proxy`.  
- **Trazabilidad:** conservar columnas `_isna`, `_winsorized`, `_imputed`.  
- **Privacidad:** códigos personales en `tags/eventos_estresores/notas` **no se decodifican**, solo se analizan **patrones repetidos**.

---

## 13) Glosario rápido
- **z / z_r:** estandarización (clásica / robusta).  
- **EMA:** media móvil exponencial.  
- **WB:** bienestar compuesto.  
- **AX/HYPO/DEP:** índices **proxy** de ansiedad / hipomanía / depresión.  
- **scale/norm:** mapeos a 0–10 por cuantiles o z a 0–10.  
- **LOCF:** imputación por último valor observado.

---

## 14) Ejemplo de fórmulas en bloque (copiable)
```text
# EMA genérica
S_t = alpha * x_t + (1 - alpha) * S_{t-1}
alpha ≈ 0.30, lookback ≈ 14, warm-up ≥ 7-10

# WB (pesos sugeridos)
WB_t = 0.25*H_t + 0.20*V_t + 0.15*C_t + 0.20*P_t + 0.20*S^-_t

# Ansiedad-proxy (AX)
AX_t = 0.40*z(estrés) + 0.25*z(ansiedad) + 0.15*zr(irritabilidad)
       + 0.10*scale(cafeina_mg) + 0.10*scale(despertares_nocturnos)
```

## 15) Límites y futuros módulos

- Hormonas y marcadores biológicos: aún no modelados (dopamina, serotonina, cortisol).

- Geometría social detallada: tamaño grupo, ruido, alcohol, horario, duración.

- Señales somáticas rápidas (MTC/fisio): palpitaciones, calor vespertino, aftas, etc.

- Integraciones externas: wearables (sueño, HRV), apps mindfulness y actividad.

> **Cierre**: Este documento estandariza cómo derivamos y suavizamos señales, cómo agregamos índices compuestos y cómo comparamos modelos cuando existe historial suficiente (≥30). Todo está diseñado para preservar contexto personal, mantener explicabilidad y sostener una evolución iterativa del BDP.