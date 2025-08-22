

# Oh Long Day - Modelado del Bienestar Dinámico Personal (BDP)
### Autor: fabocv

El **Bienestar Dinámico Personal (BDP)** es un marco teórico-práctico diseñado para **medir, modelar y acompañar** la experiencia de las personas con foco en el **bienestar cotidiano**, particularmente útil en contextos de **trastornos del ánimo** (depresión, bipolaridad, TLP).

BDP no busca diagnosticar, sino **detectar patrones individuales**, **dar retroalimentación humanista** y **proponer micro-hábitos** que impacten positivamente en la vida diaria.

---

## 📌 Resumen técnico

El **BDP** integra **psicometría ligera**, **modelos dinámicos** y **mensajes humanistas** para acompañar el bienestar en tiempo real.

-   Fórmulas basadas en **z-scores idiográficos**.
    
-   Índices diseñados para **trastornos del ánimo (Depresión, Ansiedad, Bipolaridad, TLP)**.
    
-   Alertas definidas por **percentiles personales** y **duración mínima**.
    
-   Mensajes orientados a **hábitos concretos**, no a diagnósticos.

---

## 🎯 Objetivos

- **Medición dinámica:** capturar de manera continua y ligera el estado afectivo, energético y relacional de una persona.  
- **Identificación de patrones:** detectar tendencias relevantes como hipomanía, labilidad afectiva o desregulación ansiosa/depresiva.  
- **Integración de micro-hábitos:** vincular pequeños cambios de conducta (sueño, movimiento, conexión, mindfulness) con su impacto real en el bienestar.  
- **Alertas humanistas:** ofrecer mensajes amables y neutrales que reflejen los datos sin caer en diagnósticos invasivos.  
- **Empoderamiento personal:** entregar al usuario una herramienta de autoobservación que complemente (pero no reemplace) la terapia profesional.

---

## 🧩 Marco teórico

### Bienestar como constructo dinámico
El bienestar se entiende como la **capacidad de mantener un estado afectivo positivo, un funcionamiento cotidiano estable y un sentido de conexión y propósito, incluso ante estresores**.

Se operacionaliza en 5 **dominios base**:

1. **Hedónico (H):** ánimo, placer, valencia emocional.  
2. **Vitalidad (V):** energía, sueño, actividad física.  
3. **Conexión (C):** vínculos, apoyo social, interacción significativa.  
4. **Propósito / Flow (P):** sentido, claridad, tareas con significado.  
5. **Estrés invertido (S⁻):** tensión percibida, rumiación, sobrecarga.

### Extensiones clínicas
- **Bipolaridad:**  
  - **HI (Hipomanía):** ánimo ↑ + energía ↑ + sueño ↓ + impulsividad ↑ + distractibilidad.  
  - **MI (Mixto):** ánimo negativo + activación alta + irritabilidad.  
- **TLP:**  
  - **ALI (Labilidad Afectiva):** alta variación intradiaria del ánimo.  
  - **IPI (Impulsividad):** impulsividad + ira + disociación.  
  - **RI (Relacional):** sensibilidad al abandono + conflictos + baja conexión.

---

## 📊 Índices clave

- **DI (Depresivo):** ánimo bajo, anhedonia, fatiga, sueño alterado.  
- **AI (Ansiedad):** activación elevada, preocupación, tensión, insomnio.  
- **HI (Hipomanía):** energía alta con poco sueño y conductas impulsivas.  
- **MI (Mixto):** ánimo negativo con alta activación e irritabilidad.  
- **ALI (Affective Lability):** variación emocional rápida.  
- **IPI (Impulsividad):** urgencia de actuar sin plan.  
- **RI (Relacional):** sensibilidad y conflictos interpersonales.  
- **SI (Estabilidad):** ausencia de índices elevados → eutimia.

---

## 🤝 Mensajes humanistas

Los resultados nunca se expresan como “diagnóstico”, sino como **reflejos benevolentes** que invitan a la autoexploración y el autocuidado.

Ejemplos:

- **DI alto:** “Tu ánimo estuvo bajo varios días. Quizá sea buen momento para reforzar autocuidados suaves.”  
- **AI alto:** “Parece que tu cuerpo estuvo en tensión constante. Una pausa de respiración podría ayudarte.”  
- **HI alto:** “Tu energía y actividad aumentaron bastante con poco sueño. ¿Quieres priorizar descanso esta semana?”  
- **ALI alto:** “Tus emociones variaron mucho hoy. ¿Te animas a probar una técnica de calma rápida?”  
- **RI alto:** “Notaste sensibilidad en tus vínculos. ¿Quieres enviar un mensaje amable a alguien de confianza?”

---

## 🚦 Limitaciones

- **No es diagnóstico médico.** Es un complemento para la autoobservación y el bienestar.  
- **No reemplaza terapia ni psiquiatría.**  
- **Mensajes controlados:** máximo 1–2 por día, siempre opcionales.  
- **Enfoque idiográfico:** cada persona se compara solo consigo misma (su propia línea base).  

---

# ⚙️ Documentación técnica

## Variables básicas (escala 0–10 salvo indicado)
- `animo` (valencia)  
- `activacion` (energía corporal)  
- `sueno` (horas, invertido en algunos índices)  
- `estres` (tensión percibida, invertido en S⁻)  
- `claridad` (foco mental)  
- `conexion` (interacción/vínculos)  
- `proposito` (sentido/flow)  
- `impulsividad`  
- `irritabilidad`  
- `distractibilidad`  
- `abandono` (sensibilidad al abandono)  
- `vacio` (dolor psíquico, vacío)  
- `conflictos` (conteo o intensidad)  

---

## Fórmulas de dominios
$$
\begin{aligned}
H_t = z(animo_t) \\
V_t = z(activacion_t) + z(sueno_t) \\
C_t = z(conexion_t) \\
P_t = z(proposito_t) + z(claridad_t) \\
S_t^- = -z(estres_t)
\end{aligned}
$$


---

## Índices de síntomas

### Depresivo (DI)
$$DI_t = -z(animo_t) + -z(energia_t) + -z(sueno_t) + -z(claridad_t)$$

### Ansiedad (AI)
$$AI_t = z(activacion_t) + z(estres_t) + -z(sueno\_inicio_t)$$

### Hipomanía (HI)
$$HI_t = z(animo_t) + z(activacion_t) - z(sueno_t) + z(impuls_t) + z(distracti_t) + z(irrit_t)$$

### Mixto (MI)
$$MI_t = -z(animo_t) + z(activacion_t) + z(irritabilidad_t)$$


### Labilidad afectiva (ALI)
$$ALI_{dia} = Var(animo_{intradia}) + Var(activacion_{intradia}) $$

### Impulsividad (IPI)
$$
IPI_t = z(impulsividad_t) + z(irritabilidad_t) + z(disociacion_t)
$$

### Relacional (RI)
$$
RI_t = z(abandono_t) + z(conflictos_t) - z(conexion_t)
$$

### Estabilidad (SI)
$$
SI_t = 1 - \frac{DI_t + AI_t + HI_t + MI_t + ALI_t + IPI_t + RI_t}{n\_indices}
$$

---

## Reglas de decisión (umbral relativo)
- Se usan **percentiles personales (idiográficos)**:  
  - Alto = >p85, Bajo = <p15.  
- Ejemplo:  
  - **HI alerta:** ánimo >p85 + activación >p85 + sueño <p15 durante ≥4 días.  
  - **DI alerta:** ánimo <p15 + energía <p15 durante ≥5 días.  
  - **ALI alerta:** varianza de ánimo >p90 en un día.  

---

## Pseudocódigo (Python-style)

```python
def compute_HI(data):
    return (
        z(data['animo']) +
        z(data['activacion']) -
        z(data['sueno']) +
        z(data['impulsividad']) +
        z(data['distractibilidad']) +
        z(data['irritabilidad'])
    )

def detect_alerts(series, threshold=1.5, days=4):
    hi = compute_HI(series)
    if (hi > threshold).rolling(days).sum().max() >= days:
        return "HI alerta: energía alta y poco sueño sostenido"
    return None
```
---
## Reportes automáticos

-   **Diario:** valores de dominios + índice general de bienestar (W_t).
    
-   **Semanal:** promedio de índices, top 3 variables con mayor contribución.
    
-   **Mensual:** actualización de pesos predictivos, gráfico de redes (VAR/DSEM).
    

----------


# 📘 Guía para Rellenar el CSV del BDP

Este archivo CSV es el **autoregistro diario** para el modelo **BDP (Bienestar – Datos – Procesamiento)**.  
Cada fila corresponde a **un registro en un momento del día** (idealmente 3: mañana, tarde y noche).  

---

## 🗂️ Estructura del CSV

| Columna                        | Tipo          | Escala / Valores válidos             | Descripción                                                                 |
|--------------------------------|--------------|---------------------------------------|-----------------------------------------------------------------------------|
| `entry_id`                     | texto        | libre (ej. `bdp-0001`)                | Identificador único del registro.                                           |
| `fecha`                        | texto        | `dd-MM-YYYY`                          | Fecha del registro.                                                         |
| `hora`                         | texto        | `HH:MM`                               | Hora del registro (solo horas y minutos).                                   |
| `animo`                        | número       | 0–10                                  | Estado emocional general (0 = muy bajo, 10 = muy alto).                     |
| `activacion`                   | número       | 0–10                                  | Nivel de energía / motivación.                                              |
| `conexion`                     | número       | 0–10                                  | Sensación de conexión con otras personas.                                   |
| `proposito`                    | número       | 0–10                                  | Sentido de dirección / propósito en el día.                                 |
| `claridad`                     | número       | 0–10                                  | Claridad mental o foco.                                                     |
| `estres`                       | número       | 0–10                                  | Estrés percibido (0 = nada, 10 = máximo).                                   |
| `sueno_calidad`                | número       | 0–10                                  | Calidad subjetiva del sueño de la última noche.                             |
| `horas_sueno`                  | número       | 0–24                                  | Horas de sueño.                                                             |
| `siesta_min`                   | número       | minutos (0–600)                       | Minutos de siesta durante el día.                                           |
| `autocuidado`                  | número       | 0–10                                  | Autocuidado percibido (higiene, orden, pausas, etc.).                       |
| `alimentacion`                 | número       | 0–10                                  | Calidad nutricional del día.                                                |
| `movimiento`                   | número       | 0–10                                  | Nivel de actividad física / movimiento.                                     |
| `dolor_fisico`                 | número       | 0–10                                  | Dolor físico percibido.                                                     |
| `ansiedad`                     | número       | 0–10                                  | Ansiedad percibida.                                                         |
| `irritabilidad`                | número       | 0–10                                  | Irritabilidad percibida.                                                    |
| `meditacion_min`               | número       | minutos (0–600)                       | Tiempo en meditación / respiración / mindfulness.                           |
| `exposicion_sol_min`           | número       | minutos (0–600)                       | Minutos de exposición al sol o luz natural.                                 |
| `agua_litros`                  | número       | litros (0–10)                         | Consumo de agua en litros.                                                  |
| `cafeina_mg`                   | número       | mg (0–1000)                           | Miligramos aproximados de cafeína.                                          |
| `alcohol_ud`                   | número       | unidades (0–20)                       | Unidades de alcohol (1 ud ≈ 10g etanol).                                    |
| `medicacion_tomada`            | texto        | `si` / `no`                           | Si se tomó la medicación pautada.                                           |
| `medicacion_tipo`              | texto        | libre                                 | Nombre/dosis de la medicación, si aplica.                                   |
| `otras_sustancias`             | texto        | libre                                 | Ej. nicotina, THC u otras.                                                  |
| `interacciones_significativas` | texto        | libre                                 | Personas o interacciones importantes del día.                               |
| `eventos_estresores`           | texto        | libre                                 | Eventos que hayan generado estrés.                                          |
| `tags`                         | texto        | lista separada por comas              | Palabras clave libres (ej. `trabajo,familia,ejercicio`).                     |
| `notas`                        | texto        | libre                                 | Comentarios adicionales.                                                    |

---

## 🎚️ Cómo calificar (0–10)

- **0 = mínimo / ausencia** (ej. nada de energía, sin dolor, sin ansiedad).  
- **10 = máximo / extremo** (ej. ansiedad máxima, dolor insoportable, estrés muy alto).  
- **5 ≈ punto medio** (ej. ánimo regular, energía suficiente, estrés manejable).  

👉 Lo importante es que uses **siempre la misma escala interna** para que tus datos sean comparables en el tiempo.  

---

## ✍️ Reglas prácticas

1. **Registros por día**: idealmente 3 (mañana, tarde, noche).  
2. **Campos opcionales**: si no aplica o no recuerdas, puedes dejarlo en **blanco**. El sistema lo interpretará como *dato faltante*.  
3. **Texto libre**: usa `notas`, `eventos_estresores`, `interacciones_significativas` para detallar lo que no cabe en números.  
4. **Tags**: no están predefinidos. Escribe las etiquetas que quieras (ej. `trabajo`, `familia`, `ejercicio`, `ocio`, `logro`, `conflicto`).  

---

## 🧾 Ejemplo de 3 registros en un día

```csv
entry_id,fecha,hora,animo,activacion,conexion,proposito,claridad,estres,sueno_calidad,horas_sueno,siesta_min,autocuidado,alimentacion,movimiento,dolor_fisico,ansiedad,irritabilidad,meditacion_min,exposicion_sol_min,agua_litros,cafeina_mg,alcohol_ud,medicacion_tomada,medicacion_tipo,otras_sustancias,interacciones_significativas,eventos_estresores,tags,notas
bdp-0001,21-08-2025,08:00,5,4,3,6,5,3,7,7.5,0,6,7,3,1,2,1,5,10,0,80,0,si,sertralina 50mg,,"Desayuno con mi pareja","Tráfico en la mañana","mañana,familia","Me sentí con energía media"
bdp-0002,21-08-2025,15:00,6,6,5,7,6,5,7,7.5,0,7,6,5,1,3,2,0,20,2.0,80,0,si,sertralina 50mg,,"Reunión laboral","Plazo entrega","trabajo,estres","Me costó concentrarme"
bdp-0003,21-08-2025,22:00,7,5,7,8,7,2,7,7.5,0,8,7,4,0,1,1,10,0,2.5,0,0,si,sertralina 50mg,,"Cena en familia","Ninguno","familia,ocio","Día cerró tranquilo"
```