

# 📘 Bienestar Dinámico Personal (BDP)

El **Bienestar Dinámico Personal (BDP)** es un marco teórico-práctico diseñado para **medir, modelar y acompañar** la experiencia de las personas con foco en el **bienestar cotidiano**, particularmente útil en contextos de **trastornos del ánimo** (depresión, bipolaridad, TLP).

BDP no busca diagnosticar, sino **detectar patrones individuales**, **dar retroalimentación humanista** y **proponer micro-hábitos** que impacten positivamente en la vida diaria.

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

## 📌 Resumen técnico

El **BDP** integra **psicometría ligera**, **modelos dinámicos** y **mensajes humanistas** para acompañar el bienestar en tiempo real.

-   Fórmulas basadas en **z-scores idiográficos**.
    
-   Índices diseñados para **trastornos del ánimo (Depresión, Ansiedad, Bipolaridad, TLP)**.
    
-   Alertas definidas por **percentiles personales** y **duración mínima**.
    
-   Mensajes orientados a **hábitos concretos**, no a diagnósticos.