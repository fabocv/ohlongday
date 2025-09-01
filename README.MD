# 📒 Bienestar Data Project (BDP)

## 1. Introducción  
El **Bienestar Data Project (BDP)** es una iniciativa personal de autotracking orientada a la **observación, análisis y reflexión sobre el bienestar integral**.  
Busca integrar **datos objetivos y subjetivos** en torno a emociones, hábitos, fisiología y contexto, para apoyar tanto el **autoconocimiento** como procesos de **terapia psicológica**.  

⚠️ **Nota:** Este modelo **no reemplaza profesionales de la salud**, sino que actúa como complemento.  

### Objetivos  
- Crear una **base sólida de datos** sobre hábitos, emociones y bienestar.  
- Identificar patrones y correlaciones significativas.  
- Generar informes amigables con **interpretaciones técnicas y humanas**.  
- Facilitar el acompañamiento en procesos terapéuticos.  

---

## 2. Marco Teórico  
El modelo integra perspectivas desde:  
- **Psicología y neurociencia** (estado de ánimo, estrés, regulación emocional).  
- **Teoría polivagal** (impacto de sistema nervioso autónomo en conductas).  
- **Fenomenología y enfoque humanista** (vivencia subjetiva del bienestar).  
- **Hábitos cotidianos** y su influencia en bienestar a corto y largo plazo.  

---

## 3. Variables  

### 3.1 Núcleo emocional  
- Ánimo  
- Activación  
- Conexión  
- Propósito  
- Claridad  
- Estrés  

### 3.2 Fisiológicas  
- Sueño (hora_dormir, calidad, despertares nocturnos)  
- Movimiento (intensidad, ejercicio, siestas)  
- Glicemia  
- Autocuidado  
- Alimentación  

### 3.3 Hábitos y consumo  
- Café (cucharaditas, última hora)  
- Alcohol (unidades, última hora)  
- Otras sustancias  
- Medicación (tipo, tomada)  
- Agua (litros)  

### 3.4 Contextuales / Sociales  
- Interacciones (calidad, significativas, códigos personales)  
- Tiempo pantalla (total, noche, juegos dispositivo/persona)  
- Exposición solar  
- Espiritualidad (etiquetas personalizadas)  

---

## 4. Metodología de Procesamiento  
- **Normalización de datos** para homogeneizar escalas.  
- **Manejo de vacíos/outliers** mediante imputación suave.  
- **Modelos aplicados:**  
  - Z-score  
  - Media móvil exponencial (EMA)  
  - Correlaciones  
  - Clusters interpretados  
- Uso de **Escala fenomenológica 0–3** (bloqueo → avance).  

---

## 5. Reportes Generados  

### 5.1 BDP Report Coach  
- **Cards**: stoic_day, hard_day, retro_now, week_trendly  
- **Influencias**: delta stress influence, delta positive influence  
- **Micro-motivaciones**: frases humanas según nivel  
- **Focos personalizables**: (ej. Estrés/Carga 😵‍💫, Energía ⚖️, Recuperación 💪)  

### 5.2 Gráficos e Interpretaciones  
- Tendencias técnicas + interpretación humana  
- Frecuencias espirituales/cotidianas  
- Clusters de bienestar  

---

## 6. Configuración y Personalización  
- Uso de **códigos personales** (ej: `ABL01`, `FJJ`, `FKC`) para registrar experiencias privadas.  
- Definición de **umbrales automáticos** (reglas si/entonces).  
- Selección de focos adaptados por informe.  

---

## 7. Complemento con Otras Herramientas  
- Integración con **terapia psicológica** (insumo de datos para diálogo clínico).  
- Posible exportación a apps de **journaling**, **mindfulness** y **actividad física**.  
- Futuro: interoperabilidad con wearables (relojes, sensores, apps de sueño).  

---

## 8. Implementación Actual  
- **Captura de datos:** Google Forms (versión inicial).  
  - Motivo: ahorro de tiempo y rápida recolección de datos.  
  - Resultado: permite consolidar un **dataset base sólido** para futuros proyectos.  
- Procesamiento en **Python + Pandas**.  
- Visualización en **HTML/PDF**.  

---

## 9. Limitaciones y Futuras Mejoras  
- Falta integración hormonal (dopamina, serotonina, oxitocina).  
- Posible módulo de **geometría social**.  
- Señales fisiológicas rápidas (ej. MTC, indicadores corporales).  
- BDP 2.0: ampliar a modelos para TLP y bipolaridad.  

---

## 10. Fuentes Bibliográficas Seleccionadas  

- **Hábitos y Bienestar:**  
  - Lally, P., & Gardner, B. (2013). *Habits: A repeat performance*. Current Opinion in Psychology, 2, 1–6.  
  - Galla, B. M., & Duckworth, A. L. (2015). *More than resisting temptation: Beneficial habits mediate the relationship between self‐control and positive life outcomes*. JPSP, 109(3), 508.  
  - Oaten, M., & Cheng, K. (2016). *Longitudinal gains in self‐regulation from regular physical exercise*. British Journal of Health Psychology, 11(4), 717–733.  
  - Robinson, T. E., & Berridge, K. C. (2014). *The incentive sensitization theory of addiction*. Addiction, 109(9), 1590–1604.  
  - WHO (2020). *Guidelines on mental health, well-being and habits*.  

---

## 11. Conclusión  
El **BDP** en su versión actual entrega un marco de **autoobservación de hábitos y emociones**, complementa procesos terapéuticos y genera reportes accesibles.  
El uso de **Google Forms** como herramienta inicial permitió acelerar la construcción del dataset y sentar las bases para futuros desarrollos más robustos.  

---
