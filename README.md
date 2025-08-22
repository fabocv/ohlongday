
# Oh Long Day - Registro manual de BDP
### Autor: fabocv
> versión 0.3

El proyecto **Oh Long Day** tiene como objetivo aplicar el marco BDP para estudiar el bienestar de una persona mediante registros (por ahora en csv, manual) y crear un dataset propio para propósitos de investigación y autoconocimiento de personas con trastornos de ánimo.

El **Bienestar Dinámico Personal (BDP)** es un marco teórico-práctico diseñado para **medir, modelar y acompañar** la experiencia de las personas con foco en el **bienestar cotidiano**, particularmente útil en contextos de **trastornos del ánimo** (depresión, bipolaridad, TLP).

BDP no busca diagnosticar, sino **detectar patrones individuales**, **dar retroalimentación humanista** y **proponer micro-hábitos** que impacten positivamente en la vida diaria.

---

## 🎯 Objetivos

- **Medición dinámica:** capturar de manera continua y ligera el estado afectivo, energético y relacional de una persona.  
- **Identificación de patrones:** detectar tendencias relevantes como hipomanía, labilidad afectiva o desregulación ansiosa/depresiva.  
- **Integración de micro-hábitos:** vincular pequeños cambios de conducta (sueño, movimiento, conexión, mindfulness) con su impacto real en el bienestar.  
- **Alertas humanistas:** ofrecer mensajes amables y neutrales que reflejen los datos sin caer en diagnósticos invasivos.  
- **Empoderamiento personal:** entregar al usuario una herramienta de autoobservación que complemente (pero no reemplace) la terapia profesional.

Mas detalles sobre el modelo: [Modelado del Bienestar Dinámico Personal (BDP)](docs/BDP.md)

---


# 📘 BDP – Implementación

Este proyecto implementa el **BDP (Balance Diario Psicoemocional)** en Python, de forma modular y con salida en **HTML enriquecido**. El sistema permite registrar, analizar y visualizar datos emocionales y de autocuidado, generando reportes semanales con un tono **humano, motivador y no clínico**.

> Mas detalles de como se rellena el registro CSV: [📘 Guía para Rellenar el CSV del BDP](docs/BDP Registro.md)

---

## 📂 Estructura del CSV

El sistema trabaja sobre un archivo CSV con las siguientes columnas:

| fecha (dd-MM-YYYY) | hora (MM\:SS) | animo | activacion | sueno | sueno\_horas | conexion | proposito | claridad | estres | autocuidado | notas             |
| ------------------ | ------------- | ----- | ---------- | ----- | ------------ | -------- | --------- | -------- | ------ | ----------- | ----------------- |
| 18-08-2025         | 08:30         | 2     | 1          | 2     | 7.5          | 2        | 2         | 1        | 0      | 7           | Descripción libre |

### Escalas sugeridas

* **Ánimo, activación, sueño, conexión, propósito, claridad, estrés**: escala Likert 0–3
* **Autocuidado**: escala 0–10
* **Sueño (horas)**: número decimal
* **Notas**: texto libre, opcional

🔹 Se recomienda **3 registros por día** (mañana, tarde, noche).
🔹 Se pueden dejar celdas en blanco: el sistema ignora los valores vacíos en sus cálculos.

---

## ⚙️ Configuración (`bdp_config.json`)

Archivo de parámetros principales:

```json
{
  "weights": {
    "sueno_en_vitalidad": 1.2
  },
  "thresholds": {
    "animo_bajo_z": -0.3,
    "animo_bajo_streak": 3,
    "estres_alto_z": 0.6,
    "estres_alto_ratio_min": 0.3,
    "sueno_irregular_std_horas": 1.2,
    "sueno_calidad_media_min": -0.2,
    "tendencia_positiva_delta": 0.3,
    "autocuidado_media_min": 5
  },
  "report": {
    "days_window": 7,
    "messages_path": "mensajes.json"
  }
}
```

* `weights`: pesos relativos de algunas variables (ej. impacto del sueño en vitalidad).
* `thresholds`: umbrales para definir posibles hallazgos.
* `report.days_window`: número de días recientes a incluir en el informe (default 7).
* `report.messages_path`: ruta a un archivo JSON externo de mensajes.

---

## 💬 Reporte Coach (humanizado + niveles)

El reporte coach comunica hallazgos con **incertidumbre amable** y en tono humano, evitando lenguaje diagnóstico:

* `Indicador: posible ánimo bajo (leve|moderado|alto) — …`
* `Indicador: posible estrés alto (leve|moderado|alto) — …`
* `Indicador: posible sueño alterado (leve|moderado|alto) — …`
* `Indicador: posible autocuidado bajo (leve|moderado|alto) — …`
* `Indicador: posible tendencia positiva (leve|moderado|alto) — …`

Además un resumen de las tendencias semanales por cada **área**. Mas detalles sobre el áreas del modelo: [Modelado del Bienestar Dinámico Personal (BDP)](docs/BDP.md)

### Niveles

* **Leve** → señales puntuales
* **Moderado** → tendencia sostenida o promedio claro
* **Alto** → rachas largas o valores extremos

---

## 📊 Panel-resumen en el HTML

El informe incluye un **panel de lectura rápida** con conteos por indicador y nivel en la ventana seleccionada:

* Ánimo bajo → leve: N | moderado: N | alto: N
* Estrés alto → leve: N | moderado: N | alto: N
* Sueño alterado → leve: N | moderado: N | alto: N
* Autocuidado bajo → leve: N | moderado: N | alto: N
* Tendencia positiva → leve: N | moderado: N | alto: N

> ⚠️ El panel no muestra diagnósticos, solo **señales posibles** con incertidumbre (“posible …”), reforzando el carácter **no clínico** del sistema.

---

## 📑 Mensajes humanizados (`mensajes.json`)

El sistema selecciona aleatoriamente frases según **indicador** + **nivel**. Ejemplo:

```json
{
  "animo_bajo": {
    "leve": [
      "A veces el cuerpo pide calma. Está bien reconocerlo.",
      "Hoy puede ser un buen día para regalarte un pequeño descanso.",
      "Aunque sea leve, cuidar tu ánimo suma mucho en el camino."
    ],
    "moderado": [
      "Han sido varios días desafiantes, date permiso de suavizar el paso.",
      "Tu ánimo merece espacios de ternura y escucha.",
      "Pequeños gestos de autocuidado pueden abrir un respiro en medio del peso."
    ],
    "alto": [
      "Se siente una racha de bajón fuerte. No estás solo en este proceso.",
      "Quizá sea buen momento para pedir apoyo o hablar con alguien cercano.",
      "Es valioso reconocer este estado: también pasará, y tu cuidado importa."
    ]
  }
}
```

📌 Puedes ampliar este JSON con más indicadores (`estres_alto`, `sueno_alterado`, etc.) siguiendo el mismo patrón.

---


## 🚀 Ejecución del sistema

```bash
python bdp_main.py --csv datos.csv
```

### Parámetros opcionales

* `--days N` → cambia la ventana de días (default: config).
* `--tags` → filtra indicadores específicos. Ej: trabajo, estrés, fiesta, etc. (default: todos). Mas detalles sobre los tags del modelo: [📘 Guía para Rellenar el CSV del BDP](docs/BDP Registro.md)

Si no se entregan parámetros, el sistema toma `days_window` y todos los indicadores.

---

## 🔗 Notas finales

* El sistema está en evolución: busca acompañar, no diagnosticar.
* Se recomienda **uso personal, exploratorio y reflexivo**.
* Las salidas son HTML enriquecidos, fácilmente convertibles a PDF si se requiere.



