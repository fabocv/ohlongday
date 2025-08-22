# 📘 BDP – Implementación

Este proyecto implementa el **BDP (Balance Diario Psicoemocional)** en Python, de forma modular y con salida en **HTML enriquecido**. El sistema permite registrar, analizar y visualizar datos emocionales y de autocuidado, generando reportes semanales con un tono **humano, motivador y no clínico**.

---

## 📂 Estructura del CSV

El sistema trabaja sobre un archivo CSV con las siguientes columnas:

| fecha (dd-MM-YYYY) | hora (MM:SS) | animo | activacion | sueno | sueno_horas | conexion | proposito | claridad | estres | autocuidado | notas |
|--------------------|--------------|-------|------------|-------|-------------|----------|-----------|----------|--------|-------------|-------|
| 18-08-2025         | 08:30        | 2     | 1          | 2     | 7.5         | 2        | 2         | 1        | 0      | 7           | Descripción libre |

### Escalas sugeridas
- **Ánimo, activación, sueño, conexión, propósito, claridad, estrés**: escala Likert 0–3
- **Autocuidado**: escala 0–10
- **Sueño (horas)**: número decimal
- **Notas**: texto libre, opcional

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

- `weights`: pesos relativos de algunas variables (ej. impacto del sueño en vitalidad).  
- `thresholds`: umbrales para definir posibles hallazgos.  
- `report.days_window`: número de días recientes a incluir en el informe (default 7).  
- `report.messages_path`: ruta a un archivo JSON externo de mensajes.

---

## 💬 Reporte Coach (humanizado + niveles)

El reporte coach comunica hallazgos con **incertidumbre amable** y en tono humano, evitando lenguaje diagnóstico:

- `Indicador: posible ánimo bajo (leve|moderado|alto) — …`
- `Indicador: posible estrés alto (leve|moderado|alto) — …`
- `Indicador: posible sueño alterado (leve|moderado|alto) — …`
- `Indicador: posible autocuidado bajo (leve|moderado|alto) — …`
- `Indicador: posible tendencia positiva (leve|moderado|alto) — …`

### Niveles
- **Leve** → señales puntuales
- **Moderado** → tendencia sostenida o promedio claro
- **Alto** → rachas largas o valores extremos

---

## 📊 Panel-resumen en el HTML

El informe incluye un **panel de lectura rápida** con conteos por indicador y nivel en la ventana seleccionada:

- Ánimo bajo → leve: N | moderado: N | alto: N  
- Estrés alto → leve: N | moderado: N | alto: N  
- Sueño alterado → leve: N | moderado: N | alto: N  
- Autocuidado bajo → leve: N | moderado: N | alto: N  
- Tendencia positiva → leve: N | moderado: N | alto: N

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

## ✅ Buenas prácticas de comunicación

- Usar verbos **amables**, evitar imperativos.  
- Sugerir **micro-acciones concretas** y factibles.  
- Limitar la frecuencia: máximo 1–2 mensajes/día.  
- Recordar: **no diagnóstico**, **no reemplaza terapia profesional**.

---

## 🚀 Ejecución del sistema

```bash
python bdp_main.py --csv datos.csv
```

### Parámetros opcionales
- `--days N` → cambia la ventana de días (default: config).  
- `--tags` → filtra indicadores específicos (default: todos).  

Si no se entregan parámetros, el sistema toma `days_window` y todos los indicadores.

---

## 🔗 Notas finales

- El sistema está en evolución: busca acompañar, no diagnosticar.  
- Se recomienda **uso personal, exploratorio y reflexivo**.  
- Las salidas son HTML enriquecidos, fácilmente convertibles a PDF si se requiere.

