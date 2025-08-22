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