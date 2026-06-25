# Documento técnico · SmartPark Madrid

> **TFG:** Modelo Analítico para la Reducción de la Congestión Derivada de la Búsqueda de Estacionamiento en Madrid
> **Autora:** Noa Moliner Montoya · Universidad Alfonso X el Sabio · 2025

Este documento explica **cómo funciona el prototipo por dentro**, en lenguaje sencillo. Es complementario a la memoria escrita: en la memoria se argumenta el **por qué** académico del proyecto; aquí se cuenta el **cómo** está construido.

---

## 1. ¿Qué hace la aplicación, en una frase?

> A partir de millones de tickets de aparcamiento del centro de Madrid, la app aprende cuándo y dónde **suele haber más rotación de plazas**, y se lo cuenta al conductor antes de que llegue, recomendándole calles y plazas concretas que probablemente se queden libres justo cuando él aparezca.

Es un **prototipo de demostración**: simula el comportamiento que tendría una app integrada en el coche conectada con los datos del Ayuntamiento.

---

## 2. ¿De dónde salen los datos?

Todo viene del portal de datos abiertos del Ayuntamiento de Madrid (`datos.madrid.es`). Se usan **tres archivos**:

| Archivo | Para qué sirve |
|---|---|
| **Tickets SER del 1er trimestre 2025** (un CSV de 1,4 GB) | Es el "diario" de cada vez que alguien pagó un parquímetro: cuándo empezó, cuándo terminó, en qué calle, cuánto pagó, etc. |
| **Inventario de parquímetros** (Excel) | Lista de los parquímetros físicos con sus coordenadas (latitud/longitud) para poder dibujarlos en el mapa. |
| **Calles y plazas SER** (Excel) | Catálogo de calles donde existe zona regulada. Se usa en el análisis exploratorio. |

### Datos después de filtrar al interior de la M-30

El estudio se centra en **7 distritos del centro** con actividad SER en el interior de la M-30: Centro, Arganzuela, Retiro, Salamanca, Chamartín, Tetuán y Chamberí.

- 9 699 519 tickets (~76 % del total trimestral)
- 3 616 parquímetros con coordenadas
- 76 días con datos (2 enero – 1 abril 2025)

### Cosas a tener en cuenta sobre los datos (limitaciones honestas)

- **Coches con etiqueta CERO no aparecen**: aparcan gratis, no generan ticket. Son invisibles para el sistema.
- **Tickets pagados desde el móvil (EasyPark, Telpark…) no están atados a un parquímetro físico**: existen, pero no sabemos en qué plaza concreta están. Se excluyen al simular reservas.
- **Solo vemos quién pagó, no quién buscó y no encontró**. Estimamos demanda satisfecha, no demanda total.
- **Los nombres de calle vienen mal codificados** (las "Ñ" aparecen como `�`). El código lo arregla automáticamente al cargar.

---

## 3. ¿Cómo está organizado el código?

La idea principal es: **un único archivo de configuración manda sobre todo, y el código está partido en piezas pequeñas que hacen una sola cosa cada una.**

```
                       config.yaml
              (todos los parámetros aquí)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     run.py          prepare_data.py          app.py
   (lanzador)       (genera los datos)    (interfaz web)
                            │                   │
                            └────────┬──────────┘
                                     ▼
                              pipeline/
                       (el "motor" del proyecto)
```

### El paquete `pipeline/` por dentro

Cada módulo se encarga de un solo paso. Pensado para que se pueda explicar uno por uno en la defensa.

| Archivo | Qué hace, en cristiano |
|---|---|
| `config.py` | Lee `config.yaml` y reparte sus valores al resto del código. |
| `ingest.py` | Lee los archivos crudos (CSV y Excel) en trozos pequeños para no reventar la memoria. |
| `clean.py` | Filtra solo los datos del centro de Madrid y arregla formatos raros. |
| `transform.py` | Calcula columnas nuevas útiles (la hora, el día de la semana, etc.) y guarda todo en un formato más rápido (Parquet). |
| `io.py` | Lee solo los datos que la app necesita en cada momento (por ejemplo, "solo el 13 de febrero"). |
| `scoring.py` | El **cerebro**: estima la probabilidad de encontrar plaza. |
| `reservation.py` | Encuentra plazas concretas que están a punto de quedar libres. |
| `profiling.py` | Describe cada barrio (¿es comercial?, ¿residencial?). |
| `routing.py` | Calcula cuánto tardas en llegar y sincroniza la plaza con tu llegada. |
| `validation.py` | Comprueba con datos reales que el modelo acierta. |

### ¿Por qué partirlo así?

1. **Para la defensa:** cada decisión de la memoria tiene un archivo concreto al que apuntar.
2. **Sin números mágicos en el código:** si quiero cambiar un umbral, edito `config.yaml` y listo.
3. **Reutilizable:** el motor funciona desde la app, desde un notebook o desde un script.

---

## 4. La preparación de los datos (paso previo)

Esto se hace **una sola vez** al descargar los archivos crudos. Tarda unos 2 minutos.

```
Paso 1 — INGEST     leer el CSV en trozos de 500 000 filas
Paso 2 — CLEAN      quedarse solo con los distritos del interior de la M-30
                    convertir las fechas y los importes ('0,15' → 0.15)
Paso 3 — TRANSFORM  calcular hora, día de la semana, fecha
                    guardar en Parquet (formato comprimido y rápido)
```

### ¿Por qué Parquet en lugar del CSV?

- **Pesa 5 veces menos** (1,4 GB → 270 MB).
- **Se lee 10 veces más rápido**.
- **La app puede leer solo el día que necesita**, sin abrir el archivo entero. Esto es lo que permite que la app responda al instante al cambiar de fecha.

---

## 5. El "cerebro" del modelo: cómo se estima la probabilidad

### 5.1. El problema y por qué este enfoque

El gran reto: **no existen datos reales de "encontró/no encontró aparcamiento"**. Los datos solo dicen "alguien pagó". Por eso **no se puede entrenar un modelo predictivo clásico** (sería trampa: lo entrenaríamos con la misma señal que queremos predecir).

**La solución elegida** es un sistema de **puntuación por reglas**, sencillo y transparente. La gran ventaja: cualquier persona del tribunal puede entender de dónde sale un número concreto.

### 5.2. Cómo funciona el scoring, paso a paso

El usuario rellena 4 datos en la app:

1. **Distrito** al que va
2. **Hora** a la que llega
3. **Tipo de zona** que busca (azul, verde, comercial…)
4. **Etiqueta DGT** de su coche (B, C, ECO…)

Para cada uno de esos 4 datos, el sistema mira el histórico y se pregunta:

> *"¿Cuántos tickets se generaron normalmente en este distrito / a esta hora / en este tipo de zona / con esta etiqueta?"*

Y los clasifica en tres niveles según el volumen de actividad:

| Nivel | Significado | Puntos |
|---|---|---|
| **Baja** | Poca actividad → hay menos competencia, **más fácil aparcar** | 1 |
| **Media** | Actividad moderada | 2 |
| **Alta** | Mucha actividad → hay **más competencia** | 3 |

> **Idea clave:** mucha demanda = más difícil aparcar = menos probabilidad estimada.

Después se **suman** los puntos de las 4 dimensiones, dando un total entre **4 y 12**:

- Total = 4 → escenario muy fácil
- Total = 12 → escenario muy complicado

Ese total se traduce a un porcentaje de probabilidad, **siempre entre 10 % y 90 %**:

- Total mínimo (4) → 90 %
- Total máximo (12) → 10 %
- Valores intermedios → repartidos proporcionalmente

### 5.3. ¿Por qué nunca decimos 0 % ni 100 %?

A propósito. **El sistema nunca debe afirmar certezas absolutas**, porque trabaja con estimaciones. Mostrar siempre un margen (mínimo 10 %, máximo 90 %) es:

- **Honesto** con el usuario: es una orientación, no una garantía.
- **Defendible** ante el tribunal: no se vende falsa precisión.

Finalmente, la probabilidad se traduce a una etiqueta visual:

| Probabilidad | Etiqueta |
|---|---|
| 65 % o más | Alta |
| Entre 40 % y 65 % | Media |
| Menos del 40 % | Baja |

---

## 6. El simulador de reserva

### 6.1. ¿Por qué es una simulación?

Porque **el Ayuntamiento no publica el estado de cada plaza en tiempo real**. Para que esto fuera 100 % real haría falta una de estas dos cosas (declarado como línea futura en la memoria):

1. **Sensores en el suelo de cada plaza** (modelo "Smart Parking" clásico).
2. **Una API municipal** que diga qué plazas están ocupadas en cada momento.

Mientras tanto, simulamos ese comportamiento partiendo del histórico, que es lo más cercano que tenemos a la realidad.

### 6.2. ¿Cómo encuentra plazas que están a punto de liberarse?

Imagina que son las 12:00 y el usuario pregunta: *"¿qué plazas se van a liberar pronto en Salamanca?"*. El sistema hace lo siguiente:

1. **Busca todos los tickets activos en ese momento** (los que empezaron antes de las 12:00 y terminan después).
2. **Filtra los que terminan en los próximos 20 minutos** → estas son las plazas candidatas.
3. **Estima si esa persona va a renovar el ticket** mirando lo que suele hacer ese parquímetro: si normalmente la gente renueva sin moverse, la plaza no se libera de verdad. Si normalmente se van, sí.
4. **Combina las dos cosas** (cuánto falta para que termine + probabilidad de que no renueve) en una **puntuación de oportunidad**.
5. **Cruza con las coordenadas** del parquímetro para mostrarlo en el mapa.

### 6.3. Por qué se descartan los pagos desde el móvil

Las apps EasyPark, Telpark, etc. generan tickets pero **no están ligadas a una plaza física concreta**. Se excluyen del simulador filtrando por matrícula numérica del parquímetro (las apps tienen nombres tipo "EASYPARK", "TELPARK", que no son números).

---

## 7. La aplicación web (5 pestañas operativas + 2 técnicas)

Todo se construye con **Streamlit** (Python para hacer interfaces). Hay cinco pestañas operativas y, al activar el modo desarrollo, dos pestañas técnicas adicionales. En orden:

| Pestaña | Para qué sirve |
|---|---|
| **Navegación** | Funciona como un asistente de **dos pantallas** simulando el cuadro de mandos de un coche. **Pantalla 1** (estado *idle*): el conductor introduce origen y destino con tres modos de búsqueda (por calle libre que detecta el barrio automáticamente, por distrito completo, o por calle concreta dentro de un distrito). **Pantalla 2** (estado *navegando*): aparece un dashboard oscuro tipo Tesla/Android Auto con gauges de distancia, ETA y minutos hasta la liberación, además del botón grande de reservar plaza y el mapa con la ruta. |
| **Recomendación** | Compara los 7 distritos para el escenario que ha elegido el usuario y los ordena del más fácil al más difícil. |
| **Mapa en vivo** | Muestra en un mapa de Madrid las plazas candidatas a quedar libres en el instante elegido. Cuando se aplica el filtro de barrio (barra lateral), añade una ficha con el perfil del barrio (comercial, residencial, oficinas…) deducido de los datos. |
| **Calor por calle** | Mapa de calor por calle: dónde hay más presión de demanda en una franja horaria. |
| **Reservar plaza** | Simulación de reserva con cuenta atrás de 5 minutos. |
| **Validación** *(modo desarrollo)* | Resultado de comprobar que el modelo acierta sobre días que no había visto. |
| **Modelo** *(modo desarrollo)* | Trazabilidad: muestra los puntos asignados a cada dimensión, para que el tribunal pueda seguir el razonamiento. |

### Pequeños detalles de la app

- **Caché inteligente:** la app guarda en memoria los datos pesados (tickets, modelo) y solo los recarga si cambia algo importante. Por eso responde al instante.
- **Si se elige un domingo o festivo** en el que no hubo actividad SER, la app avisa de que no hay datos en lugar de fallar.
- **Diseño "cuadro de mandos del coche"** en la pestaña Navegación: la interfaz simula la pantalla de un vehículo (bisel oscuro, gauges con tipografía monoespaciada, LED pulsante de estado) para reforzar el discurso del TFG: el sistema está pensado para integrarse en el coche, no como una app aparte que mira el conductor.
- **Wizard de dos pasos**: separar la captura del destino del resultado evita la sobrecarga visual y permite que el usuario vea primero "¿a dónde voy?" y después "¿dónde aparco?". Es coherente con la experiencia real de un navegador.
- **Detección automática del barrio**: cuando el usuario solo conoce el nombre de la calle, el sistema infiere distrito y barrio mirando el inventario de parquímetros. Útil para conductores que no son de Madrid o no conocen la división administrativa.

---

## 8. ¿Cómo sé que el modelo funciona?

El módulo de validación (`pipeline/validation.py`) hace algo muy honesto:

1. **Esconde un día entero del histórico** (por ejemplo, el 15 de marzo).
2. **Reentrena el modelo sin ese día.**
3. **Pide al modelo que prediga el ranking de actividad para ese día.**
4. **Compara con lo que pasó realmente.**

Para medir el acierto se usa una correlación llamada **Spearman** (mide si el orden predicho coincide con el orden real).

### Resultados de la validación cruzada temporal (5 días)

| Día | Día de la semana | Correlación (modelo) | Aleatorio |
|---|---|---|---|
| 20 de enero | Lunes | -0.64 | -0.11 |
| 5 de febrero | Miércoles | -0.61 | -0.14 |
| 22 de febrero | Sábado | -0.77 | -0.12 |
| 10 de marzo | Lunes | -0.60 | -0.09 |
| 15 de marzo | Sábado | -0.74 | -0.13 |
| **Media** | | **-0.67 ± 0.08** | **-0.12** |

Como prueba de robustez, al repetir el día de mayor peso (15 de marzo) **excluyendo el distrito de Chamartín** la correlación sube a **-0.82**, lo que confirma que el modelo no depende de un único distrito.

> **Cómo se lee:** la correlación es **negativa** porque cuando el modelo dice "score alto = difícil aparcar", coincide con que ese día efectivamente hubo más actividad real (más demanda). Cuanto **más cercano a -1**, mejor. **Un sistema aleatorio daría alrededor de -0.12** (el ruido del propio reparto). Conseguir una media de -0.67 significa que el modelo es **unas 5 veces mejor que tirar los dados.**

---

## 9. ¿Cómo lanzar el proyecto?

```powershell
# 1. Instalar las librerías necesarias
py -m pip install -r requirements.txt

# 2. Descargar los 3 archivos crudos a la raíz del proyecto

# 3. Lanzar (todo en uno: comprueba, prepara y arranca la app)
py run.py
```

Todos los parámetros (rutas, filtros, umbrales, colores…) se editan en `config.yaml`, sin tocar código.

---

## 10. Mapeo: qué capítulo de la memoria está en qué archivo

| Capítulo de la memoria | Archivo de código |
|---|---|
| Cap. 1.2 — Ámbito de estudio (M-30) | `config.yaml` + `pipeline/clean.py` |
| Cap. 4 — Análisis exploratorio | `notebooks/analisis_exploratorio.ipynb` |
| Cap. 6.2 — Integración de datasets | `prepare_data.py` + `pipeline/ingest.py` |
| Cap. 6.3 — Indicador de presión | `pipeline/scoring.py` |
| Cap. 6.4 — Modelo de probabilidad | `pipeline/scoring.py` |
| Cap. 6.5 — Sistema de recomendación | `pipeline/scoring.py` + `pipeline/reservation.py` |
| Cap. 6.6 — Visualización en mapa | `app.py` (pestañas Mapa y Calor) |
| Cap. 6.7 — Sincronización con ETA | `pipeline/routing.py` + pestaña Navegación |
| Cap. 6.8 — Perfil por barrio | `pipeline/profiling.py` + filtro de barrio (barra lateral) y ficha en Mapa en vivo |
| Cap. 7 — Validación del modelo | `pipeline/validation.py` + pestaña Validación |
| Cap. 9 — Limitaciones | Secciones 2 y 6.1 de este documento |

---

## 11. Resumen 

- **El proyecto convierte 9,7 millones de tickets reales en una recomendación útil para el conductor.**
- **No se ha entrenado un modelo de caja negra**: se usa un sistema de puntuación transparente y defendible.
- **Las plazas se eligen mirando dos cosas**: cuánto falta para que el ticket termine y si la persona suele renovar o no.
- **La probabilidad nunca es 0 ni 100**: el sistema es honesto, ofrece estimación, no certeza.
- **Está validado con datos reales:** el modelo es 5 veces mejor que el azar.
- **Todo el comportamiento se controla desde un único archivo de configuración**, sin tocar código.
- **Las limitaciones (etiqueta CERO invisible, apps móviles, falta de ocupación real) están reconocidas**, no escondidas.
