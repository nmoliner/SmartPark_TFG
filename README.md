# SmartPark Madrid

**Modelo analítico para la reducción de la congestión derivada de la búsqueda de estacionamiento en Madrid**

Trabajo Fin de Grado · Grado en Business Analytics · Universidad Alfonso X el Sabio (UAX) · Noa Moliner Montoya · 2025-2026.

---

## 1. Descripción general

SmartPark Madrid es un prototipo funcional desarrollado en el marco del presente Trabajo Fin de Grado. Su finalidad es demostrar la viabilidad de un sistema analítico capaz de estimar la probabilidad de encontrar aparcamiento en el interior de la M-30 a partir, exclusivamente, de los datos abiertos publicados por el Ayuntamiento de Madrid. La aplicación se ejecuta sobre Python y Streamlit, replica visualmente la estética de un asistente embarcado en el vehículo y cubre el ciclo completo del proyecto: captación de fuentes, depuración, construcción del modelo, validación cuantitativa, generación de recomendaciones y simulación de reserva de plaza.

El prototipo se apoya en cuatro fuentes principales del portal `datos.madrid.es`:

- Tickets del Servicio de Estacionamiento Regulado (SER), primer trimestre de 2025.
- Inventario geolocalizado de parquímetros.
- Inventario de calles y plazas del SER.
- Datos de intensidad de tráfico para contextualización agregada.

---

## 2. Funcionalidades de la interfaz

La interfaz se estructura en una barra lateral con los parámetros de entrada y un área principal organizada en cinco pestañas operativas. Adicionalmente, un selector de modo desarrollo habilita dos pestañas técnicas de auditoría del modelo.

### Pestañas operativas

| Pestaña | Función |
|---|---|
| Navegación | Vista principal del asistente. Reúne la probabilidad estimada, el mapa de candidatos próximos al destino y la tabla de plazas sincronizadas con el tiempo estimado de llegada del conductor. |
| Recomendación | Ranking completo de zonas alternativas ordenadas por probabilidad para las condiciones de entrada introducidas. |
| Mapa en vivo | Mapa interactivo de plazas candidatas a liberación, con ficha resumen del barrio cuando se aplica el filtro correspondiente. |
| Calor por calle | Representación agregada de la presión por tramo de calle en la franja horaria consultada. |
| Reservar plaza | Mapa interactivo con marcadores numerados sobre el que el usuario selecciona una plaza concreta. Confirmación de reserva simulada con cuenta atrás. |

### Pestañas técnicas (modo desarrollo)

| Pestaña | Función |
|---|---|
| Validación | Resultados de la validación holdout temporal: correlación de Spearman, comparación con baselines, detalle por celda distrito × hora. |
| Modelo | Tablas internas del scorer: terciles de cada variable, mapeos de puntuación y umbrales de categorización. |

---

## 3. Instalación y ejecución

### 3.1 Requisitos previos

- Python 3.12 o superior.
- Sistema operativo Windows, macOS o Linux.
- Aproximadamente 4 GB de espacio en disco para los datasets crudos y los Parquet derivados.

### 3.2 Instalación de dependencias

```powershell
py -m pip install -r requirements.txt
```

### 3.3 Obtención de los datos

Los datasets utilizados deben descargarse desde el [Portal de Datos Abiertos del Ayuntamiento de Madrid](https://datos.madrid.es) y colocarse en la raíz del proyecto con los siguientes nombres:

- `Primertrimestre2025 SER tickets aparcamiento.csv` (aproximadamente 1,4 GB).
- `300481-3-ser-parquimetros-xlsx.xlsx`.
- `218228-0-ser-calles SER CALLES Y PLAZAS-xlsx.xlsx`.

### 3.4 Ejecución

```powershell
py run.py
```

El script de lanzamiento verifica el entorno, genera los ficheros Parquet derivados la primera vez que se ejecuta (proceso que tarda aproximadamente dos minutos en una máquina estándar) y abre la aplicación en el navegador predeterminado del sistema.

---

## 4. Arquitectura del proyecto

El código se organiza en torno a una separación deliberada entre la lógica de negocio (paquete `pipeline/`) y la capa de presentación (`app.py`). Esta separación facilita la prueba automatizada del modelo y permite reutilizar el motor analítico desde otros canales sin reescribir su lógica.

```
                +-------------------------------+
                |          config.yaml          |
                |  (fuente única de parámetros) |
                +-------------------------------+
                              |
        +---------------------+---------------------+
        v                     v                     v
    +--------+         +--------------+         +--------+
    | run.py | ------> | prepare_data |         | app.py |
    +--------+         +--------------+         +--------+
                              |                       |
                              v                       v
              +----------------------------------------+
              |                pipeline/               |
              |  ingest · clean · transform · io ·     |
              |  scoring · reservation · profiling ·   |
              |  routing · validation                  |
              +----------------------------------------+
                              |
                              v
              +----------------------------------------+
              |      data/  (Parquet, generado)        |
              +----------------------------------------+
```

Para una descripción técnica detallada del pipeline y de cada módulo, véase [docs/TECNICO.md](docs/TECNICO.md).

---

## 5. Estructura del repositorio

```
TFG/
├── config.yaml              parámetros del sistema (rutas, scoring, reserva)
├── run.py                   lanzador principal con verificación de entorno
├── prepare_data.py          orquestador del pipeline batch
├── app.py                   interfaz Streamlit
├── pipeline/                paquete reutilizable del motor analítico
│   ├── config.py            carga y validación del fichero de configuración
│   ├── ingest.py            lectura por bloques de CSV y Excel
│   ├── clean.py             filtrado al ámbito M-30 y saneamiento
│   ├── transform.py         construcción de variables derivadas y Parquet
│   ├── io.py                lecturas optimizadas con pushdown
│   ├── scoring.py           ParkingScorer (estimación de probabilidad)
│   ├── reservation.py       ReservationSimulator (simulador de reserva de plaza)
│   ├── profiling.py         BarrioProfiler
│   ├── routing.py           ETA y sincronización con la liberación de plaza
│   └── validation.py        validación holdout y correlación de Spearman
├── docs/
│   ├── TECNICO.md           documentación técnica del pipeline
│   └── figuras/             figuras y métricas generadas por los scripts de análisis
├── scripts/                 scripts de análisis y exportación de figuras
│   ├── regenerar_figuras_cap6.py       regenera todas las figuras del análisis exploratorio
│   ├── exportar_figura_cap6_trafico.py exporta la figura de evolución del tráfico
│   ├── exportar_figura_cap7.py         exporta la figura de validación del modelo
│   └── experimentos_extra_cap7.py      validación cruzada, sensibilidad y robustez
├── notebooks/               análisis exploratorio en cuadernos Jupyter
│   ├── analisis_exploratorio.ipynb     análisis completo (tráfico, demanda, modelo)
│   └── intensidad días laborables 2021-2025.ipynb
├── tests/                   suite de pruebas automatizadas (pytest)
└── requirements.txt         dependencias del proyecto
```

---

## 6. Comandos disponibles

```powershell
py run.py                    # Lanza la aplicación (uso habitual).
py run.py --check            # Verifica el entorno sin lanzar la aplicación.
py run.py --prepare          # Regenera los ficheros Parquet sin lanzar la aplicación.
py run.py --force-prepare    # Regenera los Parquet y a continuación lanza la aplicación.
```

Para ejecutar la suite de pruebas:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## 7. Métricas del prototipo

Los siguientes valores se obtienen de la ejecución del pipeline sobre los datos del primer trimestre de 2025:

- **9.699.519 tickets** procesados en el ámbito M-30 (entrenamiento: 9.627.875; test holdout: 71.644).
- **3.616 parquímetros** activos con coordenadas geográficas.
- **76 días** con datos disponibles en el rango analizado.
- **Correlación de Spearman = −0,74** (validación holdout sobre el 15 de marzo de 2025; baseline aleatorio: −0,13).
- **Correlación de Spearman = −0,67 ± 0,08** en la validación cruzada temporal sobre cinco días representativos.
- **−0,82** sin el distrito de Chamartín (análisis de robustez geográfica).

El signo negativo del coeficiente es el esperado por construcción: a mayor probabilidad predicha de encontrar aparcamiento, menor presión observada en los datos. La magnitud absoluta del valor sitúa la correlación dentro del rango habitualmente considerado fuerte en la literatura.

---

## 8. Documentación académica

La memoria del TFG se entrega de forma independiente como documento Word. Este repositorio contiene exclusivamente el prototipo y los materiales reproducibles que la sustentan:

- El análisis exploratorio completo (tráfico, demanda por hora/distrito/zona/distintivo, indicador de presión y modelo de probabilidad) en `notebooks/analisis_exploratorio.ipynb`.
- Las figuras y métricas que aparecen en la memoria, regenerables con los scripts de `scripts/` y almacenadas en `docs/figuras/`.
- La documentación técnica del pipeline en `docs/TECNICO.md`.

Para regenerar las figuras del análisis:

```powershell
.\.venv\Scripts\python.exe scripts\regenerar_figuras_cap6.py
```

---

## 9. Fuentes de datos

Todas las fuentes utilizadas son de carácter público y proceden del [Portal de Datos Abiertos del Ayuntamiento de Madrid](https://datos.madrid.es), publicado bajo la política municipal de reutilización de la información del sector público.

---

## 10. Licencia y atribución

El código fuente se publica con finalidad estrictamente académica, vinculado al Trabajo Fin de Grado descrito. Los datasets utilizados pertenecen al Ayuntamiento de Madrid y se distribuyen conforme a las condiciones de su portal de datos abiertos.
