"""SmartPark Madrid - pipeline de datos y modelado.

Paquete que orquesta el prototipo del TFG en cinco etapas:

    1. ingest      Lectura de los datos crudos (CSV de tickets, Excel de
                   parquimetros e inventario SER).
    2. clean       Filtrado al ambito de estudio (interior de la M-30) y
                   saneamiento de tipos.
    3. transform   Generacion de variables derivadas (hora, dia_semana...) y
                   serializacion a Parquet.
    4. scoring     Modelo de probabilidad estimada de encontrar aparcamiento.
    5. reservation Simulador de reserva basado en tickets activos.

Cada etapa esta en su propio modulo y solo depende de la configuracion
centralizada en `config.yaml`. Esto permite trazar cualquier decision de la
memoria (cap. 6) hasta la linea de codigo correspondiente.
"""

from pipeline.config import Config, load_config
from pipeline.scoring import ParkingScorer, ScoringResult
from pipeline.reservation import ReservationSimulator
from pipeline.profiling import BarrioProfile, BarrioProfiler
from pipeline.validation import ValidationResult, validar_holdout
from pipeline.routing import (
    distancia_haversine_km,
    eta_minutos,
    plazas_sincronizadas,
)
from pipeline.io import (
    cargar_tickets_fecha,
    fechas_disponibles,
    cargar_parquimetros,
    cargar_historico,
)

__all__ = [
    "Config",
    "load_config",
    "ParkingScorer",
    "ScoringResult",
    "ReservationSimulator",
    "BarrioProfile",
    "BarrioProfiler",
    "ValidationResult",
    "validar_holdout",
    "distancia_haversine_km",
    "eta_minutos",
    "plazas_sincronizadas",
    "cargar_tickets_fecha",
    "fechas_disponibles",
    "cargar_parquimetros",
    "cargar_historico",
]
