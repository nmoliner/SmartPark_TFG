"""Etapa 6 del pipeline: perfilado descriptivo de barrios.

Caracteriza cada barrio del ambito M-30 a partir del histórico de tickets
y del inventario de parquimetros. Devuelve un perfil cualitativo (tipologia
y descripcion) **derivado de datos**, no escrito a mano: cada afirmacion es
trazable a una metrica.

Metricas calculadas por barrio:
  - pct_azul, pct_verde: distribucion del tipo de zona (parquimetros)
  - duracion_media_min: duracion media de los tickets emitidos
  - pico_horario: franja con mayor volumen de tickets
  - tasa_no_renovacion: % de tickets no renovados en la ventana
  - importe_medio: tarifa media abonada
  - num_parquimetros: tamaño del inventario en el barrio

Tipologia inferida segun reglas declaradas en config.yaml::profiling:
  - COMERCIAL   : zona azul mayoritaria + duracion corta
  - OFICINAS    : duracion media-larga + pico mañana/tarde laboral
  - RESIDENCIAL : zona verde mayoritaria + baja rotacion
  - MIXTO       : ningun patron claro
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.config import Config, load_config


@dataclass(frozen=True)
class BarrioProfile:
    """Perfil de un barrio derivado de datos historicos."""

    distrito: str
    barrio: str
    tipologia: str
    descripcion: str
    pct_azul: float
    pct_verde: float
    duracion_media_min: float
    pico_horario: int
    tasa_no_renovacion: float
    importe_medio: float
    num_parquimetros: int
    num_tickets: int


class BarrioProfiler:
    """Genera perfiles cualitativos por barrio a partir del histórico."""

    def __init__(
        self,
        tickets: pd.DataFrame,
        parquimetros: pd.DataFrame,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.tickets = tickets
        self.parqs = parquimetros
        self._cache: dict[tuple[str, str], BarrioProfile] = {}

    # ------------------------------------------------------------------ #

    def perfil(self, distrito: str, barrio: str) -> BarrioProfile:
        """Devuelve el perfil del barrio (cacheado)."""
        key = (distrito, barrio)
        if key in self._cache:
            return self._cache[key]

        # Parquimetros del barrio
        parqs_b = self.parqs[
            (self.parqs["distrito"] == distrito)
            & (self.parqs["barrio"] == barrio)
        ]
        num_parqs = len(parqs_b)
        if num_parqs == 0:
            return self._perfil_vacio(distrito, barrio)

        # Tickets del barrio (cruce por matricula del parquimetro)
        matriculas = set(parqs_b["matricula"].astype(str))
        tickets_b = self.tickets[
            self.tickets["matricula_parquimetro"].astype(str).isin(matriculas)
        ]
        num_tickets = len(tickets_b)

        if num_tickets == 0:
            return self._perfil_vacio(
                distrito, barrio, num_parquimetros=num_parqs,
            )

        # Distribucion zona (azul = rotacion, verde = residentes).
        # tipo_zona vive en los tickets, no en el inventario de parquimetros.
        zona_counts = tickets_b["tipo_zona"].value_counts(normalize=True)
        pct_azul = float(zona_counts.get("AZUL", 0.0)) * 100
        pct_verde = float(zona_counts.get("VERDE", 0.0)) * 100

        # Duracion media e importe medio
        duracion_media = float(tickets_b["minutos_tique"].mean())
        importe_medio = float(tickets_b["importe_tique"].mean())

        # Pico horario
        if "hora" in tickets_b.columns:
            pico = int(tickets_b["hora"].mode().iloc[0])
        else:
            pico = int(tickets_b["fecha_inicio"].dt.hour.mode().iloc[0])

        # Tasa de no-renovacion: tickets cuyo siguiente del mismo parquimetro
        # llega despues de la ventana de renovacion (renewal_window_min).
        tasa_no_renov = self._tasa_no_renovacion(tickets_b)

        tipologia = self._inferir_tipologia(
            pct_azul=pct_azul,
            pct_verde=pct_verde,
            duracion=duracion_media,
            pico=pico,
        )
        descripcion = self._descripcion(
            tipologia,
            pct_azul=pct_azul,
            pct_verde=pct_verde,
            duracion=duracion_media,
            pico=pico,
        )

        perfil = BarrioProfile(
            distrito=distrito, barrio=barrio,
            tipologia=tipologia, descripcion=descripcion,
            pct_azul=pct_azul, pct_verde=pct_verde,
            duracion_media_min=duracion_media,
            pico_horario=pico,
            tasa_no_renovacion=tasa_no_renov,
            importe_medio=importe_medio,
            num_parquimetros=num_parqs,
            num_tickets=num_tickets,
        )
        self._cache[key] = perfil
        return perfil

    # ------------------------------------------------------------------ #

    def _tasa_no_renovacion(self, tickets: pd.DataFrame) -> float:
        ventana = self.cfg.raw["reservation"]["renewal_window_min"]
        df = tickets.sort_values(
            ["matricula_parquimetro", "fecha_inicio"]
        ).copy()
        df["siguiente"] = df.groupby("matricula_parquimetro")[
            "fecha_inicio"
        ].shift(-1)
        df["gap_min"] = (
            df["siguiente"] - df["fecha_fin"]
        ).dt.total_seconds() / 60
        renovados = ((df["gap_min"] >= 0) & (df["gap_min"] <= ventana)).sum()
        total_con_siguiente = df["siguiente"].notna().sum()
        if total_con_siguiente == 0:
            return 0.0
        return float(1 - renovados / total_con_siguiente) * 100

    def _inferir_tipologia(
        self, pct_azul: float, pct_verde: float,
        duracion: float, pico: int,
    ) -> str:
        rules = self.cfg.raw.get("profiling", {})
        thr = rules.get("thresholds", {})
        # Umbrales por defecto si no estan en config
        azul_alto = thr.get("azul_alto", 60)
        verde_alto = thr.get("verde_alto", 60)
        duracion_corta = thr.get("duracion_corta_min", 90)
        duracion_larga = thr.get("duracion_larga_min", 120)
        pico_oficina_min = thr.get("pico_oficina_min", 9)
        pico_oficina_max = thr.get("pico_oficina_max", 11)

        if pct_azul >= azul_alto and duracion <= duracion_corta:
            return "COMERCIAL"
        if (
            duracion >= duracion_larga
            and pico_oficina_min <= pico <= pico_oficina_max
        ):
            return "OFICINAS"
        if pct_verde >= verde_alto:
            return "RESIDENCIAL"
        return "MIXTO"

    def _descripcion(
        self, tipologia: str, pct_azul: float, pct_verde: float,
        duracion: float, pico: int,
    ) -> str:
        textos = {
            "COMERCIAL": (
                f"Barrio con alta rotación: {pct_azul:.0f}% de zona AZUL y "
                f"duración media de tickets de {duracion:.0f} min. "
                f"Probabilidad de encontrar plaza relativamente alta."
            ),
            "OFICINAS": (
                f"Barrio de oficinas: pico horario a las {pico}:00 y "
                f"duración media de {duracion:.0f} min (estacionamiento "
                f"prolongado). Mayor dificultad en horario laboral."
            ),
            "RESIDENCIAL": (
                f"Barrio residencial: {pct_verde:.0f}% de zona VERDE "
                f"(plazas reservadas a residentes). Mayor dificultad para "
                f"encontrar plaza libre y duración media de {duracion:.0f} min."
            ),
            "MIXTO": (
                f"Barrio con perfil mixto: {pct_azul:.0f}% AZUL / "
                f"{pct_verde:.0f}% VERDE, duración media {duracion:.0f} min, "
                f"pico a las {pico}:00."
            ),
        }
        return textos.get(tipologia, "")

    @staticmethod
    def _perfil_vacio(
        distrito: str, barrio: str,
        num_parquimetros: int = 0,
    ) -> BarrioProfile:
        return BarrioProfile(
            distrito=distrito, barrio=barrio,
            tipologia="SIN DATOS",
            descripcion="No hay tickets suficientes para perfilar este barrio.",
            pct_azul=0.0, pct_verde=0.0,
            duracion_media_min=0.0, pico_horario=0,
            tasa_no_renovacion=0.0, importe_medio=0.0,
            num_parquimetros=num_parquimetros, num_tickets=0,
        )
