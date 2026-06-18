"""Etapa 5 del pipeline: simulador de reserva.

Implementa el `ReservationSimulator` descrito en el cap. 6.5 de la memoria.

Conceptos clave (parametros en config.yaml):
  - Tickets activos en el instante T: aquellos cuyo intervalo
    [fecha_inicio, fecha_fin] cubre T.
  - Candidatos a liberacion: activos cuya fecha_fin esta a menos de
    `liberation_window_min` minutos.
  - Probabilidad de no-renovacion: % historico de tickets en cada
    parquimetro que no fueron renovados en los `renewal_window_min`
    minutos posteriores a expirar.
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import Config, load_config


class ReservationSimulator:
    """Modulo de prediccion de liberacion de plazas."""

    def __init__(
        self,
        tickets_dia: pd.DataFrame,
        parquimetros: pd.DataFrame,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg or load_config()

        # Excluimos tickets de apps moviles: no estan ligados a un parquimetro
        # fisico, asi que no se puede reservar plaza desde ellos. Se detecta
        # mediante la regex declarada en config.yaml (matriculas numericas).
        es_fisico = tickets_dia["matricula_parquimetro"].str.fullmatch(
            self.cfg.physical_meter_regex, na=False
        )
        self.tickets = tickets_dia[es_fisico].copy()

        for c in ("fecha_inicio", "fecha_fin"):
            self.tickets[c] = pd.to_datetime(self.tickets[c])

        self.tickets = self.tickets.sort_values(
            ["matricula_parquimetro", "fecha_inicio"]
        ).reset_index(drop=True)

        self.parqs = parquimetros.set_index("matricula")
        self.prob_no_renovacion = self._precompute_no_renewal()

    # ----- Probabilidad de no-renovacion (precomputada) ----------------- #

    def _precompute_no_renewal(self) -> pd.Series:
        """Probabilidad de no-renovacion por parquimetro (media historica simple).

        Para cada parquimetro se calcula el porcentaje de tickets que NO fueron
        renovados en los `renewal_window_min` minutos posteriores a expirar:

            P(no renueva) = n_no_renov / n_total

        Cuanto mayor es este porcentaje, mas probable es que la plaza quede
        libre cuando el ticket actual expire.
        """
        ventana = pd.Timedelta(minutes=self.cfg.renewal_window_min)
        df = self.tickets.copy()
        df["sig_inicio"] = df.groupby(
            "matricula_parquimetro"
        )["fecha_inicio"].shift(-1)
        diff = df["sig_inicio"] - df["fecha_fin"]
        df["renovado"] = (
            df["sig_inicio"].notna()
            & (diff <= ventana)
            & (diff >= pd.Timedelta(0))
        )
        agg = df.groupby("matricula_parquimetro").agg(
            n_total=("renovado", "size"),
            n_renov=("renovado", "sum"),
        )
        n_no_renov = agg["n_total"] - agg["n_renov"]
        prob = n_no_renov / agg["n_total"]
        prob.name = "prob_no_renovacion"
        return prob

    # ----- Snapshot ----------------------------------------------------- #

    def _activos_en(self, instante: pd.Timestamp) -> pd.DataFrame:
        return self.tickets[
            (self.tickets["fecha_inicio"] <= instante)
            & (self.tickets["fecha_fin"] > instante)
        ]

    def estado_instante(self, instante: pd.Timestamp) -> dict[str, int]:
        """Resumen del estado del sistema en `instante` (para la UI)."""
        activos = self._activos_en(instante)
        ventana = self.cfg.liberation_window_min
        candidatos = activos[
            (activos["fecha_fin"] - instante).dt.total_seconds() / 60 <= ventana
        ]
        return {
            "tickets_activos": int(len(activos)),
            "candidatos_liberacion": int(len(candidatos)),
            "parquimetros_unicos_activos": int(
                activos["matricula_parquimetro"].nunique()
            ),
        }

    def candidatos(
        self,
        instante: pd.Timestamp,
        distrito: str | None = None,
    ) -> pd.DataFrame:
        """Devuelve los parquimetros candidatos a liberacion en `instante`.

        Cada fila incluye coordenadas, minutos hasta liberacion, probabilidad
        historica de no-renovacion y un score de oportunidad combinado.
        """
        activos = self._activos_en(instante)
        if distrito:
            activos = activos[activos["distrito"] == distrito]

        ventana = self.cfg.liberation_window_min
        activos = activos.assign(
            min_hasta_liberacion=(activos["fecha_fin"] - instante).dt.total_seconds() / 60
        )
        candidatos = activos[activos["min_hasta_liberacion"] <= ventana].copy()

        # Cruce con coordenadas reales del parquimetro
        candidatos = candidatos.merge(
            self.parqs[["longitud", "latitud", "calle", "numero_finca"]],
            left_on="matricula_parquimetro",
            right_index=True,
            how="left",
        )

        candidatos["prob_no_renovacion"] = (
            candidatos["matricula_parquimetro"]
            .map(self.prob_no_renovacion)
            .fillna(0.5)  # default neutro si no hay historico para este parq
        )

        # Score: cuanto antes libere y mas probable que no renueve, mejor.
        candidatos["score_oportunidad"] = (
            (1 - candidatos["min_hasta_liberacion"] / ventana)
            * candidatos["prob_no_renovacion"]
        )

        cols = [
            "matricula_parquimetro", "distrito", "barrio",
            "calle", "numero_finca", "longitud", "latitud",
            "fecha_inicio", "fecha_fin", "min_hasta_liberacion",
            "prob_no_renovacion", "score_oportunidad",
            "tipo_zona", "distintivo",
        ]
        return (
            candidatos[cols]
            .sort_values("score_oportunidad", ascending=False)
            .reset_index(drop=True)
        )

    # ----- Presion por calle (vista cromatica) -------------------------- #

    def presion_por_calle(
        self,
        hora: int,
        distrito: str | None = None,
    ) -> pd.DataFrame:
        """Tickets por calle en una franja horaria, clasificados en terciles.

        Devuelve una fila por parquimetro etiquetada con el nivel de presion
        de su calle, lista para ser pintada en el mapa.
        """
        df = self.tickets.merge(
            self.parqs[["calle", "longitud", "latitud", "distrito"]].reset_index(),
            left_on="matricula_parquimetro",
            right_on="matricula",
            how="left",
            suffixes=("_t", ""),
        )
        if distrito:
            df = df[df["distrito"] == distrito]
        df = df[df["hora"] == hora]
        if df.empty:
            return pd.DataFrame(columns=[
                "calle", "matricula_parquimetro", "num_tickets",
                "nivel", "longitud", "latitud",
            ])

        por_calle = (
            df.groupby("calle", observed=True).size()
              .reset_index(name="num_tickets")
        )
        try:
            por_calle["nivel"] = pd.qcut(
                por_calle["num_tickets"], q=3,
                labels=list(self.cfg.score_map.keys()),
                duplicates="drop",
            )
        except ValueError:
            por_calle["nivel"] = "Media"
        por_calle["nivel"] = por_calle["nivel"].astype(str)

        return (
            df[["matricula_parquimetro", "calle", "longitud", "latitud"]]
            .drop_duplicates(subset=["matricula_parquimetro"])
            .merge(por_calle, on="calle", how="left")
            .dropna(subset=["latitud", "longitud", "calle"])
            .reset_index(drop=True)
        )

    # ----- Calles disponibles por barrio -------------------------------- #

    def calles_disponibles(
        self,
        instante: pd.Timestamp,
        distrito: str,
        barrio: str | None = None,
    ) -> pd.DataFrame:
        """Ranking de calles con disponibilidad en un barrio concreto.

        Para cada calle del distrito (o del barrio si se especifica) devuelve:
          - num_parquimetros: parquimetros totales en la calle
          - candidatos: cuantos liberan plaza en la ventana
          - score_medio: oportunidad media de los candidatos
          - mejor_score: maxima oportunidad disponible
          - latitud / longitud: centro de la calle (para mapa)
        """
        cand = self.candidatos(instante, distrito=distrito)
        if barrio:
            cand = cand[cand["barrio"] == barrio]
        if cand.empty:
            return pd.DataFrame(columns=[
                "calle", "candidatos", "score_medio", "mejor_score",
                "min_proxima_liberacion", "latitud", "longitud",
            ])

        agg = (
            cand.groupby("calle", observed=True)
            .agg(
                candidatos=("matricula_parquimetro", "count"),
                score_medio=("score_oportunidad", "mean"),
                mejor_score=("score_oportunidad", "max"),
                min_proxima_liberacion=("min_hasta_liberacion", "min"),
                latitud=("latitud", "mean"),
                longitud=("longitud", "mean"),
            )
            .reset_index()
            .sort_values("mejor_score", ascending=False)
            .reset_index(drop=True)
        )
        return agg

    # ----- Listado de barrios ------------------------------------------- #

    def barrios_de(self, distrito: str) -> list[str]:
        """Devuelve los barrios del distrito presentes en el inventario."""
        df = self.parqs[self.parqs["distrito"] == distrito]
        return sorted(df["barrio"].dropna().unique().tolist())
