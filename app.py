"""SmartPark Madrid - aplicacion Streamlit.

Interfaz que simula la experiencia de un sistema de aparcamiento integrado
en el vehiculo del usuario. Toda la logica de modelado y simulacion vive
en el paquete `pipeline`; este modulo se limita a la presentacion (UI/UX).

Uso:
    streamlit run app.py

Justificacion: la separacion entre logica (pipeline/) y presentacion (app.py)
sigue el principio de capas y permite reutilizar el motor desde un notebook,
un script batch o cualquier otra interfaz futura sin tocar la app.
"""

from __future__ import annotations

import time

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from streamlit_folium import st_folium

from pipeline import (
    ParkingScorer,
    ReservationSimulator,
    BarrioProfiler,
    validar_holdout,
    plazas_sincronizadas,
    distancia_haversine_km,
    cargar_parquimetros,
    cargar_historico,
    cargar_tickets_fecha,
    fechas_disponibles,
    load_config,
)

# --------------------------------------------------------------------------- #
# Configuracion de pagina
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="SmartPark Madrid - Prototipo TFG",
    page_icon="🅿️",
    layout="wide",
)

cfg = load_config()

DIAS_SEMANA_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes",
    "Saturday": "Sábado", "Sunday": "Domingo",
}


# --------------------------------------------------------------------------- #
# Capa de cache: lecturas pesadas se calculan una vez por sesion
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Catalogando fechas disponibles...")
def _fechas_disponibles_cached() -> list[pd.Timestamp]:
    return fechas_disponibles()


def _folium_html(folium_map: folium.Map) -> str:
    """Renderiza un mapa folium a HTML embebible (envuelve _repr_html_)."""
    return folium_map._repr_html_()  # type: ignore[attr-defined]  # noqa: SLF001


@st.cache_resource(show_spinner="Cargando inventario de parquimetros...")
def _parquimetros_cached() -> pd.DataFrame:
    return cargar_parquimetros()


@st.cache_resource(show_spinner="Entrenando modelo de scoring...")
def _scorer_cached() -> ParkingScorer:
    return ParkingScorer().fit(cargar_historico())


@st.cache_resource(show_spinner="Calculando perfiles de barrio...")
def _profiler_cached() -> BarrioProfiler:
    return BarrioProfiler(cargar_historico(), _parquimetros_cached())


@st.cache_data(show_spinner="Validando modelo (holdout)...")
def _validacion_cached(fecha: str):
    return validar_holdout(cargar_historico(), pd.Timestamp(fecha))


@st.cache_data(show_spinner="Cargando tickets del dia seleccionado...")
def _tickets_dia_cached(fecha: str) -> pd.DataFrame:
    return cargar_tickets_fecha(pd.Timestamp(fecha))


def _build_simulator(fecha: str) -> ReservationSimulator:
    return ReservationSimulator(
        _tickets_dia_cached(fecha),
        _parquimetros_cached(),
    )


_CANDIDATOS_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _candidatos_en_momento(
    sim: ReservationSimulator,
    momento_ts: pd.Timestamp,
    distrito: str | None = None,
) -> pd.DataFrame:
    """Evita recalcular candidatos idénticos en el mismo render."""
    key = (momento_ts.isoformat(), distrito or "")
    if key not in _CANDIDATOS_CACHE:
        _CANDIDATOS_CACHE[key] = sim.candidatos(momento_ts, distrito=distrito)
    return _CANDIDATOS_CACHE[key].copy()


# --------------------------------------------------------------------------- #
# Sidebar: parametros del escenario
# --------------------------------------------------------------------------- #

st.sidebar.title("SmartPark Madrid")
st.sidebar.caption("Sistema de aparcamiento integrado · Prototipo TFG · Noa Moliner Montoya · UAX")

fechas = _fechas_disponibles_cached()
fecha_min, fecha_max = fechas[0].date(), fechas[-1].date()

fecha_sel = st.sidebar.date_input(
    "Día simulado",
    value=pd.Timestamp(cfg.raw["app"]["default_date"]).date(),
    min_value=fecha_min,
    max_value=fecha_max,
    help=f"Cualquier día entre {fecha_min} y {fecha_max} (1T 2025).",
)
fecha_iso = pd.Timestamp(fecha_sel).strftime("%Y-%m-%d")
dia_semana_es = DIAS_SEMANA_ES.get(
    pd.Timestamp(fecha_sel).day_name(), pd.Timestamp(fecha_sel).day_name()
)
st.sidebar.caption(dia_semana_es)

distrito_destino = st.sidebar.selectbox(
    "Distrito destino", cfg.app_options["distritos"]
)

# Selector opcional de barrio dentro del distrito (filtra todas las vistas)
_parqs_sidebar = _parquimetros_cached()
_barrios_distrito = sorted(
    _parqs_sidebar.loc[
        _parqs_sidebar["distrito"] == distrito_destino, "barrio"
    ].dropna().unique().tolist()
)
_opciones_barrio = ["(Todos los barrios)"] + _barrios_distrito
barrio_destino_sel = st.sidebar.selectbox(
    "Barrio (opcional)", _opciones_barrio,
    help="Si eliges un barrio concreto, todas las vistas se filtran por ese barrio.",
)
barrio_filtro: str | None = (
    None if barrio_destino_sel == "(Todos los barrios)" else barrio_destino_sel
)

tipo_zona = st.sidebar.selectbox(
    "Tipo de zona",
    ["Cualquiera"] + list(cfg.app_options["tipos_zona"]),
    help="Elige 'Cualquiera' si te valen plazas de cualquier tipología (azul, verde, etc.).",
)
distintivo = st.sidebar.selectbox(
    "Distintivo ambiental DGT", cfg.app_options["distintivos"]
)

hour_min, hour_max = cfg.raw["app"]["hour_range"]
hora = st.sidebar.slider(
    "Hora simulada",
    min_value=hour_min,
    max_value=hour_max,
    value=cfg.raw["app"]["default_hour"],
)
minuto = st.sidebar.slider(
    "Minuto", min_value=0, max_value=55, step=5,
    value=cfg.raw["app"]["default_minute"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "El modelo se ha entrenado con todo el histórico del 1T 2025 "
    "filtrado al interior de la M-30."
)

modo_dev = st.sidebar.checkbox(
    "Mostrar pestañas técnicas",
    value=False,
    help=(
        "Añade las pestañas 'Validación' (correlación Spearman vs realidad) y "
        "'Modelo' (desglose del cálculo de la probabilidad). "
        "Pensadas para la defensa del TFG; no son necesarias para usar la app."
    ),
)


# --------------------------------------------------------------------------- #
# Carga de modelos
# --------------------------------------------------------------------------- #

scorer = _scorer_cached()
profiler = _profiler_cached()
simulator = _build_simulator(fecha_iso)

if len(simulator.tickets) == 0:
    st.warning(
        f"No hay tickets registrados el {fecha_sel.strftime('%d/%m/%Y')} "
        f"({dia_semana_es}). En Madrid el sistema SER **no opera los "
        f"domingos ni festivos**, así que no se emiten tickets. "
        f"Selecciona un día laborable o un sábado."
    )
    st.stop()

instante = pd.Timestamp(f"{fecha_iso} {hora:02d}:{minuto:02d}:00")

res = scorer.predict(
    distrito=distrito_destino, hora=hora,
    tipo_zona=tipo_zona, distintivo=distintivo,
)


# --------------------------------------------------------------------------- #
# Cabecera: contexto del escenario y KPIs
# --------------------------------------------------------------------------- #

# Barra superior estilo infotainment del vehículo
st_html(
    f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0; padding: 14px 22px; border-radius: 10px;
                font-family: 'Segoe UI', sans-serif;
                display: flex; align-items: center; justify-content: space-between;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);">
      <div style="display:flex; align-items:center; gap:18px;">
        <div style="font-size: 11px; letter-spacing: 2px; color:#64748b;
                    text-transform: uppercase;">SmartPark Madrid · Cuadro de mando</div>
      </div>
      <div style="display:flex; align-items:center; gap:28px;
                  font-family: 'Consolas', 'Courier New', monospace;">
        <div style="font-size: 13px; color:#94a3b8;">{dia_semana_es.upper()}</div>
        <div style="font-size: 13px; color:#94a3b8;">{fecha_sel.strftime('%d/%m/%Y')}</div>
        <div style="font-size: 22px; color:#f1f5f9; font-weight: 600;">
          {hora:02d}:{minuto:02d}
        </div>
        <div style="font-size: 13px; color:#94a3b8;
                    text-transform: uppercase; letter-spacing: 1px;">
          {distrito_destino}
        </div>
      </div>
    </div>
    """,
    height=70,
)

estado = simulator.estado_instante(instante)
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Plazas activas", f"{estado['tickets_activos']:,}")
col_b.metric(
    f"Liberándose en < {cfg.liberation_window_min} min",
    f"{estado['candidatos_liberacion']:,}",
)
col_c.metric("📍 Parquímetros activos", f"{estado['parquimetros_unicos_activos']:,}")
col_d.metric(
    "Probabilidad estimada",
    f"{res.probabilidad:.0f}%",
    delta=res.nivel,
    delta_color="off",
)

# --------------------------------------------------------------------------- #
# Tarjeta de etiqueta DGT (privilegios y restricciones del usuario)
# --------------------------------------------------------------------------- #

_aviso = cfg.raw["app"].get("distintivo_avisos", {}).get(distintivo)
if _aviso:
    _palette = {
        "success": {"bg": "#dcfce7", "border": "#16a34a", "title": "#15803d"},
        "info":    {"bg": "#dbeafe", "border": "#2563eb", "title": "#1d4ed8"},
        "warning": {"bg": "#fef3c7", "border": "#d97706", "title": "#b45309"},
        "error":   {"bg": "#fee2e2", "border": "#dc2626", "title": "#b91c1c"},
    }.get(_aviso.get("type", "info"), {"bg": "#f3f4f6", "border": "#6b7280", "title": "#374151"})

    tarjeta_dgt = f"""
    <div style="background: {_palette['bg']};
                border-left: 6px solid {_palette['border']};
                border-radius: 10px;
                padding: 18px 24px;
                margin: 18px 0 8px 0;
                font-family: 'Segoe UI', sans-serif;
                box-shadow: 0 1px 4px rgba(0,0,0,0.05);">
      <div style="display:flex; align-items:center; gap:14px; margin-bottom:8px;">
        <div style="font-size: 32px; line-height:1;">{_aviso.get('icon', 'ℹ️')}</div>
        <div>
          <div style="color:{_palette['title']}; font-size:11px;
                      font-weight:700; letter-spacing:1px;
                      text-transform:uppercase; margin-bottom:2px;">
            Tu vehículo · etiqueta {distintivo}
          </div>
          <div style="font-size:18px; font-weight:600; color:#111827;">
            {_aviso['title']}
          </div>
        </div>
      </div>
      <div style="color:#374151; font-size:14px; line-height:1.55;
                  margin-left: 46px;">
        {_aviso['body']}
      </div>
    </div>
    """
    st_html(tarjeta_dgt, height=170)

st.markdown("---")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

_tab_labels = [
    "Navegación",
    "Recomendación",
    "Mapa en vivo",
    "Calor por calle",
    "Reservar plaza",
]
if modo_dev:
    _tab_labels += ["Validación", "Modelo"]

_tabs = st.tabs(_tab_labels)
tab_nav, tab_recom, tab_mapa, tab_calor, tab_reserva = _tabs[:5]
if modo_dev:
    tab_validacion, tab_modelo = _tabs[5], _tabs[6]
else:
    tab_validacion = tab_modelo = None


def _color_nivel_prob(nivel: str) -> str:
    return {
        "Alta probabilidad": "background-color: #c6f6d5",
        "Probabilidad media": "background-color: #feebc8",
        "Baja probabilidad": "background-color: #fed7d7",
    }.get(nivel, "")


# ----- Tab 1: Recomendacion ------------------------------------------------ #

with tab_recom:
    st.subheader("Ranking de distritos para tu escenario")
    ranking = scorer.ranking(
        ranking_dim="distrito",
        hora=hora, tipo_zona=tipo_zona, distintivo=distintivo,
    )
    ranking["Destino"] = ranking["distrito"].apply(
        lambda d: "▶ seleccionado" if d == distrito_destino else ""
    )
    ranking_display = ranking.rename(columns={
        "distrito": "Distrito",
        "probabilidad": "Probabilidad",
        "nivel": "Nivel",
        "score_total": "Puntuación",
    })
    st.dataframe(
        ranking_display.style.map(_color_nivel_prob, subset=["Nivel"]).format(
            {"Probabilidad": "{:.1f}%"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "El modelo combina cuatro variables (distrito, hora, tipo de zona, "
        "distintivo) mediante scoring lineal saturado en [10%, 90%]. "
        "Ver capítulo 5.7 de la memoria."
    )


# ----- Tab 2: Mapa en vivo ------------------------------------------------- #

with tab_mapa:
    _ambito_label = (
        f"{distrito_destino} · {barrio_filtro}" if barrio_filtro else distrito_destino
    )
    st.subheader(f"Plazas candidatas a liberación · {_ambito_label}")

    # ---- Tarjeta de perfil del barrio (si hay barrio seleccionado) ----- #
    if barrio_filtro:
        perfil = profiler.perfil(distrito_destino, barrio_filtro)
        color_tipologia = {
            "COMERCIAL":   "#16a34a",
            "OFICINAS":    "#dc2626",
            "RESIDENCIAL": "#f59e0b",
            "MIXTO":       "#6366f1",
            "SIN DATOS":   "#6b7280",
        }.get(perfil.tipologia, "#6b7280")

        tarjeta = f"""
        <div style="background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
                    border-left: 5px solid {color_tipologia};
                    border-radius: 10px;
                    padding: 18px 22px;
                    margin: 10px 0 18px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                    font-family: 'Segoe UI', sans-serif;">
          <div style="display:flex; align-items:center; gap:10px;
                      margin-bottom: 8px;">
            <span style="background:{color_tipologia}; color:white;
                         padding:4px 12px; border-radius:20px;
                         font-size:12px; font-weight:600;
                         letter-spacing:0.5px;">
              {perfil.tipologia}
            </span>
            <span style="font-size: 17px; font-weight: 600; color:#111827;">
              {perfil.barrio}
            </span>
          </div>
          <div style="color:#374151; font-size:14px; line-height:1.5;
                      margin-bottom: 14px;">
            {perfil.descripcion}
          </div>
          <div style="display:flex; gap:24px; flex-wrap:wrap;
                      color:#6b7280; font-size:12px;">
            <div><b style="color:#111827">{perfil.num_parquimetros}</b> parquímetros</div>
            <div><b style="color:#111827">{perfil.num_tickets:,}</b> tickets (1T 2025)</div>
            <div>AZUL <b style="color:#111827">{perfil.pct_azul:.0f}%</b></div>
            <div>VERDE <b style="color:#111827">{perfil.pct_verde:.0f}%</b></div>
            <div>Duración media <b style="color:#111827">{perfil.duracion_media_min:.0f} min</b></div>
            <div>Pico <b style="color:#111827">{perfil.pico_horario}:00 h</b></div>
            <div>Importe medio <b style="color:#111827">{perfil.importe_medio:.2f}€</b></div>
          </div>
          <div style="color:#9ca3af; font-size:11px; margin-top:10px;
                      font-style:italic;">
            Perfil derivado automáticamente del histórico 1T 2025.
          </div>
        </div>
        """
        st_html(tarjeta, height=240)

    cand = _candidatos_en_momento(simulator, instante, distrito=distrito_destino)
    if barrio_filtro:
        cand = cand[cand["barrio"] == barrio_filtro]

    if len(cand) == 0:
        st.info(
            f"No hay candidatos a liberación en **{_ambito_label}** "
            f"en este instante exacto ({instante.strftime('%H:%M')}). "
            f"Prueba a desplazar el slider de hora unos minutos."
        )
    else:
        st.caption(
            f"{len(cand)} parquímetros en {_ambito_label} cuyo ticket "
            f"expira en menos de {cfg.liberation_window_min} min."
        )
        cand_geo = cand.dropna(subset=["latitud", "longitud"])
        if len(cand_geo) == 0:
            st.warning("Ninguno de los candidatos tiene coordenadas asociadas.")
        else:
            centro = [cand_geo["latitud"].mean(), cand_geo["longitud"].mean()]
            zoom_base = cfg.app_map["default_zoom"] + (1 if barrio_filtro else 0)
            mapa = folium.Map(
                location=centro,
                zoom_start=zoom_base,
                tiles=cfg.app_map["tiles"],
            )

            colors = cfg.app_map["color_scale"]
            max_markers = cfg.app_map["max_markers_reservation"]
            for _, row in cand_geo.head(max_markers).iterrows():
                s = row["score_oportunidad"]
                color = (
                    colors["high"] if s >= 0.7
                    else colors["medium"] if s >= 0.4
                    else colors["low"]
                )
                popup = (
                    f"<b>{row['calle']}</b><br>"
                    f"Libera en: {row['min_hasta_liberacion']:.1f} min<br>"
                    f"Prob. no renovar: {row['prob_no_renovacion']*100:.0f}%<br>"
                    f"Score: {s:.2f}<br>"
                    f"Zona: {row['tipo_zona']}"
                )
                folium.CircleMarker(
                    location=[row["latitud"], row["longitud"]],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=250),
                ).add_to(mapa)

            st_html(_folium_html(mapa), height=500)

            # ---- Ranking de calles cuando hay barrio seleccionado ----- #
            if barrio_filtro:
                calles_rank = simulator.calles_disponibles(
                    instante, distrito=distrito_destino, barrio=barrio_filtro,
                )
                if not calles_rank.empty:
                    st.markdown("##### Ranking de calles del barrio")
                    _calles_display = calles_rank.rename(columns={
                        "score_medio": "Puntuación media",
                        "mejor_score": "Mejor puntuación",
                        "min_proxima_liberacion": "Min. próx. liberación",
                        "latitud": "Latitud",
                        "longitud": "Longitud",
                    })
                    st.dataframe(
                        _calles_display.style.format({
                            "Puntuación media": "{:.2f}",
                            "Mejor puntuación": "{:.2f}",
                            "Min. próx. liberación": "{:.1f} min",
                            "Latitud": "{:.5f}",
                            "Longitud": "{:.5f}",
                        }),
                        hide_index=True,
                        width="stretch",
                    )

            with st.expander("Ver listado de candidatos"):
                _cand_display = cand[
                    ["calle", "numero_finca", "min_hasta_liberacion",
                     "prob_no_renovacion", "score_oportunidad", "tipo_zona"]
                ].head(max_markers).rename(columns={
                    "calle": "Calle",
                    "numero_finca": "Nº",
                    "min_hasta_liberacion": "Min. hasta liberarse",
                    "prob_no_renovacion": "Fiabilidad",
                    "score_oportunidad": "Oportunidad",
                    "tipo_zona": "Tipo de zona",
                })
                st.dataframe(
                    _cand_display.style.format({
                        "Min. hasta liberarse": "{:.1f} min",
                        "Fiabilidad": "{:.0%}",
                        "Oportunidad": "{:.2f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )


# ----- Tab 3: Calor por calle ---------------------------------------------- #

with tab_calor:
    _ambito_calor = (
        f"{distrito_destino} · {barrio_filtro}" if barrio_filtro else distrito_destino
    )
    st.subheader(f"Presión por calle · {_ambito_calor} · {hora}:00 h")
    st.caption(
        "Cada parquímetro se representa como un punto coloreado según el "
        "nivel de presión (número de tickets) que registra **su calle** "
        "en esta franja horaria."
    )
    presion = simulator.presion_por_calle(hora=hora, distrito=distrito_destino)

    # Filtro por barrio: cruzamos con el inventario para conocer el barrio de cada parq
    if barrio_filtro and not presion.empty:
        _parq_barrio = _parqs_sidebar[["matricula", "barrio"]].rename(
            columns={"matricula": "matricula_parquimetro"}
        )
        presion = presion.merge(_parq_barrio, on="matricula_parquimetro", how="left")
        presion = presion[presion["barrio"] == barrio_filtro]

    if presion.empty:
        st.info("No hay datos de tickets en este distrito y franja horaria.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Calles analizadas", f"{presion['calle'].nunique()}")
        c2.metric("Parquímetros", f"{len(presion)}")
        c3.metric(
            "Tickets totales (esa hora)", f"{int(presion['num_tickets'].sum()):,}"
        )

        centro = [presion["latitud"].mean(), presion["longitud"].mean()]
        mapa = folium.Map(
            location=centro,
            zoom_start=cfg.app_map["default_zoom"],
            tiles=cfg.app_map["tiles"],
        )
        colors = cfg.app_map["color_scale"]
        nivel_color = {
            "Alta": colors["low"],
            "Media": colors["medium"],
            "Baja": colors["high"],
        }

        for _, row in presion.iterrows():
            color = nivel_color.get(row["nivel"], "#718096")
            popup_html = (
                f"<b>{row['calle']}</b><br>"
                f"Tickets a las {hora}h: <b>{int(row['num_tickets'])}</b><br>"
                f"Nivel: <b>{row['nivel']}</b>"
            )
            folium.CircleMarker(
                location=[row["latitud"], row["longitud"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1,
                popup=folium.Popup(popup_html, max_width=240),
            ).add_to(mapa)

        leyenda = f"""
        <div style="position: fixed; bottom: 30px; left: 30px;
                    background: rgba(255,255,255,0.95); padding: 12px 16px;
                    border: none; border-radius: 8px;
                    font-size: 13px; font-family: 'Segoe UI', sans-serif;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
                    line-height: 1.6;">
          <div style="font-weight:600; margin-bottom:6px;">Presión por calle</div>
          <div><span style="display:inline-block;width:12px;height:12px;
                background:{colors['low']};border-radius:50%;
                vertical-align:middle;margin-right:6px;"></span>Alta</div>
          <div><span style="display:inline-block;width:12px;height:12px;
                background:{colors['medium']};border-radius:50%;
                vertical-align:middle;margin-right:6px;"></span>Media</div>
          <div><span style="display:inline-block;width:12px;height:12px;
                background:{colors['high']};border-radius:50%;
                vertical-align:middle;margin-right:6px;"></span>Baja</div>
        </div>
        """
        mapa.get_root().html.add_child(folium.Element(leyenda))
        st_html(_folium_html(mapa), height=520)

        with st.expander("Ranking de calles más saturadas"):
            top = (
                presion.drop_duplicates("calle")[["calle", "num_tickets", "nivel"]]
                .sort_values("num_tickets", ascending=False)
                .head(20)
            )
            st.dataframe(top, hide_index=True, width="stretch")


# ----- Tab 5: Navegación con ETA ------------------------------------------ #

with tab_nav:
    # =========================================================== #
    #  WIZARD DE NAVEGACION (2 pantallas)                          #
    #  - Pantalla 1: seleccion de origen + destino + CONFIRMAR     #
    #  - Pantalla 2: dashboard del coche + reserva + mapa          #
    # =========================================================== #

    # ---- ESTILOS GLOBALES (botones del coche y dashboard embebido) ----- #
    st.markdown(
        """
        <style>
        /* Boton primario destacado (CTA verde grande) */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
            color: white;
            font-size: 18px;
            font-weight: 700;
            padding: 16px 24px;
            border: none;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35);
            transition: transform 0.1s ease, box-shadow 0.1s ease;
            width: 100%;
            letter-spacing: 0.5px;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(22, 163, 74, 0.45);
        }
        /* Boton secundario (volver / cambiar destino) */
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: #f1f5f9;
            color: #334155;
            font-weight: 500;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 8px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Estado del wizard
    if "nav_step" not in st.session_state:
        st.session_state.nav_step = 1
    if "nav_data" not in st.session_state:
        st.session_state.nav_data = {}

    parqs_df = _parquimetros_cached()

    # =========================================================== #
    #  PANTALLA 1 — SELECCION DE ORIGEN Y DESTINO                  #
    # =========================================================== #
    if st.session_state.nav_step == 1:
        # ---- Pantalla del coche en estado IDLE (cabecera HTML) -------- #
        hora_disp_idle = f"{hora:02d}:{minuto:02d}"
        idle_html = f"""
        <!DOCTYPE html>
        <html><head><style>
          body {{ margin: 0; padding: 6px; font-family: 'Segoe UI', sans-serif; }}
          .car-bezel {{
              background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
              border-radius: 28px;
              padding: 18px;
              box-shadow:
                  inset 0 1px 0 rgba(255,255,255,0.08),
                  inset 0 -2px 4px rgba(0,0,0,0.6),
                  0 14px 40px rgba(0,0,0,0.55);
              border: 1px solid #2a2a2a;
              position: relative;
          }}
          .car-bezel::before {{
              content: "● ● ●";
              position: absolute;
              top: 8px; left: 50%;
              transform: translateX(-50%);
              color: #2a2a2a;
              font-size: 9px;
              letter-spacing: 5px;
          }}
          .car-screen {{
              background: radial-gradient(circle at top right, #1e293b 0%, #0b1020 70%);
              border-radius: 16px;
              padding: 28px 32px;
              color: #e5e7eb;
              box-shadow:
                  inset 0 0 0 1px rgba(255,255,255,0.04),
                  inset 0 0 30px rgba(0,0,0,0.4);
              text-align: center;
          }}
          .header-row {{
              display: flex;
              justify-content: space-between;
              align-items: center;
              border-bottom: 1px solid rgba(148,163,184,0.18);
              padding-bottom: 12px;
              margin-bottom: 22px;
          }}
          .header-title {{
              font-size: 11px;
              letter-spacing: 2px;
              color: #94a3b8;
              font-weight: 600;
          }}
          .header-clock {{
              font-size: 14px;
              color: #e5e7eb;
              font-variant-numeric: tabular-nums;
              font-weight: 500;
          }}
          .greeting {{
              font-size: 32px;
              font-weight: 700;
              color: #ffffff;
              letter-spacing: -0.5px;
              margin-bottom: 8px;
          }}
          .subtitle {{
              font-size: 14px;
              color: #94a3b8;
              max-width: 480px;
              margin: 0 auto;
              line-height: 1.5;
          }}
          .led {{
              display: inline-block;
              width: 8px; height: 8px;
              border-radius: 50%;
              background: #f59e0b;
              box-shadow: 0 0 8px #f59e0b;
              margin-right: 8px;
              animation: pulse 1.6s infinite;
          }}
          .icon-stack {{
              display: flex;
              justify-content: center;
              gap: 32px;
              margin: 18px 0 8px 0;
              font-size: 28px;
              opacity: 0.85;
          }}
          .icon-stack .arrow {{
              color: #64748b;
              font-size: 22px;
              align-self: center;
          }}
          @keyframes pulse {{
              0%, 100% {{ opacity: 1; }}
              50% {{ opacity: 0.4; }}
          }}
        </style></head>
        <body>
          <div class="car-bezel">
            <div class="car-screen">
              <div class="header-row">
                <div class="header-title">
                  <span class="led"></span>SMARTPARK · ESPERANDO DESTINO
                </div>
                <div class="header-clock">{hora_disp_idle}</div>
              </div>
              <div class="greeting">¿A dónde vamos hoy?</div>
              <div class="subtitle">
                Indica tu origen y tu destino. Calcularé el tiempo de viaje
                y te propondré plazas que se liberen justo cuando llegues.
              </div>
              <div class="icon-stack">
                <span title="Origen">📍</span>
                <span class="arrow">→</span>
                <span title="Destino">◉</span>
              </div>
            </div>
          </div>
        </body></html>
        """
        st_html(idle_html, height=320)

        calles_origen = sorted(parqs_df["calle"].dropna().unique().tolist())

        # ---- Origen ---------------------------------------------------- #
        st.markdown("##### 📍 Origen")
        calle_origen = st.selectbox(
            "Tu ubicación actual",
            options=calles_origen,
            index=calles_origen.index("VELAZQUEZ, CALLE, DE")
            if "VELAZQUEZ, CALLE, DE" in calles_origen else 0,
            help="Empieza a escribir para autocompletar.",
        )

        # ---- Destino: 3 modos ----------------------------------------- #
        st.markdown("##### Destino")
        modo_destino = st.radio(
            "¿Cómo quieres definir tu destino?",
            options=[
                "Buscar por calle (escribe el nombre de la calle)",
                f"Aparcar en cualquier punto de **{distrito_destino}**",
            ],
            horizontal=False,
            help=(
                "• *Buscar por calle*: escribe la calle y detectamos automáticamente "
                "el barrio y distrito; filtramos las plazas por distancia caminable. "
                "• *Aparcar en…*: te valen todas las plazas del distrito."
            ),
        )

        calle_destino = None
        distrito_efectivo = distrito_destino

        if modo_destino.startswith("Buscar"):
            calles_m30 = sorted(
                parqs_df.loc[
                    parqs_df["distrito"].isin(cfg.distritos_m30), "calle"
                ].dropna().unique().tolist()
            )
            calle_destino = st.selectbox(
                "Escribe la calle de destino",
                options=calles_m30,
                help="Se buscan calles en los 8 distritos del centro.",
            )
            destino_row = parqs_df[parqs_df["calle"] == calle_destino].iloc[0]
            distrito_efectivo = str(destino_row["distrito"])
            barrio_detectado = str(destino_row.get("barrio", "—"))
            st.info(
                f"Esa calle está en el barrio **{barrio_detectado}**, "
                f"distrito **{distrito_efectivo}**."
            )
        # ---- Boton de confirmar -------------------------------------- #
        st.markdown("---")
        col_confirm_l, col_confirm_c, col_confirm_r = st.columns([1, 2, 1])
        with col_confirm_c:
            if st.button(
                "CONFIRMAR DESTINO Y BUSCAR PLAZAS",
                type="primary",
                key="btn_confirmar_destino",
                use_container_width=True,
            ):
                st.session_state.nav_data = {
                    "calle_origen": calle_origen,
                    "calle_destino": calle_destino,
                    "distrito_efectivo": distrito_efectivo,
                    "modo_destino": modo_destino,
                }
                st.session_state.nav_step = 2
                st.rerun()

    # =========================================================== #
    #  PANTALLA 2 — DASHBOARD + RESERVA + MAPA                     #
    # =========================================================== #
    else:
        nav = st.session_state.nav_data
        calle_origen = nav["calle_origen"]
        calle_destino = nav["calle_destino"]
        distrito_efectivo = nav["distrito_efectivo"]

        # Boton volver
        col_back_l, col_back_r = st.columns([1, 6])
        with col_back_l:
            if st.button("← Cambiar", key="btn_back_nav", type="secondary"):
                st.session_state.nav_step = 1
                st.rerun()
        with col_back_r:
            st.markdown(
                f"**{calle_origen.split(',')[0]}** &nbsp;→&nbsp; "
                f"**{calle_destino.split(',')[0] if calle_destino else distrito_efectivo}**"
            )

        origen_row = parqs_df[parqs_df["calle"] == calle_origen].iloc[0]
        origen_lat = float(origen_row["latitud"])
        origen_lon = float(origen_row["longitud"])

        destino_lat = destino_lon = None
        if calle_destino is not None:
            destino_row = parqs_df[parqs_df["calle"] == calle_destino].iloc[0]
            destino_lat = float(destino_row["latitud"])
            destino_lon = float(destino_row["longitud"])

        # Aviso ZBEDEP
        if distrito_efectivo == "CENTRO" and distintivo in ("B", "C"):
            st.error(
                f"**Restricción ZBEDEP — Distrito Centro**\n\n"
                f"Tu vehículo (etiqueta **{distintivo}**) **no puede aparcar en "
                f"calle SER** dentro del Distrito Centro salvo que seas residente. "
                f"Se recomienda dirigirse a un **parking público rotacional** "
                f"(p.ej. Plaza Mayor, Plaza del Carmen, Sevilla, Descalzas)."
            )

        # Candidatos
        cand_dest = _candidatos_en_momento(simulator, instante, distrito=distrito_efectivo)

        # Filtro por radio si hay calle concreta
        if (
            calle_destino is not None
            and not cand_dest.empty
            and not nav["modo_destino"].startswith("Aparcar")
        ):
            radio_m = cfg.raw["routing"].get("destino_calle_radio_m", 400)
            radio_km = radio_m / 1000.0
            cand_dest = cand_dest.copy()
            cand_dest["distancia_destino_km"] = cand_dest.apply(
                lambda r: distancia_haversine_km(
                    destino_lat, destino_lon, r["latitud"], r["longitud"],
                ),
                axis=1,
            )
            cand_dest = cand_dest[cand_dest["distancia_destino_km"] <= radio_km]

        sync = plazas_sincronizadas(cand_dest, origen_lat, origen_lon)

        if sync.empty:
            st.warning(
                "Ninguna plaza candidata se sincroniza con tu llegada. "
                "Prueba a cambiar la hora simulada o el destino."
            )
        else:
            mejor = sync.iloc[0]
            diff = mejor["diff_llegada_min"]
            if abs(diff) < 0.5:
                sync_msg = "● Sincronía perfecta"
                sync_color = "#10b981"
            elif diff > 0:
                sync_msg = f"▼ Libera {diff:.1f} min después de tu llegada"
                sync_color = "#f59e0b"
            else:
                sync_msg = f"▲ Libera {abs(diff):.1f} min antes de tu llegada"
                sync_color = "#fb923c"
            # ---- DASHBOARD: HTML auto-contenido (CSS + HTML juntos) ---- #
            hora_disp = f"{hora:02d}:{minuto:02d}"
            calle_d = mejor["calle"].split(",")[0]
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
              body {{
                  margin: 0;
                  padding: 8px;
                  font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
                  background: transparent;
              }}
              .car-bezel {{
                  background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%);
                  border-radius: 28px;
                  padding: 22px;
                  box-shadow:
                      inset 0 1px 0 rgba(255,255,255,0.08),
                      inset 0 -2px 4px rgba(0,0,0,0.6),
                      0 14px 40px rgba(0,0,0,0.55);
                  border: 1px solid #2a2a2a;
                  position: relative;
              }}
              .car-bezel::before {{
                  content: "● ● ●";
                  position: absolute;
                  top: 8px;
                  left: 50%;
                  transform: translateX(-50%);
                  color: #2a2a2a;
                  font-size: 9px;
                  letter-spacing: 5px;
              }}
              .car-screen {{
                  background: radial-gradient(circle at top right, #1e293b 0%, #0b1020 70%);
                  border-radius: 16px;
                  padding: 26px 30px;
                  color: #e5e7eb;
                  box-shadow:
                      inset 0 0 0 1px rgba(255,255,255,0.04),
                      inset 0 0 30px rgba(0,0,0,0.4);
              }}
              .header-row {{
                  display: flex;
                  justify-content: space-between;
                  align-items: center;
                  border-bottom: 1px solid rgba(148,163,184,0.18);
                  padding-bottom: 12px;
                  margin-bottom: 18px;
              }}
              .header-title {{
                  font-size: 11px;
                  letter-spacing: 2px;
                  color: #94a3b8;
                  font-weight: 600;
              }}
              .header-clock {{
                  font-size: 14px;
                  color: #e5e7eb;
                  font-variant-numeric: tabular-nums;
                  font-weight: 500;
              }}
              .destination {{
                  font-size: 30px;
                  font-weight: 700;
                  color: #ffffff;
                  letter-spacing: -0.5px;
                  margin-bottom: 4px;
                  line-height: 1.1;
              }}
              .destination .sub {{
                  display: block;
                  font-size: 13px;
                  color: #94a3b8;
                  font-weight: 400;
                  letter-spacing: 0.5px;
                  margin-top: 6px;
                  text-transform: uppercase;
              }}
              .gauges {{
                  display: grid;
                  grid-template-columns: repeat(3, 1fr);
                  gap: 14px;
                  margin: 22px 0 8px 0;
              }}
              .gauge {{
                  background: rgba(148,163,184,0.08);
                  border: 1px solid rgba(148,163,184,0.15);
                  border-radius: 14px;
                  padding: 16px 14px;
                  text-align: center;
              }}
              .gauge .label {{
                  font-size: 10px;
                  letter-spacing: 1.5px;
                  color: #94a3b8;
                  text-transform: uppercase;
                  margin-bottom: 8px;
              }}
              .gauge .value {{
                  font-size: 30px;
                  font-weight: 700;
                  color: #ffffff;
                  font-variant-numeric: tabular-nums;
              }}
              .gauge .unit {{
                  font-size: 14px;
                  color: #94a3b8;
                  font-weight: 500;
                  margin-left: 4px;
              }}
              .sync-bar {{
                  margin-top: 18px;
                  padding: 12px 16px;
                  border-radius: 10px;
                  background: rgba(255,255,255,0.04);
                  border-left: 4px solid {sync_color};
                  font-size: 13px;
                  color: #e2e8f0;
              }}
              .footer {{
                  margin-top: 16px;
                  display: flex;
                  justify-content: space-between;
                  font-size: 11px;
                  color: #64748b;
                  letter-spacing: 0.5px;
              }}
              .led {{
                  display: inline-block;
                  width: 8px;
                  height: 8px;
                  border-radius: 50%;
                  background: #10b981;
                  box-shadow: 0 0 8px #10b981;
                  margin-right: 8px;
                  animation: pulse 1.6s infinite;
              }}
              @keyframes pulse {{
                  0%, 100% {{ opacity: 1; }}
                  50% {{ opacity: 0.4; }}
              }}
            </style>
            </head>
            <body>
              <div class="car-bezel">
                <div class="car-screen">
                  <div class="header-row">
                    <div class="header-title">
                      <span class="led"></span>SMARTPARK · NAVEGACIÓN ACTIVA
                    </div>
                    <div class="header-clock">{hora_disp}</div>
                  </div>
                  <div class="destination">
                    {calle_d}
                    <span class="sub">{distrito_efectivo} · plaza recomendada</span>
                  </div>
                  <div class="gauges">
                    <div class="gauge">
                      <div class="label">Distancia</div>
                      <div class="value">{mejor['distancia_km']:.1f}<span class="unit">km</span></div>
                    </div>
                    <div class="gauge">
                      <div class="label">Llegada en</div>
                      <div class="value">{mejor['eta_min']:.0f}<span class="unit">min</span></div>
                    </div>
                    <div class="gauge">
                      <div class="label">Libera en</div>
                      <div class="value">{mejor['min_hasta_liberacion']:.0f}<span class="unit">min</span></div>
                    </div>
                  </div>
                  <div class="sync-bar">
                    {sync_msg} · Probabilidad de no renovación:
                    <b style="color:#fff">{mejor['prob_no_renovacion']*100:.0f}%</b>
                  </div>
                  <div class="footer">
                    <span>{len(sync)} plazas sincronizadas en el área</span>
                    <span>Velocidad media urbana: {cfg.raw['routing']['urban_speed_kmh']} km/h</span>
                  </div>
                </div>
              </div>
            </body>
            </html>
            """
            st_html(dashboard_html, height=470)

            # ---- BOTÓN DE RESERVA --------------------------------- #
            if "reserva_nav" not in st.session_state:
                st.session_state.reserva_nav = None

            col_btn_l, col_btn_c, col_btn_r = st.columns([1, 3, 1])
            with col_btn_c:
                reservar_clicked = st.button(
                    f"RESERVAR PLAZA Y ARRANCAR  ·  Llegada en {mejor['eta_min']:.0f} min",
                    type="primary",
                    key="btn_nav",
                    use_container_width=True,
                    help=(
                        f"Bloqueo garantizado de {cfg.reservation_hold_min} min."
                    ),
                )

            if reservar_clicked:
                st.session_state.reserva_nav = {
                    "plaza": mejor.to_dict(),
                    "origen": calle_origen,
                    "instante_reserva": instante,
                }
                st.success(
                    f"Plaza reservada en **{mejor['calle']}** · "
                    f"llegada estimada en **{mejor['eta_min']:.1f} min** · "
                    f"bloqueo durante **{cfg.reservation_hold_min} min**"
                )

            # ---- KPIs secundarios -------------------------------- #
            st.markdown("##### Resumen del área")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Distancia más corta", f"{sync['distancia_km'].min():.2f} km")
            c2.metric("ETA medio", f"{sync['eta_min'].mean():.1f} min")
            c3.metric("Plazas sincronizadas", f"{len(sync)}")
            c4.metric("Mejor sincronía", f"{mejor['score_sincronizado']:.2f}")

            # ---- MAPA -------------------------------------------- #
            st.markdown("##### Ruta y plazas en el mapa")
            centro = [
                (origen_lat + mejor["latitud"]) / 2,
                (origen_lon + mejor["longitud"]) / 2,
            ]
            mapa_n = folium.Map(
                location=centro,
                zoom_start=cfg.app_map["default_zoom"] - 1,
                tiles=cfg.app_map["tiles"],
            )
            if calle_destino is not None:
                folium.Marker(
                    [destino_lat, destino_lon],
                    popup=f"<b>Destino</b><br>{calle_destino}",
                    tooltip="Destino",
                    icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
                ).add_to(mapa_n)
                radio_m = cfg.raw["routing"].get("destino_calle_radio_m", 400)
                folium.Circle(
                    [destino_lat, destino_lon],
                    radius=radio_m,
                    color="#dc2626", weight=1, opacity=0.4,
                    fill=True, fill_color="#dc2626", fill_opacity=0.05,
                ).add_to(mapa_n)
            folium.Marker(
                [origen_lat, origen_lon],
                popup=f"<b>Origen</b><br>{calle_origen}",
                tooltip="Origen",
                icon=folium.Icon(color="blue", icon="play", prefix="fa"),
            ).add_to(mapa_n)
            folium.PolyLine(
                locations=[
                    [origen_lat, origen_lon],
                    [mejor["latitud"], mejor["longitud"]],
                ],
                color="#3b82f6", weight=4, opacity=0.7, dash_array="8,8",
            ).add_to(mapa_n)
            folium.Marker(
                [mejor["latitud"], mejor["longitud"]],
                popup=(
                    f"<b>{mejor['calle']}</b><br>"
                    f"Libera en {mejor['min_hasta_liberacion']:.1f} min<br>"
                    f"ETA: {mejor['eta_min']:.1f} min"
                ),
                tooltip="Mejor opción",
                icon=folium.Icon(color="green", icon="star", prefix="fa"),
            ).add_to(mapa_n)
            for _, row in sync.iloc[1:30].iterrows():
                folium.CircleMarker(
                    location=[row["latitud"], row["longitud"]],
                    radius=5,
                    color="#16a34a",
                    fill=True, fill_color="#16a34a", fill_opacity=0.7,
                    weight=1,
                    popup=(
                        f"<b>{row['calle']}</b><br>"
                        f"ETA {row['eta_min']:.1f} min · "
                        f"libera en {row['min_hasta_liberacion']:.1f} min"
                    ),
                ).add_to(mapa_n)
            st_html(_folium_html(mapa_n), height=480)

            with st.expander(f"Ver las {len(sync)} plazas sincronizadas"):
                _tabla = sync.copy()

                def _sincronia(diff: float) -> str:
                    if abs(diff) < 0.5:
                        return "Justo a tu llegada"
                    if diff > 0:
                        return f"Esperas {diff:.0f} min"
                    return f"Te adelantas {abs(diff):.0f} min"

                def _fiabilidad(p: float) -> str:
                    if p >= 0.70:
                        return "Alta"
                    if p >= 0.40:
                        return "Media"
                    return "Baja"

                _tabla["Sincronía"] = _tabla["diff_llegada_min"].apply(_sincronia)
                _tabla["Fiabilidad"] = _tabla["prob_no_renovacion"].apply(_fiabilidad)
                _tabla = _tabla.rename(columns={
                    "calle": "Calle",
                    "distancia_km": "Distancia (km)",
                    "eta_min": "Llegas en (min)",
                    "min_hasta_liberacion": "Plaza libre en (min)",
                })

                st.dataframe(
                    _tabla[[
                        "Calle", "Distancia (km)", "Llegas en (min)",
                        "Plaza libre en (min)", "Sincronía", "Fiabilidad",
                    ]].style.format({
                        "Distancia (km)": "{:.2f}",
                        "Llegas en (min)": "{:.1f}",
                        "Plaza libre en (min)": "{:.1f}",
                    }),
                    hide_index=True, width="stretch",
                )
                st.caption(
                    "**Sincronía**: cuánto desfase hay entre que llegas y "
                    "que la plaza se libera. **Fiabilidad**: probabilidad "
                    "histórica de que el conductor anterior no renueve el ticket."
                )


# ----- Tab 6: Reserva simulada --------------------------------------------- #

with tab_reserva:
    _ambito_res = (
        f"{distrito_destino} · {barrio_filtro}" if barrio_filtro else distrito_destino
    )
    st.subheader(f"Reservar plaza · {_ambito_res}")

    # ----- Capas del mapa --------------------------------------------- #
    cand = _candidatos_en_momento(simulator, instante, distrito=distrito_destino)
    if barrio_filtro:
        cand = cand[cand["barrio"] == barrio_filtro]
    cand_geo = cand.dropna(subset=["latitud", "longitud"]).head(20).reset_index(drop=True)

    presion_full = simulator.presion_por_calle(hora=hora, distrito=distrito_destino)
    if barrio_filtro and not presion_full.empty:
        _pb = _parqs_sidebar[["matricula", "barrio"]].rename(
            columns={"matricula": "matricula_parquimetro"}
        )
        presion_full = presion_full.merge(_pb, on="matricula_parquimetro", how="left")
        presion_full = presion_full[presion_full["barrio"] == barrio_filtro]

    if len(cand_geo) == 0:
        st.info(
            "No hay plazas candidatas a reservar en este escenario. "
            "Prueba a cambiar la hora o el barrio."
        )
    else:
        st.markdown(
            f"Las **calles del distrito** se colorean según su nivel de saturación "
            f"a las {hora}:00 h. Las **plazas reservables** aparecen marcadas con un "
            f"número verde. Haz clic en cualquier marcador verde para preseleccionarla; "
            f"el botón **Confirmar reserva** bloqueará la plaza durante "
            f"**{cfg.reservation_hold_min} min**."
        )

        # Índice de la plaza preseleccionada (para centrar el mapa en ella)
        if "plaza_preseleccionada" not in st.session_state:
            st.session_state.plaza_preseleccionada = 0

        def _normalize_plaza_idx(raw_value, n_items: int) -> int:
            """Devuelve un índice válido [0, n_items-1] para el selector de plazas."""
            idx = 0
            if isinstance(raw_value, int):
                idx = raw_value
            elif isinstance(raw_value, str):
                txt = raw_value.strip()
                if txt.isdigit():
                    idx = int(txt)
                elif txt.startswith("Plaza nº"):
                    # Ejemplo: "Plaza nº 1 · HERMOSILLA..." -> índice 0
                    try:
                        numero = int(txt.split("·")[0].replace("Plaza nº", "").strip())
                        idx = numero - 1
                    except ValueError:
                        idx = 0
            if n_items <= 0:
                return 0
            return max(0, min(idx, n_items - 1))

        sel_idx = _normalize_plaza_idx(st.session_state.plaza_preseleccionada, len(cand_geo))
        if sel_idx < 0 or sel_idx >= len(cand_geo):
            sel_idx = 0
            st.session_state.plaza_preseleccionada = 0

        # Centro del mapa: sobre la plaza seleccionada, con zoom más cercano
        centro = [
            float(cand_geo.iloc[sel_idx]["latitud"]),
            float(cand_geo.iloc[sel_idx]["longitud"]),
        ]
        zoom = cfg.app_map["default_zoom"] + (2 if barrio_filtro else 1)
        mapa_r = folium.Map(
            location=centro, zoom_start=zoom, tiles=cfg.app_map["tiles"],
        )

        # Capa 1: calles del distrito coloreadas por presión (mapa de calor)
        colors = cfg.app_map["color_scale"]
        nivel_color = {
            "Alta": colors["low"],     # rojo (saturado)
            "Media": colors["medium"], # naranja
            "Baja": colors["high"],    # verde claro
        }
        if not presion_full.empty:
            for _, row in presion_full.iterrows():
                c = nivel_color.get(row["nivel"], "#cbd5e1")
                folium.CircleMarker(
                    location=[row["latitud"], row["longitud"]],
                    radius=3,
                    color=c,
                    fill=True, fill_color=c, fill_opacity=0.45,
                    weight=0,
                    popup=folium.Popup(
                        f"<b>{row['calle']}</b><br>"
                        f"Saturación: <b>{row['nivel']}</b><br>"
                        f"Tickets a las {hora}h: {int(row['num_tickets'])}",
                        max_width=220,
                    ),
                ).add_to(mapa_r)

        # Capa 2: plazas candidatas a reservar (marcadores numerados verdes)
        for i, row in cand_geo.iterrows():
            num = i + 1
            es_sel = (i == sel_idx)
            popup_html = (
                f"<b>Plaza nº {num}</b><br>"
                f"<b>{row['calle']}</b><br>"
                f"Libera en: {row['min_hasta_liberacion']:.1f} min<br>"
                f"Fiabilidad: {row['prob_no_renovacion']*100:.0f}%<br>"
                f"<i>Selecciónala abajo para reservar.</i>"
            )
            # La plaza seleccionada se muestra más grande, en azul y con borde resaltado
            bg = "#2563eb" if es_sel else "#16a34a"
            size = 38 if es_sel else 28
            font_size = 16 if es_sel else 13
            border = "3px solid #fbbf24" if es_sel else "2px solid white"
            folium.Marker(
                location=[row["latitud"], row["longitud"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="background:{bg}; color:white;
                                width:{size}px; height:{size}px; border-radius:50%;
                                display:flex; align-items:center;
                                justify-content:center; font-weight:700;
                                font-family:'Segoe UI',sans-serif;
                                font-size:{font_size}px; border:{border};
                                box-shadow:0 2px 8px rgba(0,0,0,0.45);">
                      {num}
                    </div>
                    """
                ),
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"Plaza nº {num} · {row['calle'].split(',')[0]}",
            ).add_to(mapa_r)

        # Render del mapa con captura de click.
        # La key incluye sel_idx: cambia cuando el usuario elige otra plaza,
        # forzando que el iframe se reconstruya centrado en la nueva.
        click_data = st_folium(
            mapa_r,
            height=520,
            width=None,
            returned_objects=["last_object_clicked_tooltip", "last_object_clicked"],
            key=f"mapa_reserva_{sel_idx}",
        )

        # Leyenda fuera del iframe del mapa para que siempre se vea completa
        st.markdown(
            f"""
            <div style="display:flex; flex-wrap:wrap; gap:18px; align-items:center;
                        padding:10px 14px; margin-top:6px;
                        background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.08);
                        border-radius:8px; font-size:13px; line-height:1.4;">
              <div style="font-weight:600;">Saturación de la calle:</div>
              <div><span style="display:inline-block;width:11px;height:11px;
                    background:{colors['low']};border-radius:50%;
                    margin-right:6px;vertical-align:middle;"></span>Alta</div>
              <div><span style="display:inline-block;width:11px;height:11px;
                    background:{colors['medium']};border-radius:50%;
                    margin-right:6px;vertical-align:middle;"></span>Media</div>
              <div><span style="display:inline-block;width:11px;height:11px;
                    background:{colors['high']};border-radius:50%;
                    margin-right:6px;vertical-align:middle;"></span>Baja</div>
              <div style="margin-left:auto;">
                <span style="display:inline-block;width:16px;height:16px;
                      background:#16a34a;border-radius:50%;color:white;
                      text-align:center;font-size:10px;line-height:16px;
                      font-weight:700;margin-right:6px;vertical-align:middle;">N</span>
                Plaza reservable nº N
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----- Selección de plaza por click o por selector ------------ #
        # Si el usuario ha clicado un marcador verde, leemos el tooltip "Plaza nº N · ..."
        tooltip = (click_data or {}).get("last_object_clicked_tooltip") or ""
        if tooltip.startswith("Plaza nº "):
            try:
                num_clic = int(tooltip.split("·")[0].replace("Plaza nº", "").strip())
                if 1 <= num_clic <= len(cand_geo) and (num_clic - 1) != sel_idx:
                    st.session_state.plaza_preseleccionada = num_clic - 1
                    st.rerun()
            except ValueError:
                pass

        # Callback del selector: actualiza la plaza preseleccionada
        # para que en el siguiente render el mapa se recentre sobre ella.
        def _on_plaza_select_change() -> None:
            st.session_state.plaza_preseleccionada = _normalize_plaza_idx(
                st.session_state.select_plaza_reserva,
                len(cand_geo),
            )

        # Selector de respaldo (también funciona sin clicar en el mapa)
        st.selectbox(
            "Plaza seleccionada",
            options=list(range(len(cand_geo))),
            index=sel_idx,
            format_func=lambda i: (
                f"Plaza nº {i+1} · {cand_geo.iloc[i]['calle']} · "
                f"libera en {cand_geo.iloc[i]['min_hasta_liberacion']:.1f} min · "
                f"fiabilidad {cand_geo.iloc[i]['prob_no_renovacion']*100:.0f}%"
            ),
            key="select_plaza_reserva",
            on_change=_on_plaza_select_change,
        )
        plaza_idx = _normalize_plaza_idx(st.session_state.plaza_preseleccionada, len(cand_geo))

        if "reserva_activa" not in st.session_state:
            st.session_state.reserva_activa = None

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            if st.button("Confirmar reserva", type="primary", use_container_width=True):
                st.session_state.reserva_activa = {
                    "plaza": cand_geo.iloc[plaza_idx].to_dict(),
                    "inicio": time.time(),
                    "duracion": cfg.reservation_hold_min * 60,
                }
                st.rerun()

        if st.session_state.reserva_activa:
            r = st.session_state.reserva_activa
            transcurrido = time.time() - r["inicio"]
            restante = max(0, r["duracion"] - transcurrido)
            mins, segs = divmod(int(restante), 60)
            if restante > 0:
                st.success(
                    f"Reserva activa en **{r['plaza']['calle']}**\n\n"
                    f"Tiempo restante: **{mins:02d}:{segs:02d}**"
                )
                st.progress(restante / r["duracion"])
                if st.button("Cancelar reserva"):
                    st.session_state.reserva_activa = None
                    st.rerun()
            else:
                st.error("Reserva expirada.")
                st.session_state.reserva_activa = None

        st.caption(
            "Nota académica: la reserva es **simulada**. Una implementación "
            "real requeriría integración con sensores en plaza o con una API "
            "de gestión municipal del SER. Ver cap. 9 (Limitaciones y líneas futuras)."
        )


# ----- Tab 7: Validación --------------------------------------------------- #

if modo_dev:
    with tab_validacion:
        st.subheader("¿Funciona el modelo? Validación con datos reales")
        st.caption(
            "Escondemos un día entero, entrenamos el modelo sin él y comprobamos "
            "si lo que predice coincide con lo que pasó de verdad ese día."
        )

        fecha_val = st.selectbox(
            "Elige el día de prueba",
            options=[f.strftime("%Y-%m-%d") for f in fechas[::5]],
            index=4,
            help=(
                "Ese día no se usa para entrenar: el modelo lo ve por primera vez. "
                "Prueba a cambiarlo y verás que el resultado se mantiene."
            ),
        )

        validacion_res = _validacion_cached(fecha_val)
        mejora = (
            abs(validacion_res.spearman_modelo) / abs(validacion_res.spearman_baseline_aleatorio)
            if validacion_res.spearman_baseline_aleatorio else float("nan")
        )

        c1, c2 = st.columns(2)
        c1.metric(
            "Acierto del modelo",
            f"{validacion_res.spearman_modelo:+.2f}",
            help="Cuanto más cerca de −1, mejor ordena las zonas de fácil a difícil.",
        )
        c2.metric(
            "Acierto si fuera al azar",
            f"{validacion_res.spearman_baseline_aleatorio:+.2f}",
            help="Un sistema que adivinara al azar daría un valor cercano a 0.",
        )

        interpretacion = (
            f"El modelo acierta el orden de las zonas y es **{mejora:.0f} veces mejor "
            "que adivinar al azar**."
            if validacion_res.spearman_modelo <= -0.5
            else "El modelo acierta de forma moderada: es orientativo."
            if validacion_res.spearman_modelo <= -0.2
            else "El modelo apenas mejora al azar en este día."
        )
        st.info(interpretacion)

        st.markdown("##### Lo que predijo el modelo frente a lo que pasó de verdad")
        st.caption(
            "Cada punto es una zona a una hora. A la derecha, lo que el modelo "
            "predijo más fácil; arriba, los tickets que de verdad se vendieron. "
            "La línea roja resume la tendencia: como baja de izquierda a derecha, "
            "donde el modelo predijo más probabilidad se vendieron menos tickets. "
            "Eso es justo lo que tiene que pasar si acierta."
        )

        df_plot = validacion_res.detalle.copy()
        df_plot["zona"] = (
            df_plot["distrito"].str.title() + " · " + df_plot["hora"].astype(str) + "h"
        )
        df_plot = df_plot.rename(
            columns={
                "prob_predicha": "Probabilidad predicha (%)",
                "presion_observada": "Tickets reales",
            }
        )

        puntos = (
            alt.Chart(df_plot)
            .mark_circle(size=90, opacity=0.7, color="#2563eb")
            .encode(
                x=alt.X("Probabilidad predicha (%):Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Tickets reales:Q"),
                tooltip=["zona", "Probabilidad predicha (%)", "Tickets reales"],
            )
        )
        # Linea de tendencia (recta de ajuste) para visualizar la correlacion.
        x = df_plot["Probabilidad predicha (%)"].to_numpy(dtype=float)
        y = df_plot["Tickets reales"].to_numpy(dtype=float)
        pendiente, interseccion = np.polyfit(x, y, 1)
        linea_df = pd.DataFrame(
            {
                "Probabilidad predicha (%)": [x.min(), x.max()],
                "Tickets reales": [
                    pendiente * x.min() + interseccion,
                    pendiente * x.max() + interseccion,
                ],
            }
        )
        linea = (
            alt.Chart(linea_df)
            .mark_line(color="#dc2626", size=3)
            .encode(x="Probabilidad predicha (%):Q", y="Tickets reales:Q")
        )
        st.altair_chart((puntos + linea).properties(height=380), use_container_width=True)

        with st.expander("Ver la tabla con todos los números"):
            st.dataframe(
                validacion_res.detalle.style.format({
                    "prob_predicha": "{:.1f}%",
                    "presion_observada": "{:,}",
                }),
                hide_index=True,
                width="stretch",
            )

        st.caption(
            f"Día escondido: {fecha_val}. Entrenado con "
            f"{validacion_res.n_train_tickets:,} tickets del resto del trimestre y "
            f"probado sobre {validacion_res.n_observaciones} combinaciones de "
            "distrito y hora. Métrica: correlación de Spearman (capítulo 7)."
        )


# ----- Tab 8: Modelo ------------------------------------------------------- #

if modo_dev:
    with tab_modelo:
        st.subheader("¿De dónde sale la probabilidad? El modelo por dentro")
        st.caption(
            "El modelo no es una caja negra: suma puntos por cada factor y los "
            "convierte en una probabilidad. Aquí se ve el cálculo de tu escenario."
        )
        st.markdown(f"""
Para el escenario que tienes elegido ahora:

- **Distrito** ({distrito_destino}): {res.detalle["distrito"]} puntos
- **Hora** ({hora}h): {res.detalle["hora"]} puntos
- **Tipo de zona** ({tipo_zona}): {res.detalle["tipo_zona"]} puntos
- **Distintivo** ({distintivo}): {res.detalle["distintivo"]} puntos

Sumando los cuatro factores salen **{res.score_total} puntos**. Cuantos más
puntos, más presión esperada y por tanto **menos probabilidad** de encontrar
plaza. Al traducir esos puntos a porcentaje, queda:

**Probabilidad estimada: {res.probabilidad:.0f}%** → {res.nivel}
""")

    # Grafico de barras: contribucion de cada dimension al score total
    st.markdown("##### Cuánto suma cada factor")
    detalle_df = pd.DataFrame(
        {
            "dimensión": [
                f"Distrito ({distrito_destino})",
                f"Hora ({hora}h)",
                f"Tipo de zona ({tipo_zona})",
                f"Distintivo ({distintivo})",
            ],
            "score": [
                res.detalle["distrito"],
                res.detalle["hora"],
                res.detalle["tipo_zona"],
                res.detalle["distintivo"],
            ],
        }
    )
    st.bar_chart(detalle_df, x="dimensión", y="score", height=260)
    st.caption(
        "Cuanto más alta la barra, más presión aporta ese factor y menos "
        "probabilidad de aparcar."
    )

    with st.expander("Ver las tablas de presión que usa el modelo por dentro"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Por distrito**")
            st.dataframe(scorer.por_distrito, hide_index=True)
            st.markdown("**Por tipo de zona**")
            st.dataframe(scorer.por_zona, hide_index=True)
        with col2:
            st.markdown("**Por hora**")
            st.dataframe(scorer.por_hora, hide_index=True)
            st.markdown("**Por distintivo**")
            st.dataframe(scorer.por_distintivo, hide_index=True)


# --------------------------------------------------------------------------- #
# Pie
# --------------------------------------------------------------------------- #

st.markdown("---")
st.caption(
    "TFG · Modelo Analítico para la Reducción de la Congestión Derivada de la "
    "Búsqueda de Estacionamiento en Madrid · UAX 2025"
)
