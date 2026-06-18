"""Exporta a PNG las 8 figuras referenciadas en el capítulo 6.

Cada figura se guarda en `docs/figuras/` con un nombre coherente
con el índice de figuras del TFG (cap_06_fig_XX.png).

Uso:
    .\.venv\Scripts\python.exe scripts\exportar_figuras_cap6.py

Requisitos: los datasets crudos deben estar en la raíz del proyecto, con los
mismos nombres que los detectados por el pipeline (Primertrimestre2025 SER
tickets aparcamiento.csv, 218228-0-ser-calles SER CALLES Y PLAZAS-xlsx.xlsx,
300481-3-ser-parquimetros-xlsx.xlsx).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

stats: dict = {}


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "docs" / "figuras"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    "tickets": BASE_DIR / "Primertrimestre2025 SER tickets aparcamiento.csv",
    "plazas": BASE_DIR / "218228-0-ser-calles SER CALLES Y PLAZAS-xlsx.xlsx",
    "parquimetros": BASE_DIR / "300481-3-ser-parquimetros-xlsx.xlsx",
}

print("Cargando tickets...")
df_tiques = pd.read_csv(DATA_FILES["tickets"], sep=";")
df_tiques["fecha_operacion"] = pd.to_datetime(df_tiques["fecha_operacion"])
df_tiques["hora"] = df_tiques["fecha_operacion"].dt.hour

print("Cargando plazas y parquímetros...")
df_plazas = pd.read_excel(DATA_FILES["plazas"])
df_parq = pd.read_excel(DATA_FILES["parquimetros"])


def _guardar(fig, nombre):
    ruta = OUT_DIR / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


def _nice_upper_bound(max_value: float) -> float:
    if max_value <= 0:
        return 1.0
    target = max_value * 1.10
    exponent = math.floor(math.log10(target))
    base = 10 ** exponent
    normalized = target / base
    if normalized <= 1:
        step = 1
    elif normalized <= 2:
        step = 2
    elif normalized <= 5:
        step = 5
    else:
        step = 10
    return step * base


def _fmt_thousands(x, _pos):
    return f"{int(x):,}".replace(",", ".")


def _set_axis_max(ax, values, axis: str = "y"):
    upper = _nice_upper_bound(float(np.max(values)))
    if axis == "y":
        ax.set_ylim(0, upper)
    else:
        ax.set_xlim(0, upper)


def _style_count_axis(ax, axis: str = "y"):
    formatter = FuncFormatter(_fmt_thousands)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


# Figura 6.1: tickets por hora
print("Figura 6.1: tickets por hora...")
tph = df_tiques.groupby("hora").size()
fig, ax = plt.subplots(figsize=(9, 5))
tph.plot(kind="bar", ax=ax, color="#2563eb")
ax.set_title("Número de tickets por hora del día")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Número de tickets")
_set_axis_max(ax, tph.values, axis="y")
_style_count_axis(ax, axis="y")
_guardar(fig, "cap_06_fig_01_tickets_por_hora.png")

# Figura 6.2: top 10 distritos
print("Figura 6.2: top 10 distritos...")
tpd = df_tiques.groupby("distrito").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
tpd.head(10).plot(kind="bar", ax=ax, color="#0ea5e9")
ax.set_title("Distritos con mayor demanda de aparcamiento (top 10)")
ax.set_xlabel("Distrito")
ax.set_ylabel("Número de tickets")
ax.tick_params(axis="x", rotation=35)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right")
_set_axis_max(ax, tpd.head(10).values, axis="y")
_style_count_axis(ax, axis="y")
_guardar(fig, "cap_06_fig_02_top10_distritos.png")

# Figura 6.3: ranking distritos con terciles
print("Figura 6.3: presión por distrito (terciles)...")
ranking = tpd.reset_index()
ranking.columns = ["distrito", "num_tickets"]
ranking["nivel_presion"] = pd.qcut(
    ranking["num_tickets"], q=3, labels=["Baja", "Media", "Alta"]
)
colores = {"Alta": "#dc2626", "Media": "#f59e0b", "Baja": "#16a34a"}
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    ranking["distrito"],
    ranking["num_tickets"],
    color=[colores[n] for n in ranking["nivel_presion"]],
)
ax.set_title("Presión de aparcamiento por distrito (terciles Alta/Media/Baja)")
ax.set_xlabel("Distrito")
ax.set_ylabel("Número de tickets")
ax.tick_params(axis="x", rotation=35)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right")
from matplotlib.patches import Patch

legend = [Patch(color=c, label=l) for l, c in colores.items()]
ax.legend(handles=legend, title="Nivel")
_set_axis_max(ax, ranking["num_tickets"].values, axis="y")
_style_count_axis(ax, axis="y")
_guardar(fig, "cap_06_fig_03_presion_distrito_terciles.png")

# Figura 6.4: tickets por tipo de zona
print("Figura 6.4: tipo de zona...")
tpz = df_tiques.groupby("tipo_zona").size().sort_values(ascending=False)
tpz_plot = tpz.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(tpz_plot.index, tpz_plot.values, color="#10b981")
ax.set_title("Número de tickets por tipo de zona regulada")
ax.set_xlabel("Número de tickets")
ax.set_ylabel("Tipo de zona")
_set_axis_max(ax, tpz_plot.values, axis="x")
_style_count_axis(ax, axis="x")
_guardar(fig, "cap_06_fig_04_tipo_zona.png")

# Figura 6.5: plazas por distrito
print("Figura 6.5: plazas SER por distrito...")
ppd = df_plazas.groupby("distrito")["numero_plazas"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ppd.plot(kind="bar", ax=ax, color="#6366f1")
ax.set_title("Número de plazas SER por distrito")
ax.set_xlabel("Distrito")
ax.set_ylabel("Número de plazas")
ax.tick_params(axis="x", rotation=35)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right")
_set_axis_max(ax, ppd.values, axis="y")
_style_count_axis(ax, axis="y")
_guardar(fig, "cap_06_fig_05_plazas_distrito.png")

# Figura 6.6: parquímetros por distrito
print("Figura 6.6: parquímetros por distrito...")
parq = df_parq.groupby("distrito").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
parq.plot(kind="bar", ax=ax, color="#8b5cf6")
ax.set_title("Número de parquímetros por distrito")
ax.set_xlabel("Distrito")
ax.set_ylabel("Número de parquímetros")
ax.tick_params(axis="x", rotation=35)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right")
_set_axis_max(ax, parq.values, axis="y")
_style_count_axis(ax, axis="y")
_guardar(fig, "cap_06_fig_06_parquimetros_distrito.png")

# Figura 6.7: ratio demanda/oferta
print("Figura 6.7: ratio demanda/oferta...")
df_dem = ranking.rename(columns={"num_tickets": "demanda"})
df_of = ppd.reset_index().rename(columns={"numero_plazas": "plazas"})
df_cmp = pd.merge(df_dem, df_of, on="distrito", how="inner")
df_cmp["ratio"] = df_cmp["demanda"] / df_cmp["plazas"]
df_cmp_top = df_cmp.sort_values("ratio", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_cmp_top["distrito"], df_cmp_top["ratio"], color="#ef4444")
ax.set_title("Distritos con mayor presión relativa (demanda / oferta)")
ax.set_xlabel("Distrito")
ax.set_ylabel("Ratio tickets / plazas")
ax.tick_params(axis="x", rotation=35)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right")
_set_axis_max(ax, df_cmp_top["ratio"].values, axis="y")
_guardar(fig, "cap_06_fig_07_ratio_demanda_oferta.png")

# Figura 6.8: comparación normalizada
print("Figura 6.8: comparación normalizada...")
df_plot = df_cmp.copy()
df_plot["demanda_norm"] = df_plot["demanda"] / df_plot["demanda"].max()
df_plot["plazas_norm"] = df_plot["plazas"] / df_plot["plazas"].max()
df_plot = df_plot.sort_values("demanda", ascending=False)
x = np.arange(len(df_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, df_plot["demanda_norm"], width, label="Demanda (normalizada)", color="#ef4444")
ax.bar(x + width / 2, df_plot["plazas_norm"], width, label="Oferta (normalizada)", color="#0ea5e9")
ax.set_xticks(x)
ax.set_xticklabels(df_plot["distrito"], rotation=35, ha="right")
ax.set_title("Comparación normalizada entre demanda y oferta por distrito")
ax.set_xlabel("Distrito")
ax.set_ylabel("Nivel relativo (0-1)")
ax.set_ylim(0, 1.05)
ax.legend()
_guardar(fig, "cap_06_fig_08_comparacion_normalizada.png")

# ---------------------------------------------------------------------------
# Estadísticos auxiliares para insertar en el capítulo 6
# ---------------------------------------------------------------------------
print("Calculando estadísticos para el capítulo 6...")

total_tickets = int(df_tiques.shape[0])

# 6.1 — Horaria
horaria = tph.sort_index()
top3_horas = horaria.sort_values(ascending=False).head(3)
manana = horaria.loc[10:14].sum()
tarde = horaria.loc[17:19].sum()
noche = horaria.loc[list(range(0, 8)) + [22, 23]].sum()
stats["fig_01_horaria"] = {
    "hora_pico": int(top3_horas.index[0]),
    "tickets_hora_pico": int(top3_horas.iloc[0]),
    "pct_manana_10_14": round(100 * manana / total_tickets, 1),
    "pct_tarde_17_19": round(100 * tarde / total_tickets, 1),
    "pct_franja_nocturna": round(100 * noche / total_tickets, 1),
}

# 6.2 — Top 10 distritos
total_distritos = tpd.sum()
top10 = tpd.head(10)
stats["fig_02_top10_distritos"] = {
    "lider": str(top10.index[0]),
    "tickets_lider": int(top10.iloc[0]),
    "segundo": str(top10.index[1]),
    "tickets_segundo": int(top10.iloc[1]),
    "tercero": str(top10.index[2]),
    "tickets_tercero": int(top10.iloc[2]),
    "pct_top3_sobre_total": round(100 * top10.head(3).sum() / total_distritos, 1),
    "pct_top10_sobre_total": round(100 * top10.sum() / total_distritos, 1),
    "ratio_lider_vs_decimo": round(float(top10.iloc[0] / top10.iloc[-1]), 1),
}

# 6.3 — Terciles
conteo_terciles = ranking["nivel_presion"].value_counts().to_dict()
stats["fig_03_terciles"] = {
    "n_distritos": int(ranking.shape[0]),
    "n_alta": int(conteo_terciles.get("Alta", 0)),
    "n_media": int(conteo_terciles.get("Media", 0)),
    "n_baja": int(conteo_terciles.get("Baja", 0)),
}

# 6.4 — Tipo zona
pct_zonas = (tpz / tpz.sum() * 100).round(1).to_dict()
stats["fig_04_tipo_zona"] = {
    "ranking": [str(z) for z in tpz.index.tolist()],
    "pct": {str(k): float(v) for k, v in pct_zonas.items()},
}

# 6.5 — Plazas
stats["fig_05_plazas"] = {
    "total_plazas": int(ppd.sum()),
    "distrito_max_plazas": str(ppd.index[0]),
    "plazas_max": int(ppd.iloc[0]),
    "distrito_min_plazas": str(ppd.index[-1]),
    "plazas_min": int(ppd.iloc[-1]),
}

# 6.6 — Parquímetros
stats["fig_06_parquimetros"] = {
    "total_parquimetros": int(parq.sum()),
    "distrito_max_parq": str(parq.index[0]),
    "parq_max": int(parq.iloc[0]),
    "correlacion_plazas_parq": round(
        float(
            pd.concat([ppd.rename("plazas"), parq.rename("parq")], axis=1)
            .dropna()
            .corr()
            .iloc[0, 1]
        ),
        3,
    ),
}

# 6.7 — Ratio demanda/oferta
df_cmp_sorted = df_cmp.sort_values("ratio", ascending=False)
top3_ratio = df_cmp_sorted.head(3)
stats["fig_07_ratio"] = {
    "top3": [
        {"distrito": str(r["distrito"]), "ratio": round(float(r["ratio"]), 1)}
        for _, r in top3_ratio.iterrows()
    ],
    "ratio_mediano": round(float(df_cmp_sorted["ratio"].median()), 1),
    "ratio_min": round(float(df_cmp_sorted["ratio"].min()), 1),
    "ratio_lider_vs_mediano": round(
        float(top3_ratio.iloc[0]["ratio"] / df_cmp_sorted["ratio"].median()), 1
    ),
}

# 6.8 — Desequilibrio normalizado
df_plot["gap"] = df_plot["demanda_norm"] - df_plot["plazas_norm"]
desequilibrados = df_plot[df_plot["gap"] > 0.15].sort_values("gap", ascending=False)
equilibrados = df_plot[df_plot["gap"].abs() <= 0.10]
stats["fig_08_normalizado"] = {
    "n_desequilibrados": int(desequilibrados.shape[0]),
    "n_equilibrados": int(equilibrados.shape[0]),
    "top3_gap": [
        {"distrito": str(r["distrito"]), "gap": round(float(r["gap"]), 2)}
        for _, r in desequilibrados.head(3).iterrows()
    ],
}

stats_path = OUT_DIR / "figuras_cap6_stats.json"
with stats_path.open("w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)
print(f"  -> {stats_path}")

print(f"\nLISTO. Figuras guardadas en: {OUT_DIR}")
