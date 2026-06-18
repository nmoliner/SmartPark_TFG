"""Genera la figura del ratio de presión relativa (tickets/plazas) por distrito
para el capítulo 6, con los datos reales del interior de la M-30.

Marca la mediana con una línea horizontal y destaca el distrito de mayor y
menor ratio.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# --- Datos reales -------------------------------------------------------------
tickets = pd.read_parquet(RAIZ / "data" / "tickets_m30.parquet")
tickets_distrito = tickets.groupby("distrito").size()

plazas = pd.read_excel(
    RAIZ / "datasets" / "218228-0-ser-calles SER CALLES Y PLAZAS-xlsx.xlsx"
)
plazas_distrito = plazas.groupby("distrito")["numero_plazas"].sum()

df = pd.DataFrame({"tickets": tickets_distrito, "plazas": plazas_distrito}).dropna()
df["ratio"] = df["tickets"] / df["plazas"]
df = df.sort_values("ratio", ascending=False)

mediana = df["ratio"].median()

# --- Colores: destaca máximo (rojo) y mínimo (verde) -------------------------
colores = []
for distrito in df.index:
    if df.loc[distrito, "ratio"] == df["ratio"].max():
        colores.append("#d62728")  # rojo: el más presionado
    elif df.loc[distrito, "ratio"] == df["ratio"].min():
        colores.append("#2ca25f")  # verde: el más tranquilo
    else:
        colores.append("#3182bd")  # azul: resto

# --- Figura -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
etiquetas = [d.capitalize() for d in df.index]
barras = ax.bar(etiquetas, df["ratio"], color=colores, width=0.65)

# Línea de la mediana
ax.axhline(mediana, color="#636363", linestyle="--", linewidth=1.5)
ax.text(
    len(df) - 0.5,
    mediana + 1.5,
    f"Mediana: {mediana:.1f}",
    ha="right",
    va="bottom",
    fontsize=10,
    color="#636363",
    fontstyle="italic",
)

# Valor encima de cada barra
for barra, valor in zip(barras, df["ratio"]):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_title(
    "Presión relativa por distrito (tickets por plaza)\nInterior de la M-30",
    fontsize=12,
)
ax.set_ylabel("Tickets por plaza (trimestre)")
ax.set_ylim(0, df["ratio"].max() * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

fig.tight_layout()

salida = RAIZ / "docs" / "figuras"
salida.mkdir(parents=True, exist_ok=True)
destino = salida / "cap_06_fig_presion_relativa_distrito.png"
fig.savefig(destino, dpi=150, bbox_inches="tight")

print(f"Figura guardada en: {destino}")
print(f"\nMediana: {mediana:.1f} | Max: {df['ratio'].max():.1f} ({df['ratio'].idxmax()}) | Min: {df['ratio'].min():.1f} ({df['ratio'].idxmin()})")
