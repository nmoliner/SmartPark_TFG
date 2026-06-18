"""Genera la figura comparativa de oferta (plazas) y demanda (tickets) por tipo
de zona SER para el capítulo 6.

Datos reales:
    - Plazas: dataset "SER calles y plazas" (datasets/218228-...).
    - Demanda: tickets del interior de la M-30 (data/tickets_m30.parquet).

Para que la comparación sea limpia se agrupan las categorías en Verde, Azul y
"Otros" (el resto de tipos, minoritarios y con correspondencia no directa entre
ambos datasets).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# --- Demanda: tickets por tipo de zona (define el ámbito M-30) ----------------
df_tickets = pd.read_parquet(RAIZ / "data" / "tickets_m30.parquet")
distritos_m30 = df_tickets["distrito"].unique()

# --- Oferta: plazas por color, restringidas al ámbito M-30 --------------------
df_plazas = pd.read_excel(
    RAIZ / "datasets" / "218228-0-ser-calles SER CALLES Y PLAZAS-xlsx.xlsx"
)
df_plazas = df_plazas[df_plazas["distrito"].isin(distritos_m30)]
df_plazas["color_norm"] = (
    df_plazas["color"].str.split().str[1:].str.join(" ").str.upper()
)
plazas = df_plazas.groupby("color_norm")["numero_plazas"].sum()

plazas_verde = int(plazas.get("VERDE", 0))
plazas_azul = int(plazas.get("AZUL", 0))
plazas_otros = int(plazas.sum() - plazas_verde - plazas_azul)

# --- Demanda: tickets por tipo de zona ---------------------------------------
df_tickets = pd.read_parquet(RAIZ / "data" / "tickets_m30.parquet")
tickets = df_tickets.groupby("tipo_zona").size()
tickets_verde = int(tickets.get("VERDE", 0))
tickets_azul = int(tickets.get("AZUL", 0))
tickets_otros = int(tickets.sum() - tickets_verde - tickets_azul)

categorias = ["Verde", "Azul", "Otros"]
plazas_vals = [plazas_verde, plazas_azul, plazas_otros]
tickets_vals = [tickets_verde, tickets_azul, tickets_otros]

plazas_pct = [v / sum(plazas_vals) * 100 for v in plazas_vals]
tickets_pct = [v / sum(tickets_vals) * 100 for v in tickets_vals]

# --- Figura con dos paneles ---------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colores = ["#2ca25f", "#3182bd", "#bdbdbd"]

# Panel A: número de plazas por tipo
barras1 = ax1.bar(categorias, plazas_vals, color=colores, width=0.6)
ax1.set_title("Oferta: número de plazas por tipo", fontsize=12)
ax1.set_ylabel("Número de plazas")
ax1.set_ylim(0, max(plazas_vals) * 1.15)
for barra, valor in zip(barras1, plazas_vals):
    ax1.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor:,.0f}".replace(",", "."),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Panel B: % de plazas vs % de demanda
x = np.arange(len(categorias))
ancho = 0.38
b1 = ax2.bar(x - ancho / 2, plazas_pct, ancho, label="% de plazas (oferta)", color="#a1d99b")
b2 = ax2.bar(x + ancho / 2, tickets_pct, ancho, label="% de tickets (demanda)", color="#08519c")
ax2.set_title("Oferta vs demanda (en %)", fontsize=12)
ax2.set_ylabel("Porcentaje sobre el total")
ax2.set_xticks(x)
ax2.set_xticklabels(categorias)
ax2.set_ylim(0, 100)
ax2.legend()
for barras in (b1, b2):
    for barra in barras:
        altura = barra.get_height()
        ax2.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f"{altura:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()

salida = RAIZ / "docs" / "figuras"
salida.mkdir(parents=True, exist_ok=True)
destino = salida / "cap_06_fig_tipo_zona_oferta_demanda.png"
fig.savefig(destino, dpi=150, bbox_inches="tight")

print(f"Figura guardada en: {destino}")
print("\n--- Datos usados ---")
print(f"Plazas  -> Verde: {plazas_verde:,} | Azul: {plazas_azul:,} | Otros: {plazas_otros:,}")
print(f"Plazas %% -> Verde: {plazas_pct[0]:.1f} | Azul: {plazas_pct[1]:.1f} | Otros: {plazas_pct[2]:.1f}")
print(f"Tickets -> Verde: {tickets_verde:,} | Azul: {tickets_azul:,} | Otros: {tickets_otros:,}")
print(f"Tickets %% -> Verde: {tickets_pct[0]:.1f} | Azul: {tickets_pct[1]:.1f} | Otros: {tickets_pct[2]:.1f}")
