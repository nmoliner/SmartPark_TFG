"""Regenera TODAS las figuras del capítulo 6 con los datos reales del prototipo
(ámbito interior de la M-30, 7 distritos) e imprime todas las cifras necesarias
para el texto de la memoria.

Fuentes reales:
    - data/tickets_m30.parquet      (demanda: tickets ya filtrados a la M-30)
    - datasets/218228-...xlsx       (oferta: plazas SER)
    - datasets/300481-...xlsx       (parquímetros SER)

Sobrescribe los PNG existentes (cap_06_fig_01..08) para que las referencias del
.md sigan siendo válidas.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
FIG = RAIZ / "docs" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

AZUL = "#3182bd"
GRIS = "#bdbdbd"

# Formateador de eje Y: numeros normales con separador de miles (formato espanol)
_MILES = FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))


def formato_miles(ax):
    """Aplica separador de miles al eje Y y desactiva la notacion cientifica."""
    ax.yaxis.set_major_formatter(_MILES)


def guardar(fig, nombre):
    destino = FIG / nombre
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardada: {nombre}")


# ===========================================================================
# Cargar datos reales
# ===========================================================================
t = pd.read_parquet(RAIZ / "data" / "tickets_m30.parquet")
distritos_m30 = t["distrito"].unique()
total_tickets = len(t)

plazas = pd.read_excel(RAIZ / "datasets" / "218228-0-ser-calles SER CALLES Y PLAZAS-xlsx.xlsx")
plazas = plazas[plazas["distrito"].isin(distritos_m30)]

parq = pd.read_excel(RAIZ / "datasets" / "300481-3-ser-parquimetros-xlsx.xlsx")
parq = parq[parq["distrito"].isin(distritos_m30)]

print(f"\nTotal tickets M-30: {total_tickets:,} | distritos: {len(distritos_m30)}")
print("Distritos:", sorted(distritos_m30))

# ===========================================================================
# FIGURA 6.1 - Distribución horaria
# ===========================================================================
print("\n=== 6.1 Distribución horaria ===")
por_hora = t.groupby("hora").size()
pct_hora = por_hora / total_tickets * 100
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(por_hora.index, por_hora.values, color=AZUL)
ax.set_title("Distribución horaria de los tickets (interior de la M-30)")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Número de tickets")
ax.set_xticks(range(int(por_hora.index.min()), int(por_hora.index.max()) + 1))
ax.spines[["top", "right"]].set_visible(False)
formato_miles(ax)
guardar(fig, "cap_06_fig_01_tickets_por_hora.png")
print(f"  Pico: {pct_hora.idxmax()}h ({pct_hora.max():.1f}%)")
print(f"  9-14h: {pct_hora[(pct_hora.index>=9)&(pct_hora.index<=14)].sum():.1f}%")
print(f"  17-20h: {pct_hora[(pct_hora.index>=17)&(pct_hora.index<=20)].sum():.1f}%")
print(f"  Franja 15h (valle): {pct_hora.get(15,0):.1f}%")

# ===========================================================================
# FIGURA 6.2 - Demanda por distrito
# ===========================================================================
print("\n=== 6.2 Demanda por distrito ===")
dem = t.groupby("distrito").size().sort_values(ascending=False)
pct_dem = (dem / total_tickets * 100).round(1)
fig, ax = plt.subplots(figsize=(9, 5))
barras = ax.bar([d.capitalize() for d in dem.index], dem.values, color=AZUL)
ax.set_title("Demanda por distrito (interior de la M-30)")
ax.set_ylabel("Número de tickets")
ax.set_ylim(0, dem.max() * 1.12)
for ba, d in zip(barras, dem.index):
    ax.text(ba.get_x() + ba.get_width() / 2, ba.get_height(),
            f"{pct_dem[d]:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.spines[["top", "right"]].set_visible(False)
formato_miles(ax)
guardar(fig, "cap_06_fig_02_top10_distritos.png")
for d in dem.index:
    print(f"  {d:12s} {dem[d]:>9,}  {pct_dem[d]:5.1f}%")
print(f"  Top-3: {pct_dem.head(3).sum():.1f}%")
print(f"  Chamartín/Centro: {dem.iloc[0]/dem.iloc[-1]:.1f}x")

# ===========================================================================
# FIGURA 6.3 - Presión por distrito (terciles)
# ===========================================================================
print("\n=== 6.3 Terciles ===")
niveles = pd.qcut(dem, 3, labels=["Baja", "Media", "Alta"])
color_map = {"Alta": "#d62728", "Media": "#fd8d3c", "Baja": "#2ca25f"}
fig, ax = plt.subplots(figsize=(9, 5))
barras = ax.bar([d.capitalize() for d in dem.index], dem.values,
       color=[color_map[niveles[d]] for d in dem.index])
ax.set_title("Presión por distrito segmentada en terciles (interior de la M-30)")
ax.set_ylabel("Número de tickets")
ax.set_ylim(0, dem.max() * 1.12)
for ba, d in zip(barras, dem.index):
    ax.text(ba.get_x() + ba.get_width() / 2, ba.get_height(),
            f"{pct_dem[d]:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
ax.legend(handles, color_map.keys(), title="Presión")
ax.spines[["top", "right"]].set_visible(False)
formato_miles(ax)
guardar(fig, "cap_06_fig_03_presion_distrito_terciles.png")
for nivel in ["Alta", "Media", "Baja"]:
    ds = [d.capitalize() for d in dem.index if niveles[d] == nivel]
    print(f"  {nivel}: {len(ds)} -> {ds}")

# ===========================================================================
# FIGURA 6.4 - Tipo de zona: oferta vs demanda
# ===========================================================================
print("\n=== 6.4 Tipo de zona ===")
plazas["cn"] = plazas["color"].str.split().str[1:].str.join(" ").str.upper()
pl = plazas.groupby("cn")["numero_plazas"].sum()
pl_v, pl_a = int(pl.get("VERDE", 0)), int(pl.get("AZUL", 0))
pl_o = int(pl.sum() - pl_v - pl_a)
tk = t.groupby("tipo_zona").size()
tk_v, tk_a = int(tk.get("VERDE", 0)), int(tk.get("AZUL", 0))
tk_o = int(tk.sum() - tk_v - tk_a)
cats = ["Verde", "Azul", "Otros"]
pv = [pl_v, pl_a, pl_o]
tv = [tk_v, tk_a, tk_o]
pp = [v / sum(pv) * 100 for v in pv]
tp = [v / sum(tv) * 100 for v in tv]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
cols = ["#2ca25f", AZUL, GRIS]
b = ax1.bar(cats, pv, color=cols, width=0.6)
ax1.set_title("Oferta: número de plazas por tipo")
ax1.set_ylabel("Número de plazas")
ax1.set_ylim(0, max(pv) * 1.15)
for ba, va in zip(b, pv):
    ax1.text(ba.get_x() + ba.get_width() / 2, va, f"{va:,.0f}".replace(",", "."),
             ha="center", va="bottom", fontsize=10, fontweight="bold")
formato_miles(ax1)
x = np.arange(len(cats))
w = 0.38
b1 = ax2.bar(x - w / 2, pp, w, label="% de plazas (oferta)", color="#a1d99b")
b2 = ax2.bar(x + w / 2, tp, w, label="% de tickets (demanda)", color="#08519c")
ax2.set_title("Oferta vs demanda (en %)")
ax2.set_ylabel("Porcentaje sobre el total")
ax2.set_xticks(x)
ax2.set_xticklabels(cats)
ax2.set_ylim(0, 100)
ax2.legend()
for bars in (b1, b2):
    for ba in bars:
        ax2.text(ba.get_x() + ba.get_width() / 2, ba.get_height(),
                 f"{ba.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
guardar(fig, "cap_06_fig_04_tipo_zona.png")
print(f"  Plazas: Verde {pl_v:,} ({pp[0]:.1f}%) Azul {pl_a:,} ({pp[1]:.1f}%) Otros {pl_o:,} ({pp[2]:.1f}%)")
print(f"  Tickets: Verde {tp[0]:.1f}% Azul {tp[1]:.1f}% Otros {tp[2]:.1f}%")
print(f"  TOTAL plazas M-30: {int(pl.sum()):,}")

# ===========================================================================
# FIGURA 6.5 - Plazas por distrito
# ===========================================================================
print("\n=== 6.5 Plazas por distrito ===")
pl_dist = plazas.groupby("distrito")["numero_plazas"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([d.capitalize() for d in pl_dist.index], pl_dist.values, color=AZUL)
ax.set_title("Plazas SER por distrito (interior de la M-30)")
ax.set_ylabel("Número de plazas")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.spines[["top", "right"]].set_visible(False)
formato_miles(ax)
guardar(fig, "cap_06_fig_05_plazas_distrito.png")
print(f"  Más: {pl_dist.index[0].capitalize()} {pl_dist.iloc[0]:,} | Menos: {pl_dist.index[-1].capitalize()} {pl_dist.iloc[-1]:,}")

# ===========================================================================
# FIGURA 6.6 - Parquímetros por distrito
# ===========================================================================
print("\n=== 6.6 Parquímetros ===")
pq_dist = parq.groupby("distrito").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([d.capitalize() for d in pq_dist.index], pq_dist.values, color=AZUL)
ax.set_title("Parquímetros por distrito (interior de la M-30)")
ax.set_ylabel("Número de parquímetros")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.spines[["top", "right"]].set_visible(False)
formato_miles(ax)
guardar(fig, "cap_06_fig_06_parquimetros_distrito.png")
comb = pd.DataFrame({"plazas": pl_dist, "parq": pq_dist}).dropna()
r = comb["plazas"].corr(comb["parq"])
print(f"  Correlación plazas-parquímetros: r = {r:.2f}")
print(f"  Total parquímetros: {len(parq):,}")

# ===========================================================================
# FIGURA 6.7 - Ratio de presión relativa
# ===========================================================================
print("\n=== 6.7 Ratio presión relativa ===")
ratio = (dem / pl_dist).dropna().sort_values(ascending=False)
mediana = ratio.median()
cols = []
for d in ratio.index:
    if ratio[d] == ratio.max():
        cols.append("#d62728")
    elif ratio[d] == ratio.min():
        cols.append("#2ca25f")
    else:
        cols.append(AZUL)
fig, ax = plt.subplots(figsize=(9, 5))
barras = ax.bar([d.capitalize() for d in ratio.index], ratio.values, color=cols)
ax.axhline(mediana, color="#636363", ls="--", lw=1.5)
ax.text(len(ratio) - 0.5, mediana + 1.5, f"Mediana: {mediana:.1f}", ha="right",
        color="#636363", fontstyle="italic")
for ba, va in zip(barras, ratio.values):
    ax.text(ba.get_x() + ba.get_width() / 2, va, f"{va:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Presión relativa por distrito: tickets por plaza (interior de la M-30)")
ax.set_ylabel("Tickets por plaza (trimestre)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.set_ylim(0, ratio.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
guardar(fig, "cap_06_fig_07_ratio_demanda_oferta.png")
for d in ratio.index:
    print(f"  {d:12s} {ratio[d]:6.1f}")
print(f"  Mediana: {mediana:.1f} | Máx: {ratio.idxmax().capitalize()} {ratio.max():.1f} | Mín: {ratio.idxmin().capitalize()} {ratio.min():.1f}")

# ===========================================================================
# FIGURA 6.8 - Comparación normalizada oferta-demanda
# ===========================================================================
print("\n=== 6.8 Normalizada ===")
dfn = pd.DataFrame({"demanda": dem, "plazas": pl_dist}).dropna()
dfn["dn"] = dfn["demanda"] / dfn["demanda"].max()
dfn["pn"] = dfn["plazas"] / dfn["plazas"].max()
dfn = dfn.sort_values("demanda", ascending=False)
x = np.arange(len(dfn))
w = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w / 2, dfn["dn"], w, label="Demanda (normalizada)", color="#08519c")
ax.bar(x + w / 2, dfn["pn"], w, label="Oferta de plazas (normalizada)", color="#a1d99b")
ax.set_xticks(x)
ax.set_xticklabels([d.capitalize() for d in dfn.index], rotation=25, ha="right")
ax.set_title("Comparación normalizada entre oferta y demanda (interior de la M-30)")
ax.set_ylabel("Nivel relativo (0-1)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
guardar(fig, "cap_06_fig_08_comparacion_normalizada.png")

# ===========================================================================
# DISTINTIVO (texto, sin figura propia)
# ===========================================================================
print("\n=== Distintivo (texto) ===")
dist = (t.groupby("distintivo").size() / total_tickets * 100).round(1).sort_values(ascending=False)
print(dist.to_string())

print("\nLISTO. Todas las figuras regeneradas con datos reales M-30.")
