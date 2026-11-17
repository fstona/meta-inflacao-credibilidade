"""
Estática comparativa de longo prazo (estado estacionário, sem dinâmica).

  No equilíbrio:  pe = meta (crível)          ->  expectativas ancoradas na meta
                  r  = r* (juro neutro)       ->  hiato zero, juro real no neutro
                  i  = r* + meta              ->  equação de Fisher

O juro real de equilíbrio não depende da meta; a Selic nominal sobe 1 para 1 com ela.
"""
import numpy as np
import matplotlib.pyplot as plt

r_star = 10.0
meta = np.linspace(2, 6, 100)
selic = r_star + meta
real = np.full_like(meta, r_star)

m0, m1 = 3.0, 4.5
cor_nom, cor_real = "#2a78d6", "#eb6834"

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(meta, selic, color=cor_nom, lw=2)
ax.plot(meta, real, color=cor_real, lw=2)
ax.text(6.05, selic[-1], "Selic nominal\n= r* + meta", va="center", fontsize=9.5, color="#0b0b0b")
ax.text(6.05, real[-1], "Juro real\n= r*", va="center", fontsize=9.5, color="#0b0b0b")

for m, rot in [(m0, "meta 3,0%"), (m1, "meta 4,5%")]:
    for y in (r_star + m, r_star):
        ax.plot(m, y, "o", ms=8, color=cor_nom if y > r_star else cor_real, mec="white", mew=2, zorder=3)
    ax.vlines(m, 8.5, r_star + m, color="#b9b8b2", lw=0.8, ls=":")
    ax.text(m, r_star + m + 0.35, f"{r_star + m:.1f}%".replace(".", ","), ha="center", fontsize=9.5, color="#0b0b0b")
    ax.text(m, r_star - 0.55, "10,0%", ha="center", fontsize=9.5, color="#0b0b0b")

ax.annotate("", xy=(m1, r_star + m1), xytext=(m0, r_star + m0),
            arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.2))
ax.text(3.75, 14.35, "+1,5 p.p.", ha="center", fontsize=10, color="#0b0b0b")
ax.annotate("", xy=(m1, r_star + 0.15), xytext=(m0, r_star + 0.15),
            arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.2))
ax.text(3.75, r_star + 0.4, "0 p.p.", ha="center", fontsize=10, color="#0b0b0b")

ax.set_xlim(1.8, 7.0)
ax.set_ylim(8.5, 17)
ax.set_xlabel("Meta de inflação (% a.a.)", color="#52514e", fontsize=10)
ax.set_ylabel("% a.a.", color="#52514e", fontsize=10)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color("#b9b8b2")
ax.tick_params(colors="#52514e", labelsize=9)
ax.set_title("Equilíbrio de longo prazo: a meta move a Selic, não o juro real",
             fontsize=13, loc="left", color="#0b0b0b", pad=14)
fig.text(0.01, 0.01, "Estado estacionário com meta crível: expectativa = meta, hiato = 0, i = r* + meta (r* = 10%). Ilustrativo.",
         fontsize=8, color="#52514e")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig("equilibrio_meta.png", dpi=160)
