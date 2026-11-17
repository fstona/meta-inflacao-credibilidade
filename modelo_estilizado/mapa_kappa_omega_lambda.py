"""
Mapas κ × ω × λ — meta de 3% para 4,5%, γ = 0 (melhor caso para a mudança).

Modelo (estado estacionário):
  pe - meta = (1-λ)·K,   K = κ - (1-ω)·Δmeta
  i = r* + θ·(pe - meta) + pe
  (A) λ exógena
  (B) λ endógena: λ = λbar - η·(pe - meta)  ->  gap = (1-λbar)·K / (1-ηK)

Resultado analítico do caso exógeno:
  ΔSelic = Δmeta · [1 - (1+θ)(1-λ)(1-ω)]     -> κ não aparece!
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

meta0, dmeta, theta, rstar, lam_bar, eta = 3.0, 1.5, 0.5, 5.0, 0.8, 0.2
div = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#b9b8b2"

def endog(K):
    den = 1 - eta * K
    gap = np.where(den > 0, (1 - lam_bar) * K / np.where(den > 0, den, 1), np.inf)
    lam = lam_bar - eta * gap
    bad = ~np.isfinite(gap) | (lam < 0)
    lam = np.where(bad, 0.0, lam)
    gap = (1 - lam) * K
    return lam, gap

def selic(meta, gap):
    return rstar + theta * gap + meta + gap

def estilo(ax, tit):
    ax.set_title(tit, fontsize=10.5, loc="left", color=INK)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)

# ======================= Figura B: λ endógena, eixos κ × ω
kap = np.linspace(1.0, 4.0, 121)
om = np.linspace(0.0, 1.0, 101)
KK, OO = np.meshgrid(kap, om)
lam0, gap0 = endog(KK)                      # hoje (Δmeta = 0)
lam1, gap1 = endog(KK - (1 - OO) * dmeta)   # depois
dS = selic(meta0 + dmeta, gap1) - selic(meta0, gap0)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[0.85, 1, 1]))
# painel 1: κ -> credibilidade de hoje
estilo(axs[0], "Credibilidade hoje (λ) em função de κ")
axs[0].plot(kap, lam0[0], color="#2a78d6", lw=2)
for k in (1.5, 3.0, 3.75):
    l, g = endog(np.array(k))
    axs[0].plot(k, l, "o", ms=7, color="#2a78d6", mec="white", mew=2)
    axs[0].annotate("desancoragem hoje " + f"{float(g):.1f}".replace(".", ",") + " p.p.", (k, float(l)),
                    xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK)
axs[0].set_xlabel("κ: leniência percebida (p.p.)", color=INK2, fontsize=9)
axs[0].set_ylabel("λ hoje", color=INK2, fontsize=9)
axs[0].grid(axis="y", color="#e6e5e0", lw=0.8)
axs[0].spines[["top", "right"]].set_visible(False)

# painel 2: λ depois
estilo(axs[1], "Credibilidade depois da mudança (λ)")
im1 = axs[1].pcolormesh(kap, om, lam1, cmap=seq, vmin=0, vmax=lam_bar, shading="auto")
cs = axs[1].contour(kap, om, lam1, levels=[0.2, 0.4, 0.6], colors="white", linewidths=0.8)
axs[1].clabel(cs, fmt=lambda v: f"λ={v:.1f}".replace(".", ","), fontsize=8)
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02).ax.tick_params(labelsize=8, colors=INK2)

# painel 3: ΔSelic
estilo(axs[2], "Δ Selic de equilíbrio (p.p.)\nazul = cai, vermelho = sobe")
lim = np.nanmax(np.abs(dS))
im2 = axs[2].pcolormesh(kap, om, dS, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
c0 = axs[2].contour(kap, om, dS, levels=[0], colors=INK, linewidths=1.4)
axs[2].clabel(c0, fmt=lambda v: "Δ Selic = 0", fontsize=8.5)
c1 = axs[2].contour(kap, om, dS, levels=[0.5, 1.0, 1.5, 2.0], colors=INK2, linewidths=0.6, linestyles=":")
axs[2].clabel(c1, fmt=lambda v: f"+{v:.1f}".replace(".", ","), fontsize=8)
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02).ax.tick_params(labelsize=8, colors=INK2)

for ax in axs[1:]:
    ax.set_xlabel("κ: leniência percebida (p.p.)", color=INK2, fontsize=9)
    ax.set_ylabel("ω: quanto a tolerância anda com a meta", color=INK2, fontsize=9)
    ax.plot(3.0, 1.0, "o", ms=7, color="#eb6834", mec="white", mew=2, clip_on=False, zorder=5)
    ax.annotate("calibração base", (3.0, 1.0), xytext=(-4, -14), textcoords="offset points",
                fontsize=8.5, color=INK, ha="right")
fig.suptitle("λ endógena: com o BC já percebido como leniente, o resultado depende de a tolerância acompanhar ou não a meta",
             fontsize=12.5, x=0.01, ha="left", color=INK)
fig.text(0.01, 0.005, "λ = 0,8 − 0,2·(pe − meta);  pe − meta = (1−λ)·[κ − (1−ω)·Δmeta];  i = 5 + 0,5·(pe − meta) + pe;  meta 3 → 4,5;  γ = 0.  Ilustrativo.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig("mapa_lambda_endogena.png", dpi=160)

# ======================= Figura A: λ exógena, eixos ω × λ (κ sai da conta)
lam = np.linspace(0.0, 1.0, 101)
OO2, LL = np.meshgrid(om, lam)
dSx = dmeta * (1 - (1 + theta) * (1 - LL) * (1 - OO2))
fig2, ax = plt.subplots(figsize=(7.2, 5.2))
estilo(ax, "Δ Selic de equilíbrio (p.p.) com λ exógena — vale para qualquer κ")
lim2 = np.abs(dSx).max()
im = ax.pcolormesh(om, lam, dSx, cmap=div, norm=TwoSlopeNorm(0, -lim2, lim2), shading="auto")
c0 = ax.contour(om, lam, dSx, levels=[0], colors=INK, linewidths=1.4)
ax.clabel(c0, fmt=lambda v: "Δ Selic = 0", fontsize=8.5)
c1 = ax.contour(om, lam, dSx, levels=[0.5, 1.0], colors=INK2, linewidths=0.6, linestyles=":")
ax.clabel(c1, fmt=lambda v: f"+{v:.1f}".replace(".", ","), fontsize=8)
for l in (0.3, 0.5, 0.7):
    ax.axhline(l, color="white", lw=0.6, ls="--")
    ax.text(1.01, l, f"λ = {l:.1f}".replace(".", ","), va="center", fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform())
ax.plot(1.0, 0.5, "o", ms=7, color="#eb6834", mec="white", mew=2, clip_on=False, zorder=5)
ax.annotate("calibração base", (1.0, 0.5), xytext=(-6, 8), textcoords="offset points", fontsize=8.5, color=INK, ha="right")
ax.set_xlabel("ω: quanto a tolerância anda com a meta", color=INK2, fontsize=9)
ax.set_ylabel("λ: credibilidade", color=INK2, fontsize=9)
fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.1).ax.tick_params(labelsize=8, colors=INK2)
fig2.text(0.01, 0.005, "ΔSelic = Δmeta·[1 − (1+θ)(1−λ)(1−ω)], θ = 0,5, Δmeta = 1,5, γ = 0.  Ilustrativo.", fontsize=8, color=INK2)
fig2.tight_layout(rect=[0, 0.03, 1, 1])
fig2.savefig("mapa_lambda_exogena.png", dpi=160)

# ---- números para o texto
for k, o in [(3, 1), (3, 0.5), (3, 0), (1.5, 0), (2, 0), (3.75, 0), (3.75, 1)]:
    l0, g0 = endog(np.array(float(k)))
    l1, g1 = endog(np.array(k - (1 - o) * dmeta))
    print(f"κ={k:<4} ω={o:<4} λ0={float(l0):.2f} gap0={float(g0):.2f}  λ1={float(l1):.2f} gap1={float(g1):.2f}  ΔS={float(selic(4.5,g1)-selic(3,g0)):+.2f}")
