"""
Toy model com credibilidade endógena à desancoragem.
Estado estacionário (sem dinâmica). Taxas em % a.a.

Sistema
  (1) Expectativas:   pe  = lam*meta + (1-lam)*piL
  (2) Tolerância:     piL = meta0 + kappa + omega*(meta - meta0)
  (3) Credibilidade:  lam = lam_bar - eta*(pe - meta) - gamma*(meta - meta0)     <- ENDÓGENA
  (4) Juro real:      r   = rstar + theta*(pe - meta)
  (5) Fisher:         i   = r + pe

Solução fechada (ponto fixo de 1+3), com K = kappa - (1-omega)*Δmeta:
  gap = (1 - lam_bar + gamma*Δmeta) * K / (1 - eta*K)
  -> multiplicador 1/(1 - eta*K): desancoragem corrói credibilidade, que aumenta a desancoragem.
  -> se eta*K >= 1: não há âncora (lam = 0, gap = K).

Calibração: lam_bar = 0,8 (credibilidade "plena" possível), kappa = 3,
  eta escolhido para reproduzir hoje: meta 3, pe 4,5 -> eta = 0,2, lam hoje = 0,5.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

meta0, kappa, theta, rstar, lam_bar = 3.0, 3.0, 0.5, 5.0, 0.8
gap0 = 1.5
eta = (1 - (1 - lam_bar) * kappa / gap0) / kappa      # = 0,2

def solve(meta, gamma=0.0, omega=1.0, endog=True, lam_exo=0.5):
    d = meta - meta0
    K = kappa - (1 - omega) * d
    if endog:
        den = 1 - eta * K
        gap = (1 - lam_bar + gamma * d) * K / den if den > 0 else np.inf
        lam = lam_bar - eta * gap - gamma * d
        if not np.isfinite(gap) or lam < 0:
            lam, gap = 0.0, K
        lam = min(lam, 1.0)
        gap = (1 - lam) * K
    else:
        lam = max(lam_exo - gamma * d, 0.0)
        gap = (1 - lam) * K
    pe = meta + gap
    i = rstar + theta * gap + pe
    return dict(lam=lam, gap=gap, pe=pe, i=i)

hoje = solve(meta0)
print(f"eta = {eta:.2f} | hoje: lam {hoje['lam']:.2f}, pe {hoje['pe']:.2f}, gap {hoje['gap']:.2f}, Selic {hoje['i']:.2f}")
print(f"multiplicador 1/(1-eta*kappa) = {1/(1-eta*kappa):.2f}  (sem feedback a desancoragem seria {(1-lam_bar)*kappa:.2f})\n")

print(f"{'omega':>5s} {'gamma':>5s} | {'λ exóg.':>8s} {'ΔSelic':>7s} | {'λ endóg.':>8s} {'gap':>5s} {'ΔSelic':>7s}")
for omega in (1.0, 0.5, 0.0):
    for gamma in (0.0, 0.05, 0.1):
        ex = solve(4.5, gamma, omega, endog=False)
        en = solve(4.5, gamma, omega, endog=True)
        print(f"{omega:5.1f} {gamma:5.2f} | {ex['lam']:8.2f} {ex['i']-hoje['i']:+7.2f} | {en['lam']:8.2f} {en['gap']:5.2f} {en['i']-hoje['i']:+7.2f}")

# ---------- figura
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw=dict(width_ratios=[1, 1.1]))

# painel 1: desancoragem vs nova meta, exógeno x endógeno (omega=1, gamma=0,05)
metas = np.linspace(3, 6, 61)
g = 0.05
for endog, c, lb in [(False, "#86b6ef", "λ exógena"), (True, "#104281", "λ endógena à desancoragem")]:
    axs[0].plot(metas, [solve(m, g, 1.0, endog)["gap"] for m in metas], color=c, lw=2, label=lb)
axs[0].plot(meta0, gap0, "o", ms=8, color="#eb6834", mec="white", mew=2, zorder=4)
axs[0].text(3.1, gap0 + 0.06, "hoje", fontsize=9, color="#0b0b0b")
axs[0].set_title("Desancoragem (expectativa − meta, p.p.)\nω = 1, γ = 0,05", fontsize=11, loc="left", color="#0b0b0b")
axs[0].set_xlabel("Nova meta de inflação (% a.a.)", color="#52514e", fontsize=9)
axs[0].legend(frameon=False, fontsize=9.5, loc="upper left")
axs[0].grid(axis="y", color="#e6e5e0", lw=0.8)
for sp in ["top", "right"]:
    axs[0].spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    axs[0].spines[sp].set_color("#b9b8b2")
axs[0].tick_params(colors="#52514e", labelsize=9)

# painel 2: mapa de ΔSelic (meta 3 -> 4,5) com λ endógena
omegas = np.linspace(0, 1, 5)
gammas = np.linspace(0, 0.1, 5)
Z = np.array([[solve(4.5, gm, om)["i"] - hoje["i"] for gm in gammas] for om in omegas])
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
lim = np.abs(Z).max()
im = axs[1].imshow(Z, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), origin="lower", aspect="auto")
for a in range(len(omegas)):
    for b in range(len(gammas)):
        axs[1].text(b, a, f"{Z[a, b]:+.2f}".replace(".", ","), ha="center", va="center", fontsize=9.5, color="#0b0b0b")
axs[1].set_xticks(range(len(gammas)), [f"{x:.3f}".rstrip("0").rstrip(".").replace(".", ",") for x in gammas])
axs[1].set_yticks(range(len(omegas)), [f"{x:.2f}".replace(".", ",") for x in omegas])
axs[1].set_xlabel("γ: perda de credibilidade por p.p. de meta", color="#52514e", fontsize=9)
axs[1].set_ylabel("ω: quanto a tolerância anda com a meta", color="#52514e", fontsize=9)
axs[1].set_title("Δ Selic de equilíbrio (p.p.), meta 3 → 4,5, λ endógena\nazul = Selic cai, vermelho = Selic sobe",
                 fontsize=11, loc="left", color="#0b0b0b")
for sp in axs[1].spines.values():
    sp.set_visible(False)
axs[1].tick_params(colors="#52514e", labelsize=9, length=0)

fig.suptitle("Com credibilidade endógena, a desancoragem se autoalimenta", fontsize=13, x=0.01, ha="left", color="#0b0b0b")
fig.text(0.01, 0.005, "λ = 0,8 − 0,2·(pe − meta) − γ·Δmeta;  pe = meta + (1−λ)·(κ − (1−ω)·Δmeta), κ = 3;  i = 5 + 0,5·(pe − meta) + pe.  Ilustrativo.",
         fontsize=8, color="#52514e")
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig("toy_lambda_endogeno.png", dpi=160)
