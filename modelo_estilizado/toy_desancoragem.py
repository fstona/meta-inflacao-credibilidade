"""
Toy model: mudar a meta quando as expectativas já estão desancoradas.
Estado estacionário (sem dinâmica). Taxas em % a.a.

Sistema de equações
  (1) Expectativas:   pe   = lam*meta + (1-lam)*piL
  (2) Tolerância:     piL  = meta0 + kappa + omega*(meta - meta0)
  (3) Credibilidade:  lam  = max(lam0 - gamma*(meta - meta0), 0)
  (4) Prêmio de risco: r   = rstar + theta*(pe - meta)
  (5) Fisher:         i    = r + pe

Parâmetros
  lam0  credibilidade inicial                       0,5
  kappa leniência percebida (p.p. acima da meta)    3,0   -> hoje pe = 4,5 com meta 3
  omega quanto da tolerância "anda" com a meta      1 = relativa à meta; 0 = nível fixo
  gamma perda de credibilidade por p.p. de meta     sensibilidade testada
  theta prêmio de risco por p.p. de desancoragem    0,5
  rstar juro real neutro                            5,0
"""
import numpy as np
import matplotlib.pyplot as plt

meta0, lam0, kappa, theta, rstar = 3.0, 0.5, 3.0, 0.5, 5.0

def modelo(meta, gamma=0.0, omega=1.0):
    d = meta - meta0
    lam = np.maximum(lam0 - gamma * d, 0.0)
    piL = meta0 + kappa + omega * d
    pe = lam * meta + (1 - lam) * piL
    gap = pe - meta
    r = rstar + theta * gap
    i = r + pe
    return dict(lam=lam, pe=pe, gap=gap, r=r, i=i)

hoje = modelo(np.array(meta0))
print(f"Hoje: meta 3,0 | pe {hoje['pe']:.2f} | desancoragem {hoje['gap']:.2f} | juro real {hoje['r']:.2f} | Selic {hoje['i']:.2f}\n")

# --- tabela de sensibilidade: meta 3 -> 4,5
print(f"{'omega':>6s} {'gamma':>6s} {'lambda':>7s} {'pe':>6s} {'desanc.':>8s} {'Selic':>6s} {'ΔSelic':>7s}")
for omega in (1.0, 0.5, 0.0):
    for gamma in (0.0, 0.1, 0.2):
        s = modelo(np.array(4.5), gamma, omega)
        print(f"{omega:6.1f} {gamma:6.1f} {float(s['lam']):7.2f} {float(s['pe']):6.2f} {float(s['gap']):8.2f} "
              f"{float(s['i']):6.2f} {float(s['i'] - hoje['i']):+7.2f}")

# --- gráfico: sensibilidade a gamma (omega = 1)
metas = np.linspace(3.0, 6.0, 61)
gammas = [0.0, 0.1, 0.2]
cores = ["#86b6ef", "#2a78d6", "#104281"]   # rampa sequencial: gamma maior = mais escuro
rotulos = ["γ = 0 (credibilidade não muda)", "γ = 0,1", "γ = 0,2 (perda forte)"]

fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
paineis = [("pe", "Expectativa de inflação (% a.a.)"),
           ("gap", "Desancoragem: expectativa − meta (p.p.)"),
           ("i", "Selic de equilíbrio (% a.a.)")]
for ax, (v, tit) in zip(axs, paineis):
    if v == "pe":
        ax.plot(metas, metas, color="#b9b8b2", lw=1, ls="--")
        ax.text(5.95, 5.75, "expectativa = meta", ha="right", fontsize=8.5, color="#52514e")
    for g, c, lb in zip(gammas, cores, rotulos):
        ax.plot(metas, modelo(metas, g)[v], color=c, lw=2, label=lb)
    ax.axvline(4.5, color="#b9b8b2", lw=0.8, ls=":")
    ax.plot(meta0, float(hoje[v]), "o", ms=8, color="#eb6834", mec="white", mew=2, zorder=4)
    ax.set_title(tit, fontsize=11, loc="left", color="#0b0b0b")
    ax.set_xlabel("Nova meta de inflação (% a.a.)", color="#52514e", fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#b9b8b2")
    ax.tick_params(colors="#52514e", labelsize=9)
axs[0].text(3.1, float(hoje["pe"]) - 0.35, "hoje", fontsize=9, color="#0b0b0b")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_[-3:], l_[-3:], frameon=False, fontsize=9.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.9))
fig.suptitle("Subir a meta com expectativas desancoradas: a desancoragem não some, e a Selic sobe",
             fontsize=13, x=0.01, ha="left", color="#0b0b0b")
fig.text(0.01, 0.005, "pe = λ·meta + (1−λ)·(meta + κ);  λ = 0,5 − γ·Δmeta;  κ = 3;  i = r* + θ·(pe − meta) + pe, r* = 5, θ = 0,5.  Estado estacionário, ilustrativo.",
         fontsize=8, color="#52514e")
fig.tight_layout(rect=[0, 0.03, 1, 0.84])
fig.savefig("toy_desancoragem.png", dpi=160)
