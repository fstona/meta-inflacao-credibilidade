"""Calibração de γ pelo episódio de 2023 (Focus diário, IPCA 2026 e 2027).
Identificação:
  pico − base       = p·D·b + E        (E = desancoragem extra, perda de credibilidade)
  queda na decisão  = p·D·b            (supõe que E não se reverte quando a meta é mantida)
  pós − base        = E
  e = E/(p·D)  : desancoragem extra por p.p. de mudança esperada da meta
γ é escolhido para que o modelo (λ endógena, base 2026T3) gere a mesma desancoragem extra por p.p. de mudança."""
import numpy as np, pandas as pd
p = pd.read_csv("parametros_trim.csv", index_col=0).squeeze(); q = pd.read_csv("base_trimestral.csv", index_col=0)
D, b = 1.5, p.b
casos = {  # série: (base, pico, véspera da decisão (29/06), pós)
    "IPCA 2027 (base 18/01/23)": (3.00, 4.00, 3.60, 3.50),
    "IPCA 2026 (base 18/01/23)": (3.25, 4.00, 3.70, 3.50),
    "IPCA 2026 (base dez/22)":   (3.00, 4.00, 3.70, 3.50),
}
lb, eta, om = p.lam_bar, p.eta, p.omega
gap0 = q.at["2026Q3", "gap36"]; kn = gap0 / ((1 - lb) + eta * gap0)
def gap(g):
    K = kn - (1 - om) * D; den = 1 - eta * K
    gp = (1 - lb + g * D) * K / den
    lam = lb - eta * gp - g * D
    if lam < 0: lam, gp = 0.0, K
    return (1 - lam) * K, lam
g_ref = gap(0)[0]
rows = []
for nome, (base, pico, vesp, pos) in casos.items():
    queda_total = pico - pos                  # inclui sinais antes de 29/06 (7 e 16/06)
    pD = queda_total / b; prob = pD / D
    E = pos - base; e = E / pD
    gs = np.linspace(0, 0.6, 6001)
    extra = np.array([gap(g)[0] - g_ref for g in gs]) / D
    g_hat = gs[np.argmin(np.abs(extra - e))]
    rows.append(dict(serie=nome, subida=pico - base, queda=queda_total, prob=round(prob, 2), E=E, e=round(e, 2),
                     gamma=round(g_hat, 3), lam1=round(gap(g_hat)[1], 2)))
print(pd.DataFrame(rows).to_string(index=False))
