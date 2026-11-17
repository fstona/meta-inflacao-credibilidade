"""Simulação meta 3 -> 4,5 com parâmetros trimestrais (horizonte de 36 meses). Base: 2026T3."""
import numpy as np, pandas as pd
q = pd.read_csv("base_trimestral.csv", index_col=0); p = pd.read_csv("parametros_trim.csv", index_col=0).squeeze()
theta, rstar, d = 0.5, 5.0, 1.5
def run(base):
    lam0, gap0 = q.at[base, "lam"], q.at[base, "gap36"]
    out = []
    # λ exógena
    kx = gap0 / (1 - lam0); i0 = rstar + theta * gap0 + 3 + gap0
    for om_lab, om in [("ω estimado", p.omega), ("ω = 0", 0.0)]:
        for g in (0.0, 0.05, 0.10):
            l1 = max(lam0 - g * d, 0); g1 = (1 - l1) * (kx - (1 - om) * d)
            out.append(("exógena", om_lab, g, round(l1, 2), round(4.5 + g1, 2), round(rstar + theta * g1 + 4.5 + g1 - i0, 2)))
    # λ endógena
    lb, eta = p.lam_bar, p.eta
    kn = gap0 / ((1 - lb) + eta * gap0)
    def solve(dd, g, om):
        K = kn - (1 - om) * dd; den = 1 - eta * K
        gap = (1 - lb + g * dd) * K / den if den > 0 else np.inf
        lam = lb - eta * gap - g * dd
        if not np.isfinite(gap) or lam < 0: lam, gap = 0.0, K
        return lam, (1 - lam) * K
    l0n, g0n = solve(0, 0, 1); i0n = rstar + theta * g0n + 3 + g0n
    for om_lab, om in [("ω estimado", p.omega), ("ω = 0", 0.0)]:
        for g in (0.0, 0.05, 0.10):
            l1, g1 = solve(d, g, om)
            out.append(("endógena", om_lab, g, round(l1, 2), round(4.5 + g1, 2), round(rstar + theta * g1 + 4.5 + g1 - i0n, 2)))
    info = dict(base=base, lam0=round(lam0, 2), gap0=round(gap0, 2), kappa_exo=round(kx, 2), kappa_endo=round(kn, 2),
                lam0_endo=round(l0n, 2), mult=round(1 / (1 - eta * kn), 2))
    return info, pd.DataFrame(out, columns=["versao", "omega", "gamma", "lam1", "pe1", "dSelic"])
for b in ["2026Q3", "2026Q4"]:
    info, res = run(b); print(info); print(res.to_string(index=False)); res.to_csv(f"resultado_trim_{b}.csv", index=False)
