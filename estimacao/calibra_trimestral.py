"""
Calibração trimestral com horizonte constante (interpolação dos anos-calendário do Focus).

Horizonte constante de 36 meses (inflação dos 12 meses que terminam em t+36):
  no mês m do ano Y, esse período cobre (12−m) meses de Y+2 e m meses de Y+3, logo
      π36_t = (12−m)/12 · Focus(Y+2) + m/12 · Focus(Y+3)
  e a meta de mesmo horizonte é interpolada da mesma forma.
Horizonte de 12 meses: Focus 12 meses à frente (suavizada), já com horizonte constante.

Estimações (trimestral = média dos meses do trimestre; janela móvel de 16 trimestres):
  λ_q  = 1 − β,   Δπ36_q = c + β·Δπ12_q
  λ_q  = λ̄ − η·gap̄_(q−4)   (gap̄ = média dos 16 trimestres anteriores, defasada 4 trimestres; EP Newey-West)
  ω    = (b − λ)/(1 − λ), b = repasse no estudo de evento dos anúncios do CMN (por ano de referência)
"""
import numpy as np
import pandas as pd
import importlib.util, sys

spec = importlib.util.spec_from_file_location("cal", "calibra.py")   # reaproveita leitura de dados e metas
src = open("calibra.py").read().split("# ------------------------------------------------------------------ λ_t por janelas móveis")[0]
ns = {}; exec(src, ns)
df, meta_conhecida, anuncios = ns["df"], ns["meta_conhecida"], ns["anuncios"]

# ---------- horizonte constante de 36 meses
m = np.array([t.month for t in df.index])
w2, w3 = (12 - m) / 12, m / 12
df["pi36"] = w2 * df["h2"] + w3 * df["h3"]
df["meta36"] = [ (12 - t.month) / 12 * meta_conhecida(t, t.year + 2) + t.month / 12 * meta_conhecida(t, t.year + 3)
                 for t in df.index]
df["gap36"] = df["pi36"] - df["meta36"]

# ---------- trimestral
q = df[["e12", "pi36", "meta36", "gap36", "ipca"]].copy()
q.index = q.index.asfreq("Q")
q = q.groupby(level=0).mean()
q = q.loc["2002Q1":]
ultimo_completo = pd.Period("2026Q3", "Q")

W = 16
dq = q[["pi36", "e12"]].diff()
lam = pd.Series(np.nan, index=q.index)
for i in range(W, len(q)):
    w = dq.iloc[i - W + 1:i + 1].dropna()
    X = np.c_[np.ones(len(w)), w["e12"]]
    lam.iloc[i] = np.clip(1 - np.linalg.lstsq(X, w["pi36"].values, rcond=None)[0][1], 0, 1)
q["lam"] = lam

# ---------- η e λ̄
q["gapL"] = q["gap36"].rolling(W).mean().shift(4)
reg = q[["lam", "gapL"]].dropna()
reg = reg.loc[:ultimo_completo]
X = np.c_[np.ones(len(reg)), reg["gapL"]]
coef = np.linalg.lstsq(X, reg["lam"].values, rcond=None)[0]
lam_bar, eta = coef[0], -coef[1]
e = reg["lam"].values - X @ coef
def nw_se(X, e, L):
    n = len(e); S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for l in range(1, L + 1):
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None]) / n
        S += (1 - l / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    return np.sqrt(np.diag(Q @ S @ Q / n))
se = nw_se(X, e, L=8)

# ---------- ω: estudo de evento (ano de referência), λ do trimestre anterior ao anúncio
eventos = [
    ("jun/2017: 2019 4,5→4,25", 2019, 4.50, 4.25, "2017-02", "2017-12"),
    ("jun/2017: 2020 4,5→4,0", 2020, 4.50, 4.00, "2017-02", "2017-12"),
    ("jun/2018: 2021 4,0→3,75", 2021, 4.00, 3.75, "2018-05", "2018-12"),
    ("jun/2019: 2022 3,75→3,5", 2022, 3.75, 3.50, "2019-05", "2019-12"),
    ("jun/2020: 2023 3,5→3,25", 2023, 3.50, 3.25, "2020-05", "2020-12"),
    ("jun/2021: 2024 3,25→3,0", 2024, 3.25, 3.00, "2021-05", "2021-12"),
]
def expect(t, Y):
    col = {1: "h1", 2: "h2", 3: "h3", 4: "h4"}.get(Y - t.year)
    return df.at[t, col]
ev = []
for lab, Y, m0, m1, pre, pos in eventos:
    pre, pos = pd.Period(pre, "M"), pd.Period(pos, "M")
    de = expect(pos, Y) - expect(pre, Y)
    lq = q.at[pd.Period(f"{pre.year}Q{(pre.month - 1)//3 + 1}", "Q") - 1, "lam"]
    ev.append(dict(evento=lab, d_meta=m1 - m0, d_exp=round(de, 2), repasse=round(de / (m1 - m0), 2), lam_pre=round(lq, 2)))
ev = pd.DataFrame(ev)
b = (ev.d_exp * ev.d_meta).sum() / (ev.d_meta ** 2).sum()
lam_ev = ev.lam_pre.mean()
omega = (b - lam_ev) / (1 - lam_ev)

# ---------- saída
pd.set_option("display.width", 140)
print(ev.to_string(index=False))
print(f"\nrepasse b = {b:.2f} | λ médio pré-evento = {lam_ev:.2f} | ω = {omega:.2f}")
print(f"λ̄ = {lam_bar:.2f} (EP {se[0]:.2f})  η = {eta:.2f} (EP {se[1]:.2f})  n = {len(reg)}  t(η) = {eta/se[1]:.2f}")
print(q.loc["2025Q1":, ["e12", "pi36", "meta36", "gap36", "lam"]].round(2))
print("\nmédias por período:")
for a, z in [("2002Q1", "2005Q4"), ("2006Q1", "2014Q4"), ("2015Q1", "2016Q4"), ("2017Q1", "2021Q4"), ("2022Q1", "2026Q3")]:
    print(a, z, q.loc[a:z, ["gap36", "lam"]].mean().round(2).to_dict())

q.to_csv("base_trimestral.csv"); ev.to_csv("eventos_trim.csv", index=False)
pd.Series(dict(lam_bar=lam_bar, eta=eta, se_lam_bar=se[0], se_eta=se[1], b=b, lam_ev=lam_ev, omega=omega,
               n=len(reg))).to_csv("parametros_trim.csv")
