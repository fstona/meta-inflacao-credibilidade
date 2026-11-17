"""
Calibração do toy model de meta de inflação com dados do Brasil (2002-2026).
Fontes: Focus/BCB (medianas anuais, média mensal das diárias; expectativa 12m suavizada) e SGS 13522 (IPCA 12m).

Parâmetros estimados
  λ_t  : ancoragem = 1 − sensibilidade das expectativas de longo prazo (t+3) a revisões de curto prazo (12m),
         regressão móvel de 36 meses:  Δπe(t+3) = c + β·Δπe(12m),  λ = 1 − β
  κ_t  : leniência percebida, da identidade do modelo  gap = (1−λ)·κ  ->  κ = gap / (1−λ)
  η, λ̄ : equação de credibilidade  λ_t = λ̄ − η·gap_{t−12}
  ω    : estudo de evento dos anúncios do CMN. Repasse  b = Δπe/Δmeta  =  λ + (1−λ)·ω  ->  ω = (b−λ)/(1−λ)
"""
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ dados
def le(path, cols):
    rows = [r.split(",") for r in open(path).read().strip().split(";")]
    df = pd.DataFrame(rows, columns=cols)
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("ym"), format="%Y%m"), freq="M")
    return df.replace("", np.nan).astype(float)

foc = le("focus.txt", ["ym", "e12", "h1", "h2", "h3", "h4"])
ipca = le("ipca.txt", ["ym", "ipca"])["ipca"]
df = foc.join(ipca, how="left")

# ------------------------------------------------------------------ metas (anúncio CMN -> ano de referência)
anuncios = [  # (mês do anúncio, ano de referência, meta)
    ("2000-06", 2002, 3.50), ("2001-06", 2003, 3.25), ("2002-06", 2003, 4.00), ("2002-06", 2004, 3.75),
    ("2003-01", 2003, 8.50), ("2003-01", 2004, 5.50),            # metas ajustadas (carta aberta jan/2003)
    ("2003-06", 2004, 5.50), ("2003-06", 2005, 4.50),
] + [(f"{y-2}-06", y, 4.50) for y in range(2006, 2019)] + [
    ("2017-06", 2019, 4.25), ("2017-06", 2020, 4.00), ("2018-06", 2021, 3.75), ("2019-06", 2022, 3.50),
    ("2020-06", 2023, 3.25), ("2021-06", 2024, 3.00), ("2022-06", 2025, 3.00), ("2023-06", 2026, 3.00),
]
anuncios = [(pd.Period(m, "M"), y, v) for m, y, v in anuncios]

def meta_conhecida(t, Y):
    """Meta para o ano Y conhecida no mês t. Sem anúncio para Y: extrapola a última meta anunciada
    (a partir de 2023-06, meta contínua de 3%)."""
    cand = [(m, y, v) for m, y, v in anuncios if m <= t]
    exato = [(m, v) for m, y, v in cand if y == Y]
    if exato:
        return max(exato)[1]
    if t >= pd.Period("2023-06", "M"):
        return 3.0
    return max(cand, key=lambda x: (x[1], x[0]))[2]

df["meta_h3"] = [meta_conhecida(t, t.year + 3) for t in df.index]
df["meta12"] = [0.5 * meta_conhecida(t, t.year) + 0.5 * meta_conhecida(t, t.year + 1) for t in df.index]
df["gap"] = df["h3"] - df["meta_h3"]                  # desancoragem de longo prazo

# ------------------------------------------------------------------ λ_t por janelas móveis
d = df[["h3", "e12"]].diff()
W = 36
lam = pd.Series(np.nan, index=df.index)
for i in range(W, len(df)):
    w = d.iloc[i - W + 1:i + 1].dropna()
    X = np.c_[np.ones(len(w)), w["e12"]]
    b = np.linalg.lstsq(X, w["h3"].values, rcond=None)[0][1]
    lam.iloc[i] = np.clip(1 - b, 0, 1)
df["lam"] = lam
df["gap_suave"] = df["gap"].rolling(W).mean()
df["kappa"] = df["gap_suave"] / (1 - df["lam"]).clip(lower=0.05)

# ------------------------------------------------------------------ η e λ̄:  λ_t = λ̄ − η·gap_{t−12}
reg = pd.concat([df["lam"], df["gap_suave"].shift(12)], axis=1, keys=["lam", "gapL"]).dropna()
X = np.c_[np.ones(len(reg)), reg["gapL"]]
coef, res, *_ = np.linalg.lstsq(X, reg["lam"].values, rcond=None)
lam_bar, eta = coef[0], -coef[1]
e = reg["lam"].values - X @ coef
# erro-padrão HAC (Newey-West, 12 lags) — janelas móveis geram autocorrelação forte
def nw_se(X, e, L=12):
    n = len(e); S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for l in range(1, L + 1):
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None]) / n
        S += (1 - l / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    return np.sqrt(np.diag(Q @ S @ Q / n))
se = nw_se(X, e)

# ------------------------------------------------------------------ ω: estudo de evento
def expect(t, Y):
    h = Y - t.year
    col = {1: "h1", 2: "h2", 3: "h3", 4: "h4"}.get(h)
    return df.at[t, col] if col else np.nan

eventos = [  # (rótulo, ano ref., meta anterior, meta nova, mês pré, mês pós)
    ("jan/2003: 2004 3,75→5,5", 2004, 3.75, 5.50, "2002-12", "2003-06"),
    ("jun/2017: 2019 4,5→4,25", 2019, 4.50, 4.25, "2017-02", "2017-12"),
    ("jun/2017: 2020 4,5→4,0", 2020, 4.50, 4.00, "2017-02", "2017-12"),
    ("jun/2018: 2021 4,0→3,75", 2021, 4.00, 3.75, "2018-05", "2018-12"),
    ("jun/2019: 2022 3,75→3,5", 2022, 3.75, 3.50, "2019-05", "2019-12"),
    ("jun/2020: 2023 3,5→3,25", 2023, 3.50, 3.25, "2020-05", "2020-12"),
    ("jun/2021: 2024 3,25→3,0", 2024, 3.25, 3.00, "2021-05", "2021-12"),
]
ev = []
for lab, Y, m0, m1, pre, pos in eventos:
    pre, pos = pd.Period(pre, "M"), pd.Period(pos, "M")
    de = expect(pos, Y) - expect(pre, Y)
    l_ev = df.at[pre, "lam"]
    ev.append(dict(evento=lab, d_meta=m1 - m0, d_exp=round(de, 2), repasse=round(de / (m1 - m0), 2),
                   lam_pre=round(l_ev, 2), gap_pre=round(expect(pre, Y) - m0, 2)))
ev = pd.DataFrame(ev)
pos_2017 = ev.iloc[1:]
b_all = (ev.d_exp * ev.d_meta).sum() / (ev.d_meta ** 2).sum()
b_17 = (pos_2017.d_exp * pos_2017.d_meta).sum() / (pos_2017.d_meta ** 2).sum()
lam_17 = pos_2017.lam_pre.mean()
omega_17 = (b_17 - lam_17) / (1 - lam_17)

# ------------------------------------------------------------------ episódio 2023: ameaça à meta, sem mudança
amea = df.loc["2022-12":"2023-12", ["h3", "h4", "meta_h3"]]

# ------------------------------------------------------------------ saída
pd.set_option("display.width", 140)
print("=== Estudo de evento (Δ expectativa para o ano cuja meta mudou) ===")
print(ev.to_string(index=False))
print(f"\nRepasse médio (MQO na origem): todos = {b_all:.2f}; 2017-2021 = {b_17:.2f}")
print(f"λ médio antes dos eventos 2017-21 = {lam_17:.2f}  ->  ω = (b − λ)/(1 − λ) = {omega_17:.2f}")
print(f"\n=== Credibilidade endógena: λ_t = λ̄ − η·gap_(t−12)  (n = {len(reg)}) ===")
print(f"λ̄ = {lam_bar:.2f} (EP {se[0]:.2f})   η = {eta:.2f} (EP {se[1]:.2f})")
print("\n=== Médias por período ===")
per = {"2002-05": ("2002-01", "2005-12"), "2006-14": ("2006-01", "2014-12"), "2015-16": ("2015-01", "2016-12"),
       "2017-21": ("2017-01", "2021-12"), "2022-26": ("2022-01", "2026-10")}
print(pd.DataFrame({k: df.loc[a:b, ["gap", "lam", "kappa"]].mean() for k, (a, b) in per.items()}).T.round(2))
print("\n=== Hoje ===")
print(df.loc["2026-06":, ["e12", "h3", "h4", "meta_h3", "gap", "lam", "kappa"]].round(2))
print("\n=== 2023: debate sobre mudar a meta (meta de 3% mantida em jun/2023) ===")
print(amea.round(2).T)

df.to_csv("base_calibracao.csv")
ev.to_csv("eventos.csv", index=False)
pd.Series(dict(lam_bar=lam_bar, eta=eta, se_lam_bar=se[0], se_eta=se[1], b_all=b_all, b_17=b_17,
               lam_17=lam_17, omega=omega_17)).to_csv("parametros.csv")
