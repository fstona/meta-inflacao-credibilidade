import pandas as pd, matplotlib.pyplot as plt
rows = [r.split(",") for r in open("focus_2023_diario.txt").read().strip().split(";")]
df = pd.DataFrame(rows, columns=["d", "y25", "y26", "y27"]).replace("", None)
df.index = pd.to_datetime(df.pop("d"), format="%y%m%d"); df = df.astype(float)
INK, INK2, GR = "#0b0b0b", "#52514e", "#b9b8b2"
fig, ax = plt.subplots(figsize=(11, 4.8))
for c, cor, lb in [("y25", "#86b6ef", "IPCA 2025"), ("y26", "#2a78d6", "IPCA 2026"), ("y27", "#104281", "IPCA 2027")]:
    s = df[c].dropna(); ax.step(s.index, s, where="post", color=cor, lw=2, label=lb)
ax.axhline(3.0, color=GR, lw=0.8, ls="--"); ax.text(pd.Timestamp("2022-11-03"), 3.03, "meta: 3,0%", fontsize=8.5, color=INK2)
ev = [("2023-01-18", "18/01: Lula defende\nmeta de 4,5%"), ("2023-02-09", "09/02: governo estuda\nantecipar debate"),
      ("2023-02-16", "16/02: reunião\ndo CMN"), ("2023-06-29", "29/06: CMN mantém 3%\n(meta contínua)")]
for i, (d, t) in enumerate(ev):
    ax.axvline(pd.Timestamp(d), color=INK2, lw=0.8, ls=":")
    ax.text(pd.Timestamp(d), 4.24, str(i + 1), fontsize=9, color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc=INK2, ec="none"))
ax.set_ylim(2.9, 4.32); ax.set_ylabel("mediana Focus (%)", color=INK2, fontsize=9)
ax.set_title("2023: expectativas de longo prazo durante o debate sobre a meta (Focus, diário)", fontsize=11, loc="left", color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=3)
ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color(GR)
ax.tick_params(colors=INK2, labelsize=8.5)
fig.tight_layout(); fig.savefig("fig_2023.png", dpi=160)
