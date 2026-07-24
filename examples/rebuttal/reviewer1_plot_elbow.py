import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(HERE, "results", "cgm_elbow.json")
d = json.load(open(SC))
KS = d["KS"]
panels = [("intraday", "Intraday $H^1$ curves", 2),
          ("correlation", "Correlation matrices $\\mathrm{Sym}(24)$", 3)]

fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2))
for ax, (key, title, ksel) in zip(axes, panels):
    c = np.array(d[key]["curve"])
    ax.plot(KS, c, "o-", color="tab:blue", lw=2, ms=6)
    ax.axvline(ksel, color="tab:red", ls="--", lw=1.5, label=f"selected $K={ksel}$")
    ax.set_xlabel("Number of components $K$")
    ax.set_ylabel("Within-cluster MMD$^2$")
    ax.set_title(f"{title}\n($M={d[key]['M']}$, $n={d[key]['n']}$)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
plt.tight_layout()
out = os.path.join(ROOT, "paper", "images", "rebuttal_cgm_elbow.pdf")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
# print marginal decreases for inspection
for key, title, _ in panels:
    c = np.array(d[key]["curve"])
    dec = -np.diff(c)
    print(title, "curve:", [f"{v:.3e}" for v in c])
    print("   marginal drops:", [f"{v:.3e}" for v in dec])
