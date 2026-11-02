"""Regenerate the README figures. Run from the repository root:  python docs/make_figures.py
Uses only the code in this repository plus a few published numbers (cited below); no third-party datasets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from dga.rules.duval import classify_pct, ZONE_ORDER
from twin.dingwang import DingWangAgeing

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)
COLORS = dict(zip(ZONE_ORDER, ["#8ecae6", "#ffd166", "#f4a261", "#e76f51", "#b5e48c", "#52b788", "#adb5bd"]))

def xy(p_ch4, p_c2h4, p_c2h2):
    return (np.asarray(p_c2h4) + 0.5 * np.asarray(p_ch4)) / 100.0, (np.sqrt(3) / 2) * np.asarray(p_ch4) / 100.0

# ---- 1. Duval triangle with published worked examples (ppm CH4, C2H4, C2H2) ----
EXAMPLES = [((10, 35, 5), "T3"), ((140, 120, 0), "T2"), ((101.2, 158.8, 0), "T3"), ((260, 0, 0), "PD"),
            ((255.2, 63.7, 95.2), "D1"), ((294.5, 0, 55.8), "D1"), ((261.2, 125.8, 75), "D2"), ((32, 32, 36), "D2")]
fig, ax = plt.subplots(figsize=(6.2, 5.6))
pts = [(a, b, classify_pct(100 - a - b, a, b)) for a in np.arange(0, 100.01, 0.5) for b in np.arange(0, 100.01 - a, 0.5)]
for z in ZONE_ORDER:
    s = [p for p in pts if p[2] == z]
    x, y = xy([100 - p[0] - p[1] for p in s], [p[0] for p in s], [p[1] for p in s])
    ax.scatter(x, y, s=9, color=COLORS[z], marker="s", label=z, linewidths=0)
for (ch4, c2h4, c2h2), expected in EXAMPLES:
    t = ch4 + c2h4 + c2h2
    x, y = xy(100 * ch4 / t, 100 * c2h4 / t, 100 * c2h2 / t)
    ax.scatter([x], [y], color="k", marker="x", s=60, zorder=5)
ax.scatter([], [], color="k", marker="x", label="published worked examples")
ax.text(0.5, 0.9, "CH4", ha="center"); ax.text(1.0, -0.04, "C2H4", ha="center"); ax.text(0.0, -0.04, "C2H2", ha="center")
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper right", fontsize=8, markerscale=1.6)
ax.set_title("Duval triangle 1 as implemented, with published worked examples (all classified as in their sources)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "duval_triangle.png", dpi=150); plt.close(fig)

# ---- 2. Method comparison on the 20 real cases with a stated class (results of run_compare.py) ----
methods = ["Duval triangle 1", "IEC ratios", "Random forest"]
all_cases, non_pd = [0.60, 0.50, 0.57], [0.62, 0.56, 0.71]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar(x - w / 2, all_cases, w, label="all 20 cases"); ax.bar(x + w / 2, non_pd, w, label="excluding PD (16)")
for i, (a, b) in enumerate(zip(all_cases, non_pd)):
    ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8); ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
ax.set_title("Real cases: differences are within noise (n = 16 to 20)", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "method_comparison.png", dpi=150); plt.close(fig)

# ---- 3. Conformal coverage: inside the training distribution vs real cases (results of run_uncertainty.py) ----
fig, ax = plt.subplots(figsize=(5.6, 3.6))
vals, labels = [0.89, 0.80], ["inside the\naugmented set", "16 real cases"]
ax.bar(labels, vals, color=["#52b788", "#e76f51"], width=0.5); ax.axhline(0.9, color="k", ls="--", lw=1, label="target 90%")
for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel("empirical coverage"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Conformal sets hold in-distribution, fall short on real cases", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "conformal_coverage.png", dpi=150); plt.close(fig)

# ---- 4. Luo et al. (2023) Table 2 row I: published values vs model with ONE rate fitted on year 1 ----
TABLE2_I = {1: 1167.52, 10: 920.92, 25: 648.27, 40: 486.52, 48: 429.17}
m = DingWangAgeing(0.18); m.advance(np.array([61.29]), 365.0); m.calibrate_to_measurement(TABLE2_I[1])
yrs = np.linspace(0, 50, 101); curve = []
for y in yrs:
    mm = DingWangAgeing(0.18); mm.k_scale = m.k_scale; mm.advance(np.array([61.29]), 365.0 * y); curve.append(mm.dp())
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(yrs, curve, label="model (rate fitted on year 1)"); ax.scatter(list(TABLE2_I), list(TABLE2_I.values()), color="k", zorder=3, label="Luo et al. 2023, Table 2, row I")
ax.axhline(DingWangAgeing(0.18).dp_levelling_off, color="gray", ls="--", label="levelling-off DP (252)")
ax.set_xlabel("operating years"); ax.set_ylabel("DP"); ax.set_ylim(0, 1250); ax.legend(fontsize=8)
ax.set_title("Model form reproduces the published row; the rate is fitted, not predicted", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "luo_table2.png", dpi=150); plt.close(fig)

# ---- 5. Architecture ----
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 3.4)
def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333")); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
box(0.1, 1.3, 1.6, 0.9, "Gas data (ppm)\nH2 CH4 C2H6 C2H4 C2H2\n(CO, CO2, 2-FAL)", "#f1f1f1")
box(2.2, 2.3, 1.9, 0.8, "Duval triangle 1", "#8ecae6"); box(2.2, 1.3, 1.9, 0.8, "IEC 60599 ratios", "#8ecae6"); box(2.2, 0.3, 1.9, 0.8, "Random forest\n(optional, needs data)", "#ffd166")
box(4.6, 0.3, 1.7, 0.8, "Conformal\nprediction sets", "#ffd166"); box(4.6, 1.8, 1.7, 0.9, "Ageing indicators\nCO2/CO, 2-FAL to DP", "#b5e48c")
box(6.8, 0.3, 2.0, 1.4, "Twin core\nhot-spot model\nDP: Ekenstam and\nDing-Wang kinetics\nrecalibrated from\nmeasured DP", "#52b788")
for (x0, y0, x1, y1) in [(1.7, 1.75, 2.2, 2.7), (1.7, 1.75, 2.2, 1.7), (1.7, 1.75, 2.2, 0.7), (4.1, 0.7, 4.6, 0.7), (1.7, 1.9, 4.6, 2.25), (6.3, 2.25, 7.8, 1.7), (6.3, 0.7, 6.8, 0.9)]:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#555"))
ax.set_title("Pipeline overview", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "architecture.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
