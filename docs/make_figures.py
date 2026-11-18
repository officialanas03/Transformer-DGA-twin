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
# ---- 6. Luo et al. Table 2, all three rows: only row I is fitted ----
from twin import report as rp, luo_chain as lc
frame = rp.luo_table2_frame()
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ks = lc.fit_k_scale(rp._paper_row(67)[1]); yy = list(range(1, 49))
for (label, var, ol, first, status), col in zip(rp.CASES, ["#1d3557", "#e76f51", "#f4a261", "#2a9d8f", "#52b788"]):
    ax.plot(yy, [lc.table2_row(yy, var, ks, overload_hours=ol)[y] for y in yy], color=col, lw=1.4, ls="-" if var == "none" else "--", label="model " + label)
for first, mk, name in [(67, "o", "paper I"), (72, "s", "paper II"), (77, "^", "paper III")]:
    ax.scatter(rp.YEARS, [rp._paper_row(first)[y] for y in rp.YEARS], color="k", marker=mk, zorder=4, s=28, label=name)
ax.set_xlabel("operating years"); ax.set_ylabel("DP at the hot spot"); ax.set_ylim(200, 1250); ax.legend(fontsize=6, ncol=2)
ax.set_title("Table 2: row I is fitted; rows II and III use ASSUMED moisture and do not match (printed Eq. 11 cannot reproduce them)", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "luo_rows_variants.png", dpi=150); plt.close(fig)

# ---- 7. Temperature -> moisture -> DP maps from the ANSYS exports (ASSUMED geometry), smoothed by Delaunay surface rebuild ----
from twin import ansys_io as ai, pipeline_dp as pdp, surface as sf
EXD = Path(__file__).resolve().parents[1] / "examples" / "ansys"
snap, snap13 = ai.read_snapshot(EXD / "T_load0p74.csv"), ai.read_snapshot(EXD / "T_load1p30.csv")
P = snap[["x_m", "y_m"]].to_numpy(); TRI = sf.mask_long_edges(P, sf.delaunay(P), 0.02)
def draw(a, z, title, cmap, vmin=None, vmax=None):
    gx, gy, Z = sf.resample(P[:, 0], P[:, 1], z, TRI, 0.002)
    im = a.imshow(Z, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    a.set_title(title, fontsize=8); a.set_xlabel("radius (m)"); a.set_ylabel("height (m)")
    return im
m48 = pdp.dp_map_with_moisture(snap, 48)
fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.4))
for a, z, ttl, cm in [(axs[0], snap.T_C.to_numpy(), "temperature (C), load 0.74", "inferno"),
                      (axs[1], m48.C_pct.to_numpy(), "moisture (%), year 48", "Blues"),
                      (axs[2], m48.DP.to_numpy(), "DP, year 48 (moisture variant)", "viridis")]:
    fig.colorbar(draw(a, z, ttl, cm), ax=a, shrink=0.8)
fig.suptitle("ANSYS export, ASSUMED geometry, film coefficient fitted (LV left, HV right). Pipeline demonstration, not a finding. "
             "Surface rebuilt by Delaunay triangulation", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "ansys_dp_map.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(12, 4.2), sharey=True)
for a, yr in zip(axs, [1, 10, 25, 40, 48]):
    m = pdp.dp_map_with_moisture(snap, yr, df_overload=snap13, overload_hours=24)
    im = draw(a, m.DP.to_numpy(), f"year {yr}", "turbo", 200, 1200)
axs[0].set_ylabel("height (m)")
for a in axs[1:]: a.set_ylabel("")
fig.colorbar(im, ax=axs, shrink=0.8, label="DP")
fig.suptitle("DP maps, case III-like (24 h per year at load 1.3), same assumed model; compare the layout of Luo et al. Fig. 13", fontsize=8)
fig.savefig(OUT / "dp_maps_years.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 8. SYNTHETIC live calibration: works with a correct model, hides a wrong one ----
import numpy as np
from twin.surrogate import WindingSurrogate
L, HFS = np.linspace(0.3, 1.4, 12), np.linspace(50, 110, 8)
filt = WindingSurrogate(nz=60).fit_from_model(L, HFS)
truth = WindingSurrogate(nz=120, nr_lv=14, nr_hv=18).fit_from_model(L, HFS)
wrong = WindingSurrogate(nz=60).fit_from_model(L, HFS, t_top=47.0)
H = 24 * 60
a_ = rp.calibration_demo(filt, truth, np.linspace(72, 55, H), hours=H, qx=1e-3)
b_ = rp.calibration_demo(filt, wrong, 72.0, hours=H, qx=1e-3)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for a, d, ttl in [(axs[0], a_, "model correct, film coefficient drifting"), (axs[1], b_, "model WRONG (oil 7 K warmer than assumed)")]:
    a.plot(d["hf_true"], "k", label="true (hidden)"); a.plot(d["hf_est"], color="#e76f51", label="filter estimate")
    a.set_title(f"{ttl}\nhot-spot RMSE: filter {d['rmse_filter']:.2f} K, sensor noise {d['rmse_sensor']:.2f} K; error at load 1.3: {d['overload_error']:+.1f} K", fontsize=7)
    a.set_xlabel("hour")
axs[0].set_ylabel("film coefficient (W/m2K)"); axs[0].legend(fontsize=8)
fig.suptitle("SYNTHETIC data (invented load, drift and noise): calibration is not validation", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "kalman_synthetic.png", dpi=150); plt.close(fig)
print(f"fig8: drift case filter {a_['rmse_filter']:.2f} K vs uncalibrated {a_['rmse_uncal']:.2f} K; wrong-model case: hf est {b_['hf_est'][-1]:.1f} vs 72, overload error {b_['overload_error']:+.1f} K")
# ---- 9. Sensitivity of the LV DP profile (year 48) to the ASSUMED thermal model ----
from twin import thermal_variants as tv
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for name, tt, qf, col in [("A  oil 20-40 C, equal loss density", 40, 1.0, "#999999"), ("B  oil 20-40 C, HV x0.6", 40, 0.6, "#1d3557"),
                          ("C  oil 20-50 C, HV x0.6", 50, 0.6, "#e76f51"), ("D  oil 20-60 C, HV x0.6", 60, 0.6, "#2a9d8f")]:
    v = tv.variant(tt, qf); m = ai.winding(pdp.dp_map_with_moisture(v["base"], 48)); lv = m[m.winding == "LV"]
    g = lv.groupby(lv.y_m.round(4)).DP.min()
    ax.plot(g.values, g.index / 1.2, color=col, label=f"{name} (1.3 pu: {v['lv_overload']:.0f} C)")
ax.scatter([252.15, 676.33, 927.5], [0.9, 0.5, 0.1], color="k", zorder=5, label="paper Table 3, LV (1.3 pu: 112.84 C)")
ax.set_xlabel("DP at year 48 (moisture-field variant)"); ax.set_ylabel("relative height of LV winding"); ax.legend(fontsize=6.5)
ax.set_title("A wider temperature range fits Table 3 better but breaks the 1.3 pu check; all thermal models are placeholders", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "thermal_sensitivity.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
