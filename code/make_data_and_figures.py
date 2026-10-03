"""Build the data tables (data/*.csv) and the figures (figures/*.pdf, *.png) of the paper.

Exact values that required heavy computation (ranks over F2) are copied from results/RESULTS.md;
everything else (d(m), S_m, the maximal-rank baseline B_m) is recomputed here.

    python code/make_data_and_figures.py
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# palette (validated categorical order: blue, orange, aqua) and inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- structure constants
def d_k(k):
    return 2 if k % 3 == 0 else 3


def N(m):
    return 3 * (m - 1) - (m - 1) // 3


def H_set(m):
    s = {k for k in range(1, m) if k % 3}
    r = 0
    while 3 * 2 ** r < m:
        s.add(3 * 2 ** r)
        r += 1
    return s


def d_of(m):
    return sum(H_set(m))


def baseline(m):
    """B_m/|G_m| with B_m = sum_n max(0, h_n - h_{n+3}), h = prod_{k<m} (1+z^k)^{d_k} (as probabilities)."""
    p = np.array([1.0])
    for k in range(1, m):
        for _ in range(d_k(k)):
            q = np.zeros(len(p) + k)
            q[: len(p)] += 0.5 * p
            q[k:] += 0.5 * p
            p = q
    shifted = np.concatenate([p[3:], np.zeros(3)])
    return float(np.clip(p - shifted, 0, None).sum())


# ---------------------------------------------------------------- exact data (results/RESULTS.md)
dimC_exact = {2: 7, 3: 37, 4: 100, 5: 547, 6: 3170, 7: 10226}
graded = {2: 7, 3: 37, 4: 100, 5: 547, 6: 3182, 7: 10226, 8: 64216, 9: 423412, 10: 1526176}
k_m = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 35, 10: 36, 11: 46}
excess = {3: 0.04688, 4: 0.01172, 5: 0.01465, 6: 0.00952, 7: 0.00247, 8: 0.00217, 9: 0.00267, 10: 0.00411}
excess_sigma_abs = {4: 3, 5: 30, 6: 156, 7: 162, 8: 1136, 9: 11200}
excess_generic_abs = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 18}
lift_density = [  # m, element, kernel density in F2[G_m] (exact), graded kernel density
    (5, "lift of v (v^2=0 in gr)", 0.3750, 0.6270), (6, "lift of v (v^2=0 in gr)", 0.2789, 0.6252),
    (7, "lift of v (v^2=0 in gr)", 0.2336, 0.6251),
    (5, "random element of I^4", 0.3320, 0.3320), (6, "random element of I^4", 0.2440, 0.2442),
    (7, "random element of I^4", 0.2034, 0.2035),
    (5, "F", 0.2671, ""), (6, "F", 0.1935, ""), (7, "F", 0.1560, ""),
]

# ---------------------------------------------------------------- CSV tables
with open(os.path.join(DATA, "levels.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "log2_|G_m|", "n_m", "dim_C_m_exact", "graded_cokernel", "c_e_upper_bound",
                "k_m", "d_m", "S_m=m(m-1)/2"])
    for m in range(2, 12):
        n = 3 * 2 ** N(m)
        ex = dimC_exact.get(m, "")
        gr = graded.get(m, "")
        best = ex if ex != "" else gr
        bound = f"{best / n:.6f}" if best != "" else ""
        w.writerow([m, N(m), n, ex, gr, bound, k_m.get(m, ""), d_of(m), m * (m - 1) // 2])

base = {m: baseline(m) for m in range(2, 91)}
with open(os.path.join(DATA, "baseline.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "B_m/|G_m|", "B_m/n_m", "m^1.5*B_m/|G_m|"])
    for m, b in base.items():
        w.writerow([m, f"{b:.8f}", f"{b / 3:.8f}", f"{b * m ** 1.5:.4f}"])

with open(os.path.join(DATA, "excess.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "x_m=dim ker/|G_m|", "e_m (excess density)", "excess_sigma_abs", "excess_generic_abs"])
    for m in range(3, 11):
        x = graded[m] / 2 ** N(m)
        w.writerow([m, f"{x:.5f}", excess[m], excess_sigma_abs.get(m, ""), excess_generic_abs.get(m, "")])

with open(os.path.join(DATA, "lifts.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "element", "kernel_density_F2[G_m]", "graded_kernel_density"])
    w.writerows(lift_density)

with open(os.path.join(DATA, "thresholds.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "d_m", "S_m", "H_m"])
    for m in range(2, 41):
        w.writerow([m, d_of(m), m * (m - 1) // 2, " ".join(map(str, sorted(H_set(m))))])

# ---------------------------------------------------------------- Figure 1: bounds on c_e
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ms = np.arange(2, 91)
ax.plot(ms, [base[m] / 3 for m in ms], color=AQUA, lw=2, zorder=2)
mg = sorted(graded)
ax.plot(mg, [graded[m] / (3 * 2 ** N(m)) for m in mg], color=BLUE, lw=2, marker="o", ms=5,
        mec="white", mew=1, zorder=3)
me = sorted(dimC_exact)
ax.plot(me, [dimC_exact[m] / (3 * 2 ** N(m)) for m in me], ls="none", marker="s", ms=8,
        mfc="none", mec=ORANGE, mew=1.6, zorder=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1.8, 100)
ax.set_ylim(5e-5, 0.5)
ax.set_xlabel("level $m$")
ax.set_ylabel("density (per $n_m = 3|G_m|$)")
ax.set_xticks([2, 3, 5, 10, 20, 50, 90])
ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "90"])
ax.annotate("graded upper bound for $c_e$\n(exact, $m\\leq10$)", xy=(10, graded[10] / (3 * 2 ** 24)),
            xytext=(13, 0.09), color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("$\\dim C_m/n_m$ (exact, $m\\leq7$)", xy=(7, dimC_exact[7] / (3 * 2 ** 16)),
            xytext=(2.1, 0.012), color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("maximal-rank baseline $B_m/n_m\\approx0.85\\,m^{-3/2}$", xy=(40, base[40] / 3),
            xytext=(12, 1.2e-4), color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.text(10.6, 0.031, "$c_e\\leq0.0303$", color=INK, fontsize=9, va="center")
ax.set_title("Upper bounds for the cokernel density $c_e$", loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_ce_bounds.pdf"))
fig.savefig(os.path.join(FIG, "fig1_ce_bounds.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: threshold + odometer
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 5.6), gridspec_kw={"height_ratios": [1.35, 1]})
mm = np.arange(2, 31)
a1.plot(mm, mm * (mm - 1) / 2, color=MUTED, lw=1.5, ls="--", zorder=1)
a1.step(mm, [d_of(m) for m in mm], where="post", color=BLUE, lw=2, zorder=2)
mk = sorted(k_m)
a1.plot(mk, [k_m[m] for m in mk], ls="none", marker="o", ms=6, mfc=ORANGE, mec="white", mew=1, zorder=3)
a1.set_xlim(1.5, 30.5)
a1.set_ylim(0, 450)
a1.set_xlabel("level $m$")
a1.set_ylabel("principal degree")
a1.text(20.3, 345, "$m(m-1)/2$", color=INK2, fontsize=8.5, ha="right")
a1.text(24.5, 185, "$d(m)$ (Theorem C)", color=INK, fontsize=8.5)
a1.text(3, 95, "$k_m$ computed ($m\\leq11$);\n$k_9=35=d(9)-1$", color=INK, fontsize=8.5)
a1.annotate("", xy=(9, 35), xytext=(5.8, 90), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.set_title("(a) First non-injective degree of $R_{\\sigma F}$ on $u_m$", loc="left", fontsize=10, color=INK)

a2.set_xlim(0.3, 24.9)
a2.set_ylim(-1.9, 2.3)
a2.axis("off")
for k in range(1, 25):
    if k % 3:
        fc, ec, tc = BLUE, BLUE, "white"
    elif (k // 3) & (k // 3 - 1) == 0:
        fc, ec, tc = ORANGE, ORANGE, "white"
    else:
        fc, ec, tc = "white", MUTED, MUTED
    a2.add_patch(Rectangle((k - 0.42, -0.4), 0.84, 0.8, fc=fc, ec=ec, lw=1.2))
    a2.text(k, 0, str(k), ha="center", va="center", color=tc, fontsize=8)
for a, b in [(3, 6), (6, 12), (12, 24)]:
    a2.add_patch(FancyArrowPatch((a, 0.45), (b, 0.45), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", mutation_scale=9, color=ORANGE, lw=1.3))
a2.add_patch(Rectangle((1, -1.0), 0.5, 0.32, fc=BLUE, ec=BLUE))
a2.text(1.7, -0.84, "$3\\nmid k$: real column root vectors (square to 0)", va="center", fontsize=8, color=INK)
a2.add_patch(Rectangle((1, -1.6), 0.5, 0.32, fc=ORANGE, ec=ORANGE))
a2.text(1.7, -1.44, "$k=3\\cdot2^r$: imaginary chain, $c_{k}^{[2]}=c_{2k}$ (binary carry)", va="center",
        fontsize=8, color=INK)
a2.add_patch(Rectangle((15.2, -1.0), 0.5, 0.32, fc="white", ec=MUTED))
a2.text(15.9, -0.84, "9, 15, 18, 21: never reached", va="center", fontsize=8, color=INK)
a2.set_title("(b) The degrees $k\\in\\mathcal{H}_m$ (filled); $d(m)$ = sum of filled $k<m$", loc="left",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_threshold_odometer.pdf"))
fig.savefig(os.path.join(FIG, "fig2_threshold_odometer.png"), dpi=200)
plt.close(fig)
print("data and figures written to", DATA, FIG)
