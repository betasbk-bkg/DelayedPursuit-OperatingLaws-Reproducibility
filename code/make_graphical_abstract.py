# -*- coding: utf-8 -*-
r"""
The paper's graphical abstract.

Every number in the figure is written once, here, and the bar lengths are proportional
to the margins they represent, so the drawing cannot drift from the values the paper
states.

Usage:  python make_graphical_abstract_2026-09-23.py
Writes: figures/graphical_abstract.png  (1600 x 800 px, the 2:1 shape the journal asks for)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import FancyBboxPatch                     # noqa: E402
import numpy as np                                                # noqa: E402

U_HEAD, U_CURV = 0.4984, 0.3078          # d131, converged
U_QUANT = 0.7475                         # quantized operating margin (d127)
U_NAV2 = 0.2949                          # measured on Nav2, symmetric-window locator (d56/d102)
BAND = "2–8%"                       # locator spread against the derived constant
REGRET = "0.7–2.0%"                 # median tuning regret, interpolated / bowl fit (d132)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphical_abstract.png")
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(16, 8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(8, 7.25, "Two regimes, one operating-speed question",
        ha="center", va="center", fontsize=30, fontweight="bold")

PANELS = [(0.45, "Continuous delayed pursuit"),
          (5.55, "Quantized command regime"),
          (10.65, "Independent controller transfer")]
for x0, head in PANELS:
    ax.add_patch(FancyBboxPatch((x0, 1.15), 4.9, 5.35, boxstyle="round,pad=0.12,rounding_size=0.22",
                                linewidth=1.6, edgecolor="#444444", facecolor="#f7f7f7"))
    ax.text(x0 + 2.45, 6.05, head, ha="center", va="center", fontsize=19, fontweight="bold")

# ---------------------------------------------------------------- panel 1: the continuous law
x0 = 0.45
ax.text(x0 + 2.45, 5.35, r"$v^*\,\tau_{\mathrm{tot}} = u^*\,L$", ha="center", va="center", fontsize=25)
ax.text(x0 + 2.45, 4.75, "heading    $u^*$ = %.4f" % U_HEAD, ha="center", va="center", fontsize=18)
ax.text(x0 + 2.45, 4.28, "curvature  $u^*$ = %.4f" % U_CURV, ha="center", va="center", fontsize=18)
bx = fig.add_axes([0.085, 0.255, 0.20, 0.19])
u = np.linspace(-1.0, 1.0, 200)
bx.plot(u, 0.45 + u ** 2, color="black", lw=2.2)
bx.plot([0], [0.45], "o", color="black", ms=9)
bx.set_xlim(-1.15, 1.15)
bx.set_ylim(0.0, 1.7)
bx.set_xticks([])
bx.set_yticks([])
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
bx.patch.set_alpha(0)
ax.text(x0 + 2.45, 1.48, "finite tracking-error minimum", ha="center", va="center", fontsize=15)

# ---------------------------------------------------------------- panel 2: the quantized chain
x0 = 5.55
chain = [(x0 + 0.75, "$W_n$"), (x0 + 2.45, r"$\sigma_\theta(u)$"), (x0 + 4.15, "$u^*$ = %.4f" % U_QUANT)]
for cx, label in chain:
    ax.add_patch(FancyBboxPatch((cx - 0.72, 4.72), 1.44, 0.62, boxstyle="round,pad=0.06,rounding_size=0.12",
                                linewidth=1.4, edgecolor="#444444", facecolor="white"))
    ax.text(cx, 5.03, label, ha="center", va="center", fontsize=16)
for cx in (x0 + 0.75, x0 + 2.45):
    ax.annotate("", xy=(cx + 0.95, 5.03), xytext=(cx + 0.75, 5.03),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8))
t = np.linspace(0, 2 * np.pi, 400)
ox, oy = x0 + 2.45 + 1.55 * np.cos(t) * (1 + 0.16 * np.sin(t)), 3.15 + 0.95 * np.sin(t)
ax.plot(ox, oy, color="black", lw=2.0)
for k in (0.9, 2.6, 4.9):
    ax.plot([x0 + 2.45 + 1.55 * np.cos(k) * (1 + 0.16 * np.sin(k))], [3.15 + 0.95 * np.sin(k)],
            "o", color="black", ms=9)
ax.text(x0 + 2.45, 1.52, "periodic orbit → error statistics → speed",
        ha="center", va="center", fontsize=15)

# ---------------------------------------------------------------- panel 3: the third-party transfer
x0 = 10.65
ax.text(x0 + 2.45, 5.40, "Nav2 pure pursuit", ha="center", va="center", fontsize=19)
ax.text(x0 + 2.45, 4.78, "measured $u^*$ = %.4f" % U_NAV2, ha="center", va="center", fontsize=23)
ax.text(x0 + 2.45, 4.28, "prediction $u^*$ = %.4f" % U_CURV, ha="center", va="center", fontsize=19)
BAR_X, BAR_W = x0 + 0.65, 3.6
for y, label, value, shade in ((3.42, "measured", U_NAV2, "#555555"),
                               (2.50, "predicted", U_CURV, "#cccccc")):
    ax.text(BAR_X, y + 0.36, label, ha="left", va="center", fontsize=15)
    ax.add_patch(plt.Rectangle((BAR_X, y - 0.02), BAR_W * value / max(U_NAV2, U_CURV), 0.30,
                               facecolor=shade, edgecolor="#333333", linewidth=1.2))
ax.text(x0 + 2.45, 1.85, "%s agreement across estimators" % BAND, ha="center", va="center", fontsize=15)
ax.text(x0 + 2.45, 1.45, "median tuning regret %s" % REGRET, ha="center", va="center", fontsize=15)

ax.text(8, 0.55, "Operating optimum = performance target, not the stability limit",
        ha="center", va="center", fontsize=21, fontweight="bold")

fig.savefig(OUT)
plt.close(fig)
print("wrote %s (%d x %d px)" % (OUT, 1600, 800))
