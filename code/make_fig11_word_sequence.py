# -*- coding: utf-8 -*-
"""
Fig. 11 -- the word sequence of the quantized loop's section map.

For each word W_n = (-+)^n + (n = 0..5), the q-interval on which its fixed point
exists, over the delay margin u (d116b): the birth (dashed; one excursion as short
as the delay window, or at large u the last window reaching the period end) and
the death (solid; grazing).  Lines only -- overlapping translucent fills made the
first version unreadable -- with each word labelled between its own two lines.
Curves stop where the bisection range (q <= 8) ends rather than being clipped flat.

Usage:  python make_fig11_word_sequence_2026-09-13.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.abspath(os.path.join(HERE, "..", "derivation"))
OUT = os.path.join(HERE, "figures", "fig11_word_sequence.png")
rows = json.load(open(os.path.join(DERIV, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
COL = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
QTOP = 8.0

fig, ax = plt.subplots(figsize=(3.5, 2.9))
for n in range(6):
    rr = sorted((r for r in rows if r["n"] == n and r["exists"]), key=lambda r: r["u"])
    if not rr:
        continue
    ud = np.array([r["u"] for r in rr if r.get("death") is not None])
    qd = np.array([r["death"] for r in rr if r.get("death") is not None])
    ub = np.array([r["u"] for r in rr if r.get("birth") is not None])
    qb = np.array([r["birth"] for r in rr if r.get("birth") is not None])
    if len(ud):
        ax.plot(ud, qd, "-", color=COL[n], lw=1.2)
    if len(ub):
        ax.plot(ub, qb, "--", color=COL[n], lw=1.0)
    # label between the word's own birth and death at the smallest u where both exist
    both = [r for r in rr if r.get("birth") is not None and r.get("death") is not None]
    if n == 0:
        ax.text(0.62, 1.45, "W0", color=COL[0], fontsize=7, ha="center", va="center")
    elif both:
        r = both[len(both) // 4]
        ax.text(r["u"], 0.5 * (r["birth"] + r["death"]), "W%d" % n, color=COL[n], fontsize=7,
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.3))
ax.plot([], [], "-", color="gray", lw=1.2, label="word ends (grazing)")
ax.plot([], [], "--", color="gray", lw=1.0, label="word born")
ax.set_xlabel(r"delay margin  $u$")
ax.set_ylabel(r"$q=k\Delta/2$")
ax.set_xlim(0.1, 0.95)
ax.set_ylim(1.0, QTOP)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote %s" % OUT)
