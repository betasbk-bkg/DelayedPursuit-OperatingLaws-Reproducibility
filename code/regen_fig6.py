# -*- coding: utf-8 -*-
"""Regenerate Fig. 6 alone, after the Ackermann correction.

WHY ONLY FIG 6.  `make_figures_2026-07-19.py` redraws all nine figures and needs
the companion engine for several of them.  Only the transfer-ladder panel
changed, and regenerating the other eight would put eight new files into the
submission for no reason and with the engine's availability as a new dependency.
This script is a copy of that file's Fig. 6 block and nothing else; the ladder
data and the annotation are kept identical to the corrected generator, so the
two cannot drift apart silently -- the assertion below fails if they do.

WHAT CHANGED AND WHY.  The old panel carried a single bar, "Bicycle steering",
at -60% with the footnote "no interior optimum: v-proportional steering gain
self-cancels compensation".  Sec. VIII-B of the manuscript now retracts exactly
that claim, and the figure was left standing one paragraph above its own
retraction -- the coupled-artifact failure the paper's own process notes name.
The kinematic bicycle with its steering bound released reproduces the
curvature-command constant (0.3083 against the converged 0.3078, +0.2%; d131); what breaks the law
is the steering-angle bound once it falls below the corner's curvature demand
kappa = 2/L, measured at -18.5% for max_steer = 0.85 rad.

Usage:  python regen_fig6_2026-09-08.py
"""
import io
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "repo", "code", "make_figures_2026-07-19.py")
OUT = os.path.join(HERE, "figures")

BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

items = [("2nd-order actuator", 0.8, "PASS"),
         ("Ellipse (curvature varies 4x)", 1.5, "PASS"),
         ("Square, continuous cmds", 5.1, "PASS"),
         ("Control period 0.5 s", -0.2, "PASS"),
         ("Rate limit 2x margin", -2.4, "PASS"),
         ("Noise x5", -9.5, "MARGINAL"),
         ("Rate limit < demand", -41.7, "BREAK"),
         ("Ackermann, steering bound released", 0.7, "PASS"),
         ("Ackermann, bound below corner demand", -18.5, "BREAK")]

# The generator is the source of truth for this list; if someone edits one and
# not the other, stop rather than ship two different figures.
gen = io.open(GEN, encoding="utf-8").read()
for name, val, _st in items[-2:]:
    pat = r'\("%s",\s*%s,' % (re.escape(name), re.escape("%g" % val))
    if not re.search(pat, gen):
        raise SystemExit("generator and regenerator disagree on: %s" % name)
# Look only at ACTIVE lines: the correction comment in the generator quotes the
# retracted row verbatim to explain what it replaced, and a plain substring test
# fires on that quotation -- which it did on the first run.
active = [ln for ln in gen.splitlines() if not ln.lstrip().startswith("#")]
if any("Bicycle steering" in ln for ln in active):
    raise SystemExit("the retracted 'Bicycle steering' row is still active in the generator")

fig, ax = plt.subplots(figsize=(3.6, 2.8))
cols = {"PASS": GREEN, "MARGINAL": ORANGE, "BREAK": VERM}
ys = np.arange(len(items))[::-1]
for y, (nm, err, st) in zip(ys, items):
    ax.barh(y, err, color=cols[st], height=0.55, alpha=0.85)
    ax.annotate(nm, xy=(0, y), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7)
ax.axvline(0, color="black", lw=0.8)
ax.axvspan(-7, 7, color=GRAY, alpha=0.10, lw=0)
ax.set_yticks([])
ax.set_xlabel(r"$v^*$ prediction error (%)")
ax.set_xlim(-65, 20)
# The ladder gained a row (8 -> 9), which moved every bar down one slot.  The
# old annotation position, xy=(-63, 0.6), had been clear under eight bars and
# now lands on top of the two Ackermann labels.  Reserve space below the last
# bar and put it there instead.
ax.set_ylim(-2.0, len(items) - 0.4)
ax.annotate("PASS band ±7%", xy=(7.5, ys[0] + 0.1), fontsize=6.5, color=GRAY)
ax.annotate("Ackermann: the bound, not the kinematics."
            "\nReleased, it reproduces the curvature-command constant.",
            xy=(-64, -1.6), fontsize=6, color=GRAY)
fig.tight_layout()
p = os.path.join(OUT, "fig6_validity_map.png")
fig.savefig(p)
plt.close(fig)
print("rewrote %s (%d bars: %d pass, %d marginal, %d break)"
      % (p, len(items),
         sum(1 for i in items if i[2] == "PASS"),
         sum(1 for i in items if i[2] == "MARGINAL"),
         sum(1 for i in items if i[2] == "BREAK")))
