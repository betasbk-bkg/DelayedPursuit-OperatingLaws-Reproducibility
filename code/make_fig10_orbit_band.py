# -*- coding: utf-8 -*-
"""
Fig. 10 -- the coexistence band of the quantized loop's two periodic orbits.

In the (u, q = kD/2) plane: the upper edge q_bc(u) = (1 + sqrt u)^2, where the
(+) orbit ends by border collision; the lower edge q_lo(u), where the (+,-,+)
orbit is born (closed form up to u ~ 0.85, then the second border, from d107e);
the band between them, where both orbits are stable; the flow's refined switch
from d30 with its bisection bracket; and the operating segment of the system of
Sec. III (kD/2 = 1.96 over the circle optima u* = 0.655-0.753).

Reads only the derivation JSONs; house style as Figs. 8-9.

Usage:  python make_fig10_orbit_band_2026-09-13.py
"""
import importlib.util
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.abspath(os.path.join(HERE, "..", "derivation"))
OUT = os.path.join(HERE, "figures", "fig10_orbit_band.png")

_s = importlib.util.spec_from_file_location("d107e", os.path.join(DERIV, "d107e_lower_edge_closed_form_2026-09-13.py"))
d107e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d107e)

BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

u = np.linspace(0.05, 0.95, 181)
qbc = (1 + np.sqrt(u)) ** 2
q1 = np.array([d107e.q_lo1(float(x)) for x in u])
q2 = np.array([(d107e.q_lo2(float(x)) or [np.nan])[0] for x in u])
qlo = np.fmax(q1, np.nan_to_num(q2, nan=-np.inf))

meas = json.load(open(os.path.join(DERIV, "d30_results_2026-08-29.json"), encoding="utf-8"))["B_measured"]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.fill_between(u, qlo, qbc, color=GREEN, alpha=0.15, lw=0, label="both orbits stable")
ax.plot(u, qbc, "-", color=VERM, lw=1.3, label=r"$q_{bc}=(1+\sqrt{u})^2$: (+) ends")
ax.plot(u, qlo, "-", color=BLUE, lw=1.3, label=r"$q_{lo}(u)$: (+,$-$,+) born")
ax.plot(u, q1, ":", color=BLUE, lw=0.8, alpha=0.6)
for m in meas:
    ax.errorbar(m["u"], m["mid"], yerr=(m["first_pmp"] - m["last_plus"]) / 2, fmt="o", ms=4,
                color="black", mfc="white", capsize=2, lw=0.9)
ax.plot([], [], "o", ms=4, color="black", mfc="white", label="switch in the flow")
ax.plot([0.655, 0.753], [1.96, 1.96], "-", color=ORANGE, lw=3, solid_capstyle="butt",
        label=r"system of Sec. III ($q=1.96$)")
ax.set_xlabel(r"delay margin  $u$")
ax.set_ylabel(r"$q=k\Delta/2$")
ax.set_xlim(0.05, 0.95)
ax.set_ylim(1.4, 4.0)
ax.legend(frameon=False, fontsize=6.2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote %s" % OUT)
