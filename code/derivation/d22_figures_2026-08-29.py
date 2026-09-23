# -*- coding: utf-8 -*-
"""
Derivation note -- D22: figures 8 and 9 for the phase-density material
(Secs. IV-F / VI-F of manuscript v5).  Same house style as the v3 figures:
Okabe-Ito palette, distinct markers, single axis per panel, 300 dpi.

Fig. 8  the in-cell phase density is a fold caustic
        (a) rho(m) from the 150-agent engine and from the 1-DOF reduction, with
            the analytic edge m_s of Eq. (11.3) marked;
        (b) the fold test: rho^-2 is linear in m on the forward branch.

Fig. 9  sigma_theta and the attenuation are functions of the delay margin
        (a) sigma_theta(u): engine, analytic orbit (closed form), sawtooth;
        (b) the caustic edge m_s(u): flow versus Eq. (11.3), with the validity
            window 1 < k*Delta/2 < 2.55 and the published operating band shown.

Outputs: fig8_phase_density.png, fig9_attenuation.png  (in ../submission/figures)
"""
import importlib.util, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "submission", "figures")


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")

BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
J = lambda f: json.load(open(os.path.join(HERE, f), encoding="utf-8"))
K, DELTA = 5.0, math.radians(45.0)
DEG = math.degrees(1.0)

# ------------------------------------------------------------------ Fig. 8
d3 = J("d3_results_2026-08-29.json")
eng = {r["speed"]: r for r in d3["runs"] if "case" not in r}
d6 = J("d6_results_2026-08-29.json")
sur = {r["v"]: r for r in d6["speeds"]}

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

v = 2.13
ce = np.array(eng[v]["density_centers"]); we = np.array(eng[v]["density"])
ax[0].step(ce, we*100/(ce[1]-ce[0]), where="mid", color=BLUE, lw=1.2,
           label="150-agent engine")
cs = np.array([p[0] for p in sur[v]["density"]])
ws = np.array([p[1] for p in sur[v]["density"]])
ax[0].step(cs, ws*100/(cs[1]-cs[0]), where="mid", color=ORANGE, lw=1.2, ls="--",
           label="1-DOF reduction")
u_here = v*(0.433 + 0.15 + 1/12.)/2.0
ms_an = d19.m_s_closed(u_here, K, DELTA)*DEG
ax[0].axvline(ms_an, color=VERM, lw=1.0, ls=":", label=r"analytic edge $m_s$, Eq. (11.3)")
ax[0].axhline(100/45.0, color=GRAY, lw=0.8, ls="-.", label="uniform sweep")
ax[0].set_xlabel(r"in-cell phase $m$  (deg)")
ax[0].set_ylabel(r"density  (% per deg)")
ax[0].set_title(f"(a)  $v$ = {v} m/s,  $u$ = {u_here:.2f}", fontsize=9, loc="left")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")

grid = d19.sawtooth(45.0)
M, _ = d10.flow(0.0, k=K, T=900.0, mgrid=grid)
ctr, rho = d10.density(M, nbin=180)
i = int(np.argmax(rho))
sel = (ctr > ctr[i]) & (ctr <= ctr[i]+8.0) & (rho > 0)
cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
ax[1].plot(ctr[sel], rho[sel]**-2.0, "o", ms=3, color=BLUE, label="flow")
xs = np.linspace(ctr[sel].min(), ctr[sel].max(), 50)
ax[1].plot(xs, np.polyval(cf, xs), "-", color=VERM, lw=1.0,
           label=r"fold law $\rho^{-2}\propto m-m_s$")
r2 = 1 - np.sum((rho[sel]**-2-np.polyval(cf, ctr[sel]))**2) / \
     np.sum((rho[sel]**-2-np.mean(rho[sel]**-2))**2)
ax[1].set_xlabel(r"$m$  (deg)")
ax[1].set_ylabel(r"$\rho^{-2}$")
ax[1].set_title(f"(b)  fold test, $R^2$ = {r2:.4f}", fontsize=9, loc="left")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_phase_density.png"), bbox_inches="tight")
print("wrote fig8_phase_density.png  (fold R^2 = %.4f)" % r2)

# ------------------------------------------------------------------ Fig. 9
d8 = {r["u"]: r for r in J("d8_results_2026-08-29.json")["sigma_vs_u"]}
d20 = {r["u"]: r for r in J("d20_results_2026-08-29.json")["k5"]}
d21 = {r["u"]: r for r in J("d21_results_2026-08-29.json")["rows"]}
d10r = J("d10_results_2026-08-29.json")["rows"]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
us = sorted(d20.keys())
ax[0].plot(us, [d8[u]["sigma"] for u in us], "o-", ms=4, color=BLUE, lw=1.2,
           label="measured (150 agents)")
ax[0].plot(us, [d21[u]["two_piece"] for u in us], "s--", ms=3.5, color=VERM, lw=1.1,
           label="closed form, Eq. (12.1)")
ax[0].plot(us, [d20[u]["sawtooth"] for u in us], "^:", ms=3.5, color=GRAY, lw=1.0,
           label="sawtooth only")
ax[0].axvspan(0.655, 0.753, color=GREEN, alpha=0.15, lw=0)
ax[0].text(0.704, ax[0].get_ylim()[0]+0.12, "published optima", ha="center", fontsize=6.5, color=GREEN)
ax[0].set_xlabel(r"delay margin  $u = v\,\tau_{tot}/L$")
ax[0].set_ylabel(r"$\sigma_\theta$  (deg)")
ax[0].set_title("(a)  the constant that was not a constant", fontsize=9, loc="left")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")

uu = np.array([r["u"] for r in d10r])
ms_flow = np.array([r["m_s"] for r in d10r])
ax[1].plot(uu, ms_flow, "o", ms=4, color=BLUE, label="flow (true $b$)")
ug = np.linspace(0.0, 0.95, 200)
mg = np.array([d19.m_s_closed(x, K, DELTA)*DEG for x in ug])
ok = mg > -22.5
ax[1].plot(ug[ok], mg[ok], "-", color=VERM, lw=1.2,
           label=r"Eq. (11.3),  $c=k\Delta/2+(1-u)$")
ax[1].plot(ug[~ok], mg[~ok], "-", color=VERM, lw=0.8, alpha=0.3)
lin = (uu >= 0.3)
off = float(np.mean(ms_flow[lin] - np.array([d19.m_s_closed(x, K, DELTA)*DEG
                                             for x in uu[lin]])))
ax[1].plot(ug[ok], mg[ok]+off, "--", color=VERM, lw=1.0,
           label=r"Eq. (11.3) $+$ offset %.1f$^\circ$ (Sec. 11.4)" % off)
ax[1].axvspan(0.655, 0.753, color=GREEN, alpha=0.15, lw=0)
ax[1].axvspan(0.0, 0.25, color=GRAY, alpha=0.12, lw=0)
ax[1].text(0.125, -25.5, "shoulder", ha="center", fontsize=6.5, color=GRAY)
ax[1].axhline(-22.5, color=GRAY, lw=0.8, ls="-.")
ax[1].text(0.01, -21.9, r"$-\Delta/2$", fontsize=6.5, color=GRAY)
ax[1].set_xlabel(r"delay margin  $u$")
ax[1].set_ylabel(r"caustic edge  $m_s$  (deg)")
ax[1].set_title(r"(b)  edge law, valid for $1<k\Delta/2<2.55$", fontsize=9, loc="left")
ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_attenuation.png"), bbox_inches="tight")
print("wrote fig9_attenuation.png")
