# -*- coding: utf-8 -*-
"""Figures for the paper. Okabe-Ito colorblind-safe palette,
distinct markers (grayscale-survivable), single axis per panel, 300 dpi."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MAIN = os.environ.get("ENGINE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})

def load(p): return json.load(open(p, encoding="utf-8"))

ea   = load(os.path.join(MAIN, "expa_results.json"))
uv   = load(os.path.join(MAIN, "unified_vstar_results.json"))
text = load(os.path.join(BASE, "phase2_tau_ext_results_2026-07-18.json"))
p2   = load(os.path.join(BASE, "phase2_probes_results_2026-07-18.json"))
Ls   = load(os.path.join(BASE, "phase2_L_sweep_results_2026-07-18.json"))
cd   = load(os.path.join(BASE, "phase2_c_decomp_results_2026-07-18.json"))
rot  = load(os.path.join(BASE, "phase1_rotation_results_2026-07-18.json"))
alp  = load(os.path.join(BASE, "phase3_alpha_sweep_results_2026-07-19.json"))
lad  = load(os.path.join(BASE, "phase4_transfer_ladder_results_2026-07-19.json"))

C_LAG, L0 = 0.2333, 2.0

# ---------------- Fig 1: continuum law, 9 tau, zero-parameter -----------------
taus6 = np.array([0.1,0.2,0.3,0.433,0.6,1.0])
v360  = np.array([ea['results']['360'][str(int(t*1000))]['vstar'] for t in taus6])
tx    = np.array([1.5,2.0,3.0])
vx    = np.array([text['results']['360'][str(int(t*1000))]['vstar'] for t in tx])
tt    = np.linspace(0.08, 3.2, 400)

fig, ax = plt.subplots(figsize=(3.5,2.7))
ax.plot(tt, 1.0/(tt+C_LAG), color=BLUE, lw=1.6,
        label=r"law: $v^*=(L/2)/(\tau+\mathrm{WIN}/2+T_s)$  (0 free param.)")
ax.plot(tt, 1.347/np.sqrt(tt), color=GRAY, lw=1.2, ls="--",
        label=r"power law $1.347\,\tau^{-1/2}$")
ax.plot(taus6, v360, "o", color=VERM, ms=5, mfc="white", label="observed (main range)")
ax.plot(tx, vx, "s", color=VERM, ms=5, label="observed (extension, out-of-sample)")
ax.set_xlabel(r"latency $\tau$ (s)"); ax.set_ylabel(r"optimal speed $v^*$ (m/s)")
ax.set_xlim(0, 3.2); ax.set_ylim(0, 3.4)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.annotate("max err 3%\nacross 30x range", xy=(2.0,0.45), fontsize=7, color=GRAY)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig1_continuum_law.png")); plt.close(fig)

# ---------------- Fig 2: Theorem 1 V-curves ----------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.0,2.4), sharey=False)
panels = [("L = 1 m", Ls['speed_grids']['1.0'],
           [r['rmses'] for r in Ls['runs'] if r['name']=='circle_n360' and r['L']==1.0 and r['tau_ms']==433][0], 1.0),
          ("L = 2 m", p2['speeds'], p2['probeA']['360']['433']['rmses'], 2.0),
          ("L = 4 m", Ls['speed_grids']['4.0'],
           [r['rmses'] for r in Ls['runs'] if r['name']=='circle_n360' and r['L']==4.0 and r['tau_ms']==433][0], 4.0)]
for ax, (ttl, sp, rm, L) in zip(axs, panels):
    sp = np.array(sp); te = 0.433 + C_LAG
    vv = np.linspace(sp.min(), sp.max(), 300)
    ax.plot(vv, (L/10.0)*np.abs(L/2 - vv*te), color=BLUE, lw=1.5,
            label="theory (0 param.)")
    ax.plot(sp, rm, "o", color=VERM, ms=3.5, mfc="white", label="observed")
    ax.set_title(ttl, fontsize=8.5)
    ax.set_xlabel(r"$v$ (m/s)")
axs[0].set_ylabel("RMSE (m)")
axs[0].legend(frameon=False, fontsize=7)
fig.suptitle(r"RMSE$(v)\approx(L/R)\,|L/2-v\,\tau_{tot}|$   ($\tau=433$ ms, circle $R=10$ m)",
             fontsize=8.5, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig2_vcurves.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 3: c decomposition -------------------------------------
fig, ax = plt.subplots(figsize=(3.0,2.7))
names = list(cd['cases'].keys())
cp = [cd['cases'][n]['c_pred'] for n in names]
co = [cd['cases'][n]['c_obs'] for n in names]
lim = [0.08, 0.46]
ax.plot(lim, lim, color=GRAY, lw=1.0, ls="--")
ax.plot(cp, co, "o", color=BLUE, ms=6, mfc="white")
for n, x, y in zip(names, cp, co):
    ax.annotate(n, xy=(x,y), xytext=(4,-2), textcoords="offset points", fontsize=6.5)
ax.set_xlabel(r"predicted $c=\mathrm{WIN}/2+DT/\mathrm{SMOOTH}$ (s)")
ax.set_ylabel(r"observed $c=1/v^*-\tau$ (s)")
ax.set_xlim(lim); ax.set_ylim(lim)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig3_c_decomposition.png")); plt.close(fig)

# ---------------- Fig 4: polygon law u*(alpha) --------------------------------
alphas = np.array([30,45,60,72,90,120])
def umean(vals): return float(np.mean(vals))
u_all, u_1000 = {}, {}
for cname, a in [("dodecagon_a30",30),("octagon_a45",45),("pentagon_a72",72)]:
    tt3 = alp['cases'][cname]['taus']
    u_all[a]  = [tt3[k]['u_tot'] for k in tt3]
    u_1000[a] = tt3['1000']['u_tot']
for cname, a in [("hex_rot0",60),("sq_rot0",90)]:
    tt3 = rot['cases'][cname]['taus']
    us = [tt3[k]['vstar']*(int(k)/1000+C_LAG)/L0 for k in tt3]
    u_all[a] = us; u_1000[a] = tt3['1000']['vstar']*(1.0+C_LAG)/L0
tri = []
for cname in ["tri_rot0","tri_rot30"]:
    tt3 = rot['cases'][cname]['taus']
    tri += [tt3[k]['vstar']*(int(k)/1000+C_LAG)/L0 for k in tt3]
u_all[120] = tri
u_1000[120] = float(np.mean([rot['cases'][c]['taus']['1000']['vstar']*(1.0+C_LAG)/L0
                              for c in ["tri_rot0","tri_rot30"]]))
fig, ax = plt.subplots(figsize=(3.5,2.7))
aa = np.linspace(0, 130, 300)
ax.plot(aa, 0.5*np.cos(np.radians(aa/2)), color=BLUE, lw=1.6,
        label=r"law: $u^*=\frac{1}{2}\cos(\alpha/2)$  (0 free param.)")
for a in alphas:
    ax.plot([a]*len(u_all[a]), u_all[a], "o", color=GRAY, ms=3, alpha=0.5)
ax.plot(alphas, [u_1000[a] for a in alphas], "s", color=VERM, ms=5,
        label=r"observed, deep-lag ($\tau=1$ s)")
ax.plot(0, 0.494, "D", color=GREEN, ms=6, label=r"continuum limit ($\alpha\to0$)")
ax.set_xlabel(r"corner angle $\alpha$ (deg)"); ax.set_ylabel(r"delay margin $u^*=v^*\tau_{tot}/L$")
ax.set_xlim(-5, 130); ax.set_ylim(0.1, 0.62)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig4_polygon_law.png")); plt.close(fig)

# ---------------- Fig 5: collapse validity bound -----------------------------
t8 = np.array([0.1,0.2,0.3,0.433,0.6,1.0,1.5])
v8 = np.array([3.931,3.231,2.577,2.134,1.758,1.218,0.869])
x8 = v8*np.sqrt(t8)
fig, ax = plt.subplots(figsize=(3.5,2.7))
ax.axhspan(1.347*0.936, 1.347*1.064, color=BLUE, alpha=0.12, lw=0)
ax.axhline(1.347, color=BLUE, lw=1.2, label=r"$x^*=1.347$ (dither regime)")
tt2 = np.linspace(0.9, 1.8, 100)
ax.plot(tt2, 0.75*L0*np.sqrt(tt2)/(tt2+C_LAG), color=GREEN, lw=1.4,
        label=r"lag cap $u_{max}L\sqrt{\tau}/\tau_{tot}$")
ax.plot(t8, x8, "o", color=VERM, ms=5, mfc="white", label=r"observed $v^*\sqrt{\tau}$ (8-dir circle)")
ax.axvline(1.0, color=GRAY, lw=0.8, ls=":")
ax.annotate("published range", xy=(0.45,1.55), fontsize=7, color=GRAY)
ax.annotate("regime\nboundary", xy=(1.03,1.12), fontsize=7, color=GRAY)
ax.set_xlabel(r"latency $\tau$ (s)"); ax.set_ylabel(r"$x^*=v^*\sqrt{\tau}$ (m s$^{-1/2}$)")
ax.set_xlim(0, 1.7); ax.set_ylim(0.9, 1.6)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig5_collapse_validity.png")); plt.close(fig)

# ---------------- Fig 6: transfer ladder / validity map ----------------------
items = [("2nd-order actuator", 0.8, "PASS"),
         ("Ellipse (curvature varies 4x)", 1.5, "PASS"),
         ("Square, continuous cmds", 5.1, "PASS"),
         ("Control period 0.5 s", -0.2, "PASS"),
         ("Rate limit 2x margin", -2.4, "PASS"),
         ("Noise x5", -9.5, "MARGINAL"),
         ("Rate limit < demand", -41.7, "BREAK"),
         # v8 CORRECTION.  This row used to read ("Bicycle steering", -60.0,
         # "BREAK*") with the annotation "no interior optimum: v-proportional
         # steering gain self-cancels compensation".  That is true on a circle
         # and false at a corner: with the steering bound released the kinematic
         # bicycle IS the unicycle and reproduces the curvature-command constant
         # (0.3083 against the converged 0.3078, +0.2%; d131).  What breaks the law is the
         # steering-ANGLE bound, i.e. a curvature saturation, once it falls
         # below the corner's demand kappa = 2/L.  The two rows below replace
         # the single wrong one.
         ("Ackermann, steering bound released", 0.7, "PASS"),
         ("Ackermann, bound below corner demand", -18.5, "BREAK")]
fig, ax = plt.subplots(figsize=(3.6,2.8))
cols = {"PASS": GREEN, "MARGINAL": ORANGE, "BREAK": VERM, "BREAK*": VERM}
ys = np.arange(len(items))[::-1]
for y, (nm, err, st) in zip(ys, items):
    ax.barh(y, err, color=cols[st], height=0.55, alpha=0.85)
    ax.annotate(f"{nm}", xy=(0, y), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7)
ax.axvline(0, color="black", lw=0.8)
ax.axvspan(-7, 7, color=GRAY, alpha=0.10, lw=0)
ax.set_yticks([])
ax.set_xlabel(r"$v^*$ prediction error (%)")
ax.set_xlim(-65, 20)
ax.annotate("PASS band ±7%", xy=(7.5, ys[0]+0.1), fontsize=6.5, color=GRAY)
ax.annotate("Ackermann: the bound, not the kinematics."
            "\nReleased, it reproduces the"
            "\ncurvature-command constant.",
            xy=(-63, 0.4), fontsize=6, color=GRAY)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig6_validity_map.png")); plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
