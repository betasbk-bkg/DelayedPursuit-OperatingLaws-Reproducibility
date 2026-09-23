#!/usr/bin/env python3
"""
d126 -- the analytic margin driven by the DERIVED sigma_theta (d6/d7 1-DOF surrogate,
no engine), redone with the method d124/d125 showed is needed.

The manuscript quotes, from d7, a mean error of +15.5% (worst +33.9% at tau = 1 s) for
Eq. (9) driven by the derived sigma_theta.  d7 used 3 seeds, the grid v = 0.75-4.5, one
global cubic for sigma(v) and no edge check -- the same method that d124 showed turns
the engine-driven prediction into a fit artefact (with pointwise or locked-side sigma the
engine-driven Eq. (9) lands within +0.8 to +3.2% for tau >= 0.433).  d7's derived sigma
curves also jump from ~8 to ~11 deg at much lower speed than the engine's (tau = 1.0:
between v = 1.25 and 2.25; engine: 2.10-2.25), so for the surrogate the jump may matter.

For each tau (v = 0.30-4.50 step 0.15, SEEDS seeds per point):
  (1) derived sigma_theta and its standard error across seeds;
  (2) the jump (largest increase between neighbouring speeds), u at the jump, compared
      with the engine's jump (d125);
  (3) Eq. (9) margin by argmin of J with pointwise sigma, linear interpolation, and a
      quadratic fitted on the locked side only; edges flagged; compared with the
      published optimum and with the engine-driven results (d124, d125).

Usage:  python d126_derived_sigma_margin_2026-09-14.py [seeds]
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d6", os.path.join(HERE, "d6_rho_derivation_2026-08-29.py"))
d6 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d6)
L, R, WIN, T_S = d6.L, d6.R, d6.WIN, d6.T_S
OBS = {0.1: 3.93, 0.433: 2.13, 0.6: 1.76, 1.0: 1.218, 1.5: 0.869}
TAUS = [0.1, 0.433, 0.6, 1.0, 1.5]


def J_of(tau, v, sig_rad):
    tt = tau + WIN / 2 + T_S
    return (L / R) ** 2 * (L / 2 - v * tt) ** 2 + L ** 2 * sig_rad ** 2


def argmin_ref(v, y):
    i = int(np.argmin(y))
    edge = i == 0 or i == len(v) - 1
    vv = float(v[i])
    if not edge:
        c = np.polyfit(v[i - 1:i + 2], y[i - 1:i + 2], 2)
        if c[0] > 0:
            x = -c[1] / (2 * c[0])
            if v[i - 1] <= x <= v[i + 1]:
                vv = float(x)
    return vv, edge


def main():
    t0 = time.time()
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ms, b, s = d6.load_transfer()
    speeds = np.round(np.arange(0.30, 4.5001, 0.15), 3)
    eng = {}
    p125 = os.path.join(HERE, "d125_sigma_jump_margin_2026-09-14.json")
    if os.path.exists(p125):
        for r in json.load(open(p125, encoding="utf-8"))["rows"]:
            eng[round(r["tau"], 3)] = r
    out = {"seeds": seeds, "speeds": speeds.tolist(), "rows": []}
    for tau in TAUS:
        tt = tau + WIN / 2 + T_S
        sig = np.zeros(len(speeds))
        se = np.zeros(len(speeds))
        for i, v in enumerate(speeds):
            per = []
            for sd in range(seeds):
                m, e, z, w = d6.surrogate(float(v), tau, ms, b, s, seed=700 + sd)
                per.append(float(np.std(e)))
            sig[i] = np.mean(per)
            se[i] = np.std(per, ddof=1) / math.sqrt(seeds)
        inc = np.diff(sig)
        k = int(np.argmax(inc))
        z = inc[k] / math.sqrt(se[k] ** 2 + se[k + 1] ** 2 + 1e-30)
        v_jump = 0.5 * (speeds[k] + speeds[k + 1])
        sr = np.radians(sig)
        v_pw, e_pw = argmin_ref(speeds, J_of(tau, speeds, sr))
        vg = np.linspace(speeds[0], speeds[-1], 800)
        v_li, e_li = argmin_ref(vg, J_of(tau, vg, np.interp(vg, speeds, sr)))
        sel = speeds <= speeds[k]
        if sel.sum() >= 3:
            c = np.polyfit(speeds[sel], sr[sel], 2)
            vgl = np.linspace(speeds[sel][0], speeds[sel][-1], 400)
            v_lk, e_lk = argmin_ref(vgl, J_of(tau, vgl, np.polyval(c, vgl)))
        else:
            v_lk, e_lk = float("nan"), True
        rel = lambda vp: (OBS[tau] - vp) / vp
        row = {"tau": tau, "sigma_deg": sig.tolist(), "sigma_se_deg": se.tolist(),
               "jump_between": [float(speeds[k]), float(speeds[k + 1])], "jump_deg": float(inc[k]), "jump_z": float(z),
               "u_jump": float(v_jump * tt / L), "u_obs": OBS[tau] * tt / L,
               "pointwise": {"v_pred": v_pw, "edge": e_pw, "rel": rel(v_pw)},
               "linear": {"v_pred": v_li, "edge": e_li, "rel": rel(v_li)},
               "locked_quadratic": {"v_pred": v_lk, "edge": e_lk, "rel": rel(v_lk) if v_lk == v_lk else None}}
        if round(tau, 3) in eng:
            row["engine_u_jump"] = eng[round(tau, 3)]["u_jump"]
            row["engine_locked_prediction"] = eng[round(tau, 3)]["piecewise_prediction"]
        out["rows"].append(row)
        print("tau=%.3f (%.0f s): derived jump %.2f->%.2f deg at v %.2f-%.2f (z=%.1f), u_jump=%.3f (engine %s) | obs %.3f | "
              "pointwise %.3f%s (%+.1f%%), linear %.3f%s (%+.1f%%), locked quad %.3f%s (%s)"
              % (tau, time.time() - t0, sig[k], sig[k + 1], speeds[k], speeds[k + 1], z, row["u_jump"],
                 ("%.3f" % row["engine_u_jump"]) if "engine_u_jump" in row else "-", OBS[tau],
                 v_pw, " EDGE" if e_pw else "", 100 * rel(v_pw), v_li, " EDGE" if e_li else "", 100 * rel(v_li),
                 v_lk, " EDGE" if e_lk else "", ("%+.1f%%" % (100 * rel(v_lk))) if v_lk == v_lk else "n/a"), flush=True)
        json.dump(out, open(os.path.join(HERE, "d126_derived_sigma_margin_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d126_derived_sigma_margin_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
