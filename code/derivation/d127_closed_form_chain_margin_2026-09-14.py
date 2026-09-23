#!/usr/bin/env python3
"""
d127 -- does the ANALYTIC orbit (Eq. (13), closed-form sigma_theta of d21) predict the
operating speed through Eq. (9)?  This is the joint "orbit -> sigma_theta -> speed" that a
unified paper would rest on; d124-d126 tested it only with measured or numerically
integrated sigma_theta.

MATHEMATICAL CONSEQUENCE.  The closed form sigma_2piece(u; k, Delta, m_c, b_e, s_bar) depends
on the delay margin u only (k = R/L = 5, Delta = 45 deg fixed).  With u = v tau_tot/L,
    J(v) = (L/R)^2 (L/2 - v tau_tot)^2 + L^2 sigma(v)^2
         = L^2 [ (L/R)^2 (1/2 - u)^2 + sigma(u)^2 ],
so tau drops out: the chain predicts ONE u* for every tau, v*(tau) = u* L / tau_tot.
The published optima have u*_obs = 0.655 (tau = 0.1) ... 0.753 (tau = 1.5).

Search range u = 0.20-1.00: at kDelta/2 = 1.96 the (+) orbit exists only for
u > (sqrt(1.96) - 1)^2 = 0.16, and above u ~ 1.2 the loop loses lock (d125).

Checks: (0) sigma_2piece reproduces the d21 table; (1) u* and v*(tau) against the
published optima; (2) sensitivity: the quantisation channel scaled by the measured
coefficient r ~ 0.6 (d124 D) and by r(u) interpolated from d124.

Usage:  python d127_closed_form_chain_margin_2026-09-14.py
"""
import importlib.util
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


d21 = _load("d21", "d21_sigma_closed_2piece_2026-08-29.py")
L, R = 2.0, 10.0
WIN, T_S = 0.3, (1 / 60) / 0.2
OBS = {0.1: 3.93, 0.433: 2.13, 0.6: 1.76, 1.0: 1.218, 1.5: 0.869}
K, DELTA_DEG, A_ACC = 5.0, 45.0, 3.0


def main():
    DELTA = math.radians(DELTA_DEG)
    m_c = math.radians(DELTA_DEG / 2 - A_ACC)
    b_e = abs(math.radians(d21.d12.b_s_closed(DELTA_DEG / 2)[0]))
    ms_grid = np.linspace(-DELTA_DEG / 2, DELTA_DEG / 2, 361)
    s_bar = float(np.sqrt(np.mean([d21.d12.b_s_closed(v)[1] ** 2 for v in ms_grid])))

    def sig_deg(u):
        return d21.sigma_2piece(u, K, DELTA, m_c, b_e, s_bar)

    out = {"check_d21": [], "u_grid_failures": 0}
    ref = json.load(open(os.path.join(HERE, "d21_results_2026-08-29.json"), encoding="utf-8"))["rows"]
    for r in ref:
        out["check_d21"].append({"u": r["u"], "d21": r["two_piece"], "now": sig_deg(r["u"])})
    err0 = max(abs(x["d21"] - x["now"]) for x in out["check_d21"])
    print("(0) closed-form sigma reproduces d21: max |diff| = %.2e deg" % err0)

    us = np.round(np.arange(0.20, 1.0001, 0.0025), 6)
    sig = []
    for u in us:
        try:
            sig.append(sig_deg(float(u)))
        except Exception:
            sig.append(np.nan)
            out["u_grid_failures"] += 1
    sig = np.radians(np.array(sig))
    bias = (L / R) ** 2 * (0.5 - us) ** 2

    # measured coefficient r(u) from d124 (all taus pooled, binned in u)
    d124 = json.load(open(os.path.join(HERE, "d124_margin_error_diagnosis_2026-09-14.json"), encoding="utf-8"))
    sp = np.array(d124["B"]["speeds"])
    pu, pr = [], []
    for row in d124["D"]:
        tt = row["tau"] + WIN / 2 + T_S
        pu += list(sp * tt / L)
        pr += list(row["ratio_rmse2_minus_bias2_over_L2sigma2"])
    pu, pr = np.array(pu), np.array(pr)
    ok = (pu >= 0.2) & (pu <= 1.0) & np.isfinite(pr)
    edges = np.arange(0.2, 1.0001, 0.1)
    cen, val = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok & (pu >= a) & (pu < b)
        if m.sum():
            cen.append(0.5 * (a + b))
            val.append(float(np.mean(pr[m])))
    r_of_u = np.interp(us, cen, val)

    variants = {"coefficient_1 (Eq. 9 as written)": np.ones_like(us),
                "coefficient_0.6 (constant)": np.full_like(us, 0.6),
                "coefficient_r(u) (measured, d124)": r_of_u}
    out["variants"] = {}
    for name, coef in variants.items():
        f = bias + coef * sig ** 2
        i = int(np.nanargmin(f))
        u_star = float(us[i])
        edge = i == 0 or i == len(us) - 1
        rows = []
        for tau, vo in OBS.items():
            tt = tau + WIN / 2 + T_S
            vp = u_star * L / tt
            rows.append({"tau": tau, "v_obs": vo, "u_obs": vo * tt / L, "v_pred": vp, "rel": (vo - vp) / vp})
        mean_abs = float(np.mean([abs(x["rel"]) for x in rows]))
        out["variants"][name] = {"u_star": u_star, "edge": edge, "rows": rows, "mean_abs_rel": mean_abs}
        print("(1/2) %-36s u* = %.4f%s | " % (name, u_star, " EDGE" if edge else "")
              + "  ".join("tau=%.3g: %+.1f%%" % (x["tau"], 100 * x["rel"]) for x in rows)
              + " | mean |err| %.1f%%" % (100 * mean_abs))
    print("published u*_obs: " + ", ".join("%.3f" % (v * (t + WIN / 2 + T_S) / L) for t, v in OBS.items()))
    print("sigma closed form at u = 0.5, 0.65, 0.75, 0.9 [deg]: " + ", ".join("%.3f" % sig_deg(x) for x in (0.5, 0.65, 0.75, 0.9)))
    out["sigma_closed_deg_on_grid"] = {"u": us.tolist(), "sigma_deg": np.degrees(sig).tolist()}
    json.dump(out, open(os.path.join(HERE, "d127_closed_form_chain_margin_2026-09-14.json"), "w"), indent=1)
    print("-> d127_closed_form_chain_margin_2026-09-14.json")


if __name__ == "__main__":
    main()
