#!/usr/bin/env python3
"""
d125 -- does the analytic margin fail because sigma_theta(v) has a jump (loss of phase
lock), which a global cubic fit and a smooth stationarity condition cannot represent?

Reads d124 (16-seed closed-loop sigma_theta and tracking RMSE on v = 0.30-4.50, five taus).
d7's derived sigma_theta curves already show a step from ~8 deg to ~11 deg (the uniform-
sweep value is 10.95 deg), e.g. tau = 1.0 between v = 1.25 and 2.25.

For each tau:
  (1) the jump: the largest increase of sigma between neighbouring speeds, with its size
      in standard errors; the jump speed (midpoint), and the delay margin u there;
  (2) Eq. (9) with sigma fitted separately on each side of the jump (quadratic in v on
      the locked side v < v_jump and on the unlocked side v > v_jump), minimising
      J = (L/R)^2 (L/2 - v tau_tot)^2 + L^2 sigma^2 over each side, global minimum;
  (3) where the published optimum and the directly measured RMSE optimum (d124) sit
      relative to the jump.
Also: whether u at the jump is the same for every tau (a u-controlled mechanism).

Usage:  python d125_sigma_jump_margin_2026-09-14.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L, R = 2.0, 10.0
WIN, T_S = 0.3, (1 / 60) / 0.2
OBS = {0.1: 3.93, 0.433: 2.13, 0.6: 1.76, 1.0: 1.218, 1.5: 0.869}


def minimise(tau, vgrid, sig):
    tt = tau + WIN / 2 + T_S
    J = (L / R) ** 2 * (L / 2 - vgrid * tt) ** 2 + L ** 2 * sig(vgrid) ** 2
    i = int(np.argmin(J))
    edge = i == 0 or i == len(vgrid) - 1
    return float(vgrid[i]), float(J[i]), edge


def main():
    d = json.load(open(os.path.join(HERE, "d124_margin_error_diagnosis_2026-09-14.json"), encoding="utf-8"))
    B = d["B"]
    sp = np.array(B["speeds"])
    SIG = np.array(B["sigma_deg"])
    SE = np.array(B["sigma_se_deg"])
    D = {round(r["tau"], 3): r for r in d.get("D", [])}
    out = []
    for j, tau in enumerate(B["taus"]):
        tt = tau + WIN / 2 + T_S
        s, se = SIG[:, j], SE[:, j]
        inc = np.diff(s)
        z = inc / np.sqrt(se[1:] ** 2 + se[:-1] ** 2)
        k = int(np.argmax(inc))
        v_jump = 0.5 * (sp[k] + sp[k + 1])
        row = {"tau": tau, "jump_between": [float(sp[k]), float(sp[k + 1])], "v_jump": float(v_jump),
               "u_jump": float(v_jump * tt / L), "sigma_before": float(s[k]), "sigma_after": float(s[k + 1]),
               "jump_deg": float(inc[k]), "jump_z": float(z[k]), "v_obs": OBS[tau], "u_obs": OBS[tau] * tt / L}
        best = None
        for side, sel in (("locked", sp <= sp[k]), ("unlocked", sp >= sp[k + 1])):
            if sel.sum() >= 3:
                c = np.polyfit(sp[sel], np.radians(s[sel]), 2)
                vg = np.linspace(sp[sel][0], sp[sel][-1], 400)
                v, Jv, edge = minimise(tau, vg, lambda x, c=c: np.polyval(c, x))
                row[side] = {"v_min": v, "J_min": Jv, "edge_of_side": edge}
                if best is None or Jv < best[1]:
                    best = (side, Jv, v, edge)
        row["piecewise_prediction"] = {"side": best[0], "v_pred": best[2], "edge_of_side": best[3],
                                       "rel_obs": (OBS[tau] - best[2]) / best[2]}
        if round(tau, 3) in D:
            vd = D[round(tau, 3)]["v_direct_rmse"]
            row["v_direct_rmse"] = vd
            row["direct_minus_jump"] = vd - v_jump
        row["obs_minus_jump"] = OBS[tau] - v_jump
        out.append(row)
        print("tau=%.3f: jump %.2f->%.2f deg between v=%.2f and %.2f (z=%.1f), u_jump=%.3f | obs v=%.3f (u=%.3f), obs-jump=%+.3f"
              % (tau, row["sigma_before"], row["sigma_after"], sp[k], sp[k + 1], row["jump_z"], row["u_jump"], OBS[tau],
                 row["u_obs"], row["obs_minus_jump"]))
        print("          piecewise Eq.(9): best side %s, v_pred=%.3f%s (obs vs pred %+.1f%%)%s"
              % (best[0], best[2], " [edge of side]" if best[3] else "", 100 * row["piecewise_prediction"]["rel_obs"],
                 ("; direct RMSE argmin %.3f (direct-jump %+.3f)" % (row["v_direct_rmse"], row["direct_minus_jump"])) if "v_direct_rmse" in row else ""))
    uj = [r["u_jump"] for r in out]
    summary = {"u_jump_mean": float(np.mean(uj)), "u_jump_range": [float(min(uj)), float(max(uj))], "rows": out}
    print("u at the jump across taus: %.3f-%.3f (mean %.3f)" % (min(uj), max(uj), np.mean(uj)))
    json.dump(summary, open(os.path.join(HERE, "d125_sigma_jump_margin_2026-09-14.json"), "w"), indent=1)
    print("-> d125_sigma_jump_margin_2026-09-14.json")


if __name__ == "__main__":
    main()
