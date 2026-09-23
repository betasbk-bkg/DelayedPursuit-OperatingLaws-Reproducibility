#!/usr/bin/env python3
"""
d128 -- does the operating law give a practical tuning benefit on the Nav2 data?

The Nav2 validation so far shows that the curvature-command constant u* = 0.306034 is
REPRODUCED (0.284-0.303 across estimators).  A control journal will ask the practice
question: if an integrator sets the speed by the law, how much tracking error does that
cost against the best speed, and against what a practitioner would do without the law?

For every stored Nav2 speed-RMSE curve (no new runs):
  law:        v_law = u* L / tau_tot, u* = 0.306034, tau_tot the first-order budget with
              every coefficient 1 (the stored pair u_obs/v_obs gives tau_tot/L exactly and
              uses no fitted coefficient)
  oracle:     the minimum of the curve
  rules without the law (each removes one ingredient of the law):
    stab_0.5  : half the stability limit,          u = 0.5  x 0.520494
    stab_0.85 : the manuscript's "85% of u_c" cap,  u = 0.85 x 0.520494
    transport : the law with the delay budget cut to the injected transport delay only
                (drops controller hold, plant hold, actuator lag)
    heading   : the heading-servo constant 0.499496 applied to this curvature-command plant
    fixed     : one speed for every condition (FIXED_V, default taken from the command line)
Regret  R = RMSE(v_rule)/RMSE_oracle - 1, evaluated two ways:
  (L) piecewise-linear interpolation of the measured points, oracle = grid minimum;
  (Q) a parabola through every point within 4x the minimum (d102 whole-bowl), oracle = its
      vertex; the rule speed is evaluated on the parabola only inside the fitted points'
      range, otherwise by (L).
A rule speed outside the swept range is reported as "outside" (no extrapolation).

Blocks: d102's nine (the pooled constant) and, separately, all eligible blocks (corner
angle <= 90 deg; the midpoint-integrator diagnostic and the 17-point re-run excluded).

Usage:  python d128_nav2_tuning_regret_2026-09-14.py [FIXED_V]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
U_CURV, U_HEAD, U_C = 0.306034, 0.499496, 0.520494
NINE = ["nav2_block_freq_2026-09-05.json", "nav2_block_tsim_2026-09-05.json", "nav2_block_lookahead_2026-09-05.json",
        "nav2_block_alpha_2026-09-06.json", "nav2_block_nulls_2026-09-06.json", "nav2_block_replicate_2026-09-06.json",
        "nav2_block_freq_iso_2026-09-06.json", "nav2_block_freq_iso_tau005_2026-09-06.json",
        "nav2_block_freq_iso_tau020_2026-09-06.json"]
EXTRA = ["nav2_block_actuator_2026-09-08.json", "nav2_block_alpha_iso_2026-09-07.json", "nav2_block_replicate2_2026-09-07.json",
         "nav2_block_replicate_iso_2026-09-07.json", "nav2_block_tau_iso_2026-09-07.json", "nav2_block_tsim_iso_2026-09-07.json"]


def curve_eval_linear(vs, rs, v):
    if v < vs[0] or v > vs[-1]:
        return None
    return float(np.interp(v, vs, rs))


def bowl_fit(vs, rs, factor=4.0):
    lo = min(rs) * factor
    sel = [k for k in range(len(rs)) if rs[k] <= lo]
    if len(sel) < 3:
        return None
    x = np.array([vs[k] for k in sel])
    y = np.array([rs[k] for k in sel])
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return None
    vv = -c[1] / (2 * c[0])
    if not (x[0] <= vv <= x[-1]):
        return None
    return {"c": c, "xmin": float(x[0]), "xmax": float(x[-1]), "v_star": float(vv), "r_star": float(np.polyval(c, vv))}


def tau_parts(fn, r, meta):
    """transport delay injected for this row (for the transport-only rule)."""
    t = r.get("tau_inject", meta.get("tau"))
    return float(t) if t is not None else None


def analyse(files, fixed_v):
    rules = ["law", "stab_0.5", "stab_0.85", "transport", "heading", "fixed"]
    rows = []
    for fn in files:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        blk = json.load(open(p, encoding="utf-8"))
        meta = blk.get("meta", {})
        for r in blk["rows"]:
            cur = r.get("curve")
            if not cur or not r.get("u_obs") or not r.get("v_obs"):
                continue
            if "alpha" in fn and r.get("alpha_deg", 0) > 90:
                continue
            pts = sorted((c["v"], c["rmse"]) for c in cur if c.get("rmse") is not None and c["rmse"] == c["rmse"])
            if len(pts) < 5:
                continue
            vs = [a for a, _ in pts]
            rs = [b for _, b in pts]
            k = r["u_obs"] / r["v_obs"]                       # tau_tot / L
            tau_tot = r.get("tau_tot")
            L = tau_tot / k if tau_tot else None
            t_inj = tau_parts(fn, r, meta)
            speeds = {"law": U_CURV / k, "stab_0.5": 0.5 * U_C / k, "stab_0.85": 0.85 * U_C / k,
                      "heading": U_HEAD / k, "fixed": fixed_v}
            if L and t_inj:
                speeds["transport"] = U_CURV * L / t_inj
            else:
                speeds["transport"] = None
            fit = bowl_fit(vs, rs)
            oracle_L = min(rs)
            row = {"block": fn.replace("nav2_block_", "").replace(".json", ""), "k": k, "L": L, "tau_tot": tau_tot,
                   "tau_inject": t_inj, "v_grid": [vs[0], vs[-1]], "oracle_v_grid": vs[int(np.argmin(rs))],
                   "oracle_v_fit": fit["v_star"] if fit else None, "rules": {}}
            for name in rules:
                v = speeds.get(name)
                if v is None:
                    row["rules"][name] = {"v": None, "R_L": None, "R_Q": None, "status": "not defined"}
                    continue
                rl = curve_eval_linear(vs, rs, v)
                R_L = (rl / oracle_L - 1) if rl is not None else None
                if fit and fit["xmin"] <= v <= fit["xmax"]:
                    R_Q = float(np.polyval(fit["c"], v)) / fit["r_star"] - 1
                elif fit and rl is not None:
                    R_Q = rl / fit["r_star"] - 1
                else:
                    R_Q = None
                row["rules"][name] = {"v": v, "R_L": R_L, "R_Q": R_Q, "status": "outside grid" if rl is None else "ok"}
            rows.append(row)
    return rows, rules


def summarise(rows, rules, tag):
    out = {"rows": len(rows), "by_rule": {}, "paired_vs_law": {}}
    print("\n== %s: %d curves" % (tag, len(rows)))
    print("  %-10s %7s %9s %9s %9s %9s %9s" % ("rule", "inside", "med R_L", "p90 R_L", "med R_Q", "p90 R_Q", "max R_Q"))
    for name in rules:
        RL = [x["rules"][name]["R_L"] for x in rows if x["rules"][name]["R_L"] is not None]
        RQ = [x["rules"][name]["R_Q"] for x in rows if x["rules"][name]["R_Q"] is not None]
        inside = sum(1 for x in rows if x["rules"][name]["status"] == "ok")
        st = {"inside_grid": inside, "n_defined": sum(1 for x in rows if x["rules"][name]["v"] is not None),
              "median_R_L": float(np.median(RL)) if RL else None, "p90_R_L": float(np.percentile(RL, 90)) if RL else None,
              "median_R_Q": float(np.median(RQ)) if RQ else None, "p90_R_Q": float(np.percentile(RQ, 90)) if RQ else None,
              "max_R_Q": float(np.max(RQ)) if RQ else None}
        out["by_rule"][name] = st
        f = lambda z: ("%8.1f%%" % (100 * z)) if z is not None else "     n/a"
        print("  %-10s %4d/%-3d %s %s %s %s %s" % (name, inside, st["n_defined"], f(st["median_R_L"]), f(st["p90_R_L"]),
                                                   f(st["median_R_Q"]), f(st["p90_R_Q"]), f(st["max_R_Q"])))
    print("  paired: law better than rule (both inside grid), by R_L and by R_Q")
    for name in rules[1:]:
        both = [x for x in rows if x["rules"][name]["status"] == "ok" and x["rules"]["law"]["status"] == "ok"]
        wl = sum(1 for x in both if x["rules"]["law"]["R_L"] < x["rules"][name]["R_L"])
        wq = [x for x in both if x["rules"]["law"]["R_Q"] is not None and x["rules"][name]["R_Q"] is not None]
        wqq = sum(1 for x in wq if x["rules"]["law"]["R_Q"] < x["rules"][name]["R_Q"])
        outside_rule = sum(1 for x in rows if x["rules"][name]["status"] == "outside grid")
        out["paired_vs_law"][name] = {"both_inside": len(both), "law_better_L": wl, "law_better_Q": wqq, "n_Q": len(wq),
                                      "rule_outside_grid": outside_rule}
        print("    vs %-10s law better %d/%d (R_L), %d/%d (R_Q); rule speed outside the swept range in %d curves"
              % (name, wl, len(both), wqq, len(wq), outside_rule))
    return out


def main():
    fixed_v = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    res = {"fixed_v": fixed_v}
    rows9, rules = analyse(NINE, fixed_v)
    res["nine_blocks"] = summarise(rows9, rules, "d102 nine blocks (pooled constant)")
    rows_all, _ = analyse(NINE + EXTRA, fixed_v)
    res["all_eligible"] = summarise(rows_all, rules, "all eligible blocks")
    res["curves"] = rows_all
    json.dump(res, open(os.path.join(HERE, "d128_nav2_tuning_regret_2026-09-14.json"), "w"), indent=1, default=float)
    print("\n-> d128_nav2_tuning_regret_2026-09-14.json")


if __name__ == "__main__":
    main()
