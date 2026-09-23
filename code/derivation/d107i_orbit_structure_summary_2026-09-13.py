#!/usr/bin/env python3
"""
d107i -- every number the manuscript states about the orbit structure of the
quantized loop, recomputed from the d107/d30 outputs into one JSON.

  (+) orbit:     multiplier P' = (1-u-q)/(1-u+q) vs the exact map    (d107c)
                 end of the stable branch vs q_bc(u) = (1+sqrt u)^2  (d107b)
  (+,-,+) orbit: birth vs q_lo(u) = 1 + sqrt(1 - (2u-1)^2 + u^2/4),
                 re-bisected here by continuation WITH restarts, because d107e's
                 single-pass continuation failed numerically at u = 0.10, 0.55
                 main-piece multiplier: closed form vs map at trusted points,
                 and stability there                                   (d107h)
  band vs flow:  the flow's refined switch (d30 B_measured) inside
                 [q_lo, q_bc] at every u, and the distance to each edge
  operating point: q_lo over the circle optima u* in [0.655, 0.753], against
                 the system's kD/2 = 1.96

Usage:  python d107i_orbit_structure_summary_2026-09-13.py
"""
import importlib.util
import json
import math
import os

import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
d107e = _load("d107e", "d107e_lower_edge_closed_form_2026-09-13.py")


def q_bc(u):
    return (1 + math.sqrt(u)) ** 2


def q_lo1(u):
    r = 1 - (2 * u - 1) ** 2 + u * u / 4
    return 1 + math.sqrt(r) if r >= 0 else float("nan")


def q_lo(u):
    b2 = d107e.q_lo2(u)
    return float(np.nanmax([q_lo1(u), b2[0] if b2 else float("nan")]))


def continuation_end(u, dq=0.0005):
    """d26's closure continued down in q, restarting from scratch before giving up."""
    q = q_bc(u) + 0.05
    s = d26.solve_word(u, 2 * q / D, D)
    if not s:
        return None
    p = [s["z0"], s["t1"], s["t2"]]
    last = None
    while q > 0.9:
        k = 2 * q / D
        th = u / k
        sol, info, ier, msg = fsolve(d26.residuals, p, args=(u, k, D, th), full_output=True, xtol=1e-12)
        r = max(abs(x) for x in d26.residuals(sol, u, k, D, th))
        if not (ier == 1 and r < 1e-8 and 0 < sol[1] < sol[2] < D):
            s2 = d26.solve_word(u, k, D)
            if s2 and s2["resid"] < 1e-8:
                sol = np.array([s2["z0"], s2["t1"], s2["t2"]])
            else:
                break
        p = list(sol)
        last = q
        q = round(q - dq, 7)
    return last


def main():
    out = {}
    c = J("d107c_coexistence_band_2026-09-13.json")["mult_check"]
    out["plus_mult_max_err"] = max(abs(x["map"] - x["closed"]) for x in c)
    out["plus_mult_n"] = len(c)
    b = J("d107b_border_collision_2026-09-13.json")["end_of_plus"]
    out["q_bc_max_err"] = max(abs(x["q_end"] - x["q_bc"]) for x in b)
    out["q_bc_n"] = len(b)
    out["q_bc_u_range"] = [min(x["u"] for x in b), max(x["u"] for x in b)]

    lo = []
    for u in np.round(np.arange(0.10, 0.96, 0.05), 3):
        u = float(u)
        pred = q_lo(u)
        end = continuation_end(u)
        lo.append({"u": u, "q_lo": pred, "q_lo1": q_lo1(u), "continuation": end,
                   "err": (end - pred) if end else None})
        print("  u=%.2f  q_lo %.4f  continuation %s" % (u, pred, ("%.4f" % end) if end else None))
    out["q_lo_rows"] = lo
    errs = [abs(x["err"]) for x in lo if x["err"] is not None]
    out["q_lo_max_err"] = max(errs)
    out["q_lo_n"] = len(errs)
    # where the second border takes over
    out["q_lo_border2_from_u"] = min((x["u"] for x in lo if x["q_lo"] > x["q_lo1"] + 1e-9), default=None)

    h = J("d107h_piece_map_2026-09-13.json")
    trusted = 0
    stable = 0
    for us, rows in h.items():
        if abs(float(us) - 0.2) < 1e-9:
            continue      # continuation followed a different ordering there (see d107h)
        for r in rows:
            if r.get("in_band") and "sig" in r and r["section_valid"] and np.isfinite(r["mult"]) and r["spread"] < 1e-6:
                trusted += 1
                stable += abs(r["mult"]) < 1
    out["pmp_trusted_n"] = trusted
    out["pmp_trusted_stable_n"] = stable
    grazing = {}
    for us in ("0.7", "0.8", "0.9"):
        band = [r for r in h[us] if r.get("in_band") and "sig" in r]
        grazing[us] = 1 - sum(1 for r in band if r["section_valid"]) / len(band)
    out["pmp_grazing_fraction"] = grazing

    meas = J("d30_results_2026-08-29.json")["B_measured"]
    rows = []
    for m in meas:
        u = m["u"]
        rows.append({"u": u, "switch": m["mid"], "half_bracket": (m["first_pmp"] - m["last_plus"]) / 2,
                     "q_lo": q_lo(u), "q_bc": q_bc(u),
                     "inside": bool(q_lo(u) < m["last_plus"] and m["first_pmp"] < q_bc(u)),
                     "above_lo": m["mid"] - q_lo(u), "below_bc": q_bc(u) - m["mid"]})
    out["switch_vs_band"] = rows
    out["all_inside"] = all(r["inside"] for r in rows)

    uop = np.linspace(0.655, 0.753, 99)
    qop = [q_lo(float(x)) for x in uop]
    out["operating_q_lo_range"] = [float(min(qop)), float(max(qop))]
    # u at which q_lo(u) = 1.96 on the rising and falling sides
    f = lambda x: q_lo1(x) - 1.96
    grid = np.linspace(0.2, 0.84, 6401)
    vals = [f(x) for x in grid]
    out["u_where_q_lo_eq_1p96"] = [float(grid[i]) for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]

    json.dump(out, open(os.path.join(HERE, "d107i_orbit_structure_summary_2026-09-13.json"), "w"), indent=1, default=float)
    for k, v in out.items():
        if k != "q_lo_rows":
            print("%-26s %s" % (k, v))


if __name__ == "__main__":
    main()
