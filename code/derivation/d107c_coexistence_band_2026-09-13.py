#!/usr/bin/env python3
"""
d107c -- the lower edge of the coexistence band, and the (+) multiplier in closed form.

Established by d107b:
  * the stable (+) fixed point of the exact return map ends at
        q_bc(u) = (1 + sqrt u)^2      (m_s = -Delta/2, border collision)
  * differentiating the (+) orbit's crossing condition
        (1-u) T1 - k z0 T1 + k T1^2 / 2 = Delta (1-u)
    at z0 = Delta/2, T1 = Delta gives dT1/dz0 = 2q / (1-u+q), hence
        P'(+) = (1 - u - q) / (1 - u + q)
    which matched 12 samples to 5 digits.

This script:
  (1) checks P'(+) against the map on a dense (u, q) grid;
  (2) bisects, for each u, the smallest q at which a stable, 1-D-valid fixed
      point of the other word (-++ / +-+) exists, and records its multiplier
      just above that q -- a value away from +-1 means it, too, is born by a
      border collision rather than a smooth bifurcation;
  (3) tests simple closed-form candidates for that lower edge.

Usage:  python d107c_coexistence_band.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np

DERIV = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d27", os.path.join(DERIV, "d27_branch_selection_2026-08-29.py"))
d27 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d27)
D = math.radians(45.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107c_coexistence_band_2026-09-13.json")


def fps(u, q, n=900):
    return d27.fixed_points(u, 2.0 * q / D, D, n=n)


def stable_other(u, q, n=900):
    return [f for f in fps(u, q, n) if f["word"] in ("-++", "+-+") and abs(f["mult"]) < 1 and f["gap_ok"]]


def main():
    t0 = time.time()
    out = {"mult_check": [], "lower_edge": []}
    worst = 0.0
    n = 0
    for u in np.arange(0.1, 0.96, 0.1):
        qbc = (1 + math.sqrt(u)) ** 2
        for q in np.linspace(1.05, qbc - 0.02, 6):
            p = [f for f in fps(float(u), float(q), 600) if f["word"] == "+" and f["gap_ok"]]
            if not p:
                continue
            pred = (1 - u - q) / (1 - u + q)
            err = abs(p[0]["mult"] - pred)
            worst = max(worst, err)
            n += 1
            out["mult_check"].append({"u": float(u), "q": float(q), "map": p[0]["mult"], "closed": pred})
    print("(1) P'(+) = (1-u-q)/(1-u+q): %d grid points, max |map - closed form| = %.2e" % (n, worst))

    print("\n(2) lower edge of the other word (stable, 1-D valid)")
    print("    u     q_lo      mult just above   q_bc      band width")
    for u in (0.2, 0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.9):
        qbc = (1 + math.sqrt(u)) ** 2
        grid = np.round(np.arange(1.0, qbc + 0.3, 0.02), 3)
        first = None
        for q in grid:
            if stable_other(u, float(q), 600):
                first = float(q)
                break
        if first is None:
            print("    %.2f   none found up to %.2f" % (u, grid[-1]))
            continue
        lo, hi = first - 0.02, first
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if stable_other(u, mid):
                hi = mid
            else:
                lo = mid
        qlo = hi
        o = stable_other(u, qlo + 1e-4)
        m = o[0]["mult"] if o else float("nan")
        out["lower_edge"].append({"u": u, "q_lo": qlo, "mult_above": m, "q_bc": qbc})
        print("    %.2f   %.4f   %+.4f          %.4f    %.4f" % (u, qlo, m, qbc, qbc - qlo))

    print("\n(3) closed-form candidates for q_lo(u)")
    le = out["lower_edge"]
    if le:
        u = np.array([r["u"] for r in le])
        ql = np.array([r["q_lo"] for r in le])
        cands = {
            "(1+u)": 1 + u,
            "1/(1-u)": 1 / (1 - u),
            "(1+u)^2/ (1+sqrt u)": (1 + u) ** 2 / (1 + np.sqrt(u)),
            "(1-sqrt u)^2 + 1": (1 - np.sqrt(u)) ** 2 + 1,
            "1 + u + u^2": 1 + u + u * u,
            "(1+sqrt u)^2 / 2 + ...lin fit": None,
        }
        for name, c in cands.items():
            if c is None:
                a, b = np.polyfit(u, ql, 1)
                print("    linear fit q_lo = %.4f + %.4f u (rms %.4f)" % (b, a, np.sqrt(np.mean((a * u + b - ql) ** 2))))
                continue
            print("    %-24s rms %.4f  max %.4f" % (name, np.sqrt(np.mean((c - ql) ** 2)), np.max(np.abs(c - ql))))
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
