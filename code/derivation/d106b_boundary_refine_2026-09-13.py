#!/usr/bin/env python3
"""
d106b -- refine the optimum-existence boundary B*(A) and test the J'(0) criterion.

d106 found, on an integer grid, optimum for B <= 2 and none for B >= 3 at every
A = 2..7, with J'(0) < 0 agreeing in 85 of 90 cells.  This script:
  1. bisects B*(A) for existence at h = 1e-3 and a 61-point u grid;
  2. bisects the zero of J'(0; A, B) using a one-sided difference at small u;
  3. prints J(u)/J(0) on a fine u grid at the five cells where d106 disagreed,
     so the shape -- dip after an initial rise, or a flat edge -- is visible.

Usage:  python d106b_boundary_refine.py
"""
import importlib.util
import json
import os
import time

import numpy as np

DERIV = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d97", os.path.join(DERIV, "d97_graceful_class_2026-09-08.py"))
d97 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d97)
H = 1.0e-3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d106b_boundary_refine_2026-09-13.json")


def exists(A, B, n=61):
    uc = d97.u_critical(A, B)
    us = np.linspace(0.002, 0.92 * uc[0], n)
    js = np.array([d97.transit_energy_AB(float(u), A, B, h=H) for u in us])
    i = int(np.argmin(js))
    return 0 < i < n - 1, us, js


def slope0(A, B):
    a = d97.transit_energy_AB(0.002, A, B, h=H)
    b = d97.transit_energy_AB(0.010, A, B, h=H)
    return (b - a) / 0.008 / a


def bisect(f, lo, hi, it=9):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) == flo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    t0 = time.time()
    out = {"boundary": [], "shapes": []}
    print(" A    B*_exist   B*_slope0   diff")
    for A in (3.0, 4.0, 5.0, 6.0, 7.0):
        be = bisect(lambda B: exists(A, B)[0], 2.0, 3.0)
        bs = bisect(lambda B: slope0(A, B) < 0, 1.5, 3.5)
        out["boundary"].append({"A": A, "B_exist": be, "B_slope0": bs})
        print("%4.1f   %7.4f    %7.4f   %+.4f   (%.0f s)" % (A, be, bs, bs - be, time.time() - t0))
    print("\nshapes at d106's disagreement cells (J/J(u0) on a fine grid):")
    for A, B in ((1.0, 0.0), (2.0, 1.0), (8.0, 0.0), (8.0, 3.0), (9.0, 0.0)):
        ex, us, js = exists(A, B, n=61)
        rel = js / js[0]
        idx = list(range(0, 61, 6)) + [int(np.argmin(js))]
        print("  (%.0f,%.0f) exists=%s  min at u=%.4f (J/J0=%.4f)  samples: %s"
              % (A, B, ex, us[int(np.argmin(js))], rel.min(),
                 " ".join("%.3f:%.3f" % (us[i], rel[i]) for i in sorted(set(idx)))))
        out["shapes"].append({"A": A, "B": B, "exists": bool(ex),
                              "u": [float(x) for x in us], "J_rel": [float(x) for x in rel]})
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
