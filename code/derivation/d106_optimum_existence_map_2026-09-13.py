#!/usr/bin/env python3
"""
d106 -- where does the interior optimum exist in the (A, B) family?

The graceful-controller transition (Sec. VII-D) is a statement about EXISTENCE:
for the linear delayed transit

    e'' = -(A+B) e(x-u) - A e'(x-u) + (A+B) g(x-u) - B P(x-u)        (d97, eq. *)

the corner-transit energy J(u) = int e^2 dx either has an interior minimum on
(0, u_c) or rises from u = 0.  Measured (Nav2, 5 of 5): optimum at (3.5, 1),
(4, 2) twice, and (6, 0); none at (5, 4) and (6, 6).

Two candidate closures are computed on a grid and compared:
  (1) EXISTENCE: J(u) on (0, 0.92 u_c) has an interior minimum (d97's locator);
  (2) SLOPE AT ZERO: sign of J'(0+) -- if J falls from u = 0 an optimum must exist
      before the stability limit (J -> inf at u_c).  If (2) reproduces (1) over
      the plane, the transition curve is the zero set of J'(0; A, B), which is a
      linear-perturbation quantity with a closed form.

Usage:  python d106_optimum_existence_map.py
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

DERIV = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d97", os.path.join(DERIV, "d97_graceful_class_2026-09-08.py"))
d97 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d97)

H = 2.0e-3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d106_optimum_existence_map_2026-09-13.json")


def classify(A, B, n=41):
    uc = d97.u_critical(A, B)
    if uc is None:
        return None
    us = np.linspace(0.004, 0.92 * uc[0], n)
    js = np.array([d97.transit_energy_AB(float(u), A, B, h=H) for u in us])
    i = int(np.argmin(js))
    exists = 0 < i < n - 1
    # slope at zero from the two smallest delays (u = 0.004 and one grid step)
    j0 = d97.transit_energy_AB(0.004, A, B, h=H)
    j1 = d97.transit_energy_AB(0.024, A, B, h=H)
    slope0 = (j1 - j0) / 0.020
    return {"A": A, "B": B, "u_c": uc[0], "exists": bool(exists),
            "u_star_grid": float(us[i]) if exists else None,
            "J0": float(j0), "slope0": float(slope0), "slope0_rel": float(slope0 / j0)}


def main():
    t0 = time.time()
    # self-check at the RPP anchor
    r = classify(2.0, 0.0)
    print("anchor (2,0): exists=%s u*~%.3f u_c=%.4f slope0/J0=%+.3f"
          % (r["exists"], r["u_star_grid"] or -1, r["u_c"], r["slope0_rel"]))
    measured = [(3.5, 1.0, True), (4.0, 2.0, True), (6.0, 0.0, True), (5.0, 4.0, False), (6.0, 6.0, False)]
    rows = []
    print("\nmeasured points:")
    for A, B, obs in measured:
        c = classify(A, B)
        rows.append(dict(c, measured=obs))
        print("  (A,B)=(%.1f,%.1f) measured %-5s | existence %-5s slope0/J0 %+.3f"
              % (A, B, obs, c["exists"], c["slope0_rel"]))
    grid = []
    for A in np.arange(1.0, 9.01, 1.0):
        for B in np.arange(0.0, 9.01, 1.0):
            c = classify(float(A), float(B))
            if c:
                grid.append(c)
    agree = sum(1 for c in grid if c["exists"] == (c["slope0"] < 0))
    print("\ngrid %d points: existence == (J'(0)<0) in %d" % (len(grid), agree))
    for c in grid:
        if c["exists"] != (c["slope0"] < 0):
            print("  disagree at (%.0f,%.0f): exists %s slope0/J0 %+.4f" % (c["A"], c["B"], c["exists"], c["slope0_rel"]))
    # boundary B*(A): smallest B without an optimum, per A
    print("\nper A: largest B with optimum -> smallest B without")
    for A in sorted({c["A"] for c in grid}):
        col = sorted((c for c in grid if c["A"] == A), key=lambda c: c["B"])
        yes = [c["B"] for c in col if c["exists"]]
        no = [c["B"] for c in col if not c["exists"]]
        print("  A=%.0f  optimum at B=%s ; none at B=%s" % (A, yes, no))
    json.dump({"measured": rows, "grid": grid, "h": H}, open(OUT, "w"), indent=1)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    sys.exit(main())
