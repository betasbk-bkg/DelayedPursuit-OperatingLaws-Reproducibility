#!/usr/bin/env python3
"""
d107b -- is the end of the (+) branch the analytic locus q = (1 + sqrt u)^2 ?

d107 found the (+) fixed point of the exact return map stable (P' in (-1, 0)),
1-D valid (gap_ok) and then simply absent: at u = 0.30 between q = 2.32 and 2.40,
at u = 0.50 between 2.88 and 2.96, still present at q = 3.2 for u = 0.71, 0.90.
Its multiplier never reaches +1 or -1 there, so it is not a fold or a flip: the
orbit collides with a switching surface (border collision).

The no-backward-crossing condition of Eq. (12), m_s = -Delta/2 with
m_s = -k Delta^2/8 - (1-u)^2/(2k) + u Delta/2 and q = k Delta/2, reduces to

    q^2 - 2(1+u) q + (1-u)^2 = 0   ->   q_bc(u) = (1 + sqrt(u))^2       (upper root)

This script (1) bisects the numerical end of the (+) branch on a u grid and
compares it with q_bc(u); (2) maps where a STABLE, 1-D-valid fixed point of the
other word (-++, the cyclic form of +-+) exists, to find the lower edge of the
coexistence band; (3) records the (+) multiplier for a closed-form check.

Usage:  python d107b_border_collision.py
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107b_border_collision_2026-09-13.json")


def branches(u, q, n=900):
    k = 2.0 * q / D
    fps = d27.fixed_points(u, k, D, n=n)
    plus = [f for f in fps if f["word"] == "+" and abs(f["mult"]) < 1 and f["gap_ok"]]
    other = [f for f in fps if f["word"] in ("-++", "+-+") and abs(f["mult"]) < 1 and f["gap_ok"]]
    return plus, other


def main():
    t0 = time.time()
    out = {"end_of_plus": [], "coexistence": [], "mult_plus": []}
    print("(1) end of the stable (+) branch vs q_bc(u) = (1+sqrt u)^2")
    print("    u      q_end(num)   q_bc(analytic)   diff")
    for u in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.9):
        qa = (1 + math.sqrt(u)) ** 2
        lo, hi = 1.2, qa + 0.6
        if not branches(u, lo)[0] or branches(u, hi)[0]:
            print("    %.2f   bracket fails (plus at lo=%s, at hi=%s)" % (u, bool(branches(u, lo)[0]), bool(branches(u, hi)[0])))
            continue
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if branches(u, mid)[0]:
                lo = mid
            else:
                hi = mid
        qe = 0.5 * (lo + hi)
        out["end_of_plus"].append({"u": u, "q_end": qe, "q_bc": qa})
        print("    %.2f   %9.4f     %9.4f      %+.4f" % (u, qe, qa, qe - qa))
    print("\n(2) stable 1-D-valid fixed points of the other word, by q")
    for u in (0.3, 0.5, 0.71, 0.9):
        present = []
        for q in np.round(np.arange(1.2, 4.01, 0.05), 3):
            p, o = branches(u, float(q), n=600)
            present.append((float(q), bool(p), bool(o), [round(f["mult"], 3) for f in o][:3]))
        first_other = next((q for q, p, o, _ in present if o), None)
        both = [q for q, p, o, _ in present if p and o]
        out["coexistence"].append({"u": u, "first_q_other": first_other,
                                   "both_q": both, "table": present})
        print("  u=%.2f: other word first stable at q=%s; both stable on %d of %d grid q (%s..%s)"
              % (u, first_other, len(both), len(present), both[0] if both else "-", both[-1] if both else "-"))
    print("\n(3) (+) multiplier samples")
    for u in (0.3, 0.5, 0.71, 0.9):
        for q in (1.4, 1.8, 2.2):
            p, _ = branches(u, q, n=600)
            if p:
                out["mult_plus"].append({"u": u, "q": q, "mult": p[0]["mult"], "z": p[0]["z"]})
                print("  u=%.2f q=%.2f  P'=%+.5f  z*=%.5f" % (u, q, p[0]["mult"], p[0]["z"]))
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
