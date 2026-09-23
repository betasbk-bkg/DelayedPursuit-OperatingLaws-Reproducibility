#!/usr/bin/env python3
"""
d107 -- scoping the symbolic transition of the quantized loop as a bifurcation.

Uses d27's exact piecewise return map z -> P(z) (Poincare section at a forward
boundary crossing).  For each (u, q = k*Delta/2) on a grid it records:

  * every fixed point, its word and multiplier P'(z*), and whether it is stable
  * whether the 1-D reduction is valid on that orbit (d27's gap_ok: no earlier
    crossing still inside the delay window) -- where it is not, the return map
    is not one-dimensional and a 1-D bifurcation analysis does not apply
  * whether ANY stable fixed point exists (if not, the attractor is not a fixed
    point: periodic orbit of the map, or worse)

and, along each u, how the (+) branch ends as q rises: does its multiplier pass
through +1 (fold), -1 (period doubling), or does the fixed point vanish with
|P'| < 1 (a border-collision-type disappearance, typical of piecewise maps)?

Usage:  python d107_return_map_scope.py
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107_return_map_scope_2026-09-13.json")


def main():
    t0 = time.time()
    us = [0.3, 0.5, 0.71, 0.9]
    qs = np.round(np.arange(1.6, 3.21, 0.08), 3)
    rows = []
    for u in us:
        print("\nu = %.2f" % u)
        print("   q    #fp  stable words              mult(+)   gap_ok(+)  any_stable")
        for q in qs:
            k = 2.0 * q / D
            fps = d27.fixed_points(u, k, D, n=600)
            fps = [f for f in fps if abs(f["mult"]) < 50]
            plus = [f for f in fps if f["word"] == "+"]
            stable = [f for f in fps if abs(f["mult"]) < 1.0]
            words = sorted({f["word"] for f in fps})
            mp = plus[0]["mult"] if plus else None
            gp = plus[0]["gap_ok"] if plus else None
            rows.append({"u": u, "q": float(q), "n_fp": len(fps), "n_stable": len(stable),
                         "words": words, "stable_words": sorted({f["word"] for f in stable}),
                         "mult_plus": mp, "gap_ok_plus": gp,
                         "fps": fps})
            print("  %.2f  %3d  %4d   %-22s %8s   %6s     %s"
                  % (q, len(fps), len(stable), ",".join(words)[:22],
                     ("%+.3f" % mp) if mp is not None else "   -", str(gp), bool(stable)))
    # how the (+) branch ends along each u
    print("\nend of the (+) branch along q:")
    summary = []
    for u in us:
        r = [x for x in rows if x["u"] == u]
        last = None
        for x in r:
            if x["mult_plus"] is not None:
                last = x
        nxt = [x for x in r if last and x["q"] > last["q"]][:1]
        if last:
            s = {"u": u, "last_q_with_plus": last["q"], "mult_there": last["mult_plus"],
                 "gap_ok_there": last["gap_ok_plus"],
                 "next_q": nxt[0]["q"] if nxt else None,
                 "stable_words_next": nxt[0]["stable_words"] if nxt else None}
            summary.append(s)
            print("  u=%.2f: (+) last at q=%.2f with P'=%+.3f gap_ok=%s; at q=%.2f stable words %s"
                  % (u, s["last_q_with_plus"], s["mult_there"], s["gap_ok_there"],
                     s["next_q"] or float("nan"), s["stable_words_next"]))
    json.dump({"rows": rows, "summary": summary}, open(OUT, "w"), indent=1, default=float)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
