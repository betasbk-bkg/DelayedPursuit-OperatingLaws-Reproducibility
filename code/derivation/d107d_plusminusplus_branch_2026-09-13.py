#!/usr/bin/env python3
"""
d107d -- continue the (+,-,+) periodic orbit in q and find what ends it.

d107c's scan for the lower edge of the coexistence band was inconsistent: the
sign-change root finder of d27 picks up spurious roots on a discontinuous map.
This script instead continues d26's exact closure (z0, t1, t2) in q, using the
previous solution as the next initial guess, and at each q records:

  * the residual (is it a genuine periodic orbit?)
  * its multiplier, from d27's exact orbit integrator at z0 (finite difference)
  * the margins of every inequality the word needs:
        t1 > 0,  t2 - t1 > 0,  Delta - t2 > 0,
        distances from each crossing to the delay-window breakpoints t + th,
        and d27's gap_ok (no earlier crossing still inside the delay window)
    -- the constraint whose margin reaches zero where the branch ends is the
    border it collides with, and that condition can then be solved in closed form.

Continuation runs DOWN from q_bc(u) = (1 + sqrt u)^2 (where the other word is known
to exist) until the solve fails, and UP a little beyond it.

Usage:  python d107d_plusminusplus_branch.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.optimize import fsolve

DERIV = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(DERIV, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
d27 = _load("d27", "d27_branch_selection_2026-08-29.py")
D = math.radians(45.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107d_plusminusplus_branch_2026-09-13.json")


def solve_from(p0, u, k):
    th = u / k
    sol, info, ier, msg = fsolve(d26.residuals, p0, args=(u, k, D, th), full_output=True, xtol=1e-12)
    r = max(abs(x) for x in d26.residuals(sol, u, k, D, th))
    ok = ier == 1 and r < 1e-8 and 0 < sol[1] < sol[2] < D
    return sol, r, ok


def record(sol, u, k, q):
    z0, t1, t2 = sol
    th = u / k
    o = d27.orbit(z0, u, k, D, th)
    h = 1e-6
    op, om = d27.orbit(z0 + h, u, k, D, th), d27.orbit(z0 - h, u, k, D, th)
    mult = (op["z_next"] - om["z_next"]) / (2 * h) if (op and om) else float("nan")
    word = o["word"] if o else None
    crossings = [0.0, t1, t2]
    bps = [c + th for c in crossings] + [c + th - D for c in crossings]
    bp_margin = min(abs(c - b) for c in crossings + [D] for b in bps)
    return {"q": q, "z0": z0, "t1": t1, "t2": t2, "mult": float(mult), "word_map": word,
            "gap_ok": bool(o["min_gap"] > o["th"]) if o else None,
            "margin_t1": t1, "margin_t2_t1": t2 - t1, "margin_D_t2": D - t2,
            "margin_breakpoint": bp_margin, "th": th}


def main():
    t0 = time.time()
    out = {}
    for u in (0.3, 0.5, 0.71, 0.9):
        qbc = (1 + math.sqrt(u)) ** 2
        q = qbc + 0.2
        k = 2 * q / D
        s = d26.solve_word(u, k, D)
        if not s:
            print("u=%.2f: no starting solution at q=%.3f" % (u, q))
            continue
        p = [s["z0"], s["t1"], s["t2"]]
        rows = []
        dq = 0.01
        while q > 0.8:
            k = 2 * q / D
            sol, r, ok = solve_from(p, u, k)
            if not ok:
                # halve the step once before declaring the end
                sol2, r2, ok2 = solve_from(p, u, 2 * (q + dq / 2) / D)
                rows.append({"q": q, "end": True, "resid": r})
                break
            rec = record(sol, u, k, q)
            rows.append(rec)
            p = list(sol)
            q = round(q - dq, 6)
        good = [x for x in rows if not x.get("end")]
        last = good[-1] if good else None
        out[str(u)] = rows
        if last:
            ms = {kk: last[kk] for kk in ("margin_t1", "margin_t2_t1", "margin_D_t2", "margin_breakpoint")}
            tight = min(ms, key=ms.get)
            stable = [x for x in good if abs(x["mult"]) < 1]
            print("u=%.2f: branch continued from q=%.3f down to q=%.3f; tightest margin there: %s=%.4f;"
                  " mult there %+.3f; word %s; stable for q in [%.3f, %.3f]"
                  % (u, good[0]["q"], last["q"], tight, ms[tight], last["mult"], last["word_map"],
                     min(x["q"] for x in stable) if stable else float("nan"),
                     max(x["q"] for x in stable) if stable else float("nan")))
            for x in good[-4:]:
                print("     q=%.3f mult %+.4f t1 %.4f t2-t1 %.4f D-t2 %.4f bp %.4f gap_ok %s"
                      % (x["q"], x["mult"], x["margin_t1"], x["margin_t2_t1"], x["margin_D_t2"],
                         x["margin_breakpoint"], x["gap_ok"]))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
