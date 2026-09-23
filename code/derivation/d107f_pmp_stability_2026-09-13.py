#!/usr/bin/env python3
"""
d107f -- stability of the (+,-,+) branch, measured without crossing a discontinuity.

d107d's multipliers for this branch contained values like -1e5: a central
difference whose two perturbed orbits realised different event sequences, i.e.
a difference taken ACROSS a switching surface of the piecewise map.  Here:

  * the orbit through z0 +- h is integrated for several h (1e-5 .. 1e-8);
  * a difference is kept only if both perturbed orbits realise the same word and
    the same number of events as the unperturbed one (same smooth piece);
  * the multiplier is the median of the kept one-sided and central estimates,
    and the spread of those estimates is reported as its uncertainty.

Scanned from the lower edge q_lo(u) up to q_bc(u) + 1, continuing d26's closure,
so the branch's whole stability interval -- and how it loses stability, if it
does (+1 fold, -1 flip, or a border) -- is visible.

Usage:  python d107f_pmp_stability.py
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107f_pmp_stability_2026-09-13.json")


def q_lo1(u):
    return 1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4)


def multiplier(z0, u, k):
    th = u / k
    base = d27.orbit(z0, u, k, D, th)
    if not base:
        return float("nan"), float("nan"), 0
    key = (base["word"],)
    est = []
    for h in (1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8):
        op = d27.orbit(z0 + h, u, k, D, th)
        om = d27.orbit(z0 - h, u, k, D, th)
        okp = op is not None and (op["word"],) == key
        okm = om is not None and (om["word"],) == key
        if okp and okm:
            est.append((op["z_next"] - om["z_next"]) / (2 * h))
        if okp:
            est.append((op["z_next"] - base["z_next"]) / h)
        if okm:
            est.append((base["z_next"] - om["z_next"]) / h)
    if not est:
        return float("nan"), float("nan"), 0
    est = np.array(est)
    med = float(np.median(est))
    spread = float(np.percentile(est, 75) - np.percentile(est, 25))
    return med, spread, len(est)


def main():
    t0 = time.time()
    out = {}
    for u in (0.3, 0.5, 0.71, 0.9):
        qbc = (1 + math.sqrt(u)) ** 2
        q = (q_lo1(u) if u < 0.85 else 1.83) + 0.01
        k = 2 * q / D
        s = d26.solve_word(u, k, D)
        if not s:
            print("u=%.2f: no start" % u)
            continue
        p = [s["z0"], s["t1"], s["t2"]]
        rows = []
        while q < qbc + 1.0:
            k = 2 * q / D
            th = u / k
            sol, info, ier, msg = fsolve(d26.residuals, p, args=(u, k, D, th), full_output=True, xtol=1e-12)
            r = max(abs(x) for x in d26.residuals(sol, u, k, D, th))
            if not (ier == 1 and r < 1e-8 and 0 < sol[1] < sol[2] < D):
                s2 = d26.solve_word(u, k, D)
                if s2 and s2["resid"] < 1e-8:
                    sol = np.array([s2["z0"], s2["t1"], s2["t2"]])
                else:
                    rows.append({"q": q, "end": "no solution"})
                    break
            p = list(sol)
            m, spr, n = multiplier(sol[0], u, k)
            o = d27.orbit(sol[0], u, k, D, th)
            rows.append({"q": q, "mult": m, "spread": spr, "n_est": n,
                         "word": o["word"] if o else None,
                         "gap_ok": bool(o and o["min_gap"] > o["th"])})
            q = round(q + 0.02, 6)
        out[str(u)] = rows
        good = [x for x in rows if "mult" in x and np.isfinite(x["mult"])]
        stab = [x for x in good if abs(x["mult"]) < 1]
        print("\nu=%.2f  q_lo=%.3f  q_bc=%.3f   branch solved for q in [%.3f, %.3f]%s"
              % (u, q_lo1(u), qbc, good[0]["q"], good[-1]["q"],
                 "  (ended: %s)" % rows[-1]["end"] if rows and rows[-1].get("end") else ""))
        print("   stable (|P'|<1) on %d of %d points; first unstable q: %s"
              % (len(stab), len(good), next((("%.3f (P'=%+.3f)" % (x["q"], x["mult"])) for x in good if abs(x["mult"]) >= 1), "none")))
        for x in good[:: max(1, len(good) // 8)]:
            print("   q=%.3f  P'=%+.4f  (IQR %.1e, n=%d)  word %s gap_ok %s"
                  % (x["q"], x["mult"], x["spread"], x["n_est"], x["word"], x["gap_ok"]))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
