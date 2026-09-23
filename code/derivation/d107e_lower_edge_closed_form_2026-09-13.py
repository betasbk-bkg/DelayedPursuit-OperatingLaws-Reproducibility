#!/usr/bin/env python3
"""
d107e -- the lower edge of the coexistence band in closed form.

For the (+,-,+) orbit (crossings at 0 forward, t1 backward, t2 forward, period
Delta, z0 = Delta/2 + (t2 - t1)), integrating m' = (1-u) - k z(T) + k Delta J(T)
segment by segment, the jump of z at t1 and the delay-window term cancel on the
backward excursion, and the end-of-period condition is an identity.  With
tau = T/Delta, sigma = (t2 - t1)/Delta, a = 1 - u - q - 2 q sigma, q = k Delta/2:

    a tau1 + q tau1^2 + u = 0                                  (crossing at t1)
    a tau2 + q tau2^2 + 2 q sigma - 2 q max(0, sigma - u/(2q))
          ... reduces, for sigma >= u/(2q), to  a tau2 + q tau2^2 + 2 q sigma = 0

Border 1, t2 - t1 = theta_d (sigma = u/(2q)): t1, t2 are the roots of one
quadratic, and their separation gives

    q_lo1(u) = 1 + sqrt(1 - (2u - 1)^2 + u^2/4)

Border 2, t2 + theta_d = Delta (tau2 = 1 - u/(2q)): two equations in (sigma, q),
solved here numerically (it is algebraic; a closed form is attempted with sympy).

The branch ends at whichever border it meets first coming down in q, i.e. at
q_lo(u) = max(q_lo1, q_lo2) where both are defined.

Checks against direct continuation of d26's closure with a fine q step.

Usage:  python d107e_lower_edge_closed_form.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.optimize import brentq, fsolve

DERIV = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(DERIV, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
D = math.radians(45.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107e_lower_edge_closed_form_2026-09-13.json")


def q_lo1(u):
    r = 1 - (2 * u - 1) ** 2 + u * u / 4
    return 1 + math.sqrt(r) if r >= 0 else float("nan")


def q_lo2(u):
    """t2 + theta = Delta: unknowns sigma, q."""
    def eqs(p):
        sig, q = p
        a = 1 - u - q - 2 * q * sig
        t2 = 1 - u / (2 * q)
        t1 = t2 - sig
        return [a * t1 + q * t1 * t1 + u, a * t2 + q * t2 * t2 + 2 * q * sig]
    best = None
    for q0 in np.linspace(1.2, 4.0, 15):
        for s0 in (0.1, 0.2, 0.3, 0.4):
            sol, info, ier, msg = fsolve(eqs, [s0, q0], full_output=True, xtol=1e-13)
            if ier == 1 and max(abs(x) for x in eqs(sol)) < 1e-10:
                sig, q = sol
                t2 = 1 - u / (2 * q)
                t1 = t2 - sig
                if 0 < t1 < t2 < 1 and sig >= u / (2 * q) - 1e-12 and q > 1:
                    if best is None or q > best[0]:
                        best = (q, sig, t1, t2)
    return best


def continuation_end(u, dq=0.001):
    qbc = (1 + math.sqrt(u)) ** 2
    q = qbc + 0.15
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
            break
        p = list(sol)
        last = q
        q = round(q - dq, 6)
    return last


def main():
    t0 = time.time()
    rows = []
    print("   u     q_lo1(closed)  q_lo2(border 2)  predicted end  continuation end   diff")
    for u in np.round(np.arange(0.1, 0.96, 0.05), 3):
        a = q_lo1(float(u))
        b2 = q_lo2(float(u))
        b = b2[0] if b2 else float("nan")
        pred = np.nanmax([a, b])
        c = continuation_end(float(u))
        rows.append({"u": float(u), "q_lo1": a, "q_lo2": b, "pred": float(pred), "cont": c})
        print("  %.2f    %8.4f       %8.4f         %8.4f         %8s        %s"
              % (u, a, b, pred, ("%.4f" % c) if c else "   -",
                 ("%+.4f" % (c - pred)) if c else "-"))
    ok = [r for r in rows if r["cont"]]
    if ok:
        print("\n  max |continuation - prediction| = %.4f over %d u (dq = 0.001)"
              % (max(abs(r["cont"] - r["pred"]) for r in ok), len(ok)))
    try:
        import sympy as sp
        u_, q_, s_ = sp.symbols("u q sigma", positive=True)
        a = 1 - u_ - q_ - 2 * q_ * s_
        t2 = 1 - u_ / (2 * q_)
        t1 = t2 - s_
        e1 = sp.expand((a * t1 + q_ * t1 ** 2 + u_) * 4 * q_ ** 2)
        e2 = sp.expand((a * t2 + q_ * t2 ** 2 + 2 * q_ * s_) * 4 * q_)
        res = sp.factor(sp.resultant(e1, e2, s_))
        print("\n  border 2, resultant in q (sigma eliminated):\n   ", res)
        rows_sym = str(res)
    except Exception as ex:  # sympy may be absent
        rows_sym = "sympy unavailable: %s" % ex
        print("\n  " + rows_sym)
    json.dump({"rows": rows, "border2_resultant": rows_sym}, open(OUT, "w"), indent=1, default=float)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
