#!/usr/bin/env python3
"""
d107g -- the (+,-,+) multiplier in closed form, checked against the map.

Dimensionless tau = T/Delta, zeta = z0/Delta, 2q = k Delta, theta = u/(2q) in
tau units.  For a general section state zeta (not only the fixed point), with the
event order 0 < theta < tau1 < tau1+theta <= tau2 < tau2+theta < tau3, the three
crossing conditions are

  F1:  (1-u) tau1 - 2q zeta tau1 + q tau1^2 + u                  = 0
  F2:  (1-u) tau2 - 2q zeta tau2 + q tau2^2 + 2q (tau2 - tau1)   = 0
  F3:  (1-u) tau3 - 2q zeta tau3 + q tau3^2 + 2q (tau2 - tau1) + u - 1 = 0

and P(zeta) = zeta - tau3 + 1, so P' = 1 - d tau3/d zeta.  Implicit
differentiation at the fixed point (tau3 = 1, zeta = 1/2 + sigma) gives

  tau1' = 2q tau1 / D1,                 D1 = 1 - u - q - 2q sigma + 2q tau1
  tau2' = 2q (tau1' + tau2) / D2,       D2 = D1 - 2q tau1 + 2q tau2 + 2q
  P'    = 1 - 2q [1 - (tau2' - tau1')] / D3,   D3 = 1 - u + q - 2q sigma

with (tau1, tau2) the periodic solution of F1, F2 at zeta = 1/2 + sigma.

Usage:  python d107g_pmp_multiplier_closed.py
"""
import json
import math
import os

import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "d107f_pmp_stability_2026-09-13.json")
OUT = os.path.join(HERE, "d107g_pmp_multiplier_closed_2026-09-13.json")
D = math.radians(45.0)


def periodic(u, q, guess):
    def eqs(p):
        t1, t2 = p
        s = t2 - t1
        a = 1 - u - q - 2 * q * s
        return [a * t1 + q * t1 * t1 + u, a * t2 + q * t2 * t2 + 2 * q * s]
    sol, info, ier, msg = fsolve(eqs, guess, full_output=True, xtol=1e-14)
    ok = ier == 1 and max(abs(x) for x in eqs(sol)) < 1e-11
    return sol, ok


def mult_closed(u, q, t1, t2):
    s = t2 - t1
    D1 = 1 - u - q - 2 * q * s + 2 * q * t1
    D2 = D1 - 2 * q * t1 + 2 * q * t2 + 2 * q
    D3 = 1 - u + q - 2 * q * s
    t1p = 2 * q * t1 / D1
    t2p = 2 * q * (t1p + t2) / D2
    return 1 - 2 * q * (1 - (t2p - t1p)) / D3


def valid(u, q, t1, t2):
    th = u / (2 * q)
    return (th < t1) and (t2 - t1 >= th - 1e-12) and (t2 + th <= 1 + 1e-12) and (0 < t1 < t2 < 1)


def main():
    num = json.load(open(F, encoding="utf-8"))
    out = {"check": [], "scan": {}}
    print("(1) closed form vs map, at the map points whose estimate is trustworthy (IQR < 1e-4, word -++, gap_ok)")
    errs = []
    for us, rows in num.items():
        u = float(us)
        for r in rows:
            if "mult" not in r or not np.isfinite(r["mult"]) or r["spread"] > 1e-4:
                continue
            if r["word"] != "-++" or not r["gap_ok"]:
                continue
            q = r["q"]
            # periodic solution from the tau form of d26's (t1, t2): re-solve with a spread of guesses
            best = None
            for g1 in np.linspace(0.05, 0.7, 8):
                for dg in (0.05, 0.1, 0.2, 0.3):
                    sol, ok = periodic(u, q, [g1, g1 + dg])
                    if ok and valid(u, q, *sol):
                        m = mult_closed(u, q, *sol)
                        if best is None or abs(m - r["mult"]) < abs(best[0] - r["mult"]):
                            best = (m, sol)
            if best:
                errs.append(abs(best[0] - r["mult"]))
                out["check"].append({"u": u, "q": q, "map": r["mult"], "closed": best[0],
                                     "tau1": best[1][0], "tau2": best[1][1]})
    if errs:
        print("    n = %d, median |diff| %.2e, max |diff| %.2e" % (len(errs), np.median(errs), np.max(errs)))
        bad = [c for c in out["check"] if abs(c["map"] - c["closed"]) > 1e-3]
        for c in bad[:8]:
            print("    disagree u=%.2f q=%.3f map %+.4f closed %+.4f" % (c["u"], c["q"], c["map"], c["closed"]))

    print("\n(2) along the branch: stability interval and where |P'| reaches 1")
    for u in (0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.9):
        qlo = 1 + math.sqrt(max(0.0, 1 - (2 * u - 1) ** 2 + u * u / 4))
        guess = None
        rows = []
        for q in np.arange(qlo + 0.002, 6.0, 0.01):
            cands = []
            gl = [guess] if guess is not None else []
            gl += [[g1, g1 + dg] for g1 in np.linspace(0.05, 0.7, 6) for dg in (0.05, 0.15, 0.3)]
            for g in gl:
                sol, ok = periodic(u, float(q), g)
                if ok and valid(u, float(q), *sol):
                    cands.append(sol)
            if not cands:
                rows.append({"q": float(q), "end": True})
                break
            sol = min(cands, key=lambda s: np.hypot(*(s - guess)) if guess is not None else 0)
            guess = list(sol)
            rows.append({"q": float(q), "P": float(mult_closed(u, float(q), *sol)),
                         "tau1": float(sol[0]), "tau2": float(sol[1])})
        good = [r for r in rows if "P" in r]
        out["scan"][str(u)] = rows
        if not good:
            # above u ~ 0.85 the branch starts on the second border (t2 + theta = Delta),
            # not on q_lo1, so no valid main-piece solution exists just above q_lo1
            print("  u=%.2f  no main-piece solution above q_lo1 = %.3f (second border binds)" % (u, qlo))
            continue
        flip = next((r["q"] for r in good if r["P"] <= -1), None)
        fold = next((r["q"] for r in good if r["P"] >= 1), None)
        cross0 = next((r["q"] for r in good if r["P"] <= 0), None)
        out["scan"][str(u)] = rows
        print("  u=%.2f  q_lo=%.3f  valid to q=%.3f%s  P'(q_lo+)=%+.3f  P'=0 at %s  P'=-1 at %s  P'=+1 at %s  P'(end)=%+.3f"
              % (u, qlo, good[-1]["q"], " (ended)" if rows[-1].get("end") else "",
                 good[0]["P"], cross0, flip, fold, good[-1]["P"]))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
