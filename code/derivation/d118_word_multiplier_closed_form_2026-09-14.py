#!/usr/bin/env python3
"""
d118 -- the multiplier of every word W_n in closed form, and where |P'| < 1.

SECTION MAP.  From a forward crossing with section state zeta, the phase in cell
units is M(tau) = c tau + q tau^2 + 2q min(tau, theta) - 2q sum_e s_e max(0, tau - tau_e - theta),
c = 1 - u - 2 q zeta.  After the last window of a W_n closes, M = q tau^2 + c tau + u + 2q S_n
exactly, so the next forward crossing T solves q T^2 + c T + u + 2q S_n = 1, and
z regains one cell there: P(zeta) = zeta + 1 - T.  S_n = F_n(c) through the recursion
(G(tau) = -q tau^2 - c tau):

    G(b_j) = u + 2q S_(j-1)        (b_j on the rising side of G)
    q f_j^2 + (c + 2q) f_j + 2q (S_(j-1) - b_j) = 0   (f_j the upward root)
    S_j = S_(j-1) + f_j - b_j.

DERIVATIVE.  With beta_j = -2q b_j - c (downward slope of M at b_j) and
phi_j = 2q f_j + c + 2q (upward slope at f_j), differentiating in c gives the linear recursion

    S'_j = A_j S'_(j-1) - B_j,
    A_j = (1 - 2q/phi_j)(1 - 2q/beta_j),
    B_j = f_j/phi_j + (1 - 2q/phi_j) b_j/beta_j,

and, since dc/dzeta = -2q and the final slope at T = 1 is c + 2q,

    P' = (c - 4 q^2 S'_n) / (c + 2q),

which for n = 0 is (1 - u - q)/(1 - u + q).

CHECKS.  (1) P' against a central difference of P(zeta) computed from the recursion
with the word held fixed; (2) P' against the multipliers the validated integrator v3
measured in d116; (3) a scan of every existence interval of d116b (u = 0.10-0.95,
n = 0-5): min/max of P', whether |P'| < 1 everywhere, and the signs of the factors
(1 - 2q/phi_j), (1 - 2q/beta_j), S'_n that a proof would use.

Usage:  python d118_word_multiplier_closed_form_2026-09-14.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d116", os.path.join(HERE, "d116_word_sequence_closure_2026-09-13.py"))
w = importlib.util.module_from_spec(_s)
_s.loader.exec_module(w)


def rec_c(c, u, q, n):
    """recursion with c free (the word held fixed): S_n and the crossings, or None."""
    S, ev = 0.0, []
    for _ in range(n):
        D1 = c * c - 4 * q * (u + 2 * q * S)
        if D1 < 0:
            return None
        b = (-c - math.sqrt(D1)) / (2 * q)
        B2, C2 = c + 2 * q, 2 * q * (S - b)
        D2 = B2 * B2 - 4 * q * C2
        if D2 < 0:
            return None
        f = (-B2 + math.sqrt(D2)) / (2 * q)
        ev.append((b, f))
        S += f - b
    return S, ev


def P_of_zeta(zeta, u, q, n):
    c = 1 - u - 2 * q * zeta
    r = rec_c(c, u, q, n)
    if r is None:
        return None
    S = r[0]
    disc = c * c - 4 * q * (u + 2 * q * S - 1)
    if disc < 0:
        return None
    T = (-c + math.sqrt(disc)) / (2 * q)
    return zeta + 1 - T


def multiplier(u, q, n, sol):
    """closed-form P' at a verified fixed point; also the factor signs."""
    Sn = sol["S_n"]
    c = 1 - u - q - 2 * q * Sn
    r = rec_c(c, u, q, n)
    Sp = 0.0
    fac = []
    for b, f in r[1]:
        beta = -2 * q * b - c
        phi = 2 * q * f + c + 2 * q
        A = (1 - 2 * q / phi) * (1 - 2 * q / beta)
        B = f / phi + (1 - 2 * q / phi) * b / beta
        Sp = A * Sp - B
        fac.append({"beta": beta, "phi": phi, "one_m_2q_phi": 1 - 2 * q / phi, "one_m_2q_beta": 1 - 2 * q / beta, "A": A, "B": B})
    return (c - 4 * q * q * Sp) / (c + 2 * q), Sp, c, fac


def main():
    t0 = time.time()
    out = {"fd": [], "v3": [], "scan": []}

    # (2) against v3, and (1) against the finite difference, at the d116 sample points
    s116 = json.load(open(os.path.join(HERE, "d116_word_sequence_closure_2026-09-13.json"), encoding="utf-8"))
    for r in s116["v3"]:
        if not r["ok"] or r["mult"] is None:
            continue
        sol = w.exists(r["u"], r["q"], r["n"])
        if sol is None:
            continue
        Pp = multiplier(r["u"], r["q"], r["n"], sol)[0]
        z = sol["zeta"]
        h = 1e-6
        pa, pb = P_of_zeta(z + h, r["u"], r["q"], r["n"]), P_of_zeta(z - h, r["u"], r["q"], r["n"])
        fd = (pa - pb) / (2 * h) if pa is not None and pb is not None else None
        out["v3"].append({"u": r["u"], "n": r["n"], "q": r["q"], "closed": Pp, "v3": r["mult"], "fd": fd})
    ev3 = max(abs(x["closed"] - x["v3"]) for x in out["v3"])
    efd = max(abs(x["closed"] - x["fd"]) for x in out["v3"] if x["fd"] is not None)
    print("(1,2) closed-form P' at %d fixed points: max |closed - finite difference| %.1e, max |closed - v3| %.1e"
          % (len(out["v3"]), efd, ev3))

    # (3) scan of the existence intervals
    bd = json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))
    for row in bd:
        if not row["exists"]:
            continue
        u, n = row["u"], row["n"]
        lo = row["birth"] if row["birth"] is not None else 0.05
        hi = row["death"]
        if hi is None:
            continue
        qs = lo + (hi - lo) * (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, 81)))[1:-1]
        vals, signs = [], {"phi": set(), "beta": set(), "Sp": set()}
        for q in qs:
            sol = w.exists(u, float(q), n)
            if sol is None:
                continue
            Pp, Sp, c, fac = multiplier(u, float(q), n, sol)
            vals.append((float(q), Pp))
            signs["Sp"].add(int(np.sign(Sp)))
            for fc in fac:
                signs["phi"].add(int(np.sign(fc["one_m_2q_phi"])))
                signs["beta"].add(int(np.sign(fc["one_m_2q_beta"])))
        if not vals:
            continue
        Ps = [v for _, v in vals]
        out["scan"].append({"u": u, "n": n, "birth": row["birth"], "death": hi, "n_points": len(vals),
                            "P_min": min(Ps), "P_max": max(Ps), "stable_all": bool(max(abs(p) for p in Ps) < 1),
                            "P_near_birth": vals[0], "P_near_death": vals[-1],
                            "sign_1m2q_phi": sorted(signs["phi"]), "sign_1m2q_beta": sorted(signs["beta"]),
                            "sign_Sprime": sorted(signs["Sp"])})
    ok = all(r["stable_all"] for r in out["scan"])
    print("(3) %d existence intervals scanned (u = 0.10-0.95, n = 0-5): |P'| < 1 everywhere: %s"
          % (len(out["scan"]), ok))
    print("    P' range over all: %.4f .. %.4f" % (min(r["P_min"] for r in out["scan"]), max(r["P_max"] for r in out["scan"])))
    for n in range(6):
        rows = [r for r in out["scan"] if r["n"] == n]
        if rows:
            print("    n=%d: P' %.4f..%.4f; sign(1-2q/phi) %s  sign(1-2q/beta) %s  sign(S') %s"
                  % (n, min(r["P_min"] for r in rows), max(r["P_max"] for r in rows),
                     sorted(set(sum((r["sign_1m2q_phi"] for r in rows), []))),
                     sorted(set(sum((r["sign_1m2q_beta"] for r in rows), []))),
                     sorted(set(sum((r["sign_Sprime"] for r in rows), [])))))
    json.dump(out, open(os.path.join(HERE, "d118_word_multiplier_closed_form_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d118_word_multiplier_closed_form_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
