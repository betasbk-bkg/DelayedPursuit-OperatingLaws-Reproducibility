#!/usr/bin/env python3
"""
d119d -- the multiplier at a short-excursion birth border in closed form, and the
codimension-two point where the birth changes type.

AT ANY SHORT-EXCURSION BORDER (f_j - b_j = theta).  On (b_j, b_j + theta) the phase is
still the parabola q tau^2 + c tau + r_j, which vanishes at both b_j and f_j = b_j + theta,
so it is symmetric about b_j + theta/2: its slope at b_j is -q theta = -u/2 and at f_j is
+u/2.  Hence, for every n and every j at the border,

    beta_j = u/2,   phi_j = 2q + u/2,   A_j = (u - 4q)/(u + 4q),   B_j = (f_j + b_j)/phi_j.

W_1.  With S = theta and c = 1 - 2u - q, the f-equation reduces to
    b = (q - 1 + 3u/2)/(2q),   f + b = (q - 1 + 2u)/q,
the b-equation to q = q_lo(u) = 1 + sqrt(1 - (2u - 1)^2 + u^2/4), and

    P_B = (q - 1 + 2u)(4q - u) / ((4q + u)(q + 1 - 2u)).

P_B = 1 reduces to q (14u - 8) = 0: the birth changes type at u = 4/7 exactly --
below it W_1 is born at the border with P_B < 1, above it at a smooth fold just below
q_lo (the stable member born there, the unstable one running to the border).
The short-excursion and end-window borders of W_1 cross where b = 1 - 2 theta as well,
q = 7u/2 - 1, i.e. 8u^2 - 9u + 2 = 0, u = (9 + sqrt 17)/16.

CHECKS.  (a) P_B and q_B of every n = 1 short-excursion border in d119c against the
closed forms; (b) beta_j = u/2 and phi_j = 2q + u/2 at every short-excursion border of
d119c (n = 1-3); (c) the codimension-two u for n = 1-5 by continuation of the border
in u and bisection of P_B = 1; (d) the second switch of W_1 on the end-window border.

Usage:  python d119d_birth_multiplier_closed_form_2026-09-14.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d119c", os.path.join(HERE, "d119c_birth_border_2d_2026-09-14.py"))
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)
m, w = C.m, C.w


def q_lo(u):
    return 1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4)


def PB1(u, q):
    return (q - 1 + 2 * u) * (4 * q - u) / ((4 * q + u) * (q + 1 - 2 * u))


def border_at(u, n, k, guess):
    r = C.solve_border(u, n, k, guess[0], guess[1])
    if r is None:
        return None
    sl = C.slacks(r[0], r[1], u, n)
    if min(v for kk, v in sl.items() if kk != k) < -1e-9:
        return None
    return r


def first_border(u, n, k):
    bd = json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))
    rows = sorted((r["u"], r["birth"]) for r in bd if r["n"] == n and r["exists"] and r["birth"] is not None)
    if not rows or not (rows[0][0] <= u <= rows[-1][0]):
        return None          # d116b has no birth for this word near this u (its q range ends at 6)
    qg = float(np.interp(u, [x[0] for x in rows], [x[1] for x in rows]))
    for dq in (0.003, 0.01, 0.03, 0.06, 0.1):
        for sol in [s for s in w.solve_word(u, qg + dq, n) if s["pattern_ok"]]:
            r = border_at(u, n, k, (qg + dq, sol["S_n"]))
            if r:
                return r
    return None


def PB_at(u, n, k):
    """border located from scratch at this u (continuing the previous border point in u
    starts outside the recursion's domain and returns NaN), and its multiplier."""
    g = first_border(u, n, k)
    return (None, None) if g is None else (m.multiplier(u, g[0], n, {"S_n": g[1]})[0], g)


def switch(n, k, u0, u1, du):
    """scan P_B(u) on a coarse grid, then bisect P_B = 1 inside the first bracket."""
    grid = np.round(np.arange(u0, u1 + 1e-12, du), 10)
    prev = None
    for u in grid:
        P, g = PB_at(float(u), n, k)
        if P is None:
            continue
        if prev is not None and (prev[1] - 1) * (P - 1) <= 0:
            lo, hi, plo = prev[0], float(u), prev[1]
            glo = prev[2]
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                Pm, gm = PB_at(mid, n, k)
                if Pm is None:
                    return {"n": n, "border": k, "status": "border lost at u=%.10f" % mid}
                if (plo - 1) * (Pm - 1) > 0:
                    lo, plo, glo = mid, Pm, gm
                else:
                    hi = mid
            return {"n": n, "border": k, "status": "ok", "u_switch": 0.5 * (lo + hi), "q_B": glo[0], "S_B": glo[1],
                    "P_below": prev[1], "P_above": P}
        prev = (float(u), P, g)
    return {"n": n, "border": k, "status": "no crossing in [%.3f, %.3f]" % (u0, u1)}


def main():
    t0 = time.time()
    out = {}
    rows = json.load(open(os.path.join(HERE, "d119c_birth_border_2d_2026-09-14.json"), encoding="utf-8"))
    ok = [r for r in rows if r["status"] == "ok"]
    a = [r for r in ok if r["n"] == 1 and r["border"] == "short_exc"]
    out["a"] = {"n": len(a), "max_err_PB": max(abs(r["P_B"] - PB1(r["u"], r["q_B"])) for r in a),
                "max_err_qB": max(abs(r["q_B"] - q_lo(r["u"])) for r in a)}
    print("(a) W_1 short-excursion borders: %d; |P_B - closed form| <= %.1e; |q_B - q_lo| <= %.1e"
          % (out["a"]["n"], out["a"]["max_err_PB"], out["a"]["max_err_qB"]))
    errs = []
    for r in ok:
        if r["border"] != "short_exc":
            continue
        u, q, n, S = r["u"], r["q_B"], r["n"], r["S_B"]
        c = 1 - u - q - 2 * q * S
        ev = m.rec_c(c, u, q, n)[1]
        th = u / (2 * q)
        j = int(np.argmin([f - b for b, f in ev]))
        b, f = ev[j]
        errs.append(max(abs((-2 * q * b - c) - u / 2), abs((2 * q * f + c + 2 * q) - (2 * q + u / 2))))
    out["b"] = {"n": len(errs), "max_err": max(errs)}
    print("(b) beta_j = u/2 and phi_j = 2q + u/2 at %d short-excursion borders (n = 1-3): max error %.1e"
          % (len(errs), max(errs)))
    out["c"] = []
    for n in range(1, 6):
        r = switch(n, "short_exc", 0.50, 0.62, 0.02)
        out["c"].append(r)
        print("(c) n=%d short-excursion border: %s" % (n, ("u* = %.10f (q_B %.8f)" % (r["u_switch"], r["q_B"])) if r["status"] == "ok" else r["status"]))
    print("    4/7 = %.10f" % (4 / 7))
    r = switch(1, "end", 0.822, 0.87, 0.004)
    out["d"] = r
    print("(d) n=1 end-window border: %s   [(9+sqrt17)/16 = %.10f]"
          % (("switch back at u = %.10f (q_B %.8f), P below %.3f above %.3f" % (r["u_switch"], r["q_B"], r["P_below"], r["P_above"])) if r["status"] == "ok" else r["status"],
             (9 + math.sqrt(17)) / 16))
    json.dump(out, open(os.path.join(HERE, "d119d_birth_multiplier_closed_form_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d119d_birth_multiplier_closed_form_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
