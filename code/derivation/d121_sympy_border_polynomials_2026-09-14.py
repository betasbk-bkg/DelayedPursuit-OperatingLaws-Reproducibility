#!/usr/bin/env python3
"""
d121 -- birth borders and codimension-two points as polynomials (sympy elimination).

W_1, end-window border (f = 1 - theta).  c = 1 - 3q + 2qb; the b- and f-equations give
the border curve R(u, q) = 0 (quadratic in q), and P_B = 1 (equivalently B = 1/(2q))
eliminated against it gives the switch-back point as the root of one polynomial in u.

W_2, short-excursion border.  The short excursion at every W_n birth is the last one
(d119c: 58 of 58), so the parabola symmetry of d119d gives b_2 = -(c + u/2)/(2q)
linearly.  Unknowns (q, c, b_1) with S_1 = (1 - 2u - q - c)/(2q), f_1 = b_1 + S_1:
    G1: q b1^2 + c b1 + u = 0
    G2: q f1^2 + (c + 2q) f1 - 2q b1 = 0
    G3: q b2^2 + c b2 + u + 2q S_1 = 0
    G4: S'_2 = -1/(2q)   (P_B = 1), S'_2 = A_2 S'_1 - B_2, A_2 = (u - 4q)/(u + 4q),
        B_2 = (2 b2 + theta)/(2q + u/2), S'_1 from T1.
Resultants eliminate b1 then c; at every step only the irreducible factor that
vanishes at the numerical point (d119c/d119d) is kept.

Usage:  python d121_sympy_border_polynomials_2026-09-14.py
"""
import json
import math
import os
import time
import importlib.util

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d118", os.path.join(HERE, "d118_word_multiplier_closed_form_2026-09-14.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)

u, q, c, b, b1 = sp.symbols("u q c b b1")


def pick(expr, point, var_keep, tol=1e-4):
    """irreducible factor of expr that vanishes at the numerical point.

    A relative residual (value over largest term) is useless for large polynomials:
    their terms cancel to ~1e-33 everywhere (the first version picked a wrong factor
    that way).  The test is the ratio of |f| at the point to |f| at the point with
    every coordinate shifted by 1e-3, in 60-digit arithmetic: a true zero gives a ratio
    of order (point accuracy)/1e-3, a non-zero factor a ratio of order 1."""
    import mpmath as mp
    mp.mp.dps = 60
    syms = sorted(point.keys(), key=str)
    shifted = {s: point[s] + sp.Float("1e-3", 60) for s in syms}
    best = None
    for fac, _ in sp.factor_list(sp.expand(expr))[1]:
        if not fac.free_symbols:
            continue
        fn = sp.lambdify(syms, fac, "mpmath")
        at = abs(fn(*[mp.mpf(str(point[s])) for s in syms]))
        off = abs(fn(*[mp.mpf(str(shifted[s])) for s in syms]))
        ratio = at / off if off != 0 else mp.inf
        if best is None or ratio < best[0]:
            best = (ratio, fac)
    if best is None or best[0] > tol:
        raise RuntimeError("no vanishing factor (best ratio %s)" % (best[0] if best else None))
    return best[1]


def main():
    t0 = time.time()
    out = {}
    # ---------------- W_1 end-window border
    cc = 1 - 3 * q + 2 * q * b
    f = 1 - u / (2 * q)
    e1 = sp.expand(q * b ** 2 + cc * b + u)
    e2 = sp.expand(sp.numer(sp.together(q * f ** 2 + (cc + 2 * q) * f - 2 * q * b)))
    bet = -2 * q * b - cc
    ph = 2 * q * f + cc + 2 * q
    B = f / ph + (1 - 2 * q / ph) * b / bet
    cond = sp.expand(sp.numer(sp.together(B - 1 / (2 * q))))
    R1 = sp.factor(sp.resultant(e1, e2, b))
    R2 = sp.resultant(e1, cond, b)
    border1 = [g for g, _ in sp.factor_list(R1)[1] if g.has(q) and g.has(u) and sp.degree(g, q) >= 1 and g != q][0]
    polys = []
    for g2, _ in sp.factor_list(R2)[1]:
        if not (g2.has(q) and g2.has(u)):
            continue
        for h, _ in sp.factor_list(sp.resultant(border1, g2, q))[1]:
            if h.has(u):
                rts = [complex(r) for r in sp.Poly(h, u).nroots(n=25)]
                rr = [r.real for r in rts if abs(r.imag) < 1e-18 and 0.82 < r.real < 0.86]
                if rr:
                    polys.append((str(h), rr))
    out["W1_end_border_curve"] = str(border1)
    out["W1_end_switch_polynomial"] = polys
    print("W_1 end-window border curve: %s = 0" % border1)
    for h, rr in polys:
        print("W_1 end-window switch-back: root %s of  %s" % (rr, h))

    # ---------------- W_2 short-excursion border
    d = json.load(open(os.path.join(HERE, "d119d_birth_multiplier_closed_form_2026-09-14.json"), encoding="utf-8"))
    r2 = [x for x in d["c"] if x["n"] == 2 and x["status"] == "ok"][0]
    uv, qv, Sv = r2["u_switch"], r2["q_B"], r2["S_B"]
    cv = 1 - uv - qv - 2 * qv * Sv
    ev = m.rec_c(cv, uv, qv, 2)[1]
    b1v = ev[0][0]
    point = {u: sp.Float(uv, 30), q: sp.Float(qv, 30), c: sp.Float(cv, 30), b1: sp.Float(b1v, 30)}
    th = u / (2 * q)
    S1 = (1 - 2 * u - q - c) / (2 * q)
    f1 = b1 + S1
    b2 = -(c + u / 2) / (2 * q)
    G1 = sp.expand(q * b1 ** 2 + c * b1 + u)
    G2 = sp.expand(sp.numer(sp.together(q * f1 ** 2 + (c + 2 * q) * f1 - 2 * q * b1)))
    G3 = sp.expand(sp.numer(sp.together(q * b2 ** 2 + c * b2 + u + 2 * q * S1)))
    beta1 = -2 * q * b1 - c
    phi1 = 2 * q * f1 + c + 2 * q
    Sp1 = -(f1 / phi1 + (1 - 2 * q / phi1) * b1 / beta1)
    A2 = (u - 4 * q) / (u + 4 * q)
    B2 = (2 * b2 + th) / (2 * q + u / 2)
    G4 = sp.expand(sp.numer(sp.together(A2 * Sp1 - B2 + 1 / (2 * q))))
    for name, G in (("G1", G1), ("G2", G2), ("G3", G3), ("G4", G4)):
        print("  %s at the numerical point: %.2e" % (name, abs(complex(G.evalf(30, subs=point)))))
    Ha = pick(sp.resultant(G1, G2, b1), point, None)
    print("  H12(c,q,u): degree c %d, q %d  (%.0f s)" % (sp.degree(Ha, c), sp.degree(Ha, q), time.time() - t0))
    G3p = pick(G3, point, None)
    border2 = pick(sp.resultant(Ha, G3p, c), point, None)
    out["W2_short_border_curve"] = str(border2)
    print("W_2 short-excursion border curve (degrees q %d, u %d): %s = 0   (%.0f s)"
          % (sp.degree(border2, q), sp.degree(border2, u), str(border2)[:300], time.time() - t0))
    Hb = pick(sp.resultant(G1, G4, b1), point, None)
    print("  H14(c,q,u): degree c %d, q %d  (%.0f s)" % (sp.degree(Hb, c), sp.degree(Hb, q), time.time() - t0))
    Cb = pick(sp.resultant(Hb, G3p, c), point, None)
    print("  C(q,u) for P_B = 1: degree q %d, u %d  (%.0f s)" % (sp.degree(Cb, q), sp.degree(Cb, u), time.time() - t0))
    Upoly = pick(sp.resultant(border2, Cb, q), {u: point[u]}, None)
    # nroots does not converge at this degree: isolate the root by sign changes in
    # 80-digit arithmetic and bisect it
    import mpmath as mp
    mp.mp.dps = 80
    coeffs = [mp.mpf(int(x)) if x.is_Integer else mp.mpf(sp.Rational(x).p) / sp.Rational(x).q
              for x in sp.Poly(Upoly, u).all_coeffs()]
    grid = [mp.mpf("0.50") + mp.mpf("0.0001") * k for k in range(1201)]
    vals = [mp.polyval(coeffs, x) for x in grid]
    rr = []
    for k in range(len(grid) - 1):
        if vals[k] == 0 or vals[k] * vals[k + 1] < 0:
            lo_, hi_ = grid[k], grid[k + 1]
            flo = vals[k]
            for _ in range(200):
                mid = (lo_ + hi_) / 2
                fm = mp.polyval(coeffs, mid)
                if fm * flo > 0:
                    lo_, flo = mid, fm
                else:
                    hi_ = mid
            rr.append(float((lo_ + hi_) / 2))
            out.setdefault("W2_codim2_root_30digits", []).append(mp.nstr((lo_ + hi_) / 2, 30))
    out["W2_codim2_polynomial"] = str(Upoly)
    out["W2_codim2_roots_in_window"] = rr
    print("W_2 codimension-two point: degree-%d polynomial in u, real roots in (0.5, 0.62): %s  [numerical %.10f]  (%.0f s)"
          % (sp.degree(Upoly, u), rr, uv, time.time() - t0))
    json.dump(out, open(os.path.join(HERE, "d121_sympy_border_polynomials_2026-09-14.json"), "w"), indent=1)
    print("-> d121_sympy_border_polynomials_2026-09-14.json")


if __name__ == "__main__":
    main()
