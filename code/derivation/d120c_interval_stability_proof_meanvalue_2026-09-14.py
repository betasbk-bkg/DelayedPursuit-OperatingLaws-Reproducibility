#!/usr/bin/env python3
"""
d120c -- the stability proof of d120/d120b with mean-value enclosures.

WHY.  d120b's naive enclosures lose the correlation between u, q and S: on a
0.002 x 0.002 box Phi(s_m; U, Q) came out 9e-3 wide and P' 0.35 wide, so interval
Newton failed almost everywhere and needed 8.3 million sub-boxes for n = 1.

HOW.  Forward-mode automatic differentiation on outward-rounded interval vectors
(value plus three gradient components d/du, d/dq, d/dS, all intervals).  With the box
centre (u_c, q_c):

  Phi(s_m; U, Q)  in  Phi(u_c, q_c, s_m) + Phi_u(U, Q, s_m) (U - u_c) + Phi_q(U, Q, s_m) (Q - q_c)
  Phi'(S_box)     =   gradient d/dS of Phi over (U, Q, S_box)      [= X - 1]
  N = s_m - Phi(s_m; U, Q) / Phi'(S_box)  strictly inside S_box  ->  unique fixed point, in N
  P'(U, Q, N)     in  P(u_c, q_c, S_c) + grad P(U, Q, N) . ((U, Q, N) - centre)
                      intersected with the naive enclosure.
Each is a valid enclosure by the mean-value theorem; point evaluations are themselves
outward-rounded intervals.  Proof criterion as in d120: c + 2q > 0 and -1 < P' < 1.
Fold layer and the numerical domain as in d120b (not rigorous there).

Usage:  python d120c_interval_stability_proof_meanvalue_2026-09-14.py [n_max] [du] [dq]
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


B = _load("d120b", "d120b_interval_stability_proof_vectorised_2026-09-14.py")
V, add, sub, neg, mul, sq, div, isqrt, Kv = B.V, B.add, B.sub, B.neg, B.mul, B.sq, B.div, B.isqrt, B.Kv


# ------------------------------------------------------- interval forward AD
class D:
    __slots__ = ("v", "g")

    def __init__(self, v, g):
        self.v, self.g = v, g


def dconst(x, like):
    z = Kv(0, like)
    return D(Kv(x, like), [z, Kv(0, like), Kv(0, like)])


def dvar(v, k, like):
    g = [Kv(0, like), Kv(0, like), Kv(0, like)]
    g[k] = Kv(1, like)
    return D(v, g)


def dadd(a, b):
    return D(add(a.v, b.v), [add(x, y) for x, y in zip(a.g, b.g)])


def dsub(a, b):
    return D(sub(a.v, b.v), [sub(x, y) for x, y in zip(a.g, b.g)])


def dneg(a):
    return D(neg(a.v), [neg(x) for x in a.g])


def dmul(a, b):
    return D(mul(a.v, b.v), [add(mul(x, b.v), mul(a.v, y)) for x, y in zip(a.g, b.g)])


def dsq(a):
    two_a = mul(Kv(2, a.v.lo), a.v)
    return D(sq(a.v), [mul(two_a, x) for x in a.g])


def ddiv(a, b):
    val = div(a.v, b.v)
    return D(val, [div(sub(x, mul(val, y)), b.v) for x, y in zip(a.g, b.g)])


def dsqrt(a):
    val = isqrt(a.v)
    val.lo = np.where(val.lo > 0, val.lo, np.nan)
    two_v = mul(Kv(2, a.v.lo), val)
    return D(val, [div(x, two_v) for x in a.g])


def _vmeet(a, b):
    lo, hi = np.fmax(a.lo, b.lo), np.fmin(a.hi, b.hi)
    bad = lo > hi
    return V(np.where(bad, np.nan, lo), np.where(bad, np.nan, hi))


def dmeet(a, b):
    """two algebraically equal forms: intersect value and gradient enclosures."""
    return D(_vmeet(a.v, b.v), [_vmeet(x, y) for x, y in zip(a.g, b.g)])


def rec_d(u, q, S, n):
    """Phi, X, P' and c + 2q as AD intervals of (u, q, S).

    Cancellation-free forms (each quantity also in its textbook form, intersected):
      r = u + 2q S_(j-1),  beta = sqrt(c^2 - 4qr),  b = 2r/(beta - c)  [= (-c - beta)/(2q)]
      kappa = 2q (b - S_(j-1)),  B2 = c + 2q,  phi = sqrt(B2^2 + 4q kappa),
      f = 2 kappa/(B2 + phi)  [= (phi - B2)/(2q)]
      phi - 2q = (c(c + 4q) + 4q kappa)/(phi + 2q)  [= 2q f + c]
      beta - 2q = (c^2 - 4qr - 4q^2)/(beta + 2q)   [= -2q b - c - 2q]
      A = (phi - 2q)(beta - 2q)/(phi beta),  B = f/phi + (phi - 2q) b/(phi beta),
      S_j = S_(j-1) + (f - b)."""
    like = q.v.lo
    two, four = dconst(2, like), dconst(4, like)
    one = dconst(1, like)
    two_q = dmul(two, q)
    four_q = dmul(four, q)
    c = dsub(dsub(dsub(one, u), q), dmul(two_q, S))
    Sj, Sp = dconst(0, like), dconst(0, like)
    for _ in range(n):
        r = dadd(u, dmul(two_q, Sj))
        D1 = dsub(dsq(c), dmul(four_q, r))
        beta = dsqrt(D1)
        b = dmeet(ddiv(dmul(two, r), dsub(beta, c)), ddiv(dsub(dneg(c), beta), two_q))
        kappa = dmul(two_q, dsub(b, Sj))
        B2 = dadd(c, two_q)
        phi = dsqrt(dadd(dsq(B2), dmul(four_q, kappa)))
        f = dmeet(ddiv(dmul(two, kappa), dadd(B2, phi)), ddiv(dsub(phi, B2), two_q))
        phim = dmeet(ddiv(dadd(dmul(c, dadd(c, four_q)), dmul(four_q, kappa)), dadd(phi, two_q)),
                     dadd(dmul(two_q, f), c))
        betam = dmeet(ddiv(dsub(D1, dsq(two_q)), dadd(beta, two_q)),
                      dsub(dsub(dneg(dmul(two_q, b)), c), two_q))
        pb = dmul(phi, beta)
        A = ddiv(dmul(phim, betam), pb)
        Bj = dadd(ddiv(f, phi), ddiv(dmul(phim, b), pb))
        Sp = dsub(dmul(A, Sp), Bj)
        Sj = dadd(Sj, dsub(f, b))
    Phi = dsub(Sj, S)
    X = dneg(dmul(two_q, Sp))
    den = dadd(c, two_q)
    P = ddiv(dadd(c, dmul(two_q, X)), den)
    return Phi, X, P, den


def point(x):
    return V(x.copy(), x.copy())


def certify_mv(U0, U1, Q0, Q1, S0, n):
    like = Q0
    uc, qc = 0.5 * (U0 + U1), 0.5 * (Q0 + Q1)
    dU = V(np.nextafter(U0 - uc, -np.inf), np.nextafter(U1 - uc, np.inf))
    dQ = V(np.nextafter(Q0 - qc, -np.inf), np.nextafter(Q1 - qc, np.inf))
    Ubox, Qbox = V(U0, U1), V(Q0, Q1)
    # Phi at the centre (point intervals) and its parameter gradient over the box at s_m
    Phi_c, _, _, _ = rec_d(dvar(point(uc), 0, like), dvar(point(qc), 1, like), dvar(point(S0), 2, like), n)
    _, _, _, _ = None, None, None, None
    Phi_b, _, _, _ = rec_d(dvar(Ubox, 0, like), dvar(Qbox, 1, like), dvar(point(S0), 2, like), n)
    Phi_m = add(add(Phi_c.v, mul(Phi_b.g[0], dU)), mul(Phi_b.g[1], dQ))
    lo_n = np.maximum(Phi_m.lo, Phi_b.v.lo)
    hi_n = np.minimum(Phi_m.hi, Phi_b.v.hi)
    Phi_m = V(lo_n, hi_n)
    proved = np.zeros(S0.shape, bool)
    Plo = np.full(S0.shape, np.nan)
    Phi_ = np.full(S0.shape, np.nan)
    Sm = point(S0)
    for r in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        todo = ~proved
        if not todo.any():
            break
        Sb = V(S0 - r, S0 + r)
        PhiSb, _, _, _ = rec_d(dvar(Ubox, 0, like), dvar(Qbox, 1, like), dvar(Sb, 2, like), n)
        dPhi = PhiSb.g[2]
        ok1 = (dPhi.hi < 0) | (dPhi.lo > 0)
        N = sub(Sm, div(Phi_m, dPhi))
        ok2 = (N.lo > Sb.lo) & (N.hi < Sb.hi)
        Nlo = np.where(ok2, N.lo, S0)
        Nhi = np.where(ok2, N.hi, S0)
        Nb = V(Nlo, Nhi)
        Sc = 0.5 * (Nlo + Nhi)
        dS = V(np.nextafter(Nlo - Sc, -np.inf), np.nextafter(Nhi - Sc, np.inf))
        _, _, Pc, _ = rec_d(dvar(point(uc), 0, like), dvar(point(qc), 1, like), dvar(point(Sc), 2, like), n)
        _, _, PB_, denB = rec_d(dvar(Ubox, 0, like), dvar(Qbox, 1, like), dvar(Nb, 2, like), n)
        Pmv = add(add(add(Pc.v, mul(PB_.g[0], dU)), mul(PB_.g[1], dQ)), mul(PB_.g[2], dS))
        lo = np.fmax(Pmv.lo, PB_.v.lo)
        hi = np.fmin(Pmv.hi, PB_.v.hi)
        ok3 = (denB.v.lo > 0) & (lo > -1) & (hi < 1)
        new = todo & ok1 & ok2 & ok3
        proved |= new
        Plo = np.where(new, lo, Plo)
        Phi_ = np.where(new, hi, Phi_)
    return proved, Plo, Phi_


def main():
    t0 = time.time()
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    du = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    dq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    n_min = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out_path = os.path.join(HERE, "d120c_interval_stability_proof_meanvalue_2026-09-14.json")
    report = json.load(open(out_path)) if (n_min > 1 and os.path.exists(out_path)) else {"du": du, "dq": dq, "per_n": {}}
    CH = 150000
    for n in range(n_min, n_max + 1):
        edges = np.round(np.arange(0.05, 0.95 + du / 2, du), 10)
        tracks = {e: (B.P0.track(float(e), n)[0] or None) for e in edges}
        U0l, U1l, Q0l, Q1l, Sg = [], [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            ta, tb = tracks[a], tracks[b]
            if ta is None or tb is None:
                continue
            qlo, qhi = min(ta[-1][0], tb[-1][0]), max(ta[0][0], tb[0][0])
            qe = np.arange(math.floor(qlo / dq) * dq, qhi + dq, dq)
            qa_, qb_ = qe[:-1], qe[1:]
            qcc = 0.5 * (qa_ + qb_)
            ra, rb = sorted((p[0], p[1]) for p in ta), sorted((p[0], p[1]) for p in tb)
            sg = 0.5 * (np.interp(qcc, [x[0] for x in ra], [x[1] for x in ra]) + np.interp(qcc, [x[0] for x in rb], [x[1] for x in rb]))
            U0l.append(np.full(qcc.shape, a)); U1l.append(np.full(qcc.shape, b)); Q0l.append(qa_); Q1l.append(qb_); Sg.append(sg)
        U0, U1, Q0, Q1, S_guess = map(np.concatenate, (U0l, U1l, Q0l, Q1l, Sg))
        st = {"boxes": int(U0.size), "proved": 0, "proved_area": 0.0, "fold_lower": 0, "fold_unchecked": 0, "outside": 0,
              "unproved": 0, "unproved_area": 0.0, "P_min": 1.0, "P_max": -1.0, "by_depth": [], "unproved_boxes": []}
        depth = 0
        while U0.size:
            S0 = np.full(U0.shape, np.nan)
            X0 = np.full(U0.shape, np.nan)
            for i in range(0, U0.size, CH):
                sl = slice(i, i + CH)
                S0[sl], X0[sl] = B.newton_float(0.5 * (U0[sl] + U1[sl]), 0.5 * (Q0[sl] + Q1[sl]), n, S_guess[sl])
            outside = ~np.isfinite(S0)
            fold = ~outside & (X0 > 0.995)
            st["outside"] += int(outside.sum())
            if fold.any():
                fl = B.fold_lower(U0[fold], U1[fold], Q0[fold], Q1[fold], S0[fold], n)
                st["fold_lower"] += int(fl.sum()); st["fold_unchecked"] += int((~fl).sum())
            idx = np.where(~outside & ~fold)[0]
            pr = np.zeros(idx.size, bool); plo = np.full(idx.size, np.nan); phi_ = np.full(idx.size, np.nan)
            for i in range(0, idx.size, CH):
                j = idx[i:i + CH]
                pr[i:i + CH], plo[i:i + CH], phi_[i:i + CH] = certify_mv(U0[j], U1[j], Q0[j], Q1[j], S0[j], n)
            area = (U1[idx] - U0[idx]) * (Q1[idx] - Q0[idx])
            st["proved"] += int(pr.sum()); st["proved_area"] += float(area[pr].sum())
            st["by_depth"].append([depth, int(idx.size), int(pr.sum())])
            if pr.any():
                st["P_min"] = min(st["P_min"], float(np.nanmin(plo[pr]))); st["P_max"] = max(st["P_max"], float(np.nanmax(phi_[pr])))
            bad = idx[~pr]
            if depth >= 3 or bad.size == 0:
                st["unproved"] += int(bad.size)
                st["unproved_area"] += float(((U1[bad] - U0[bad]) * (Q1[bad] - Q0[bad])).sum())
                st["unproved_boxes"] = [[float(U0[i]), float(U1[i]), float(Q0[i]), float(Q1[i]), float(X0[i])] for i in bad]
                break
            um, qm = 0.5 * (U0[bad] + U1[bad]), 0.5 * (Q0[bad] + Q1[bad])
            U0, U1 = np.concatenate([U0[bad], um, U0[bad], um]), np.concatenate([um, U1[bad], um, U1[bad]])
            Q0, Q1 = np.concatenate([Q0[bad], Q0[bad], qm, qm]), np.concatenate([qm, qm, Q1[bad], Q1[bad]])
            S_guess = np.tile(S0[bad], 4)
            depth += 1
        report["per_n"][str(n)] = st
        print("n=%d: %d base boxes; proved %d (area %.5f), P' in [%.4f, %.4f]; by depth %s; fold layer P'>-1 %d, unchecked %d; "
              "unproved %d (area %.2e); outside %d   (%.0f s)"
              % (n, st["boxes"], st["proved"], st["proved_area"], st["P_min"], st["P_max"], st["by_depth"], st["fold_lower"],
                 st["fold_unchecked"], st["unproved"], st["unproved_area"], st["outside"], time.time() - t0))
        json.dump(report, open(out_path, "w"))
    print("%.0f s -> %s" % (time.time() - t0, os.path.basename(out_path)))


if __name__ == "__main__":
    main()
