#!/usr/bin/env python3
"""
d120b -- the computer-assisted stability proof of d120, vectorised over boxes.

d120's scalar code is correct (a point enclosure matches the float multiplier to
2.6e-14) but 0.01-wide boxes fail interval Newton through dependency, and subdividing
them one at a time would take hours.  Here every box of a word's numerical domain is
processed at once with numpy interval arithmetic; outward rounding is np.nextafter on
every bound of every operation (IEEE-754 double, round-to-nearest, so each computed
bound is within one ulp and the nextafter step encloses it; sqrt is correctly rounded).

STATEMENT CHECKED PER BOX U x Q (see d120 for the derivation):
  (1) 0 not in Phi'(S_box) = X - 1 and N = s_m - Phi(s_m)/Phi'(S_box) strictly inside
      S_box  ->  exactly one fixed point of the W_n recursion in S_box for every (u, q)
      in the box, lying in N;
  (2) on (U, Q, N): c + 2q > 0 and -1 < P' < 1.
Fold layer (numerical X > 0.995 at the centre): for all (u, q, S) in the box x
[S_0 - 5e-3, S_0 + 5e-3] with X <= 1, P' > -1 (no uniqueness claimed there).
Boxes whose centre has no numerical branch fixed point are outside the numerical domain.
Failing boxes are split 2 x 2 up to three times.

NOT rigorous: the numerical domain itself (death from the exact grazing condition,
birth from the tracked branch) and the identification of the tracked root with W_n's
physical stable branch.

Usage:  python d120b_interval_stability_proof_vectorised_2026-09-14.py [n_max] [du] [dq]
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


P0 = _load("d120", "d120_interval_stability_proof_2026-09-14.py")
NINF, PINF = -np.inf, np.inf
dn = lambda x: np.nextafter(x, NINF)
up = lambda x: np.nextafter(x, PINF)


class V:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi


def Kv(x, like):
    a = np.full(like.shape, float(x))
    return V(a, a.copy())


def add(a, b):
    return V(dn(a.lo + b.lo), up(a.hi + b.hi))


def sub(a, b):
    return V(dn(a.lo - b.hi), up(a.hi - b.lo))


def neg(a):
    return V(-a.hi, -a.lo)


def mul(a, b):
    p1, p2, p3, p4 = a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi
    lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return V(dn(lo), up(hi))


def sq(a):
    l2, h2 = a.lo * a.lo, a.hi * a.hi
    lo = np.where(a.lo >= 0, l2, np.where(a.hi <= 0, h2, 0.0))
    hi = np.maximum(l2, h2)
    return V(np.maximum(dn(lo), 0.0), up(hi))


def div(a, b):
    bad = (b.lo <= 0) & (b.hi >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = V(dn(1.0 / b.hi), up(1.0 / b.lo))
    r = mul(a, inv)
    r.lo = np.where(bad, np.nan, r.lo)
    r.hi = np.where(bad, np.nan, r.hi)
    return r


def isqrt(a):
    bad = ~(a.lo >= 0)
    with np.errstate(invalid="ignore"):
        lo = np.maximum(dn(np.sqrt(np.maximum(a.lo, 0.0))), 0.0)
        hi = up(np.sqrt(np.maximum(a.hi, 0.0)))
    return V(np.where(bad, np.nan, lo), np.where(bad, np.nan, hi))


def rec_iv(c, u, q, n):
    two_q = mul(Kv(2, q.lo), q)
    four_q = mul(Kv(4, q.lo), q)
    one = Kv(1, q.lo)
    S, Sp = Kv(0, q.lo), Kv(0, q.lo)
    for _ in range(n):
        beta = isqrt(sub(sq(c), mul(four_q, add(u, mul(two_q, S)))))
        beta.lo = np.where(beta.lo > 0, beta.lo, np.nan)
        b = div(sub(neg(c), beta), two_q)
        B2 = add(c, two_q)
        phi = isqrt(sub(sq(B2), mul(four_q, mul(two_q, sub(S, b)))))
        phi.lo = np.where(phi.lo > 0, phi.lo, np.nan)
        f = div(add(neg(B2), phi), two_q)
        r_phi, r_beta = div(two_q, phi), div(two_q, beta)
        A = mul(sub(one, r_phi), sub(one, r_beta))
        B = add(div(f, phi), mul(sub(one, r_phi), div(b, beta)))
        Sp = sub(mul(A, Sp), B)
        S = add(S, div(sub(add(beta, phi), two_q), two_q))
    return S, Sp


def c_of(S, u, q):
    return sub(sub(sub(Kv(1, q.lo), u), q), mul(mul(Kv(2, q.lo), q), S))


def P_encl(Sb, U, Q, n, clip_X=False):
    c = c_of(Sb, U, Q)
    _, Sp = rec_iv(c, U, Q, n)
    two_q = mul(Kv(2, Q.lo), Q)
    X = neg(mul(two_q, Sp))
    if clip_X:
        X = V(X.lo, np.minimum(X.hi, 1.0))
    den = add(c, two_q)
    one = Kv(1, Q.lo)
    P1 = div(add(c, mul(two_q, X)), den)
    P2 = sub(one, div(mul(two_q, sub(one, X)), den))
    lo = np.fmax(P1.lo, P2.lo)
    hi = np.fmin(P1.hi, P2.hi)
    return lo, hi, den.lo, X


# ---------------------------------------------------------- float, vectorised
def rec_float(c, u, q, n):
    S = np.zeros_like(c)
    Sp = np.zeros_like(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(n):
            beta = np.sqrt(c * c - 4 * q * (u + 2 * q * S))
            b = (-c - beta) / (2 * q)
            B2 = c + 2 * q
            phi = np.sqrt(B2 * B2 - 4 * q * 2 * q * (S - b))
            f = (-B2 + phi) / (2 * q)
            A = (1 - 2 * q / phi) * (1 - 2 * q / beta)
            B = f / phi + (1 - 2 * q / phi) * b / beta
            Sp = A * Sp - B
            S = S + (beta + phi - 2 * q) / (2 * q)
    return S, Sp


def newton_float(u, q, n, S):
    ok = np.isfinite(S)
    for _ in range(40):
        c = 1 - u - q - 2 * q * S
        Sn, Sp = rec_float(c, u, q, n)
        dPhi = -2 * q * Sp - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            S_new = S - (Sn - S) / dPhi
        S_new = np.where(np.isfinite(S_new), S_new, np.nan)
        with np.errstate(invalid="ignore"):
            moved = np.nanmax(np.abs(S_new - S)) if np.any(np.isfinite(S_new)) else 0.0
        S = S_new
        if not moved > 1e-15:
            break
    c = 1 - u - q - 2 * q * S
    Sn, Sp = rec_float(c, u, q, n)
    good = np.isfinite(S) & (np.abs(Sn - S) < 1e-11) & (S >= 0) & (S <= 1)
    X = -2 * q * Sp
    return np.where(good, S, np.nan), X


def certify(U0, U1, Q0, Q1, S0, n):
    """returns proved mask and the P bounds of proved boxes."""
    U, Q = V(U0, U1), V(Q0, Q1)
    proved = np.zeros(S0.shape, bool)
    Plo = np.full(S0.shape, np.nan)
    Phi_ = np.full(S0.shape, np.nan)
    two_q = mul(Kv(2, Q0), Q)
    Sm = V(S0.copy(), S0.copy())
    Snm, _ = rec_iv(c_of(Sm, U, Q), U, Q, n)
    Phi_m = sub(Snm, Sm)
    for r in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3):
        todo = ~proved
        if not todo.any():
            break
        Sb = V(S0 - r, S0 + r)
        _, Spb = rec_iv(c_of(Sb, U, Q), U, Q, n)
        dPhi = sub(neg(mul(two_q, Spb)), Kv(1, Q0))
        ok1 = (dPhi.hi < 0) | (dPhi.lo > 0)
        N = sub(Sm, div(Phi_m, dPhi))
        ok2 = (N.lo > Sb.lo) & (N.hi < Sb.hi)
        lo, hi, denlo, _ = P_encl(N, U, Q, n)
        ok3 = (denlo > 0) & (lo > -1) & (hi < 1)
        new = todo & ok1 & ok2 & ok3
        proved |= new
        Plo = np.where(new, lo, Plo)
        Phi_ = np.where(new, hi, Phi_)
    return proved, Plo, Phi_


def fold_lower(U0, U1, Q0, Q1, S0, n):
    U, Q = V(U0, U1), V(Q0, Q1)
    Sb = V(S0 - 5e-3, S0 + 5e-3)
    c = c_of(Sb, U, Q)
    _, Sp = rec_iv(c, U, Q, n)
    two_q = mul(Kv(2, Q0), Q)
    X = neg(mul(two_q, Sp))
    X = V(X.lo, np.minimum(X.hi, 1.0))
    den = add(c, two_q)
    lowP = div(add(c, mul(two_q, X)), den)
    return (den.lo > 0) & (lowP.lo > -1)


def main():
    t0 = time.time()
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    du = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    dq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    report = {"du": du, "dq": dq, "per_n": {}}
    for n in range(1, n_max + 1):
        edges = np.round(np.arange(0.05, 0.95 + du / 2, du), 10)
        tracks = {}
        for e in edges:
            pts, why = P0.track(float(e), n)
            tracks[e] = pts if pts else None
        U0l, U1l, Q0l, Q1l, Sg = [], [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            ta, tb = tracks[a], tracks[b]
            if ta is None or tb is None:
                continue
            qlo = min(ta[-1][0], tb[-1][0])
            qhi = max(ta[0][0], tb[0][0])
            qe = np.arange(math.floor(qlo / dq) * dq, qhi + dq, dq)
            qa_, qb_ = qe[:-1], qe[1:]
            qc = 0.5 * (qa_ + qb_)
            ra = sorted((p[0], p[1]) for p in ta)
            rb = sorted((p[0], p[1]) for p in tb)
            sg = 0.5 * (np.interp(qc, [x[0] for x in ra], [x[1] for x in ra]) +
                        np.interp(qc, [x[0] for x in rb], [x[1] for x in rb]))
            U0l.append(np.full(qc.shape, a)); U1l.append(np.full(qc.shape, b))
            Q0l.append(qa_); Q1l.append(qb_); Sg.append(sg)
        U0, U1, Q0, Q1, S_guess = map(np.concatenate, (U0l, U1l, Q0l, Q1l, Sg))
        st = {"boxes": int(U0.size), "proved": 0, "proved_area": 0.0, "fold_lower": 0, "fold_unchecked": 0,
              "outside": 0, "unproved": 0, "unproved_area": 0.0, "P_min": 1.0, "P_max": -1.0, "unproved_where": []}
        depth = 0
        while U0.size:
            S0, X0 = newton_float(0.5 * (U0 + U1), 0.5 * (Q0 + Q1), n, S_guess)
            outside = ~np.isfinite(S0)
            fold = ~outside & (X0 > 0.995)
            rest = ~outside & ~fold
            st["outside"] += int(outside.sum())
            if fold.any():
                fl = fold_lower(U0[fold], U1[fold], Q0[fold], Q1[fold], S0[fold], n)
                st["fold_lower"] += int(fl.sum())
                st["fold_unchecked"] += int((~fl).sum())
            idx = np.where(rest)[0]
            pr, plo, phi_ = certify(U0[idx], U1[idx], Q0[idx], Q1[idx], S0[idx], n)
            area = (U1[idx] - U0[idx]) * (Q1[idx] - Q0[idx])
            st["proved"] += int(pr.sum())
            st["proved_area"] += float(area[pr].sum())
            if pr.any():
                st["P_min"] = min(st["P_min"], float(np.nanmin(plo[pr])))
                st["P_max"] = max(st["P_max"], float(np.nanmax(phi_[pr])))
            bad = idx[~pr]
            if depth >= 3 or bad.size == 0:
                st["unproved"] += int(bad.size)
                st["unproved_area"] += float(((U1[bad] - U0[bad]) * (Q1[bad] - Q0[bad])).sum())
                for i in bad[:40]:
                    st["unproved_where"].append([float(U0[i]), float(U1[i]), float(Q0[i]), float(Q1[i]), float(X0[i])])
                break
            um, qm = 0.5 * (U0[bad] + U1[bad]), 0.5 * (Q0[bad] + Q1[bad])
            nU0 = np.concatenate([U0[bad], um, U0[bad], um]); nU1 = np.concatenate([um, U1[bad], um, U1[bad]])
            nQ0 = np.concatenate([Q0[bad], Q0[bad], qm, qm]); nQ1 = np.concatenate([qm, qm, Q1[bad], Q1[bad]])
            S_guess = np.tile(S0[bad], 4)
            U0, U1, Q0, Q1 = nU0, nU1, nQ0, nQ1
            depth += 1
        report["per_n"][str(n)] = st
        print("n=%d: %d base boxes; proved %d (area %.4f), P' in [%.4f, %.4f]; fold layer P'>-1 %d, unchecked %d; "
              "unproved %d (area %.2e); outside %d   (%.0f s)"
              % (n, st["boxes"], st["proved"], st["proved_area"], st["P_min"], st["P_max"], st["fold_lower"],
                 st["fold_unchecked"], st["unproved"], st["unproved_area"], st["outside"], time.time() - t0))
        json.dump(report, open(os.path.join(HERE, "d120b_interval_stability_proof_vectorised_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d120b_interval_stability_proof_vectorised_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
