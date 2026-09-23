#!/usr/bin/env python3
"""
d120 -- computer-assisted proof that the stable branch of every word W_n (n = 1-5)
has |P'| < 1 over u = 0.05-0.95: interval Newton + outward-rounded interval arithmetic.

WHAT IS PROVED, AND ON WHAT.  With X = -2q dS_n/dc (d118, T1),
    P' = (c + 2qX)/(c + 2q) = 1 - 2q(1 - X)/(c + 2q),     c + 2q > 0,
so P' < 1 <=> X < 1 (the fixed point lies on the stable side of any fold) and
P' > -1 <=> X > -(c + q)/q (no flip).  For each box U x Q in (u, q):

  (1) Existence and uniqueness.  With Phi(S) = F_n(c(S)) - S, c(S) = 1 - u - q - 2qS and
      Phi'(S) = X - 1, the interval Newton operator
          N = s_m - Phi(s_m; U, Q) / Phi'(S_box; U, Q)
      is evaluated with every operation outward-rounded (math.nextafter).  If
      0 is not in Phi'(S_box) and N lies inside S_box, then for EVERY (u, q) in the box
      the recursion has exactly one fixed point in S_box, and it lies in N.
  (2) Stability.  On (U, Q, N) the enclosures of c + 2q, X and P' (two algebraically
      equal forms, intersected) are computed; the box is PROVED if c + 2q > 0 and
      -1 < P' < 1.

  Boxes are generated along the numerically tracked stable branch, from the grazing
  death (exact condition, d119b) down to the birth: the border (ordering slack 0,
  overshot by 0.02 in q, harmless because the recursion is smooth across it) or the
  fold (X -> 1).  A box that fails is split in u and q up to four times.
  Fold layer: where X is within 0.005 of 1 Newton cannot succeed (Phi' -> 0).  There
  the weaker statement is checked on a tube S_box that contains the numerical root:
  for every (u, q, S) in it with X <= 1, P' > -1.

  NOT rigorous: the location of the existence interval (birth, death) and the
  assignment of the tracked root to W_n's physical branch are numerical (d116-d119).
  The recursion's own branch conditions (square roots of non-negative intervals, no
  division by an interval containing 0) are enforced by the interval code.

Usage:  python d120_interval_stability_proof_2026-09-14.py [n_max]
"""
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


m = _load("d118", "d118_word_multiplier_closed_form_2026-09-14.py")
DB = _load("d119b", "d119b_birth_type_by_S_2026-09-14.py")
INF = math.inf
dn = lambda x: math.nextafter(x, -INF)
up = lambda x: math.nextafter(x, INF)


# ------------------------------------------------------------------ intervals
class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = lo
        self.hi = lo if hi is None else hi

    def __repr__(self):
        return "[%.17g, %.17g]" % (self.lo, self.hi)


def K(x):
    return I(float(x))


def add(a, b):
    return I(dn(a.lo + b.lo), up(a.hi + b.hi))


def sub(a, b):
    return I(dn(a.lo - b.hi), up(a.hi - b.lo))


def neg(a):
    return I(-a.hi, -a.lo)


def mul(a, b):
    p = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    return I(dn(min(p)), up(max(p)))


def sq(a):
    if a.lo >= 0:
        return I(dn(a.lo * a.lo), up(a.hi * a.hi))
    if a.hi <= 0:
        return I(dn(a.hi * a.hi), up(a.lo * a.lo))
    return I(0.0, up(max(a.lo * a.lo, a.hi * a.hi)))


def div(a, b):
    if b.lo <= 0.0 <= b.hi:
        return None
    return mul(a, I(dn(1.0 / b.hi), up(1.0 / b.lo)))


def isqrt(a):
    if a.lo < 0.0:
        return None
    return I(dn(math.sqrt(a.lo)) if a.lo > 0 else 0.0, up(math.sqrt(a.hi)))


def meet(a, b):
    lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
    return None if lo > hi else I(lo, hi)


# ------------------------------------------------------------ interval recursion
def rec_iv(c, u, q, n):
    """S_n and dS_n/dc as intervals (T1), or None if a branch condition cannot be certified.
    Uses beta_j = sqrt(D1), phi_j = sqrt(D2) and f_j - b_j = (sqrt D1 + sqrt D2 - 2q)/(2q)
    (algebraically exact), which removes most of the dependency."""
    two_q = mul(K(2), q)
    four_q = mul(K(4), q)
    S, Sp = K(0), K(0)
    for _ in range(n):
        D1 = sub(sq(c), mul(four_q, add(u, mul(two_q, S))))
        beta = isqrt(D1)
        if beta is None or beta.lo <= 0:
            return None
        b = div(sub(neg(c), beta), two_q)
        B2 = add(c, two_q)
        D2 = sub(sq(B2), mul(four_q, mul(two_q, sub(S, b))))
        phi = isqrt(D2)
        if phi is None or phi.lo <= 0:
            return None
        f = div(add(neg(B2), phi), two_q)
        r_phi, r_beta = div(two_q, phi), div(two_q, beta)
        if r_phi is None or r_beta is None:
            return None
        A = mul(sub(K(1), r_phi), sub(K(1), r_beta))
        fb = div(f, phi)
        bb = div(b, beta)
        B = add(fb, mul(sub(K(1), r_phi), bb))
        Sp = sub(mul(A, Sp), B)
        S = add(S, div(sub(add(beta, phi), two_q), two_q))
    return S, Sp


def c_of(S, u, q):
    return sub(sub(sub(K(1), u), q), mul(mul(K(2), q), S))


def P_encl(Sbox, u, q, n):
    c = c_of(Sbox, u, q)
    r = rec_iv(c, u, q, n)
    if r is None:
        return None
    _, Sp = r
    two_q = mul(K(2), q)
    X = neg(mul(two_q, Sp))
    den = add(c, two_q)
    if den.lo <= 0:
        return None
    P1 = div(add(c, mul(two_q, X)), den)
    P2 = sub(K(1), div(mul(two_q, sub(K(1), X)), den))
    P = meet(P1, P2) if (P1 and P2) else (P1 or P2)
    return {"P": P, "X": X, "den": den}


def certify(U, Q, n, S0):
    """interval Newton in S with parameter boxes; returns (status, info)."""
    two_q = mul(K(2), Q)
    for r in (2e-5, 1e-4, 5e-4, 2e-3, 8e-3):
        Sb = I(S0 - r, S0 + r)
        cb = c_of(Sb, U, Q)
        rb = rec_iv(cb, U, Q, n)
        if rb is None:
            continue
        dPhi = sub(neg(mul(two_q, rb[1])), K(1))
        if dPhi.lo <= 0.0 <= dPhi.hi:
            continue
        Sm = K(S0)
        rm = rec_iv(c_of(Sm, U, Q), U, Q, n)
        if rm is None:
            continue
        Phi_m = sub(rm[0], Sm)
        step = div(Phi_m, dPhi)
        if step is None:
            continue
        N = sub(Sm, step)
        if not (N.lo > Sb.lo and N.hi < Sb.hi):
            continue
        e = P_encl(N, U, Q, n)
        if e is None or e["P"] is None:
            continue
        if e["P"].lo > -1.0 and e["P"].hi < 1.0:
            return "proved", {"P": (e["P"].lo, e["P"].hi), "X_hi": e["X"].hi, "r": r}
        return "enclosed_not_stable", {"P": (e["P"].lo, e["P"].hi)}
    return "no_newton", None


# ------------------------------------------------------------- numerical branch
def fp_newton(u, q, n, S):
    for _ in range(60):
        try:
            Pp, Sp, c, _ = m.multiplier(u, q, n, {"S_n": S})
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        r = m.rec_c(1 - u - q - 2 * q * S, u, q, n)
        if r is None:
            return None
        Phi = r[0] - S
        dPhi = -2 * q * Sp - 1
        if dPhi == 0:
            return None
        S_new = S - Phi / dPhi
        if not (0 <= S_new <= 1):
            return None
        if abs(S_new - S) < 1e-15:
            return S_new
        S = S_new
    return S if abs(Phi) < 1e-12 else None


def X_of(u, q, n, S):
    return -2 * q * m.multiplier(u, q, n, {"S_n": S})[1]


def min_slack(u, q, n, S):
    c = 1 - u - q - 2 * q * S
    r = m.rec_c(c, u, q, n)
    if r is None:
        return None
    th = u / (2 * q)
    ev = r[1]
    s = min(ev[0][0] - th, 1 - ev[-1][1] - th, min(f - b for b, f in ev) - th)
    if n > 1:
        s = min(s, min(ev[j + 1][0] - ev[j][1] for j in range(n - 1)) - th)
    return s


def track(u, n, h=0.005):
    """stable branch from death downward: list of (q, S, X, slack, end_reason)."""
    ds = DB.death(u, n)
    if not ds:
        return None, "no death"
    qd = min(d["q_death"] for d in ds)
    sq_ = math.sqrt(qd)
    S = ((sq_ - 1) ** 2 - u) / (2 * qd)
    pts = []
    q = qd + 0.01
    S = fp_newton(u, q, n, S) or S
    reason = None
    while q > 1.0:
        S2 = fp_newton(u, q, n, S)
        if S2 is None:
            reason = "newton lost"
            break
        X = X_of(u, q, n, S2)
        sl = min_slack(u, q, n, S2)
        if X > 0.995:
            reason = "fold layer"
            pts.append((q, S2, X, sl))
            break
        pts.append((q, S2, X, sl))
        if sl is not None and sl < 0 and q < qd - 0.05:
            # past the birth border: overshoot a little, then stop
            if pts and sum(1 for p in pts if p[3] is not None and p[3] < 0) * h > 0.02:
                reason = "past border"
                break
        S = S2
        q -= h
    return pts, reason


# --------------------------------------------------------------------- driver
def prove_box(u0, u1, q0, q1, n, S_guess, depth, stats, fails):
    uc, qc = 0.5 * (u0 + u1), 0.5 * (q0 + q1)
    S0 = fp_newton(uc, qc, n, S_guess)
    if S0 is None:
        stats["outside"] += 1          # no branch fixed point at the centre: outside the numerical domain
        return
    X0 = X_of(uc, qc, n, S0)
    if X0 > 0.995:
        # fold layer: weaker statement on a tube around the root
        Sb = I(S0 - 5e-3, S0 + 5e-3)
        e = P_encl(Sb, I(u0, u1), I(q0, q1), n)
        if e is not None and e["den"].lo > 0:
            X = meet(e["X"], I(-INF, 1.0))
            two_q = mul(K(2), I(q0, q1))
            c = c_of(Sb, I(u0, u1), I(q0, q1))
            lowP = div(add(c, mul(two_q, X)), e["den"]) if X else None
            if lowP is not None and lowP.lo > -1.0:
                stats["fold_layer_lower_bound"] += 1
                return
        stats["fold_layer_unchecked"] += 1
        fails.append({"n": n, "u": [u0, u1], "q": [q0, q1], "why": "fold layer", "X": X0})
        return
    st, info = certify(I(u0, u1), I(q0, q1), n, S0)
    if st == "proved":
        stats["proved"] += 1
        stats["P_min"] = min(stats["P_min"], info["P"][0])
        stats["P_max"] = max(stats["P_max"], info["P"][1])
        return
    if depth >= 4:
        stats["unproved"] += 1
        fails.append({"n": n, "u": [u0, u1], "q": [q0, q1], "why": st, "info": info, "X": X0})
        return
    um, qm = 0.5 * (u0 + u1), 0.5 * (q0 + q1)
    for a, b in ((u0, um), (um, u1)):
        for c_, d in ((q0, qm), (qm, q1)):
            prove_box(a, b, c_, d, n, S0, depth + 1, stats, fails)


def main():
    t0 = time.time()
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    du, dq = 0.01, 0.01
    report = {"per_n": {}, "fails": []}
    for n in range(1, n_max + 1):
        stats = {"proved": 0, "outside": 0, "unproved": 0, "fold_layer_lower_bound": 0,
                 "fold_layer_unchecked": 0, "P_min": INF, "P_max": -INF}
        fails = []
        ends = {}
        k = 0
        u0 = 0.05
        while u0 < 0.95 - 1e-9:
            u1 = round(u0 + du, 10)
            tr = {}
            for uu in (u0, u1):
                if uu not in ends:
                    ends[uu] = track(uu, n)
                tr[uu] = ends[uu]
            if tr[u0][0] is None or tr[u1][0] is None or not tr[u0][0] or not tr[u1][0]:
                fails.append({"n": n, "u": [u0, u1], "why": "no track", "reasons": [tr[u0][1], tr[u1][1]]})
                u0 = u1
                continue
            qs_lo = min(tr[u0][0][-1][0], tr[u1][0][-1][0])
            qs_hi = max(tr[u0][0][0][0], tr[u1][0][0][0])
            ref = tr[u0][0]
            qa = math.floor(qs_lo / dq) * dq
            while qa < qs_hi:
                qb = qa + dq
                qc = 0.5 * (qa + qb)
                S_guess = min(ref, key=lambda p: abs(p[0] - qc))[1]
                prove_box(u0, u1, qa, qb, n, S_guess, 0, stats, fails)
                qa = qb
            k += 1
            u0 = u1
        report["per_n"][str(n)] = stats
        report["fails"] += fails
        print("n=%d: proved boxes %d (P' in [%.4f, %.4f]); fold-layer boxes with P' > -1 certified %d, unchecked %d; "
              "unproved %d; box centres outside the branch %d   (%.0f s)"
              % (n, stats["proved"], stats["P_min"], stats["P_max"], stats["fold_layer_lower_bound"],
                 stats["fold_layer_unchecked"], stats["unproved"], stats["outside"], time.time() - t0))
        json.dump(report, open(os.path.join(HERE, "d120_interval_stability_proof_2026-09-14.json"), "w"), indent=1, default=str)
    print("%.0f s -> d120_interval_stability_proof_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
