#!/usr/bin/env python3
"""
d120d -- computer-assisted stability proof for W_n by the Krawczyk operator on the
square-root-free crossing system.

WHY NOT d120b/d120c.  The recursion computes b_j through sqrt(c^2 - 4q r_j).  Near a
birth the last excursion is shallow (beta_n -> u/2) while c^2 and 4q r_j are ~40, so the
discriminant is a near-cancellation of large numbers and any interval enclosure of it
is inflated by that ratio (n = 3: width 0.24 on a 0.0002 box).  That is geometry, not
code, and smaller boxes cannot remove it.

SYSTEM.  Unknowns x = (b_1..b_n, f_1..f_n, c), parameters (u, q), S_j = sum_{i<=j}(f_i - b_i):
    E_bj = q (b_j^2 + 2 S_(j-1)) + c b_j + u                     = 0
    E_fj = q (f_j^2 + 2 f_j + 2 S_(j-1) - 2 b_j) + c f_j          = 0
    E_c  = c - 1 + u + q (1 + 2 S_n)                              = 0
Written so that u and q occur once per equation (no parameter dependency).
Jacobian: dE_bj/db_j = 2q b_j + c (= -beta_j), dE_fj/df_j = 2q f_j + c + 2q (= phi_j),
dE_bj/db_i = -2q, dE_bj/df_i = +2q (i < j), dE_fj/db_j = -2q, dE_fj/db_i = -2q,
dE_fj/df_i = +2q (i < j), dE_bj/dc = b_j, dE_fj/dc = f_j, dE_c/dc = 1,
dE_c/db_i = -2q, dE_c/df_i = +2q.

PROOF PER BOX U x Q.  With x_c the float solution at the centre and Y = J(x_c)^-1 (float),
    K(X) = x_c - Y F(x_c; U, Q) + (I - Y J(X; U, Q)) (X - x_c).
If K(X) lies strictly inside X, then for every (u, q) in the box the system has exactly
one solution in X (Krawczyk).  Inside X we require beta_j > 0 and phi_j > 0 (b_j is the
downward and f_j the upward crossing: the roots of the word, not of another branch) and
c + 2q > 0, and the T1 multiplier
    P' = (c - 4q^2 S'_n)/(c + 2q),  S'_j = A_j S'_(j-1) - B_j,
    A_j = (phi_j - 2q)(beta_j - 2q)/(phi_j beta_j),  B_j = f_j/phi_j + (phi_j - 2q) b_j/(phi_j beta_j)
enclosed in (-1, 1).  Interval operations are outward rounded (np.nextafter per bound);
sums inside matrix products carry an explicit rounding bound 4 d eps sum|terms|.
X starts as the hull of the float solutions at the four box corners, inflated, and is
epsilon-inflated between Krawczyk steps (up to 6).

Fold layer (float X > 0.995 at the centre): only P' > -1 over the inflated corner hull
(no uniqueness), with P' <= 1 there by branch selection.
NOT rigorous: the numerical domain (as in d120b) and the identification of the
enclosed solution with W_n's physical branch beyond the root-side checks above.

Starting box and splitting: X starts at x_c +- 0.75 (|dx/du| du + |dx/dq| dq) from the
linear sensitivity at the centre; a failed box is split in two across the parameter
that contributes more width (near u -> 1 the solution is ~13x more sensitive to u than
to q), up to MAX_DEPTH binary splits.

Usage:  python d120d_krawczyk_stability_proof_2026-09-14.py [n_max] [du] [dq] [n_min] [max_depth] [u_lo] [u_hi] [tag]
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
V, add, sub, neg, mul, sq, div, Kv = B.V, B.add, B.sub, B.neg, B.mul, B.sq, B.div, B.Kv
EPS = np.finfo(float).eps
dn = lambda x: np.nextafter(x, -np.inf)
up = lambda x: np.nextafter(x, np.inf)


# ------------------------------------------------------------ float helpers
def crossings_float(u, q, n, S):
    """b_j, f_j (N x n each) and c from the recursion at S (vectorised)."""
    c = 1 - u - q - 2 * q * S
    bs, fs = [], []
    Sj = np.zeros_like(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(n):
            beta = np.sqrt(c * c - 4 * q * (u + 2 * q * Sj))
            b = (-c - beta) / (2 * q)
            B2 = c + 2 * q
            phi = np.sqrt(B2 * B2 + 8 * q * q * (b - Sj))
            f = (-B2 + phi) / (2 * q)
            bs.append(b)
            fs.append(f)
            Sj = Sj + f - b
    return np.stack(bs, 1), np.stack(fs, 1), c


def pack(bs, fs, c):
    return np.concatenate([bs, fs, c[:, None]], 1)


def jac_float(x, q, n):
    N, d = x.shape
    J = np.zeros((N, d, d))
    b, f, c = x[:, :n], x[:, n:2 * n], x[:, 2 * n]
    tq = 2 * q
    for j in range(n):
        J[:, j, j] = tq * b[:, j] + c
        J[:, n + j, n + j] = tq * f[:, j] + c + tq
        J[:, n + j, j] = -tq
        for i in range(j):
            J[:, j, i] = -tq
            J[:, j, n + i] = tq
            J[:, n + j, i] = -tq
            J[:, n + j, n + i] = tq
        J[:, j, 2 * n] = b[:, j]
        J[:, n + j, 2 * n] = f[:, j]
        J[:, 2 * n, j] = -tq
        J[:, 2 * n, n + j] = tq
    J[:, 2 * n, 2 * n] = 1.0
    return J


# --------------------------------------------------------- interval system
def F_iv(X, U, Q, n):
    """F with x as intervals (list of V), parameters U, Q (V); u, q appear once per equation."""
    b, f, c = X[:n], X[n:2 * n], X[2 * n]
    one, two = Kv(1, U.lo), Kv(2, U.lo)
    out = [None] * (2 * n + 1)
    S = Kv(0, U.lo)
    for j in range(n):
        out[j] = add(add(mul(Q, add(sq(b[j]), mul(two, S))), mul(c, b[j])), U)
        out[n + j] = add(mul(Q, add(add(sq(f[j]), mul(two, f[j])), mul(two, sub(S, b[j])))), mul(c, f[j]))
        S = add(S, sub(f[j], b[j]))
    out[2 * n] = add(add(sub(c, one), U), mul(Q, add(one, mul(two, S))))
    return out


def J_iv(X, Q, n):
    b, f, c = X[:n], X[n:2 * n], X[2 * n]
    d = 2 * n + 1
    z = Kv(0, Q.lo)
    tq = mul(Kv(2, Q.lo), Q)
    ntq = neg(tq)
    J = [[z] * d for _ in range(d)]
    for j in range(n):
        J[j][j] = add(mul(tq, b[j]), c)
        J[n + j][n + j] = add(add(mul(tq, f[j]), c), tq)
        J[n + j][j] = ntq
        for i in range(j):
            J[j][i] = ntq
            J[j][n + i] = tq
            J[n + j][i] = ntq
            J[n + j][n + i] = tq
        J[j][2 * n] = b[j]
        J[n + j][2 * n] = f[j]
        J[2 * n][j] = ntq
        J[2 * n][n + j] = tq
    J[2 * n][2 * n] = Kv(1, Q.lo)
    return J


def rowsum(terms_lo, terms_hi, d):
    """sum of interval terms with an explicit float rounding bound."""
    slo = np.sum(terms_lo, axis=0)
    shi = np.sum(terms_hi, axis=0)
    mag = np.sum(np.maximum(np.abs(terms_lo), np.abs(terms_hi)), axis=0)
    err = 4 * (d + 1) * EPS * mag + np.finfo(float).tiny
    return dn(slo - err), up(shi + err)


def Ymul_vec(Y, Fv, d):
    """float matrix Y (N, d, d) times interval vector Fv (list of V) -> list of (lo, hi)."""
    out = []
    for i in range(d):
        tl, th = [], []
        for k in range(d):
            y = Y[:, i, k]
            a, b_ = y * Fv[k].lo, y * Fv[k].hi
            tl.append(np.minimum(a, b_))
            th.append(np.maximum(a, b_))
        out.append(rowsum(np.array(tl), np.array(th), d))
    return out


def _sum_bound(tl, th, axis, d):
    slo = np.sum(tl, axis=axis)
    shi = np.sum(th, axis=axis)
    mag = np.sum(np.maximum(np.abs(tl), np.abs(th)), axis=axis)
    err = 4 * (d + 1) * EPS * mag + np.finfo(float).tiny
    return dn(slo - err), up(shi + err)


def krawczyk(xc, Xlo, Xhi, U, Q, n, chunk=3000):
    """one Krawczyk step, batched: (N, d, d, d) broadcasts in chunks; same bounds as krawczyk_loop."""
    N, d = xc.shape
    Klo = np.empty((N, d))
    Khi = np.empty((N, d))
    for s0 in range(0, N, chunk):
        sl = slice(s0, s0 + chunk)
        xcs, los, his = xc[sl], Xlo[sl], Xhi[sl]
        Us, Qs = V(U.lo[sl], U.hi[sl]), V(Q.lo[sl], Q.hi[sl])
        Y = np.linalg.inv(jac_float(xcs, 0.5 * (Qs.lo + Qs.hi), n))
        Xc = [V(xcs[:, k].copy(), xcs[:, k].copy()) for k in range(d)]
        Fc = F_iv(Xc, Us, Qs, n)
        Flo = np.stack([v.lo for v in Fc], 1)
        Fhi = np.stack([v.hi for v in Fc], 1)
        a, b_ = Y * Flo[:, None, :], Y * Fhi[:, None, :]
        YFlo, YFhi = _sum_bound(np.minimum(a, b_), np.maximum(a, b_), 2, d)
        Xv = [V(los[:, k], his[:, k]) for k in range(d)]
        JX = J_iv(Xv, Qs, n)
        Jlo = np.stack([np.stack([JX[l][k].lo for k in range(d)], 1) for l in range(d)], 1)   # (N, d, d)
        Jhi = np.stack([np.stack([JX[l][k].hi for k in range(d)], 1) for l in range(d)], 1)
        a = Y[:, :, :, None] * Jlo[:, None, :, :]          # (N, i, l, k)
        b_ = Y[:, :, :, None] * Jhi[:, None, :, :]
        YJlo, YJhi = _sum_bound(np.minimum(a, b_), np.maximum(a, b_), 2, d)   # (N, i, k)
        I = np.eye(d)[None, :, :]
        Mlo, Mhi = dn(I - YJhi), up(I - YJlo)
        dXlo, dXhi = dn(los - xcs)[:, None, :], up(his - xcs)[:, None, :]
        p1, p2, p3, p4 = Mlo * dXlo, Mlo * dXhi, Mhi * dXlo, Mhi * dXhi
        tl = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        th = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        mlo, mhi = _sum_bound(tl, th, 2, d)
        Klo[sl] = dn(xcs - YFhi + mlo)
        Khi[sl] = up(xcs - YFlo + mhi)
    return Klo, Khi


def krawczyk_loop(xc, Xlo, Xhi, U, Q, n):
    """one Krawczyk step (reference implementation with Python loops); returns K lo/hi arrays (N, d)."""
    N, d = xc.shape
    Y = np.linalg.inv(jac_float(xc, 0.5 * (Q.lo + Q.hi), n))
    Xc = [V(xc[:, k].copy(), xc[:, k].copy()) for k in range(d)]
    Fc = F_iv(Xc, U, Q, n)
    YF = Ymul_vec(Y, Fc, d)
    Xv = [V(Xlo[:, k], Xhi[:, k]) for k in range(d)]
    JX = J_iv(Xv, Q, n)
    # M = I - Y J(X)
    M = [[None] * d for _ in range(d)]
    for i in range(d):
        for k in range(d):
            tl, th = [], []
            for l in range(d):
                y = Y[:, i, l]
                a, b_ = y * JX[l][k].lo, y * JX[l][k].hi
                tl.append(np.minimum(a, b_))
                th.append(np.maximum(a, b_))
            lo, hi = rowsum(np.array(tl), np.array(th), d)
            delta = 1.0 if i == k else 0.0
            M[i][k] = V(dn(delta - hi), up(delta - lo))
    dX = [V(dn(Xlo[:, k] - xc[:, k]), up(Xhi[:, k] - xc[:, k])) for k in range(d)]
    Klo = np.empty((N, d))
    Khi = np.empty((N, d))
    for i in range(d):
        tl, th = [], []
        for k in range(d):
            p = mul(M[i][k], dX[k])
            tl.append(p.lo)
            th.append(p.hi)
        mlo, mhi = rowsum(np.array(tl), np.array(th), d)
        Klo[:, i] = dn(xc[:, i] - YF[i][1] + mlo)
        Khi[:, i] = up(xc[:, i] - YF[i][0] + mhi)
    return Klo, Khi


def multiplier_iv(Xlo, Xhi, U, Q, n):
    b = [V(Xlo[:, j], Xhi[:, j]) for j in range(n)]
    f = [V(Xlo[:, n + j], Xhi[:, n + j]) for j in range(n)]
    c = V(Xlo[:, 2 * n], Xhi[:, 2 * n])
    tq = mul(Kv(2, Q.lo), Q)
    ok = np.ones(Xlo.shape[0], bool)
    Sp = Kv(0, Q.lo)
    for j in range(n):
        beta = neg(add(mul(tq, b[j]), c))
        phi = add(add(mul(tq, f[j]), c), tq)
        ok &= (beta.lo > 0) & (phi.lo > 0)
        phim = add(mul(tq, f[j]), c)                      # phi - 2q
        betam = sub(beta, tq)                             # beta - 2q
        pb = mul(phi, beta)
        A = div(mul(phim, betam), pb)
        Bj = add(div(f[j], phi), div(mul(phim, b[j]), pb))
        Sp = sub(mul(A, Sp), Bj)
    den = add(c, tq)
    ok &= den.lo > 0
    four_q2 = mul(Kv(4, Q.lo), sq(Q))
    P = div(sub(c, mul(four_q2, Sp)), den)
    return ok, P


def prove(U0, U1, Q0, Q1, S_guess, n):
    """returns status arrays: 0 outside, 1 proved, 2 fold-layer lower bound, 3 fold unchecked, 4 failed."""
    N = U0.size
    status = np.full(N, 4, int)
    Pl = np.full(N, np.nan)
    Ph = np.full(N, np.nan)
    uc, qc = 0.5 * (U0 + U1), 0.5 * (Q0 + Q1)
    S0, X0 = B.newton_float(uc, qc, n, S_guess)
    outside = ~np.isfinite(S0)
    status[outside] = 0
    bs, fs, c = crossings_float(uc, qc, n, S0)
    xc = pack(bs, fs, c)
    # starting box from the linear sensitivity at the centre, dx/dp = -J^-1 dF/dp
    # (Krawczyk decides rigorously; this only sets where to look and how to split)
    d = 2 * n + 1
    split_u = np.ones(N, bool)
    with np.errstate(all="ignore"):
        good_c = np.all(np.isfinite(xc), 1)
        Jc = jac_float(np.where(good_c[:, None], xc, 0.0), qc, n)
        Jc[~good_c] = np.eye(d)
        dFdu = np.zeros((N, d))
        dFdq = np.zeros((N, d))
        Sj = np.zeros(N)
        for j in range(n):
            dFdu[:, j] = 1.0
            dFdq[:, j] = bs[:, j] ** 2 + 2 * Sj
            dFdq[:, n + j] = fs[:, j] ** 2 + 2 * fs[:, j] + 2 * Sj - 2 * bs[:, j]
            Sj = Sj + fs[:, j] - bs[:, j]
        dFdu[:, 2 * n] = 1.0
        dFdq[:, 2 * n] = 1 + 2 * Sj
        dxdu = -np.linalg.solve(Jc, dFdu[:, :, None])[:, :, 0]
        dxdq = -np.linalg.solve(Jc, dFdq[:, :, None])[:, :, 0]
        wu = np.abs(dxdu) * (U1 - U0)[:, None]
        wq = np.abs(dxdq) * (Q1 - Q0)[:, None]
        spread = 0.75 * (wu + wq)
        hull_lo = xc - spread
        hull_hi = xc + spread
        split_u = np.nanmax(wu, 1) >= np.nanmax(wq, 1)
    valid = ~outside & good_c & np.all(np.isfinite(hull_lo), 1) & np.all(np.isfinite(hull_hi), 1)
    status[~outside & ~valid] = 4
    fold = valid & (X0 > 0.995)
    U, Q = V(U0, U1), V(Q0, Q1)
    if fold.any():
        idx = np.where(fold)[0]
        rad = (hull_hi[idx] - hull_lo[idx]) * 0.5 + 1e-9
        ok, P = multiplier_iv(hull_lo[idx] - rad, hull_hi[idx] + rad, V(U0[idx], U1[idx]), V(Q0[idx], Q1[idx]), n)
        good = ok & (P.lo > -1)
        status[idx[good]] = 2
        status[idx[~good]] = 3
    idx = np.where(valid & ~fold)[0]
    if idx.size:
        Ui, Qi = V(U0[idx], U1[idx]), V(Q0[idx], Q1[idx])
        xci = xc[idx]
        rad = (hull_hi[idx] - hull_lo[idx]) * 0.5
        lo = np.minimum(hull_lo[idx], xci) - (0.5 * rad + 1e-10 + 1e-8 * np.abs(xci))
        hi = np.maximum(hull_hi[idx], xci) + (0.5 * rad + 1e-10 + 1e-8 * np.abs(xci))
        done = np.zeros(idx.size, bool)
        for it in range(6):
            todo = np.where(~done)[0]
            if todo.size == 0:
                break
            with np.errstate(all="ignore"):
                Klo, Khi = krawczyk(xci[todo], lo[todo], hi[todo], V(Ui.lo[todo], Ui.hi[todo]), V(Qi.lo[todo], Qi.hi[todo]), n)
            inside = np.all(Klo > lo[todo], 1) & np.all(Khi < hi[todo], 1)
            if inside.any():
                t = todo[inside]
                klo, khi = Klo[inside], Khi[inside]
                ok, P = multiplier_iv(klo, khi, V(Ui.lo[t], Ui.hi[t]), V(Qi.lo[t], Qi.hi[t]), n)
                good = ok & (P.lo > -1) & (P.hi < 1)
                status[idx[t[good]]] = 1
                Pl[idx[t[good]]] = P.lo[good]
                Ph[idx[t[good]]] = P.hi[good]
                done[t] = True            # enclosed; if not good it stays status 4 (enclosed, not proved stable)
            nt = todo[~inside]
            if nt.size:
                # epsilon inflation towards the Krawczyk image
                kl, kh = Klo[~inside], Khi[~inside]
                finite = np.all(np.isfinite(kl), 1) & np.all(np.isfinite(kh), 1)
                w = np.where(finite[:, None], kh - kl, hi[nt] - lo[nt])
                cen = np.where(finite[:, None], 0.5 * (kl + kh), 0.5 * (lo[nt] + hi[nt]))
                lo[nt] = np.minimum(lo[nt], cen - 0.6 * w - 1e-12)
                hi[nt] = np.maximum(hi[nt], cen + 0.6 * w + 1e-12)
    return status, Pl, Ph, S0, split_u


def main():
    t0 = time.time()
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    du = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    dq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    n_min = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    global MAX_DEPTH
    MAX_DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 12      # binary splits
    u_lo = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
    u_hi = float(sys.argv[7]) if len(sys.argv) > 7 else 0.95
    tag = sys.argv[8] if len(sys.argv) > 8 else ""
    out_path = os.path.join(HERE, "d120d_krawczyk_stability_proof%s_2026-09-14.json" % (("_" + tag) if tag else ""))
    report = json.load(open(out_path)) if (n_min > 1 and os.path.exists(out_path)) else {"du": du, "dq": dq, "per_n": {}}
    CH = 60000
    for n in range(n_min, n_max + 1):
        edges = np.round(np.arange(u_lo, u_hi + du / 2, du), 10)
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
              "failed": 0, "failed_area": 0.0, "P_min": 1.0, "P_max": -1.0, "by_depth": [], "failed_boxes": []}
        depth = 0
        while U0.size:
            status = np.empty(U0.size, int); Pl = np.full(U0.size, np.nan); Ph = np.full(U0.size, np.nan); S0 = np.full(U0.size, np.nan)
            SU = np.ones(U0.size, bool)
            for i in range(0, U0.size, CH):
                s_ = slice(i, i + CH)
                status[s_], Pl[s_], Ph[s_], S0[s_], SU[s_] = prove(U0[s_], U1[s_], Q0[s_], Q1[s_], S_guess[s_], n)
            area = (U1 - U0) * (Q1 - Q0)
            pr = status == 1
            st["proved"] += int(pr.sum()); st["proved_area"] += float(area[pr].sum())
            st["outside"] += int((status == 0).sum()); st["fold_lower"] += int((status == 2).sum()); st["fold_unchecked"] += int((status == 3).sum())
            st["by_depth"].append([depth, int(U0.size), int(pr.sum()), int((status == 4).sum())])
            if pr.any():
                st["P_min"] = min(st["P_min"], float(np.nanmin(Pl[pr]))); st["P_max"] = max(st["P_max"], float(np.nanmax(Ph[pr])))
            bad = np.where(status == 4)[0]
            if depth >= MAX_DEPTH or bad.size == 0:
                st["failed"] += int(bad.size); st["failed_area"] += float(area[bad].sum())
                st["failed_boxes"] = [[float(U0[i]), float(U1[i]), float(Q0[i]), float(Q1[i])] for i in bad[:2000]]
                break
            # split each failed box in two, across the parameter that contributes more width
            su = SU[bad]
            um, qm = 0.5 * (U0[bad] + U1[bad]), 0.5 * (Q0[bad] + Q1[bad])
            sg = np.where(np.isfinite(S0[bad]), S0[bad], S_guess[bad])
            a0 = np.where(su, U0[bad], U0[bad]); a1 = np.where(su, um, U1[bad])
            b0 = np.where(su, um, U0[bad]); b1 = np.where(su, U1[bad], U1[bad])
            c0 = np.where(su, Q0[bad], Q0[bad]); c1 = np.where(su, Q1[bad], qm)
            e0 = np.where(su, Q0[bad], qm); e1 = np.where(su, Q1[bad], Q1[bad])
            U0, U1 = np.concatenate([a0, b0]), np.concatenate([a1, b1])
            Q0, Q1 = np.concatenate([c0, e0]), np.concatenate([c1, e1])
            S_guess = np.tile(sg, 2)
            depth += 1
        report["per_n"][str(n)] = st
        print("n=%d: %d base boxes; proved %d (area %.5f), P' in [%.4f, %.4f]; depth [depth, boxes, proved, failed] %s; "
              "fold layer P'>-1 %d, unchecked %d; failed %d (area %.2e); outside %d   (%.0f s)"
              % (n, st["boxes"], st["proved"], st["proved_area"], st["P_min"], st["P_max"], st["by_depth"], st["fold_lower"],
                 st["fold_unchecked"], st["failed"], st["failed_area"], st["outside"], time.time() - t0), flush=True)
        json.dump(report, open(out_path, "w"))
    print("%.0f s -> %s" % (time.time() - t0, os.path.basename(out_path)))


def run_row(n, a, b, ta, tb, dq, CH=60000):
    """all boxes of one u-row [a, b]: prove, split failures, return stats and every failed box."""
    qlo, qhi = min(ta[-1][0], tb[-1][0]), max(ta[0][0], tb[0][0])
    qe = np.arange(math.floor(qlo / dq) * dq, qhi + dq, dq)
    Q0, Q1 = qe[:-1], qe[1:]
    qcc = 0.5 * (Q0 + Q1)
    ra, rb = sorted((p[0], p[1]) for p in ta), sorted((p[0], p[1]) for p in tb)
    S_guess = 0.5 * (np.interp(qcc, [x[0] for x in ra], [x[1] for x in ra]) + np.interp(qcc, [x[0] for x in rb], [x[1] for x in rb]))
    U0, U1 = np.full(qcc.shape, a), np.full(qcc.shape, b)
    st = {"boxes": int(U0.size), "proved": 0, "proved_area": 0.0, "fold_lower": 0, "fold_unchecked": 0, "outside": 0,
          "failed": 0, "failed_area": 0.0, "P_min": None, "P_max": None, "by_depth": []}
    failed = []
    depth = 0
    while U0.size:
        status = np.empty(U0.size, int); Pl = np.full(U0.size, np.nan); Ph = np.full(U0.size, np.nan)
        S0 = np.full(U0.size, np.nan); SU = np.ones(U0.size, bool)
        for i in range(0, U0.size, CH):
            s_ = slice(i, i + CH)
            with np.errstate(all="ignore"):
                status[s_], Pl[s_], Ph[s_], S0[s_], SU[s_] = prove(U0[s_], U1[s_], Q0[s_], Q1[s_], S_guess[s_], n)
        area = (U1 - U0) * (Q1 - Q0)
        pr = status == 1
        st["proved"] += int(pr.sum()); st["proved_area"] += float(area[pr].sum())
        st["outside"] += int((status == 0).sum()); st["fold_lower"] += int((status == 2).sum()); st["fold_unchecked"] += int((status == 3).sum())
        st["by_depth"].append([depth, int(U0.size), int(pr.sum()), int((status == 4).sum())])
        if pr.any():
            lo_, hi_ = float(np.nanmin(Pl[pr])), float(np.nanmax(Ph[pr]))
            st["P_min"] = lo_ if st["P_min"] is None else min(st["P_min"], lo_)
            st["P_max"] = hi_ if st["P_max"] is None else max(st["P_max"], hi_)
        bad = np.where(status == 4)[0]
        if depth >= MAX_DEPTH or bad.size == 0:
            st["failed"] = int(bad.size); st["failed_area"] = float(area[bad].sum())
            failed = [[float(U0[i]), float(U1[i]), float(Q0[i]), float(Q1[i])] for i in bad]
            break
        su = SU[bad]
        um, qm = 0.5 * (U0[bad] + U1[bad]), 0.5 * (Q0[bad] + Q1[bad])
        sg = np.where(np.isfinite(S0[bad]), S0[bad], S_guess[bad])
        nU0 = np.concatenate([U0[bad], np.where(su, um, U0[bad])]); nU1 = np.concatenate([np.where(su, um, U1[bad]), U1[bad]])
        nQ0 = np.concatenate([Q0[bad], np.where(su, Q0[bad], qm)]); nQ1 = np.concatenate([np.where(su, Q1[bad], qm), Q1[bad]])
        U0, U1, Q0, Q1 = nU0, nU1, nQ0, nQ1
        S_guess = np.tile(sg, 2)
        depth += 1
    # where the failures lie inside the existence interval (birth = tracked lower end, death = exact)
    qb_ = 0.5 * (ta[-1][0] + tb[-1][0])
    qd_ = 0.5 * (ta[0][0] + tb[0][0]) - 0.01
    bins = [0, 0, 0, 0, 0]
    for u0_, u1_, q0_, q1_ in failed:
        rel = (0.5 * (q0_ + q1_) - qb_) / (qd_ - qb_)
        bins[0 if rel < 0.02 else 1 if rel < 0.1 else 2 if rel <= 0.9 else 3 if rel <= 0.98 else 4] += 1
    st["failed_rel_bins_lt0.02_lt0.1_mid_le0.98_gt0.98"] = bins
    st["q_birth_track"], st["q_death"] = qb_, qd_
    return st, failed


def main_rows():
    """row-by-row version of main(): every u-row is appended to a JSONL checkpoint as soon
    as it is done, a restarted run skips rows already in the file, and every failed box is
    kept.  Same per-box algorithm as main()."""
    t0 = time.time()
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    du = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    dq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    n_min = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    global MAX_DEPTH
    MAX_DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    u_lo = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
    u_hi = float(sys.argv[7]) if len(sys.argv) > 7 else 0.95
    tag = sys.argv[8] if len(sys.argv) > 8 else "all"
    for n in range(n_min, n_max + 1):
        ck = os.path.join(HERE, "d120d_rows_%s_n%d_2026-09-14.jsonl" % (tag, n))
        done = {}
        if os.path.exists(ck):
            for line in open(ck, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                      # a line cut off by an interruption
                done[round(r["u0"], 6)] = r
        edges = np.round(np.arange(u_lo, u_hi + du / 2, du), 10)
        tracks = {}
        k_new = 0
        with open(ck, "a", encoding="utf-8") as fh:
            for a, b in zip(edges[:-1], edges[1:]):
                if round(float(a), 6) in done:
                    continue
                for e in (a, b):
                    if e not in tracks:
                        tracks[e] = B.P0.track(float(e), n)[0] or None
                ta, tb = tracks[a], tracks[b]
                if ta is None or tb is None:
                    rec = {"n": n, "u0": float(a), "u1": float(b), "no_track": True}
                else:
                    st, failed = run_row(n, float(a), float(b), ta, tb, dq)
                    rec = {"n": n, "u0": float(a), "u1": float(b), **st, "failed_boxes": failed}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[round(float(a), 6)] = rec
                tracks.pop(a, None)
                k_new += 1
                if k_new % 10 == 0:
                    print("n=%d row u0=%.3f done (%d new rows, %.0f s)" % (n, a, k_new, time.time() - t0), flush=True)
        rows = [done[k] for k in sorted(done)]
        good = [r for r in rows if not r.get("no_track")]
        tot = {"rows": len(rows), "rows_no_track": len(rows) - len(good),
               "proved": sum(r["proved"] for r in good), "proved_area": sum(r["proved_area"] for r in good),
               "failed": sum(r["failed"] for r in good), "failed_area": sum(r["failed_area"] for r in good),
               "fold_lower": sum(r["fold_lower"] for r in good), "fold_unchecked": sum(r["fold_unchecked"] for r in good),
               "outside": sum(r["outside"] for r in good),
               "P_min": min((r["P_min"] for r in good if r["P_min"] is not None), default=None),
               "P_max": max((r["P_max"] for r in good if r["P_max"] is not None), default=None),
               "failed_rel_bins": [sum(r["failed_rel_bins_lt0.02_lt0.1_mid_le0.98_gt0.98"][i] for r in good) for i in range(5)],
               "u_range": [u_lo, u_hi], "du": du, "dq": dq, "max_depth": MAX_DEPTH, "checkpoint": os.path.basename(ck)}
        json.dump(tot, open(os.path.join(HERE, "d120d_summary_%s_n%d_2026-09-14.json" % (tag, n)), "w"), indent=1)
        print("n=%d: rows %d (no track %d); proved %d, area %.6f; failed %d, area %.3e, by position %s; fold P'>-1 %d, "
              "unchecked %d; P' in [%s, %s]  (%.0f s)"
              % (n, tot["rows"], tot["rows_no_track"], tot["proved"], tot["proved_area"], tot["failed"], tot["failed_area"],
                 tot["failed_rel_bins"], tot["fold_lower"], tot["fold_unchecked"], tot["P_min"], tot["P_max"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main_rows()
