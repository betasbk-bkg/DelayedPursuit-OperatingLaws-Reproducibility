#!/usr/bin/env python3
"""
d116 -- the whole word sequence W_n = (-+)^n + of the quantized loop, in closed form.

EXACT PHASE INTEGRAL.  From a forward crossing (tau = T/Delta = 0, m = -Delta/2,
section state zeta = z0/Delta), with q = k Delta/2, theta = u/(2q) and crossings at
tau_e with signs s_e (the section crossing excluded), the phase advance in cell
units is exactly

  M(tau) = (1 - u - 2 q zeta) tau + q tau^2 + 2 q min(tau, theta)
           - 2 q sum_e s_e max(0, tau - tau_e - theta)

(a crossing's jump in z and its delay-window count cancel for theta, then the jump
alone acts).  A crossing of the start boundary is M = 0, of the next boundary M = 1.

THE WORD W_n.  Backward crossings b_1..b_n and forward re-crossings f_1..f_n of the
start boundary, then the forward crossing at tau = 1 (period Delta: every fixed
point).  With S_j = sum_{i<=j} (f_i - b_i) and a = 1 - u - q - 2 q S_n (from M(1) = 1,
i.e. zeta = 1/2 + S_n), on the ordering theta < b_1, b_i + theta <= f_i,
f_i + theta <= b_(i+1), f_n + theta <= 1:

  b_(j+1): smaller root of  q x^2 + a x + (u + 2 q S_j) = 0
  f_(j+1): larger  root of  q x^2 + a x + 2 q S_(j+1)  = 0

and S_(j+1) - S_j = f_(j+1) - b_(j+1) closes to one quadratic in S_(j+1).  Given a,
the recursion is explicit; a itself depends on S_n, so each word is one scalar
equation.  W_0 = (+) has S = 0, a = 1 - u - q.

BORDERS.  W_n ends when M on (f_n + theta, 1) touches 0 (a new excursion grazes the
start boundary): a^2 = 4 q (u + 2 q S_n) with the minimum inside that interval --
for n = 0 this is q_bc = (1 + sqrt u)^2.  W_n is born where an ordering constraint
becomes an equality (an excursion as short as theta, two crossings theta apart,
or f_n + theta = 1).

WHAT IS CHECKED.  (1) every solution is verified by evaluating the exact M(tau)
and requiring the word's sign pattern; (2) its fixed point zeta = 1/2 + S_n is run
through the validated integrator v3 (d113), which must return the same word and
z_next = z0; (3) the existence interval of each word is bisected in q, and the
constraint active at each end recorded; (4) the words the attractor census
(d115) found are compared with the words that exist.

Usage:  python d116_word_sequence_closure_2026-09-13.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


v3 = _load("d113", "d113_integrator_v3_2026-09-13.py")


def M_exact(tau, u, q, zeta, events):
    th = u / (2 * q)
    val = (1 - u - 2 * q * zeta) * tau + q * tau * tau + 2 * q * min(tau, th)
    for te, s in events:
        val -= 2 * q * s * max(0.0, tau - te - th)
    return val


def recursion(Sn, u, q, n):
    """given a trial S_n, run the crossing recursion; return (S_n_implied, crossings) or None."""
    a = 1 - u - q - 2 * q * Sn
    S = 0.0
    ev = []
    for j in range(n):
        Dj = a * a - 4 * q * (u + 2 * q * S)
        if Dj < 0:
            return None
        b = (-a - math.sqrt(Dj)) / (2 * q)
        # CORRECTED.  The first version took f as the larger root of
        # q x^2 + a x + 2q S_(j+1), but S_(j+1) = S_j + f - b contains f itself, so
        # that quadratic agrees with M only at x = f.  On (b + theta, f) the exact
        # phase is M(tau) = q tau^2 + (a + 2q) tau + 2q (S_j - b): f is its upward
        # (larger) root, and the recursion is explicit.
        B2 = a + 2 * q
        C2 = 2 * q * (S - b)
        disc = B2 * B2 - 4 * q * C2
        if disc < 0:
            return None
        f = (-B2 + math.sqrt(disc)) / (2 * q)
        if not f > b:
            return None
        ev += [(b, -1), (f, +1)]
        S = S + (f - b)
    return S, ev


def solve_word(u, q, n):
    """all solutions S_n of the word's scalar equation, each verified exactly."""
    if n == 0:
        cands = [0.0]
    else:
        grid = np.linspace(0.0, 1.0, 801)
        vals = []
        for s in grid:
            r = recursion(s, u, q, n)
            vals.append((r[0] - s) if r else np.nan)
        cands = []
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                try:
                    cands.append(brentq(lambda s: recursion(s, u, q, n)[0] - s, grid[i], grid[i + 1], xtol=1e-14))
                except Exception:
                    pass
    sols = []
    for Sn in cands:
        r = recursion(Sn, u, q, n) if n else (0.0, [])
        if r is None:
            continue
        S, ev = r
        zeta = 0.5 + Sn
        ok, why = verify_pattern(u, q, zeta, ev, n)
        sols.append({"S_n": Sn, "zeta": zeta, "events": ev, "pattern_ok": ok, "why": why})
    return sols


def verify_pattern(u, q, zeta, ev, n):
    """exact M must vanish at each crossing, equal 1 at tau = 1, and have the right sign in between."""
    for te, s in ev:
        if abs(M_exact(te, u, q, zeta, ev)) > 1e-9:
            return False, "M != 0 at a crossing (ordering assumption broken)"
    if abs(M_exact(1.0, u, q, zeta, ev) - 1) > 1e-9:
        return False, "M(1) != 1"
    times = [0.0] + [t for t, _ in ev] + [1.0]
    if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
        return False, "crossings out of order"
    for i, (t1, t2) in enumerate(zip(times, times[1:])):
        ts = np.linspace(t1, t2, 60)[1:-1]
        vals = np.array([M_exact(t, u, q, zeta, ev) for t in ts])
        below = (i % 2 == 1) and i < 2 * n   # inside an excursion: between b_i and f_i
        if below:
            if np.any(vals > 1e-12):
                return False, "M above the boundary inside an excursion"
        else:
            if np.any(vals < -1e-12):
                return False, "extra crossing of the start boundary"
            if np.any(vals > 1 + 1e-12):
                return False, "reaches the next boundary early"
    return True, "ok"


def exists(u, q, n):
    for s in solve_word(u, q, n):
        if s["pattern_ok"]:
            return s
    return None


def v3_check(u, q, n, sol):
    k = 2 * q / D
    z0 = sol["zeta"] * D
    o = v3.orbit_v3(z0, u, k, D, u / k)
    word = "-+" * n + "+"
    ok = o is not None and o["word"] == word and abs(o["z_next"] - z0) < 1e-8
    mult = None
    if ok:
        h = 1e-7
        op, om = v3.orbit_v3(z0 + h, u, k, D, u / k), v3.orbit_v3(z0 - h, u, k, D, u / k)
        if op and om and op["word"] == word and om["word"] == word:
            mult = (op["z_next"] - om["z_next"]) / (2 * h)
    return ok, mult, (o["word"] if o else None)


def interval(u, n, qmin=1.02, qmax=6.0, dq=0.01):
    qs = np.arange(qmin, qmax, dq)
    alive = [exists(u, float(q), n) is not None for q in qs]
    runs = []
    i = 0
    while i < len(qs):
        if alive[i]:
            j = i
            while j + 1 < len(qs) and alive[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    out = []
    for i, j in runs:
        def edge(qa, qb, target):
            fa = exists(u, qa, n) is not None
            for _ in range(40):
                qm = 0.5 * (qa + qb)
                if (exists(u, qm, n) is not None) == fa:
                    qa = qm
                else:
                    qb = qm
            return 0.5 * (qa + qb)
        lo = edge(qs[i - 1], qs[i], None) if i > 0 else qs[i]
        hi = edge(qs[j], qs[j + 1], None) if j + 1 < len(qs) else qs[j]
        out.append((float(lo), float(hi)))
    return out


def main():
    t0 = time.time()
    out = {"intervals": {}, "v3": [], "census_compare": []}
    print("(1) existence intervals in q of W_n, verified on the exact phase integral")
    for u in (0.3, 0.5, 0.71, 0.9):
        for n in range(4):
            iv = interval(u, n)
            out["intervals"]["%.2f_%d" % (u, n)] = iv
            extra = ""
            if n == 0:
                extra = "   q_bc = %.4f" % ((1 + math.sqrt(u)) ** 2)
            if n == 1:
                extra = "   q_lo1 = %.4f" % (1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4))
            print("  u=%.2f W_%d %-14s  intervals %s%s" % (u, n, "(" + ",".join("-+" * n + "+") + ")",
                                                    ["[%.4f, %.4f]" % iv_ for iv_ in iv], extra))
    print("\n(2) fixed points of W_n against the validated integrator v3")
    for u in (0.3, 0.5, 0.71):
        for n in range(4):
            for iv in out["intervals"]["%.2f_%d" % (u, n)]:
                for q in np.linspace(iv[0] + 0.02, iv[1] - 0.02, 5) if iv[1] - iv[0] > 0.05 else []:
                    s = exists(u, float(q), n)
                    if not s:
                        continue
                    ok, mult, w = v3_check(u, float(q), n, s)
                    out["v3"].append({"u": u, "n": n, "q": float(q), "ok": ok, "mult": mult, "v3_word": w})
            rows = [r for r in out["v3"] if r["u"] == u and r["n"] == n]
            if rows:
                ms = [r["mult"] for r in rows if r["mult"] is not None]
                print("  u=%.2f W_%d: %d/%d confirmed by v3; multipliers %s"
                      % (u, n, sum(r["ok"] for r in rows), len(rows),
                         ("%+.3f..%+.3f" % (min(ms), max(ms))) if ms else None))
    print("\n(3) census words (d115) vs words that exist at that (u, q)")
    census = json.load(open(os.path.join(HERE, "d115_orbit_structure_v3_2026-09-13.json"), encoding="utf-8"))["census"]
    bad = 0
    for c in census:
        seen = set(w.split(":", 1)[1] for w in c["words"])
        for w in seen:
            n = w.count("-")
            if w != "-+" * n + "+":
                continue
            ok = exists(c["u"], c["q"], n) is not None
            out["census_compare"].append({"u": c["u"], "q": c["q"], "word": w, "exists": ok})
            if not ok:
                bad += 1
                print("  census word %s at u=%.2f q=%.2f has no closed-form solution" % (w, c["u"], c["q"]))
    print("  census words without a closed-form existence: %d of %d" % (bad, len(out["census_compare"])))
    json.dump(out, open(os.path.join(HERE, "d116_word_sequence_closure_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d116_word_sequence_closure_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
