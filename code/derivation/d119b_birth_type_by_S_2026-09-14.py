#!/usr/bin/env python3
"""
d119b -- birth type of every word W_n, with the branch parametrised by S; and the
multiplier at every death.

STATUS (2026-09-14).  Part (B), the death multipliers, is valid: it solves the exact
death condition directly.  Part (A) FAILED: the continuation in S jumps between roots
of the fixed-point equation in q and loses the branch at almost every u.  The birth
type is computed instead by d119c, which solves the border as a two-dimensional root
problem.  Kept as the record.

WHY S.  Along a W_n branch the fixed point is a fold in q, so marching in q and
root-finding in S loses the twin roots (d119).  For fixed S the fixed-point
condition F_n(c(q, S); u, q) = S, c = 1 - u - q - 2 q S, is one smooth equation in q,
and q(S) is smooth through the fold.  P' = 1 exactly where dq/dS = 0.

(A) BIRTH.  Walk S down from a point well inside the existence interval, following
q(S) by continuation, until the ordering slack
    min( b_1 - theta,  f_j - b_j - theta,  b_(j+1) - f_j - theta,  1 - f_n - theta )
changes sign; locate the border S_B by brentq on the slack.  The physical side is
where the slack is positive.  If q increases as S moves into the physical side, the
word is born AT the border (border type) at q_B, with multiplier P_B.  If q decreases,
q(S) has an interior minimum (fold type): the stable/unstable pair is born at the
fold q_F < q_B... [the unstable member runs from the fold back to the border].
The codimension-two switch is dq/dS = 0 at the border, where P_B = 1.

(B) DEATH.  For n = 1-8 and u = 0.05-0.95, solve the exact death condition
    c = 2 sqrt(q)(1 - sqrt(q)),  F_n(c; u, q) = ((sqrt q - 1)^2 - u)/(2q)
for q, keep roots whose crossings satisfy the ordering, and evaluate P' there.

Usage:  python d119b_birth_type_by_S_2026-09-14.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d118", os.path.join(HERE, "d118_word_multiplier_closed_form_2026-09-14.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)
w = m.w


def resid(q, S, u, n):
    r = w.recursion(S, u, q, n)
    return None if r is None else r[0] - S


def q_of_S(S, u, n, q0, half=0.15, step=0.002):
    """root of resid in q nearest q0 (continuation)."""
    qs = np.arange(q0 - half, q0 + half + step, step)
    vals = [resid(float(x), S, u, n) for x in qs]
    best = None
    for i in range(len(qs) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None or a * b > 0:
            continue
        try:
            r = brentq(lambda x: resid(x, S, u, n), float(qs[i]), float(qs[i + 1]), xtol=1e-15)
        except Exception:
            continue
        if best is None or abs(r - q0) < abs(best - q0):
            best = r
    return best


def slack(S, u, q, n):
    c = 1 - u - q - 2 * q * S
    r = m.rec_c(c, u, q, n)
    if r is None:
        return None, None
    th = u / (2 * q)
    ev = r[1]
    parts = {"b1": ev[0][0] - th, "end": 1 - ev[-1][1] - th}
    parts["short_exc"] = min(f - b for b, f in ev) - th
    if n > 1:
        parts["gap"] = min(ev[j + 1][0] - ev[j][1] for j in range(n - 1)) - th
    k = min(parts, key=parts.get)
    return parts[k], k


def birth(u, n, q_inside):
    sol = w.exists(u, q_inside, n)
    if sol is None:
        return None
    S, q = sol["S_n"], q_inside
    s0, _ = slack(S, u, q, n)
    if s0 is None or s0 <= 0:
        return None
    dS = 2e-4
    # walk in the direction of S along which q decreases (towards the birth side)
    qp0 = q_of_S(S + 1e-5, u, n, q, half=0.01, step=0.0001)
    qm0 = q_of_S(S - 1e-5, u, n, q, half=0.01, step=0.0001)
    if qp0 is None or qm0 is None:
        return {"u": u, "n": n, "status": "no start slope", "S": S, "q": q}
    walk = -1.0 if qp0 > qm0 else 1.0
    prev = (S, q, s0)
    for _ in range(4000):
        S2 = prev[0] + walk * dS
        q2 = q_of_S(S2, u, n, prev[1])
        if q2 is None or not (1.0 < q2 < 30.0):
            return {"u": u, "n": n, "status": "lost continuation", "S": prev[0], "q": prev[1], "walk": walk}
        s2, _ = slack(S2, u, q2, n)
        if s2 is None:
            return {"u": u, "n": n, "status": "recursion undefined", "S": prev[0], "q": prev[1]}
        if s2 <= 0:
            lo, hi = sorted((S2, prev[0]))
            qref = [prev[1]]

            def g(Sx):
                qx = q_of_S(Sx, u, n, qref[0], half=0.02, step=0.0002)
                qref[0] = qx
                return slack(Sx, u, qx, n)[0]
            SB = brentq(g, lo, hi, xtol=1e-13)
            qB = q_of_S(SB, u, n, qref[0], half=0.02, step=0.0002)
            inward = -walk                       # the physical side lies opposite to the walk
            Sin = SB + inward * 1e-9
            _, which = slack(Sin, u, q_of_S(Sin, u, n, qB, half=0.01, step=0.0001), n)
            h = 1e-6
            qp = q_of_S(SB + h, u, n, qB, half=0.01, step=0.0001)
            qm = q_of_S(SB - h, u, n, qB, half=0.01, step=0.0001)
            dqdS = (qp - qm) / (2 * h)
            dq_inward = dqdS * inward            # > 0: q grows into the physical side, born at the border
            PB = m.multiplier(u, qB, n, {"S_n": SB})[0]
            row = {"u": u, "n": n, "status": "ok", "S_B": SB, "q_B": qB, "border": which, "dq_dS": dqdS,
                   "dq_inward": dq_inward, "walk": walk, "P_B": PB,
                   "type": "border" if dq_inward > 0 else "fold"}
            if dq_inward <= 0:
                # interior minimum of q(S) on the physical side
                Ss = np.linspace(SB, SB + inward * 0.03, 301)
                qq, qc = [], qB
                for Sx in Ss:
                    qc = q_of_S(float(Sx), u, n, qc, half=0.02, step=0.0002)
                    qq.append(qc if qc is not None else np.nan)
                i = int(np.nanargmin(qq))
                SF = float(Ss[i])
                row.update({"S_F": SF, "q_F": float(qq[i]), "fold_width": qB - float(qq[i]),
                            "P_F": m.multiplier(u, float(qq[i]), n, {"S_n": SF})[0]})
            return row
        prev = (S2, q2, s2)
    return {"u": u, "n": n, "status": "no border within walk"}


def death(u, n):
    def eq(q):
        sq = math.sqrt(q)
        c = 2 * sq * (1 - sq)
        r = m.rec_c(c, u, q, n)
        return None if r is None else r[0] - ((sq - 1) ** 2 - u) / (2 * q)
    qs = np.arange(1.05, 30.0, 0.01)
    vals = [eq(float(x)) for x in qs]
    out = []
    for i in range(len(qs) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None or a * b > 0:
            continue
        qd = brentq(eq, float(qs[i]), float(qs[i + 1]), xtol=1e-14)
        sq = math.sqrt(qd)
        S = ((sq - 1) ** 2 - u) / (2 * qd)
        sl, which = slack(S, u, qd, n)
        if sl is None or sl < -1e-9:
            continue
        # the graze must lie inside (f_n + theta, 1)
        c = 2 * sq * (1 - sq)
        fn = m.rec_c(c, u, qd, n)[1][-1][1]
        tmin = -c / (2 * qd)
        if not (fn + u / (2 * qd) < tmin < 1):
            continue
        out.append({"u": u, "n": n, "q_death": qd, "P_death": m.multiplier(u, qd, n, {"S_n": S})[0]})
    return out


def main():
    t0 = time.time()
    bd = json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))
    res = {"birth": [], "death": [], "switch": {}}
    for n in (1, 2, 3):
        rows = sorted((r["u"], r["birth"], r["death"]) for r in bd if r["n"] == n and r["exists"] and r["birth"] is not None)
        for u in np.round(np.arange(0.20, 0.951, 0.025), 3):
            qb = float(np.interp(u, [x[0] for x in rows], [x[1] for x in rows]))
            qd = float(np.interp(u, [x[0] for x in rows], [x[2] if x[2] else x[1] + 1 for x in rows]))
            row = birth(float(u), n, qb + 0.25 * (qd - qb))
            if row is None:
                continue
            res["birth"].append(row)
            if row["status"] != "ok":
                print("n=%d u=%.3f %s" % (n, u, row["status"]))
                continue
            extra = (", fold at q_F=%.6f (width %.2e), P_F=%.4f" % (row["q_F"], row["fold_width"], row["P_F"])) if row["type"] == "fold" else ""
            print("n=%d u=%.3f  %-6s border(%s) q_B=%.6f dq/dS=%+.3f P_B=%.4f%s"
                  % (n, u, row["type"], row["border"], row["q_B"], row["dq_dS"], row["P_B"], extra))
        ok = [r for r in res["birth"] if r["n"] == n and r["status"] == "ok"]
        consistent = all((r["P_B"] < 1) == (r["type"] == "border") for r in ok)
        res["switch"][str(n)] = {"P_B_lt_1_iff_border_type": consistent,
                                 "types": [(r["u"], r["type"], r["border"]) for r in ok]}
        print("  n=%d: P_B < 1 <=> border type at all %d u: %s" % (n, len(ok), consistent))
    for n in range(1, 9):
        for u in np.round(np.arange(0.05, 0.951, 0.05), 2):
            res["death"] += death(float(u), n)
    Pd = [r["P_death"] for r in res["death"]]
    print("(B) %d deaths (n = 1-8, u = 0.05-0.95): P' at death in [%.4f, %.4f]; all > -1: %s"
          % (len(Pd), min(Pd), max(Pd), min(Pd) > -1))
    json.dump(res, open(os.path.join(HERE, "d119b_birth_type_by_S_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d119b_birth_type_by_S_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
