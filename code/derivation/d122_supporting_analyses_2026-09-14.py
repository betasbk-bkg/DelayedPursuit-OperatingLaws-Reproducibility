#!/usr/bin/env python3
"""
d122 -- the analyses that were first run inline on 2026-09-14 and quoted in the
discussion, recorded so that every quoted number has a file behind it.

(a) Stable member along the existence intervals (d116b rows, n = 1-5, u = 0.10-0.95):
    at 119 cosine-spaced q per interval, every pattern-verified fixed point of W_n and
    its closed-form multiplier (d118 T1); count points with no fixed point |P'| < 1,
    the closest approach of the stable member to |P'| = 1, and the lowest P'.
(b) Collapse test: along the tracked stable branch (d120.track) for n = 1-5 and
    u = 0.2, 0.4, 0.6, 0.8, P'(s) with s = (q - q_birth)/(q_death - q_birth), and the
    spread of P' across (n, u), within each u and within each n.
(c) Boundary layer after birth: W_1 at u = 0.6 and 0.75, all pattern-verified fixed
    points and P' at q = q_birth + dq (q_birth from d116b).
(d) The n = 1 boxes that the Krawczyk proof left unproved (checkpoint
    d120d_rows_n13_n1), classified against the exact physical birth: q_lo(u) for
    u < 4/7; the fold just below q_lo for 4/7 < u < (9 + sqrt 17)/16; the end-window
    border (quadratic in q, d121) and its fold for u < 0.834459, the border above.

Usage:  python d122_supporting_analyses_2026-09-14.py
"""
import importlib.util
import json
import math
import os
import time
import warnings

import numpy as np
from scipy.optimize import fsolve

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


m = _load("d118", "d118_word_multiplier_closed_form_2026-09-14.py")
w = m.w
P0 = _load("d120", "d120_interval_stability_proof_2026-09-14.py")
C = _load("d119c", "d119c_birth_border_2d_2026-09-14.py")


def part_a():
    bd = json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))
    tot = bad = 0
    closest = (9.0, None)
    lowest = (9.0, None)
    for r in bd:
        if not r["exists"] or r["death"] is None or r["n"] == 0:
            continue
        u, n, lo, hi = r["u"], r["n"], r["birth"], r["death"]
        for q in lo + (hi - lo) * (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, 121)))[1:-1]:
            sols = [x for x in w.solve_word(u, float(q), n) if x["pattern_ok"]]
            if not sols:
                continue
            Ps = [m.multiplier(u, float(q), n, x)[0] for x in sols]
            st = [p for p in Ps if abs(p) < 1]
            tot += 1
            if not st:
                bad += 1
                continue
            mx = max(st, key=abs)
            if 1 - abs(mx) < closest[0]:
                closest = (1 - abs(mx), [u, n, float(q), mx])
            mn = min(st)
            if mn < lowest[0]:
                lowest = (mn, [u, n, float(q)])
    return {"points": tot, "points_without_stable_member": bad, "closest_margin_to_abs1": closest[0],
            "closest_at_u_n_q_P": closest[1], "lowest_P": lowest[0], "lowest_at_u_n_q": lowest[1]}


def part_b():
    sg = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    tab = []
    for n in range(1, 6):
        for u in (0.2, 0.4, 0.6, 0.8):
            tr, _ = P0.track(u, n)
            if not tr:
                continue
            pts = [(q, S, X, sl) for q, S, X, sl in tr if sl is not None and sl >= 0]
            qd = min(d["q_death"] for d in P0.DB.death(u, n))
            qb = min(p[0] for p in pts)
            qs = np.array([p[0] for p in pts])
            P = np.array([1 - 2 * q * (1 - X) / ((1 - u - q - 2 * q * S) + 2 * q) for q, S, X, sl in pts])
            o = np.argsort(qs)
            Pi = np.interp(sg, (qs[o] - qb) / (qd - qb), P[o])
            tab.append({"n": n, "u": u, "q_birth": qb, "q_death": qd, "P_at_s": [float(x) for x in Pi]})
    A = np.array([t["P_at_s"] for t in tab])
    out = {"s": sg, "rows": tab, "spread_all": [float(x) for x in A.max(0) - A.min(0)], "spread_within_u": {}, "spread_within_n": {}}
    for u in (0.2, 0.4, 0.6, 0.8):
        Bm = np.array([t["P_at_s"] for t in tab if t["u"] == u])
        out["spread_within_u"][str(u)] = [float(x) for x in Bm.max(0) - Bm.min(0)]
    for n in range(1, 6):
        Bm = np.array([t["P_at_s"] for t in tab if t["n"] == n])
        out["spread_within_n"][str(n)] = [float(x) for x in Bm.max(0) - Bm.min(0)]
    return out


def part_c():
    bd = {(round(r["u"], 2), r["n"]): r for r in json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))}
    out = []
    for u in (0.6, 0.75):
        qb = bd[(u, 1)]["birth"]
        rows = []
        for dq in (1e-6, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08):
            q = qb + dq
            th = u / (2 * q)
            sols = []
            for sol in w.solve_word(u, q, 1):
                if not sol["pattern_ok"]:
                    continue
                c = 1 - u - q - 2 * q * sol["S_n"]
                ev = m.rec_c(c, u, q, 1)[1]
                sols.append({"S": sol["S_n"], "P": m.multiplier(u, q, 1, sol)[0], "excursion_minus_theta": min(f - b for b, f in ev) - th})
            rows.append({"dq": dq, "solutions": sols})
        out.append({"u": u, "q_birth_d116b": qb, "rows": rows})
    return out


def part_d():
    n = 1
    U47, UX, UEND = 4 / 7, (9 + math.sqrt(17)) / 16, 0.8344589745557467

    def qlo(u):
        return 1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4)

    def qend(u, guess):
        A = 36 * u * u - 96 * u + 48
        Bq = -8 * u ** 3 + 64 * u * u - 32 * u
        Cq = 3 * u ** 4 - 8 * u ** 3 + 4 * u * u
        r = [x.real for x in np.roots([A, Bq, Cq]) if abs(x.imag) < 1e-12 and x.real > 1]
        return min(r, key=lambda x: abs(x - guess))

    def fold_from(u, q0, S0):
        def fs(x):
            q, S = x
            return [C.E(q, S, u, n), C.grad(lambda qq, SS: C.E(qq, SS, u, n), q, S)[1]]
        x, info, ier, _ = fsolve(fs, [q0 - 1e-4, S0 + 1e-4], full_output=True, xtol=1e-13)
        return float(x[0]) if ier == 1 else None

    rows = [json.loads(l) for l in open(os.path.join(HERE, "d120d_rows_n13_n1_2026-09-14.jsonl"), encoding="utf-8") if l.strip()]
    cnt = {"below": 0, "straddle": 0, "above": 0}
    area = {"below": 0.0, "straddle": 0.0, "above": 0.0}
    above = []
    rows_without_birth = 0
    proved_area = 0.0
    for r in rows:
        if r.get("no_track"):
            continue
        proved_area += r["proved_area"]
        if not r["failed_boxes"]:
            continue
        u = 0.5 * (r["u0"] + r["u1"])
        if u < U47:
            qp = qlo(u)
        elif u < UX:
            qb = qlo(u)
            qp = fold_from(u, qb, u / (2 * qb))
        else:
            qe = qend(u, r["q_birth_track"])
            b = min(np.roots([3 * qe, 1 - 3 * qe, u]).real)
            S = (1 - u / (2 * qe)) - b
            qp = fold_from(u, qe, S) if u < UEND else qe
        if qp is None:
            rows_without_birth += 1
            continue
        for u0, u1, q0, q1 in r["failed_boxes"]:
            a = (u1 - u0) * (q1 - q0)
            k = "below" if q1 <= qp else ("straddle" if q0 < qp else "above")
            cnt[k] += 1
            area[k] += a
            if k == "above":
                above.append((q0 - qp, u))
    d = np.array([x[0] for x in above]) if above else np.array([0.0])
    us = np.array([x[1] for x in above]) if above else np.array([0.0])
    return {"counts": cnt, "areas": area, "proved_area": proved_area, "rows_birth_not_located": rows_without_birth,
            "inside_distance_from_birth_max": float(d.max()), "inside_distance_from_birth_median": float(np.median(d)),
            "inside_by_birth_type": {"border_u_lt_4_7": int((us < U47).sum()), "fold": int(((us >= U47) & (us < UEND)).sum()),
                                     "end_window_border": int((us >= UEND).sum())}}


def main():
    t0 = time.time()
    out = {}
    for name, fn in (("c_boundary_layer", part_c), ("d_n1_unproved_boxes", part_d), ("b_collapse", part_b), ("a_stable_member_scan", part_a)):
        out[name] = fn()
        print("%s done (%.0f s): %s" % (name, time.time() - t0, json.dumps(out[name])[:400]), flush=True)
        json.dump(out, open(os.path.join(HERE, "d122_supporting_analyses_2026-09-14.json"), "w"), indent=1)
    print("-> d122_supporting_analyses_2026-09-14.json")


if __name__ == "__main__":
    main()
