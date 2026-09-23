#!/usr/bin/env python3
"""
d116b -- the borders of the word sequence W_n over u, and which condition sets each.

Uses d116's corrected explicit recursion.  For u = 0.10..0.95 and n = 0..5:
  * the existence interval [q_birth, q_death] of W_n (bisected, 1e-10 in q);
  * at each end, which condition becomes an equality:
      grazing        min of M on (f_n + theta, 1) reaches 0  (a new excursion appears)
      short_exc      some excursion f_i - b_i reaches theta
      short_gap      some gap b_(i+1) - f_i (or b_1 - 0) reaches theta
      end_window     f_n + theta reaches 1
  * candidate closed forms tested against the numbers: q_bc = (1+sqrt u)^2 for the
    death of W_0, q_lo1 for the birth of W_1, and, for the rest, the spacing of
    successive births and deaths in n.

Usage:  python d116b_word_borders_2026-09-13.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


w = _load("d116", "d116_word_sequence_closure_2026-09-13.py")


def margins(u, q, n, sol):
    th = u / (2 * q)
    ev = sol["events"]
    b = [t for t, s in ev if s < 0]
    f = [t for t, s in ev if s > 0]
    Sn = sol["S_n"]
    a = 1 - u - q - 2 * q * Sn
    out = {}
    tau_star = -a / (2 * q)
    lo = (f[-1] + th) if n else th
    if lo < tau_star < 1:
        out["grazing"] = u + 2 * q * Sn - a * a / (4 * q)
    else:
        out["grazing"] = min(w.M_exact(lo, u, q, sol["zeta"], ev), w.M_exact(1.0, u, q, sol["zeta"], ev))
    if n:
        out["short_exc"] = min(fi - bi for bi, fi in zip(b, f)) - th
        gaps = [b[0]] + [b[i + 1] - f[i] for i in range(n - 1)]
        out["short_gap"] = min(gaps) - th
        out["end_window"] = 1 - (f[-1] + th)
    return out


def edge_condition(u, n, q_in, q_out):
    """bisect between an existing q_in and a non-existing q_out; report the smallest margin at the edge."""
    a, b = q_in, q_out
    for _ in range(45):
        m = 0.5 * (a + b)
        if w.exists(u, m, n):
            a = m
        else:
            b = m
    sol = w.exists(u, a, n)
    mg = margins(u, a, n, sol) if sol else {}
    active = min(mg, key=lambda k: abs(mg[k])) if mg else None
    return a, active, mg


def main():
    t0 = time.time()
    rows = []
    for u in np.round(np.arange(0.10, 0.96, 0.05), 3):
        u = float(u)
        for n in range(6):
            qs = np.arange(1.02, 8.0, 0.02)
            alive = [w.exists(u, float(q), n) is not None for q in qs]
            if not any(alive):
                rows.append({"u": u, "n": n, "exists": False})
                continue
            i0 = alive.index(True)
            i1 = len(alive) - 1 - alive[::-1].index(True)
            gap = not all(alive[i0:i1 + 1])
            birth = death = None
            if i0 > 0:
                birth = edge_condition(u, n, float(qs[i0]), float(qs[i0 - 1]))
            if i1 < len(qs) - 1:
                death = edge_condition(u, n, float(qs[i1]), float(qs[i1 + 1]))
            rows.append({"u": u, "n": n, "exists": True, "interrupted": gap,
                         "birth": birth[0] if birth else None, "birth_by": birth[1] if birth else None,
                         "death": death[0] if death else None, "death_by": death[1] if death else None})
        line = "  u=%.2f " % u + " ".join(
            ("W%d[%s, %s]" % (r["n"],
                              ("%.4f %s" % (r["birth"], r["birth_by"])) if r.get("birth") is not None else "-",
                              ("%.4f %s" % (r["death"], r["death_by"])) if r.get("death") is not None else "-"))
            for r in rows if r["u"] == u and r["exists"])
        print(line)
    # candidate closed forms
    print("\nchecks")
    d0 = [(r["u"], r["death"]) for r in rows if r["n"] == 0 and r.get("death")]
    print("  W0 death vs (1+sqrt u)^2: max |diff| %.2e" % max(abs(d - (1 + math.sqrt(u)) ** 2) for u, d in d0))
    b1 = [(r["u"], r["birth"], r["birth_by"]) for r in rows if r["n"] == 1 and r.get("birth")]
    q1 = [(u, b) for u, b, by in b1 if by == "short_exc"]
    if q1:
        print("  W1 birth by short excursion vs q_lo1: max |diff| %.2e over %d u"
              % (max(abs(b - (1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4))) for u, b in q1), len(q1)))
    for kind in ("birth", "death"):
        by = {}
        for r in rows:
            if r.get(kind):
                by.setdefault(r[kind + "_by"], 0)
                by[r[kind + "_by"]] += 1
        print("  %s set by: %s" % (kind, by))
    print("  spacing of successive deaths in n (u=0.30, 0.50, 0.71):")
    for u in (0.3, 0.5, 0.71):
        ds = [r["death"] for r in rows if abs(r["u"] - u) < 1e-9 and r.get("death")]
        bs = [r["birth"] for r in rows if abs(r["u"] - u) < 1e-9 and r.get("birth")]
        print("    u=%.2f deaths %s  diffs %s | births %s diffs %s"
              % (u, ["%.4f" % x for x in ds], ["%.4f" % y for y in np.diff(ds)],
                 ["%.4f" % x for x in bs], ["%.4f" % y for y in np.diff(bs)]))
    json.dump(rows, open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d116b_word_borders_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
