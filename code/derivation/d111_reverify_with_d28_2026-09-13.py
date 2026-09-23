#!/usr/bin/env python3
"""
d111 -- recompute what d107h and d108 computed with d27's defective integrator,
using d28.orbit (explicit cell index; the integrator behind d29-d31).

  (1) (+,-,+) orbit across the coexistence band: at each (u, q) the periodic
      solution of d26's closure, its piece key (word, min_gap > theta), its
      multiplier from differences that stay on the same piece, stability, and
      agreement with the main-piece closed form (d107g) where that form is valid.
  (2) attractor census: iterate d28's map from 60 initial states per (u, q),
      classify period (<= 8), aperiodic, or failed (orbit None).

Usage:  python d111_reverify_with_d28_2026-09-13.py
"""
import importlib.util
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
g = _load("d107g", "d107g_pmp_multiplier_closed_2026-09-13.py")


def q_bc(u):
    return (1 + math.sqrt(u)) ** 2


def q_lo1(u):
    r = 1 - (2 * u - 1) ** 2 + u * u / 4
    return 1 + math.sqrt(r) if r >= 0 else float("nan")


def key(o, th):
    return None if o is None else (o["word"], bool(o["min_gap"] > th))


def mult_same_piece(z0, u, k, th):
    base = d28.orbit(z0, u, k, D, th)
    if base is None:
        return None, None, None
    kb = key(base, th)
    est = []
    for hh in (1e-6, 3e-7, 1e-7, 3e-8):
        op, om = d28.orbit(z0 + hh, u, k, D, th), d28.orbit(z0 - hh, u, k, D, th)
        sp, sm = key(op, th) == kb, key(om, th) == kb
        if sp and sm:
            est.append((op["z_next"] - om["z_next"]) / (2 * hh))
        elif sp:
            est.append((op["z_next"] - base["z_next"]) / hh)
        elif sm:
            est.append((base["z_next"] - om["z_next"]) / hh)
    if not est:
        return kb, float("nan"), float("nan")
    return kb, float(np.median(est)), float(np.ptp(est))


def part1():
    rows = []
    print("(1) (+,-,+) across the band with d28.orbit")
    for u in (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        qlo = q_lo1(u) if u < 0.85 else 1.83
        p = None
        stats = Counter()
        agree = []
        for q in np.arange(qlo + 0.003, q_bc(u), 0.01):
            k = 2 * q / D
            th = u / k
            sol = None
            if p is not None:
                s, info, ier, msg = fsolve(d26.residuals, p, args=(u, k, D, th), full_output=True, xtol=1e-12)
                if ier == 1 and max(abs(x) for x in d26.residuals(s, u, k, D, th)) < 1e-8 and 0 < s[1] < s[2] < D:
                    sol = s
            if sol is None:
                s2 = d26.solve_word(u, k, D)
                if s2 and s2["resid"] < 1e-8:
                    sol = np.array([s2["z0"], s2["t1"], s2["t2"]])
            if sol is None:
                stats["no orbit"] += 1
                continue
            p = list(sol)
            o = d28.orbit(sol[0], u, k, D, th)
            if o is None or abs(o["z_next"] - sol[0]) > 1e-6 or o["word"] not in ("-++", "+-+"):
                stats["closure orbit not a d28 fixed point"] += 1
                continue
            kb, m, spread = mult_same_piece(sol[0], u, k, th)
            if not np.isfinite(m) or spread > 1e-5:
                stats["untrusted multiplier"] += 1
                continue
            stats["trusted"] += 1
            stats["stable" if abs(m) < 1 else "unstable"] += 1
            stats["piece gap>th" if kb[1] else "piece gap<=th"] += 1
            t1, t2 = sol[1] / D, sol[2] / D
            if g.valid(u, q, t1, t2):
                c = g.mult_closed(u, q, t1, t2)
                agree.append(abs(c - m))
            rows.append({"u": u, "q": float(q), "mult": m, "spread": spread, "piece_gap_gt_th": kb[1],
                         "closed": (g.mult_closed(u, q, t1, t2) if g.valid(u, q, t1, t2) else None)})
        print("  u=%.2f  %s  closed-form checked %d, max |diff| %s"
              % (u, dict(stats), len(agree), ("%.1e" % max(agree)) if agree else None))
    return rows


def classify(z0, u, k, th):
    z = z0
    for _ in range(300):
        o = d28.orbit(z, u, k, D, th)
        if o is None:
            return "failed", None
        z = o["z_next"]
    rec, words = [], []
    for _ in range(64):
        rec.append(z)
        o = d28.orbit(z, u, k, D, th)
        if o is None:
            return "failed", None
        words.append(o["word"])
        z = o["z_next"]
    rec = np.array(rec)
    for per in range(1, 9):
        if np.all(np.abs(rec[per:] - rec[:-per]) < 1e-9):
            return "p%d" % per, "".join(words[:per])
    return "aperiodic", None


def part2():
    rows = []
    print("\n(2) attractor census with d28.orbit")
    for u in (0.3, 0.5, 0.71, 0.9):
        for q in np.round(np.arange(1.6, 4.01, 0.2), 2):
            k = 2 * q / D
            th = u / k
            c, wds = Counter(), Counter()
            for z0 in np.linspace(-0.2 * D, 1.2 * D, 60):
                kind, wd = classify(float(z0), u, k, th)
                c[kind] += 1
                if wd:
                    wds["%s:%s" % (kind, wd)] += 1
            rows.append({"u": u, "q": float(q), "census": dict(c), "words": dict(wds)})
            print("  u=%.2f q=%.2f  %s  %s" % (u, q, dict(c), dict(wds.most_common(3))))
    return rows


def main():
    out = {"pmp_band": part1(), "census": part2()}
    json.dump(out, open(os.path.join(HERE, "d111_reverify_with_d28_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n-> d111_reverify_with_d28_2026-09-13.json")


if __name__ == "__main__":
    main()
