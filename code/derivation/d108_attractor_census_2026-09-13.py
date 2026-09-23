#!/usr/bin/env python3
"""
d108 -- which attractors does the 1-D return map actually have?

Sec. X (ii) of the manuscript says that above kD/2 ~ 2.5 the return map acquires
attractors that are not fixed points.  d107 found stable fixed points everywhere
up to q = 3.2, so the claim needs testing directly: iterate the exact map from
many initial states and classify what each settles on.

For each (u, q): 60 initial z spread over one cell, 300 transient iterations,
then 64 recorded iterates.  Each trajectory is classified by its eventual period
p (smallest p <= 8 with |z_{n+p} - z_n| < 1e-9 over the record), or 'aperiodic'
if none, or 'escaped' if the map left its domain (orbit integration failed, or
a crossing of the previous period is still inside the delay window at the
section, where the 1-D reduction does not hold).

Usage:  python d108_attractor_census.py
"""
import importlib.util
import json
import math
import os
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d107h", os.path.join(HERE, "d107h_piece_map_2026-09-13.py"))
d107h = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d107h)
D = math.radians(45.0)
OUT = os.path.join(HERE, "d108_attractor_census_2026-09-13.json")


def step(z, u, k, th):
    o = d107h.orbit_events(z, u, k, D, th)
    if o is None:
        return None, None
    if any(t + th > o["period"] for t, s in o["events"][:-1]):
        return None, None
    word = "".join("+" if s > 0 else "-" for _, s in o["events"][1:])
    return o["z_next"], word


def classify(z0, u, k, th):
    z = z0
    for _ in range(300):
        z, _w = step(z, u, k, th)
        if z is None:
            return "escaped", None
    rec, words = [], []
    for _ in range(64):
        rec.append(z)
        z, w = step(z, u, k, th)
        if z is None:
            return "escaped", None
        words.append(w)
    rec = np.array(rec)
    for p in range(1, 9):
        if np.all(np.abs(rec[p:] - rec[:-p]) < 1e-9):
            return "p%d" % p, "".join(words[:p])
    return "aperiodic", None


def main():
    t0 = time.time()
    out = []
    print("   u     q   census (60 initial states)")
    for u in (0.3, 0.5, 0.71, 0.9):
        for q in np.round(np.arange(1.6, 4.01, 0.2), 2):
            k = 2 * q / D
            th = u / k
            c = Counter()
            words = Counter()
            for z0 in np.linspace(-0.2 * D, 1.2 * D, 60):
                kind, w = classify(float(z0), u, k, th)
                c[kind] += 1
                if w:
                    words["%s:%s" % (kind, w)] += 1
            out.append({"u": u, "q": float(q), "census": dict(c), "words": dict(words)})
            print("  %.2f  %.2f   %s   %s" % (u, q, dict(c), dict(words.most_common(4))))
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
