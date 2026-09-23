#!/usr/bin/env python3
"""
d116d -- the death of every word W_n by grazing, and its general condition.

At the death of W_n, a new excursion grazes the start boundary on (f_n + theta, 1):
min M = u + 2 q S_n - a^2/(4q) = 0.  With a = 1 - u - q - 2 q S_n this gives, for
every n,
      a = 2 sqrt(q) (1 - sqrt(q)),      S_n = ((sqrt(q) - 1)^2 - u) / (2 q).
Here each death is located as the zero of the grazing margin along the physical
solution branch (tracked continuously in q from where the word first exists,
without the sampled pattern test, which misses a shallow graze), and a and S_n
there are compared with the general condition.  Also prints the W_1 deaths against
the quartic (q + 2 sqrt(q)(1 - sqrt(u)) - (1 - u))^2 = 4 q (q - 2 sqrt(q) sqrt(u))
(d116i recomputes and saves that check; the d116c JSON is the earlier comparison
against the sampled map, about 1e-4, and is not written here).

(The check was first run inline on 2026-09-13; this file records it.)

Usage:  python d116d_general_death_condition_2026-09-13.py [output_dir]
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d116", os.path.join(HERE, "d116_word_sequence_closure_2026-09-13.py"))
w = importlib.util.module_from_spec(_s)
_s.loader.exec_module(w)


def tracked(u, q, n, S_prev):
    sols = w.solve_word(u, q, n)
    return min(sols, key=lambda x: abs(x["S_n"] - S_prev)) if sols else None


def margin(u, q, n, x):
    Sn = x["S_n"]
    a = 1 - u - q - 2 * q * Sn
    th = u / (2 * q)
    lo = [t for t, sg in x["events"] if sg > 0][-1] + th
    return u + 2 * q * Sn - a * a / (4 * q), (lo < -a / (2 * q) < 1)


def quartic(q, u):
    s, v = math.sqrt(q), math.sqrt(u)
    return (q + 2 * s * (1 - v) - (1 - u)) ** 2 - 4 * q * (q - 2 * s * v)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else HERE
    rows = []
    for u in (0.2, 0.3, 0.5, 0.71):
        for n in range(1, 6):
            qs = np.arange(1.02, 9.0, 0.01)
            start = next((q for q in qs if w.exists(u, float(q), n)), None)
            if start is None:
                continue
            x = w.exists(u, float(start), n)
            S_prev, q_prev, g_prev = x["S_n"], float(start), margin(u, float(start), n, x)[0]
            root = Sn = None
            for q in np.arange(start + 0.01, 9.0, 0.01):
                y = tracked(u, float(q), n, S_prev)
                if y is None:
                    break
                g, inside = margin(u, float(q), n, y)
                if g_prev > 0 >= g and inside:
                    a_, b_, Sa = q_prev, float(q), S_prev
                    for _ in range(60):
                        mid = 0.5 * (a_ + b_)
                        z = tracked(u, mid, n, Sa)
                        if margin(u, mid, n, z)[0] > 0:
                            a_, Sa = mid, z["S_n"]
                        else:
                            b_ = mid
                    root = 0.5 * (a_ + b_)
                    Sn = tracked(u, root, n, Sa)["S_n"]
                    break
                S_prev, q_prev, g_prev = y["S_n"], float(q), g
            if root is None:
                rows.append({"u": u, "n": n, "death": None})
                continue
            sq = math.sqrt(root)
            a = 1 - u - root - 2 * root * Sn
            rows.append({"u": u, "n": n, "death": root, "S_n": Sn,
                         "err_a": abs(a - 2 * sq * (1 - sq)),
                         "err_S": abs(Sn - ((sq - 1) ** 2 - u) / (2 * root))})
            print("u=%.2f W%d death q=%.10f  err_a %.1e  err_S %.1e" % (u, n, root, rows[-1]["err_a"], rows[-1]["err_S"]))
    json.dump(rows, open(os.path.join(out, "d116d_general_death_condition_2026-09-13.json"), "w"), indent=1)
    q1 = [abs(brentq(quartic, r["death"] - 0.05, r["death"] + 0.05, args=(r["u"],), xtol=1e-14) - r["death"])
          for r in rows if r.get("death") and r["n"] == 1]
    print("W1 quartic vs grazing zero: max |diff| %.1e over %d u" % (max(q1), len(q1)))


if __name__ == "__main__":
    main()
