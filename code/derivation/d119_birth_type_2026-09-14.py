#!/usr/bin/env python3
"""
d119 -- how each word W_n is born: at its border, or at a smooth fold just inside it.

SUPERSEDED BY d119b.  Marching in q and root-finding in S misses twin fixed points
that sit closer than the local grid, so the types printed here alternate from one u
to the next and some "exits" occur away from the border.  d119b parametrises the
branch by S, where q(S) is smooth through the fold.  Kept as the record.

d118 found that near the birth of W_n the recursion has two fixed points for some u,
with multipliers 1 + e and 1 - e, and one for others.  Two ways to be born:

  (a) border type: the fixed point reaches the short-excursion border
      (min_j (f_j - b_j) = theta) with multiplier P_B < 1 and is born there;
  (b) fold type: a smooth saddle-node (P' = 1) inside the ordering domain creates a
      stable/unstable pair; the unstable one then leaves through the border.

The switch between them is where P_B = 1 at the border (codimension two).

METHOD.  For each (n, u): march q upward from below the d116b birth until the scalar
fixed-point equation g(S) = F_n(S) - S has a pattern-verified root, bisect the
first-existence q, and count roots there on a fine local grid (twins sit ~3e-4 apart
in S, below d116's 801-point grid).  Two roots -> fold type: follow the root with the
larger multiplier until it disappears and bisect that q (the border exit), recording
its short-excursion slack and multiplier there.  One root -> border type: record its
slack and multiplier at first existence.

Usage:  python d119_birth_type_2026-09-14.py
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


def roots(u, q, n, s_lo=0.0, s_hi=1.0, npts=801):
    grid = np.linspace(s_lo, s_hi, npts)
    vals = []
    for s in grid:
        r = w.recursion(s, u, q, n)
        vals.append((r[0] - s) if r else np.nan)
    out = []
    for i in range(npts - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            try:
                out.append(brentq(lambda s: w.recursion(s, u, q, n)[0] - s, grid[i], grid[i + 1], xtol=1e-15))
            except Exception:
                pass
    return out


def verified(u, q, n, fine_window=None):
    """pattern-verified fixed points; with fine_window=(lo, hi) a dense local grid."""
    rs = roots(u, q, n) if fine_window is None else roots(u, q, n, fine_window[0], fine_window[1], 6001)
    sols = []
    for Sn in rs:
        r = w.recursion(Sn, u, q, n)
        if r is None:
            continue
        ok, _ = w.verify_pattern(u, q, 0.5 + Sn, r[1], n)
        if ok:
            c = 1 - u - q - 2 * q * Sn
            ev = m.rec_c(c, u, q, n)[1]
            slack = min(f - b for b, f in ev) - u / (2 * q)
            Pp = m.multiplier(u, q, n, {"S_n": Sn})[0]
            sols.append({"S": Sn, "P": Pp, "slack": slack})
    return sols


def all_sols(u, q, n):
    coarse = verified(u, q, n)
    if not coarse:
        # twins can hide inside one coarse cell: look near the coarse near-roots too
        grid = np.linspace(0, 1, 801)
        vals = []
        for s in grid:
            r = w.recursion(s, u, q, n)
            vals.append(abs(r[0] - s) if r else np.inf)
        i = int(np.argmin(vals))
        return verified(u, q, n, (max(0.0, grid[i] - 0.004), min(1.0, grid[i] + 0.004)))
    S0 = coarse[0]["S"]
    return verified(u, q, n, (max(0.0, S0 - 0.004), min(1.0, S0 + 0.004)))


def main():
    t0 = time.time()
    bd = json.load(open(os.path.join(HERE, "d116b_word_borders_2026-09-13.json"), encoding="utf-8"))
    out = []
    for n in (1, 2):
        pts = sorted((r["u"], r["birth"]) for r in bd if r["n"] == n and r["exists"] and r["birth"] is not None)
        us, qb = [p[0] for p in pts], [p[1] for p in pts]
        for u in np.round(np.arange(0.40, 0.905, 0.02), 2):
            guess = float(np.interp(u, us, qb))
            q = guess - 0.03
            while q < guess + 0.05 and not all_sols(u, q, n):
                q += 0.002
            if q >= guess + 0.05:
                out.append({"n": n, "u": float(u), "type": None})
                continue
            qa, qz = q - 0.002, q
            for _ in range(40):
                qm = 0.5 * (qa + qz)
                if all_sols(u, qm, n):
                    qz = qm
                else:
                    qa = qm
            q_birth = qz
            sols = all_sols(u, q_birth + 1e-9, n)
            row = {"n": n, "u": float(u), "q_birth": q_birth, "n_sols_at_birth": len(sols),
                   "sols_at_birth": sols}
            if len(sols) >= 2:
                row["type"] = "fold"
                # follow the unstable twin (largest P) until it is gone
                qe, step = q_birth + 1e-9, 1e-4
                while True:
                    ss = all_sols(u, qe + step, n)
                    if len(ss) < 2 or qe > q_birth + 0.05:
                        break
                    qe += step
                qa, qz = qe, qe + step
                for _ in range(35):
                    qm = 0.5 * (qa + qz)
                    if len(all_sols(u, qm, n)) >= 2:
                        qa = qm
                    else:
                        qz = qm
                ss = all_sols(u, qa, n)
                unst = max(ss, key=lambda x: x["P"]) if ss else None
                row.update({"q_exit": qa, "exit_width": qa - q_birth, "unstable_at_exit": unst})
            else:
                row["type"] = "border"
            out.append(row)
            desc = ("fold, width %.2e, unstable twin exits with slack %.1e, P %.3f"
                    % (row["exit_width"], row["unstable_at_exit"]["slack"], row["unstable_at_exit"]["P"])) \
                if row["type"] == "fold" and row.get("unstable_at_exit") else \
                ("border, slack %.1e, P_B %.4f" % (sols[0]["slack"], sols[0]["P"]) if sols else "?")
            print("n=%d u=%.2f q_birth=%.6f  %s" % (n, u, q_birth, desc))
    json.dump(out, open(os.path.join(HERE, "d119_birth_type_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d119_birth_type_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
