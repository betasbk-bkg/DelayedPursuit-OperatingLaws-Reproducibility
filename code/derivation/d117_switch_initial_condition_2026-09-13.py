#!/usr/bin/env python3
"""
d117 -- is the flow's switch from (+) to (+,-,+) a property of the system, or of
its initial condition?

Inside the coexistence band both orbits are stable (d115), so which one the flow
settles on can depend on where it starts.  The switch reported in App. B (d30)
was measured from one initial condition, m = w = 0.  Two tests:

  (a) PREDICTION.  At each (u, q), take the flow's FIRST forward-crossing section
      state and iterate the validated section map (v3) from it; does the word it
      converges to match the word the flow realises?
  (b) SENSITIVITY.  Repeat the flow from several initial conditions (m0, w0); inside
      the band the realised word should depend on them, outside it should not.

If (a) holds and (b) shows dependence only inside the band, the switch is set by
the initial condition and the system's intrinsic object is the band itself.

The faithful flow is d29's (sawtooth transfer b, full delay history); it is copied
here only to expose the initial condition.

Usage:  python d117_switch_initial_condition_2026-09-13.py
"""
import importlib.util
import json
import math
import os
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
v3 = _load("d113", "d113_integrator_v3_2026-09-13.py")
d107e = _load("d107e", "d107e_lower_edge_closed_form_2026-09-13.py")


def flow(u, k, m0=0.0, w0=0.0, T=600.0, dt=2.0e-3, burn=0.5):
    """d29's faithful flow with a chosen initial condition.
    Returns the z at the first forward crossing, and the realised word after burn-in."""
    mm, bb, Dl = d19.sawtooth(math.degrees(D))
    nb = len(mm); step = (mm[-1] - mm[0]) / (nb - 1); mlo = mm[0]; bl = bb.tolist()
    th = u / k; nd = max(0, int(round(th / dt))); n = int(T / dt)
    W = [w0] * (n + 1); Z = [0.0] * (n + 1)
    m = m0; w = w0; c0 = (0.5 - u) / k; half = Dl / 2
    prev = None; first_z = None; nf = nbk = 0
    for i in range(n):
        mw = ((m + half) % Dl) - half
        x = (mw - mlo) / step; j = int(x); j = 0 if j < 0 else (nb - 2 if j > nb - 2 else j)
        f = x - j
        bv = bl[j] * (1.0 - f) + bl[j + 1] * f
        z = c0 - (W[i - nd] if i >= nd else w0) + bv
        zd = Z[i - nd] if i >= nd else 0.0
        w += dt * k * z; m += dt * (1.0 - k * zd)
        W[i + 1] = w; Z[i + 1] = z
        if prev is not None:
            if mw - prev < -Dl / 2:                      # forward crossing
                if first_z is None and i * dt > th + 5 * dt:
                    first_z = z
                if i > burn * n:
                    nf += 1
            elif mw - prev > Dl / 2 and i > burn * n:
                nbk += 1
        prev = mw
    word = "+" if nbk == 0 else ("-++" if nf and abs(nbk / max(nf - nbk, 1) - 1) < 0.15 else "other")
    return first_z, word


def map_attractor(z0, u, q, iters=400):
    k = 2 * q / D
    z = z0
    for _ in range(iters):
        o = v3.orbit_v3(z, u, k, D, u / k)
        if o is None:
            return None
        zn = o["z_next"]
        if abs(zn - z) < 1e-10:
            return o["word"]
        z = zn
    return "no convergence"


def main():
    t0 = time.time()
    meas = {round(m["u"], 2): m for m in json.load(open(os.path.join(HERE, "d30_results_2026-08-29.json"), encoding="utf-8"))["B_measured"]}
    out = {"a": [], "b": []}
    print("(a) does the first section state predict the realised word?")
    for u in (0.3, 0.5, 0.71, 0.9):
        qbc = (1 + math.sqrt(u)) ** 2
        b2 = d107e.q_lo2(u)
        qlo = max(d107e.q_lo1(u), b2[0] if b2 else -1)
        hits = 0
        tot = 0
        for q in np.arange(qlo - 0.15, qbc + 0.15, 0.05):
            k = 2 * q / D
            fz, word = flow(u, k)
            pred = map_attractor(fz, u, q) if fz is not None else None
            match = (pred == word)
            tot += 1
            hits += match
            out["a"].append({"u": u, "q": float(q), "in_band": bool(qlo < q < qbc), "flow_word": word,
                             "first_z": fz, "map_word": pred, "match": bool(match)})
        rows = [r for r in out["a"] if r["u"] == u]
        sw = next((r["q"] for r in rows if r["flow_word"] != "+"), None)
        print("  u=%.2f band [%.3f, %.3f]: prediction matches %d/%d; flow first leaves (+) at q=%s (d30 switch %.3f)"
              % (u, qlo, qbc, hits, tot, ("%.3f" % sw) if sw else None, meas[round(u, 2)]["mid"]))
        for r in rows:
            if not r["match"]:
                print("     mismatch q=%.3f flow %s map %s (first z %s)" % (r["q"], r["flow_word"], r["map_word"], r["first_z"]))
    print("\n(b) realised word across initial conditions, inside vs outside the band  (%.0f s)" % (time.time() - t0))
    ics = [(m0, w0) for m0 in (0.0, 0.2, -0.2, 0.35) for w0 in (0.0, 0.05, -0.05)]
    for u in (0.3, 0.71):
        qbc = (1 + math.sqrt(u)) ** 2
        qlo = d107e.q_lo1(u)
        for q in (qlo - 0.1, 0.5 * (qlo + qbc), qbc + 0.1):
            k = 2 * q / D
            words = Counter(flow(u, k, m0, w0, T=400.0)[1] for m0, w0 in ics)
            tag = "inside band" if qlo < q < qbc else "outside band"
            out["b"].append({"u": u, "q": float(q), "where": tag, "words": dict(words)})
            print("  u=%.2f q=%.3f (%s): %s" % (u, q, tag, dict(words)))
    json.dump(out, open(os.path.join(HERE, "d117_switch_initial_condition_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d117_switch_initial_condition_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
