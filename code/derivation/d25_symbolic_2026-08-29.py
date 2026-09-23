# -*- coding: utf-8 -*-
"""
D25: the symbolic dynamics beyond k*Delta/2 = 2.55  (open problem (iii) of v5).

Inside the window the orbit crosses one cell boundary forward per period, which
is what Eq. (12) assumes.  Beyond it the fold would take the phase below
-Delta/2, so backward crossings appear and the assumed symbol sequence fails.
This instruments the sawtooth flow to read the actual crossing sequence: is it a
period-1 word like (+,-,+) that the same algebra could close, or does it
mode-lock / wander?

Records, per run: the crossing word, its period in T units (the theory says
T_p = Delta exactly whenever the net winding is +1), and how many distinct words
appear once transients are gone.
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0)


def crossings(u, k, delta, T=600.0, dt=1.0e-3, burn=0.4):
    """event list for the pure sawtooth hybrid system: (time, +1/-1)."""
    th = u/k
    nd = max(0, int(round(th/dt)))
    n = int(T/dt)
    Z = np.zeros(n+1)
    z = delta/2.0
    m = -delta/2.0
    ev = []
    for i in range(n):
        zd = Z[i-nd] if i >= nd else z + th
        m += dt*(1.0 - k*zd)
        z += -dt
        if m > delta/2:
            m -= delta; z += delta; ev.append((i*dt, +1))
        elif m < -delta/2:
            m += delta; z -= delta; ev.append((i*dt, -1))
        Z[i+1] = z
    return [e for e in ev if e[0] > burn*T]


def word_and_period(ev, delta):
    """collapse the event stream into repeating words with net winding +1."""
    if len(ev) < 6:
        return "", float("nan"), 0
    syms = "".join("+" if s > 0 else "-" for _, s in ev)
    # find the shortest repeating unit with net winding +1
    for L in range(1, 13):
        cand = syms[:L]
        if cand.count("+") - cand.count("-") != 1:
            continue
        reps = len(syms)//L
        if reps >= 4 and all(syms[j*L:(j+1)*L] == cand for j in range(min(reps, 8))):
            times = [ev[j*L][0] for j in range(min(reps, 8))]
            per = float(np.mean(np.diff(times))) if len(times) > 1 else float("nan")
            return cand, per, L
    return syms[:16], float("nan"), 0


if __name__ == "__main__":
    out = []
    for delta_deg in (45.0,):
        D = math.radians(delta_deg)
        print(f"Delta = {delta_deg} deg,  T_p predicted = Delta = {D:.4f}")
        print(f"{'k':>5} {'kD/2':>6} {'u':>5} {'word':>10} {'period':>8} "
              f"{'/Delta':>7} {'n_events':>9}")
        for k in (5.0, 6.0, 6.5, 7.0, 8.0):
            for u in (0.30, 0.50, 0.71, 0.90):
                ev = crossings(u, k, D)
                w, per, L = word_and_period(ev, D)
                out.append(dict(delta=delta_deg, k=k, kD2=k*D/2, u=u, word=w,
                                period=per, ratio=per/D if per == per else None,
                                n_events=len(ev)))
                print(f"{k:5.1f} {k*D/2:6.2f} {u:5.2f} {w:>10} "
                      f"{per if per == per else float('nan'):8.4f} "
                      f"{(per/D) if per == per else float('nan'):7.3f} {len(ev):9d}")
    json.dump(out, open(os.path.join(HERE, "d25_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d25_results_2026-08-29.json")
