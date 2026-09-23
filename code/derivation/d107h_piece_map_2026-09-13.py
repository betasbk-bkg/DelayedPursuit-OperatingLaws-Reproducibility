#!/usr/bin/env python3
"""
d107h -- which smooth piece of the return map carries the (+,-,+) orbit, across
the whole coexistence band, and is the orbit stable on each piece?

The map P(z) is piecewise smooth: its formula changes whenever the ORDER of the
events {crossings 0, t1, t2, period end} and the delay-window breakpoints
{c + theta} changes.  d107g's closed form holds on one ordering (the "main
piece"); it matched the map to 1.5e-10 there and disagreed at 7 points that sit
on other orderings.  d107f's noisy multipliers came from finite differences taken
across piece boundaries.

Here the orbit integrator (d27.orbit, copied) also returns its event times, so:
  * each periodic orbit gets an ORDERING SIGNATURE;
  * the multiplier is a difference over perturbations that keep the identical
    signature (same smooth piece), so it is the derivative of one smooth formula;
  * section validity is recorded: no crossing of the previous period may still be
    inside the delay window at the section (else the map is not 1-D).

Usage:  python d107h_piece_map.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
from scipy.optimize import fsolve

DERIV = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(DERIV, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
D = math.radians(45.0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d107h_piece_map_2026-09-13.json")


def orbit_events(z0, u, k, delta, th, max_ev=12):
    """d27.orbit, returning the event list as well (logic unchanged)."""
    m = -delta / 2.0
    z = z0
    T = 0.0
    events = [(0.0, +1)]
    guard = 0
    while guard < max_ev:
        guard += 1
        cand = [te + th for te, _ in events if te + th > T + 1e-14]
        T_bp = min(cand) if cand else float("inf")
        J = sum(s for te, s in events if T - th < te <= T)
        A = (1.0 - u) - k * (z + T) + k * delta * J
        lo, hi = -delta / 2.0, delta / 2.0
        while m > hi + 1e-12:
            lo += delta; hi += delta
        while m < lo - 1e-12:
            lo -= delta; hi -= delta
        t_hit, which = None, 0
        for bnd, w in ((hi, +1), (lo, -1)):
            a = k / 2.0
            b = A
            c = m - bnd - A * T - k * T * T / 2.0
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            r = math.sqrt(disc)
            for root in ((-b - r) / (2 * a), (-b + r) / (2 * a)):
                if root > T + 1e-12 and (t_hit is None or root < t_hit):
                    t_hit, which = root, w
        if t_hit is None or t_hit > T_bp:
            if not np.isfinite(T_bp):
                return None
            Tn = T_bp
            m = m + A * (Tn - T) + k * (Tn * Tn - T * T) / 2.0
            z -= (Tn - T)
            T = Tn
            continue
        Tn = t_hit
        m = m + A * (Tn - T) + k * (Tn * Tn - T * T) / 2.0
        z -= (Tn - T)
        T = Tn
        z += which * delta
        events.append((T, which))
        if which > 0 and m > -delta / 2.0 + delta - 1e-9:
            return {"z_next": z, "period": T, "events": events}
    return None


def signature(ev, th):
    """order of crossings and breakpoints inside one period, as a string."""
    period = ev[-1][0]
    marks = [(t, ("+" if s > 0 else "-")) for t, s in ev[:-1]]
    marks += [(t + th, "b%d" % i) for i, (t, s) in enumerate(ev[:-1]) if t + th < period]
    marks.sort()
    return "".join(x[1] if len(x[1]) == 1 else "|" for x in marks) + "/" + str(len(ev))


def piece_multiplier(z0, u, k, th):
    base = orbit_events(z0, u, k, D, th)
    if not base:
        return None
    sig = signature(base["events"], th)
    est = []
    for h in (1e-6, 3e-7, 1e-7, 3e-8, 1e-8):
        op = orbit_events(z0 + h, u, k, D, th)
        om = orbit_events(z0 - h, u, k, D, th)
        sp = op and signature(op["events"], th) == sig
        sm = om and signature(om["events"], th) == sig
        if sp and sm:
            est.append((op["z_next"] - om["z_next"]) / (2 * h))
        elif sp:
            est.append((op["z_next"] - base["z_next"]) / h)
        elif sm:
            est.append((base["z_next"] - om["z_next"]) / h)
    if not est:
        return {"sig": sig, "mult": float("nan"), "spread": float("nan")}
    est = np.array(est)
    prev_in_window = any(t + th > base["period"] for t, s in base["events"][:-1])
    return {"sig": sig, "mult": float(np.median(est)), "spread": float(np.ptp(est)),
            "section_valid": not prev_in_window, "events": base["events"]}


def main():
    t0 = time.time()
    out = {}
    for u in np.round(np.arange(0.10, 0.91, 0.10), 2):
        qbc = (1 + math.sqrt(u)) ** 2
        r = 1 - (2 * u - 1) ** 2 + u * u / 4
        qlo = 1 + math.sqrt(r)
        rows = []
        p = None
        for q in np.arange(max(qlo, 1.0) + 0.003, qbc + 0.5, 0.01):
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
                rows.append({"q": float(q), "none": True})
                continue
            p = list(sol)
            pm = piece_multiplier(sol[0], u, k, th)
            if pm is None:
                rows.append({"q": float(q), "none": True})
                continue
            rows.append({"q": float(q), "sig": pm["sig"], "mult": pm["mult"], "spread": pm["spread"],
                         "section_valid": pm.get("section_valid"), "in_band": bool(q <= qbc)})
        out[str(u)] = rows
        # summarise pieces inside the band
        band = [x for x in rows if x.get("in_band") and "sig" in x]
        pieces = []
        for x in band:
            if not pieces or pieces[-1]["sig"] != x["sig"]:
                pieces.append({"sig": x["sig"], "q0": x["q"], "q1": x["q"], "m": [x["mult"]], "valid": [x["section_valid"]]})
            else:
                pieces[-1]["q1"] = x["q"]; pieces[-1]["m"].append(x["mult"]); pieces[-1]["valid"].append(x["section_valid"])
        print("\nu=%.2f  band [%.3f, %.3f]  -- pieces carrying the (+,-,+) orbit:" % (u, qlo, qbc))
        for pc in pieces:
            m = np.array([v for v in pc["m"] if np.isfinite(v)])
            print("   %-22s q %.3f..%.3f  P' %s  stable %s  section-valid %s"
                  % (pc["sig"], pc["q0"], pc["q1"],
                     ("%+.3f..%+.3f" % (m.max(), m.min())) if len(m) else "nan",
                     bool(len(m) and np.all(np.abs(m) < 1)), all(pc["valid"])))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("\n%.0f s -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
