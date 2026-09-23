#!/usr/bin/env python3
"""
d109 -- what kind of border collision ends the (+) orbit and starts the (+,-,+) one?

A border collision of a piecewise-smooth map is classified by the map's local
form at the switching point z_s:
  * continuous with a kink (P ~ a (z - z_s) on one side, b (z - z_s) on the other):
    the classification is by the two one-sided slopes (persistence, nonsmooth
    fold, ... -- Nusse-Yorke / Feigin);
  * continuous with a square-root term (P - P(z_s) ~ c |z - z_s|^(1/2) on one side):
    a GRAZING bifurcation (Nordmark map), which typically opens period-adding
    and chaotic windows beyond the border;
  * discontinuous (a jump at z_s).

Border 1, end of (+) at q_bc(u): the orbit's fold touches the cell boundary, so a
square-root law is the a-priori candidate.  Border 2, birth of (+,-,+) at q_lo(u):
a crossing coincides with a delay-window breakpoint (t2 - t1 = theta), so a kink
is the a-priori candidate.  Neither is assumed: both are measured.

For each border and u, just on the side where the orbit exists:
  1. find z_s, the section state at which the map's formula changes
     (border 1: the word gains a '-'; border 2: the sign of t2 - t1 - theta);
  2. sample P(z) at z_s +- s, s = 1e-9 .. 1e-3;
  3. report the jump |P(z_s+) - P(z_s-)|, and on each side the exponent of
     |P(z) - P(z_s)| vs s from a log-log fit (1 = linear, 0.5 = square root),
     and the one-sided slope where the exponent is 1.
It also reports how far the fixed point is from z_s as q approaches the border.

Usage:  python d109_border_normal_form_2026-09-13.py
"""
import importlib.util
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
h = _load("d107h", "d107h_piece_map_2026-09-13.py")


def q_bc(u):
    return (1 + math.sqrt(u)) ** 2


def q_lo1(u):
    return 1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4)


def orbit(z, u, q):
    k = 2 * q / D
    return h.orbit_events(z, u, k, D, u / k)


def word(o):
    return "".join("+" if s > 0 else "-" for _, s in o["events"][1:]) if o else None


def bisect(pred, a, b, it=60):
    pa = pred(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        if pred(m) == pa:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def local_form(Pf, zs):
    out = {}
    P0m = Pf(zs - 1e-12)
    P0p = Pf(zs + 1e-12)
    out["jump"] = None if (P0m is None or P0p is None) else abs(P0p - P0m)
    for side, sg, P0 in (("left", -1, P0m), ("right", +1, P0p)):
        ss = np.logspace(-9, -3, 25)
        vals = [Pf(zs + sg * s) for s in ss]
        ok = [(s, v) for s, v in zip(ss, vals) if v is not None and P0 is not None and abs(v - P0) > 0]
        if len(ok) < 6:
            out[side] = None
            continue
        x = np.log([s for s, _ in ok])
        y = np.log([abs(v - P0) for s, v in ok])
        expo = float(np.polyfit(x, y, 1)[0])
        slope = float((ok[len(ok) // 2][1] - P0) / (sg * ok[len(ok) // 2][0]))
        out[side] = {"exponent": expo, "slope_mid": slope, "n": len(ok)}
    return out


def border1(u, dq):
    """end of (+): q just below q_bc; z_s = first z (above z*) whose orbit gains a '-'."""
    q = q_bc(u) - dq
    zstar = D / 2
    pred = lambda z: ("-" in (word(orbit(z, u, q)) or "-"))
    hi = zstar
    while not pred(hi) and hi < zstar + D:
        hi += D / 200
    lo = zstar - D
    while pred(lo) and lo < zstar:
        lo += D / 200
    if pred(zstar):
        return {"q": q, "error": "fixed point itself is not (+)"}
    # search both directions from z* for the nearest switch
    cands = []
    z = zstar
    step = D / 2000
    for direction in (+1, -1):
        z = zstar
        for _ in range(4000):
            z2 = z + direction * step
            if pred(z2) != pred(z):
                cands.append(bisect(pred, z, z2))
                break
            z = z2
    if not cands:
        return {"q": q, "error": "no switch found near z*"}
    zs = min(cands, key=lambda c: abs(c - zstar))
    Pf = lambda z: (lambda o: o["z_next"] if o else None)(orbit(z, u, q))
    lf = local_form(Pf, zs)
    return {"q": q, "dq": dq, "z_star": zstar, "z_s": zs, "dist": abs(zs - zstar), **lf}


def border2(u, dq):
    """birth of (+,-,+): q just above q_lo; z_s where t2 - t1 - theta changes sign."""
    q = q_lo1(u) + dq
    k = 2 * q / D
    s = d26.solve_word(u, k, D)
    if not s:
        return {"q": q, "error": "no (+,-,+) orbit"}
    zstar = s["z0"]
    th = u / k

    def sep(z):
        o = orbit(z, u, q)
        if not o or word(o) != "-++":
            return None
        ev = o["events"]
        return (ev[2][0] - ev[1][0]) - th

    pred = lambda z: (sep(z) is not None and sep(z) > 0)
    cands = []
    step = D / 4000
    for direction in (+1, -1):
        z = zstar
        for _ in range(8000):
            z2 = z + direction * step
            if pred(z2) != pred(z):
                cands.append(bisect(pred, z, z2))
                break
            z = z2
    if not cands:
        return {"q": q, "error": "no ordering switch found near z*"}
    zs = min(cands, key=lambda c: abs(c - zstar))
    Pf = lambda z: (lambda o: o["z_next"] if o else None)(orbit(z, u, q))
    lf = local_form(Pf, zs)
    return {"q": q, "dq": dq, "z_star": zstar, "z_s": zs, "dist": abs(zs - zstar), **lf}


def main():
    out = {"border1_plus_end": [], "border2_pmp_birth": []}
    for u in (0.3, 0.5, 0.71):
        print("\nu = %.2f" % u)
        for dq in (1e-2, 3e-3, 1e-3):
            r = border1(u, dq)
            out["border1_plus_end"].append({"u": u, **r})
            if "error" in r:
                print("  border 1 dq=%g: %s" % (dq, r["error"]))
            else:
                print("  border 1 (end of +)    dq=%-6g |z_s-z*|=%.2e jump=%s  left %s  right %s"
                      % (dq, r["dist"], ("%.1e" % r["jump"]) if r["jump"] is not None else None,
                         ("exp %.3f slope %+.3f" % (r["left"]["exponent"], r["left"]["slope_mid"])) if r["left"] else None,
                         ("exp %.3f slope %+.3f" % (r["right"]["exponent"], r["right"]["slope_mid"])) if r["right"] else None))
        for dq in (1e-2, 3e-3, 1e-3):
            r = border2(u, dq)
            out["border2_pmp_birth"].append({"u": u, **r})
            if "error" in r:
                print("  border 2 dq=%g: %s" % (dq, r["error"]))
            else:
                print("  border 2 (birth +-+)   dq=%-6g |z_s-z*|=%.2e jump=%s  left %s  right %s"
                      % (dq, r["dist"], ("%.1e" % r["jump"]) if r["jump"] is not None else None,
                         ("exp %.3f slope %+.3f" % (r["left"]["exponent"], r["left"]["slope_mid"])) if r["left"] else None,
                         ("exp %.3f slope %+.3f" % (r["right"]["exponent"], r["right"]["slope_mid"])) if r["right"] else None))
    json.dump(out, open(os.path.join(HERE, "d109_border_normal_form_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n-> d109_border_normal_form_2026-09-13.json")


if __name__ == "__main__":
    main()
