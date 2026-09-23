#!/usr/bin/env python3
"""
d113 -- a third event integrator for the section map, and its validation against
brute-force time stepping.

WHY A THIRD.  d112 showed that d28.orbit (the integrator behind the manuscript's
basin shares, d29-d31) and d110's rewrite agree with each other and are WRONG at
u = 0.71, q = 2.65, z = 0.45054: they register a forward crossing at t = 0.6072
with m' = +1.35 -- steep, not a graze -- which brute force, converged in dt
(1e-6 and 2e-7 both give 0.45053), does not have.

The mechanism: both evaluate the delay-window count J at the START of each
segment, J = #{events with T - theta < t_e <= T}.  A segment that starts exactly
where an event leaves the window (T = t_e + theta) decides membership by floating
point: T - theta can land a hair below t_e and keep the event in the window, and
if no later breakpoint follows, J stays wrong to the end of the period.  d26's
closure evaluates J at the segment MIDPOINT and is immune.

THE v3 RULES.  (1) J at the segment midpoint; (2) cell index explicit; (3) the
direction of a crossing from the sign of m' at the crossing, the cell moving by
one in that direction; (4) m snapped to the boundary after a crossing.

VALIDATION.  Brute force (dt = 1e-6) as the reference, at every point where an
event integrator has been seen to fail, and at a seeded random sample of
(u, q, z) over u in [0.1, 0.95], q in [1.2, 4.0], z in one cell.  Pass: agreement
at the brute-force truncation level (<= 5e-4) at every point with the same number
of crossings.  d28 is scored on the same sample for comparison.

Usage:  python d113_integrator_v3_2026-09-13.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def orbit_v3(z0, u, k, delta, th, max_ev=40):
    cell = 0
    m = -delta / 2.0
    z = z0
    T = 0.0
    events = [(0.0, +1)]
    fold = None
    for _ in range(4 * max_ev):
        cand = [te + th for te, _ in events if te + th > T + 1e-13]
        T_bp = min(cand) if cand else float("inf")
        lo = -delta / 2.0 + cell * delta
        hi = lo + delta
        # J on (T, T_bp): evaluate membership at an interior time, not at T
        T_probe = T + (min(T_bp, T + 1.0) - T) * 0.5
        J = sum(s for te, s in events if T_probe - th < te <= T_probe)
        A = (1.0 - u) - k * (z + T) + k * delta * J          # m'(T') = A + k T'
        t_hit, bnd_hit = None, None
        for bnd in (hi, lo):
            a2, b2 = k / 2.0, A
            c2 = m - bnd - A * T - k * T * T / 2.0
            d2 = b2 * b2 - 4 * a2 * c2
            if d2 < 0:
                continue
            r = math.sqrt(d2)
            for root in ((-b2 - r) / (2 * a2), (-b2 + r) / (2 * a2)):
                if root > T + 1e-11 and (t_hit is None or root < t_hit):
                    t_hit, bnd_hit = root, bnd
        if t_hit is not None and T_bp < t_hit:
            t_hit = None
        Tn = t_hit if t_hit is not None else T_bp
        if not np.isfinite(Tn):
            return None
        # the window may change inside (T, Tn) only at T_bp; T_probe was taken before it
        if T_probe > Tn:
            T_probe = 0.5 * (T + Tn)
            J2 = sum(s for te, s in events if T_probe - th < te <= T_probe)
            if J2 != J:
                J = J2
                continue                                       # recompute the segment with the right J
        tf = -A / k
        if T < tf < Tn:
            mf = m + A * (tf - T) + k * (tf * tf - T * T) / 2.0
            fold = mf if fold is None else min(fold, mf)
        m_new = m + A * (Tn - T) + k * (Tn * Tn - T * T) / 2.0
        z -= (Tn - T)
        T = Tn
        if t_hit is None:
            m = m_new
            continue
        slope = A + k * Tn
        if abs(slope) < 1e-13:
            return None
        which = +1 if slope > 0 else -1
        m = bnd_hit
        z += which * delta
        cell += which
        events.append((T, which))
        if which > 0 and cell == 1:
            gaps = [events[i + 1][0] - events[i][0] for i in range(len(events) - 1)]
            return {"z_next": z, "period": T, "events": events, "fold": fold,
                    "word": "".join("+" if s > 0 else "-" for _, s in events[1:]),
                    "min_gap": min(gaps) if gaps else float("inf")}
        if abs(cell) > 3:
            return None
    return None


def main():
    fx = _load("d110", "d110_orbit_integrator_fix_2026-09-13.py")
    d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
    t0 = time.time()
    pts = []
    for r in json.load(open(os.path.join(HERE, "d110c_d28_arbitration_2026-09-13.json"), encoding="utf-8"))["a"]:
        pts.append(("d110c-a", r["u"], r["q"], r["z"]))
    for r in json.load(open(os.path.join(HERE, "d110b_integrator_arbitration_2026-09-13.json"), encoding="utf-8"))["borders"]:
        for dz in (-1e-4, -1e-6, 1e-6, 1e-4):
            pts.append(("d110b-border%d" % r["border"], r["u"], r["q"], r["z_s"] + dz))
    rng = np.random.default_rng(20260913)
    for _ in range(150):
        u = float(rng.uniform(0.1, 0.95))
        q = float(rng.uniform(1.2, 4.0))
        z = float(rng.uniform(-0.2 * D, 1.2 * D))
        pts.append(("random", u, q, z))

    rows = []
    for tag, u, q, z in pts:
        k = 2 * q / D
        th = u / k
        ob = fx.orbit_bruteforce(z, u, k, D, dt=1e-6)
        o3 = orbit_v3(z, u, k, D, th)
        o28 = d28.orbit(z, u, k, D, th)
        if ob is None:
            rows.append({"tag": tag, "u": u, "q": q, "z": z, "brute": None})
            continue
        e3 = abs(o3["z_next"] - ob["z_next"]) if o3 else None
        e28 = abs(o28["z_next"] - ob["z_next"]) if o28 else None
        rows.append({"tag": tag, "u": u, "q": q, "z": z, "brute": ob["z_next"], "n_brute": len(ob["events"]),
                     "v3": o3["z_next"] if o3 else None, "err_v3": e3,
                     "n_v3": len(o3["events"]) if o3 else None,
                     "d28": o28["z_next"] if o28 else None, "err_d28": e28})
    def summary(key):
        errs = [r[key] for r in rows if r.get("brute") is not None and r.get(key) is not None]
        bad = [r for r in rows if r.get("brute") is not None and r.get(key) is not None and r[key] > 5e-4]
        none = sum(1 for r in rows if r.get("brute") is not None and r.get(key) is None)
        return len(errs), (max(errs) if errs else None), len(bad), none, bad
    for key in ("err_v3", "err_d28"):
        n, mx, nbad, nnone, bad = summary(key)
        print("%-8s compared %d  max err %.2e  > 5e-4: %d  returned None: %d" % (key, n, mx, nbad, nnone))
        for r in bad[:10]:
            print("   %-16s u=%.3f q=%.3f z=%.5f  brute %.6f  v3 %s  d28 %s"
                  % (r["tag"], r["u"], r["q"], r["z"], r["brute"], r["v3"], r["d28"]))
    json.dump(rows, open(os.path.join(HERE, "d113_integrator_v3_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d113_integrator_v3_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
