#!/usr/bin/env python3
"""
d112 -- event integrators vs brute force: who is right where they disagree, and
is the end of the (+) orbit a grazing bifurcation at scales brute force resolves?

Two event integrators (d28, used for the manuscript's basin numbers, and d110)
agree with each other and disagree with the brute-force time stepper (dt = 1e-6)
at u = 0.71, q = 2.65, z = 0.45054 and at u = 0.5 just outside border 2.  In the
d110 event list the orbit ends at a forward crossing of +Delta/2 that brute force
does not see.  If that crossing is a graze -- m touching +Delta/2 with m' ~ 0 --
its duration can be shorter than dt, brute force misses it, and the event
integrators are right: the section map then has a further discontinuity where the
orbit grazes the TARGET boundary.  Decided here by refining dt.

  (i)  event lists from d110 with m' at every crossing
  (ii) brute force at dt = 1e-6, 2e-7, 5e-8 at both points
  (iii) border 1 (end of (+)), u = 0.5: |P(z_s + dz) - P(z_s)| over dz = 1e-2..1e-4
       with brute force at dt = 1e-6 and 2e-7; exponent 0.5 = grazing, 1 = kink

Usage:  python d112_graze_arbitration_2026-09-13.py
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


fx = _load("d110", "d110_orbit_integrator_fix_2026-09-13.py")
d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
C = json.load(open(os.path.join(HERE, "d110c_d28_arbitration_2026-09-13.json"), encoding="utf-8"))
B = json.load(open(os.path.join(HERE, "d110b_integrator_arbitration_2026-09-13.json"), encoding="utf-8"))


def slopes(o, u, k):
    """m' just before each crossing, rebuilt from the event list."""
    th = u / k
    ev = o["events"]
    out = []
    z0 = None
    return out


def event_slopes(z0, u, k):
    """re-run d110's integrator and record m' at each crossing."""
    th = u / k
    o = fx.orbit_events_fixed(z0, u, k, D, th)
    if o is None:
        return None
    ev = o["events"]
    res = []
    # m'(T) = (1-u) - k z(T) + k D J(T); z(T) = z0 - T + D * (net jumps before T)
    for i, (t, s) in enumerate(ev[1:], start=1):
        net = sum(ss for tt, ss in ev[:i] if tt < t - 1e-15) - 1   # exclude the section crossing's jump
        z_before = z0 - t + D * net
        J = sum(ss for tt, ss in ev[:i] if t - th < tt < t)
        res.append({"t": t, "dir": s, "m_prime": (1 - u) - k * z_before + k * D * J})
    return {"z_next": o["z_next"], "crossings": res}


def main():
    t0 = time.time()
    out = {"points": [], "border1": []}
    pts = []
    a = [r for r in C["a"] if abs(r["u"] - 0.71) < 1e-9 and abs(r["q"] - 2.65) < 1e-9][0]
    pts.append(("u0.71_q2.65_fixedpoint", 0.71, 2.65, a["z"]))
    b2 = [r for r in B["borders"] if r["border"] == 2 and abs(r["u"] - 0.5) < 1e-9][0]
    pts.append(("u0.50_border2_dz+1e-4", 0.5, b2["q"], b2["z_s"] + 1e-4))
    for name, u, q, z in pts:
        k = 2 * q / D
        es = event_slopes(z, u, k)
        o28 = d28.orbit(z, u, k, D, u / k)
        row = {"name": name, "u": u, "q": q, "z": z,
               "d28": o28["z_next"] if o28 else None, "d110": es["z_next"] if es else None,
               "crossings": es["crossings"] if es else None, "brute": {}}
        print("\n%s  d28 %s  d110 %s" % (name, row["d28"], row["d110"]))
        for c in (es["crossings"] if es else []):
            print("   crossing t=%.6f dir %+d  m'=%+.3e" % (c["t"], c["dir"], c["m_prime"]))
        for dt in (1e-6, 2e-7, 5e-8):
            ob = fx.orbit_bruteforce(z, u, k, D, dt=dt)
            row["brute"][str(dt)] = ob["z_next"] if ob else None
            print("   brute dt=%.0e -> %s   (%.0f s)" % (dt, ob["z_next"] if ob else None, time.time() - t0))
        out["points"].append(row)

    print("\nborder 1 (end of (+)), u = 0.5: brute-force scaling")
    r1 = [r for r in C["c"] if r["border"] == 1 and abs(r["u"] - 0.5) < 1e-9][0]
    u, q, zs = r1["u"], r1["q"], r1["z_s"]
    k = 2 * q / D
    for dt in (1e-6, 2e-7):
        P0 = fx.orbit_bruteforce(zs, u, k, D, dt=dt)["z_next"]
        dzs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
        vals = []
        for dz in dzs:
            ob = fx.orbit_bruteforce(zs + dz, u, k, D, dt=dt)
            vals.append(abs(ob["z_next"] - P0) if ob else float("nan"))
        ok = [(d, v) for d, v in zip(dzs, vals) if np.isfinite(v) and v > 0]
        e = float(np.polyfit(np.log([d for d, _ in ok]), np.log([v for _, v in ok]), 1)[0]) if len(ok) >= 3 else float("nan")
        d28v = [abs(d28.orbit(zs + dz, u, k, D, u / k)["z_next"] - d28.orbit(zs, u, k, D, u / k)["z_next"]) for dz in dzs]
        e28 = float(np.polyfit(np.log(dzs), np.log(d28v), 1)[0])
        out["border1"].append({"dt": dt, "dz": dzs, "brute_dP": vals, "brute_exp": e, "d28_dP": d28v, "d28_exp": e28})
        print("  dt=%.0e  brute exponent %.3f  (d28 on the same dz: %.3f)  dP %s" % (dt, e, e28, ["%.2e" % v for v in vals]))
    json.dump(out, open(os.path.join(HERE, "d112_graze_arbitration_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d112_graze_arbitration_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
