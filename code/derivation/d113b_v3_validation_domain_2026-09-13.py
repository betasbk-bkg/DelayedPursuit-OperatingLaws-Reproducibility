#!/usr/bin/env python3
"""
d113b -- validate integrator v3 on the map's actual domain.

d113 scored v3 on random (u, q, z) including section states with m'(0+) < 0.
A forward crossing cannot have m' < 0, so those states are outside the section
map's domain and neither integrator's value means anything there; most of v3's
13 "failures" were such states.  The domain condition at the section, with the
section crossing inside the delay window (J = +1), is

    m'(0+) = (1 - u) - k z0 + k Delta > 0,   i.e.   z0 < Delta + (1 - u)/k.

This rerun samples only inside the domain (300 points), adds every earlier
disagreement point that satisfies it, and for any disagreement above 5e-4 reruns
brute force at dt = 2e-7 to see whether brute force itself was the one off
(shallow grazes can last less than one step).

Usage:  python d113b_v3_validation_domain_2026-09-13.py
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
v3 = _load("d113", "d113_integrator_v3_2026-09-13.py")
d28 = _load("d28", "d28_branch_robust_2026-08-29.py")


def in_domain(u, q, z):
    k = 2 * q / D
    return (1 - u) - k * z + k * D > 1e-9


def main():
    t0 = time.time()
    prev = json.load(open(os.path.join(HERE, "d113_integrator_v3_2026-09-13.json"), encoding="utf-8"))
    pts = [(r["tag"], r["u"], r["q"], r["z"]) for r in prev if r["tag"] != "random" and in_domain(r["u"], r["q"], r["z"])]
    n_prev_out = sum(1 for r in prev if not in_domain(r["u"], r["q"], r["z"]))
    rng = np.random.default_rng(913)
    while sum(1 for p in pts if p[0] == "random") < 300:
        u = float(rng.uniform(0.1, 0.95))
        q = float(rng.uniform(1.2, 4.0))
        k = 2 * q / D
        zmax = D + (1 - u) / k
        z = float(rng.uniform(-0.2 * D, zmax))
        if in_domain(u, q, z):
            pts.append(("random", u, q, z))
    print("earlier sample points outside the domain (m'(0+) <= 0): %d" % n_prev_out)
    rows = []
    for tag, u, q, z in pts:
        k = 2 * q / D
        th = u / k
        ob = fx.orbit_bruteforce(z, u, k, D, dt=1e-6)
        o3 = v3.orbit_v3(z, u, k, D, th)
        o28 = d28.orbit(z, u, k, D, th)
        row = {"tag": tag, "u": u, "q": q, "z": z,
               "brute": ob["z_next"] if ob else None,
               "v3": o3["z_next"] if o3 else None, "d28": o28["z_next"] if o28 else None}
        if ob and o3:
            row["err_v3"] = abs(o3["z_next"] - ob["z_next"])
            if row["err_v3"] > 5e-4:
                ob2 = fx.orbit_bruteforce(z, u, k, D, dt=2e-7)
                row["brute_2e-7"] = ob2["z_next"] if ob2 else None
                row["err_v3_fine"] = abs(o3["z_next"] - ob2["z_next"]) if ob2 else None
        if ob and o28:
            row["err_d28"] = abs(o28["z_next"] - ob["z_next"])
        rows.append(row)

    def score(name):
        have = [r for r in rows if r["brute"] is not None and r.get("err_" + name) is not None]
        none = sum(1 for r in rows if r["brute"] is not None and r.get(name) is None)
        bad = [r for r in have if r["err_" + name] > 5e-4]
        return len(have), none, bad

    n3, none3, bad3 = score("v3")
    still = [r for r in bad3 if r.get("err_v3_fine") is None or r["err_v3_fine"] > 5e-4]
    print("v3 : compared %d, returned None %d, > 5e-4 vs brute(1e-6) %d, of which still > 5e-4 vs brute(2e-7) %d"
          % (n3, none3, len(bad3), len(still)))
    for r in still[:10]:
        print("   %-16s u=%.3f q=%.3f z=%.5f  brute %.6f / %s  v3 %.6f" % (r["tag"], r["u"], r["q"], r["z"], r["brute"], r.get("brute_2e-7"), r["v3"]))
    n28, none28, bad28 = score("d28")
    print("d28: compared %d, returned None %d, > 5e-4 %d  (fraction wrong or None: %.1f%%)"
          % (n28, none28, len(bad28), 100 * (len(bad28) + none28) / max(1, n28 + none28)))
    json.dump({"rows": rows, "v3_still_bad": len(still), "d28_bad": len(bad28), "d28_none": none28},
              open(os.path.join(HERE, "d113b_v3_validation_domain_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d113b_v3_validation_domain_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
