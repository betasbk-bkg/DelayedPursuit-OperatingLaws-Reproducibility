#!/usr/bin/env python3
"""
d110c -- use the project's validated integrator (d28.orbit) and arbitrate it
against brute force; re-measure both borders with it.

Why.  d107-d108 reused d27.orbit, whose cell-frame defect d28 had already
documented and fixed; d110's own rewrite still disagreed with brute force at one
border point (u = 0.5, border 2, dz = +1e-4: 0.6688 vs 0.4896).  d28.orbit carries
the cell index explicitly and is the integrator behind the manuscript's basin
numbers (d29-d31).  Before anything else is recomputed, it is checked against the
brute-force time stepper at every point where a defect has been seen.

  (a) the five (+,-,+) fixed points where d27 and brute force disagreed (d110b)
  (b) both sides of border 2 at u = 0.5 and 0.71, dz = +-1e-6, +-1e-4
  (c) local exponents at both borders with d28.orbit, the smooth piece being
      identified by (word, min_gap > theta)

Usage:  python d110c_d28_arbitration_2026-09-13.py
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


d28 = _load("d28", "d28_branch_robust_2026-08-29.py")
fx = _load("d110", "d110_orbit_integrator_fix_2026-09-13.py")
d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
B = json.load(open(os.path.join(HERE, "d110b_integrator_arbitration_2026-09-13.json"), encoding="utf-8"))


def q_bc(u):
    return (1 + math.sqrt(u)) ** 2


def q_lo1(u):
    return 1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4)


def P28(z, u, q):
    k = 2 * q / D
    o = d28.orbit(z, u, k, D, u / k)
    return o


def key28(o, u, q):
    if o is None:
        return None
    th = u / (2 * q / D)
    return (o["word"], bool(o["min_gap"] > th))


def main():
    out = {"a": [], "b": [], "c": []}
    print("(a) d28 vs brute force at d110b's disagreement points")
    worst = 0.0
    for r in B["disagreements"]:
        u, q, z = r["u"], r["q"], r["z"]
        o = P28(z, u, q)
        bf = r["brute"]["z_next"]
        e = abs(o["z_next"] - bf) if (o and bf is not None) else float("nan")
        worst = max(worst, e if np.isfinite(e) else 0)
        out["a"].append({"u": u, "q": q, "z": z, "d28": o["z_next"] if o else None, "brute": bf, "err": e,
                         "word": o["word"] if o else None})
        print("  u=%.2f q=%.3f  d28 %s  brute %.6f  |diff| %.2e  word %s"
              % (u, q, ("%.6f" % o["z_next"]) if o else None, bf, e, o["word"] if o else None))
    print("    max |d28 - brute| = %.2e (brute-force step 1e-6)" % worst)

    print("\n(b) border 2 neighbourhood, d28 vs brute force")
    for r in B["borders"]:
        if r["border"] != 2 or r["u"] not in (0.5, 0.71):
            continue
        u, q, zs = r["u"], r["q"], r["z_s"]
        k = 2 * q / D
        for dz in (-1e-4, -1e-6, 1e-6, 1e-4):
            o = P28(zs + dz, u, q)
            ob = fx.orbit_bruteforce(zs + dz, u, k, D)
            e = abs(o["z_next"] - ob["z_next"]) if (o and ob) else float("nan")
            out["b"].append({"u": u, "dz": dz, "d28": o["z_next"] if o else None,
                             "brute": ob["z_next"] if ob else None, "err": e})
            print("  u=%.2f dz=%+.0e  d28 %s  brute %s  |diff| %.2e"
                  % (u, dz, ("%.6f" % o["z_next"]) if o else None, ("%.6f" % ob["z_next"]) if ob else None, e))

    print("\n(c) local form at both borders with d28.orbit")
    for u in (0.3, 0.5, 0.71):
        for border in (1, 2):
            if border == 1:
                q = q_bc(u) - 3e-3
                zstar = D / 2
            else:
                q = q_lo1(u) + 3e-3
                s = d26.solve_word(u, 2 * q / D, D)
                if not s:
                    continue
                zstar = s["z0"]
            K = lambda z: key28(P28(z, u, q), u, q)
            Pf = lambda z: (lambda o: o["z_next"] if o else None)(P28(z, u, q))
            k0 = K(zstar)
            zs = None
            step = D / 4000
            for direction in (+1, -1):
                z = zstar
                for _ in range(8000):
                    z2 = z + direction * step
                    if K(z2) != k0:
                        a, b = z, z2
                        for _ in range(70):
                            mdl = 0.5 * (a + b)
                            if K(mdl) == k0:
                                a = mdl
                            else:
                                b = mdl
                        c = 0.5 * (a + b)
                        if zs is None or abs(c - zstar) < abs(zs - zstar):
                            zs = c
                        break
                    z = z2
            if zs is None:
                print("  u=%.2f border %d: no switch found" % (u, border))
                continue
            row = {"u": u, "border": border, "q": q, "z_star": zstar, "z_s": zs,
                   "key_in": str(k0), "key_left": str(K(zs - 1e-9)), "key_right": str(K(zs + 1e-9))}
            for side, sg in (("left", -1), ("right", +1)):
                P0 = Pf(zs + sg * 1e-13)
                ss = np.logspace(-11, -5, 19)
                pts = [(s_, Pf(zs + sg * s_)) for s_ in ss]
                pts = [(s_, v) for s_, v in pts if v is not None and P0 is not None and abs(v - P0) > 1e-15]
                if len(pts) >= 6:
                    x = np.log([p[0] for p in pts]); y = np.log([abs(p[1] - P0) for p in pts])
                    row[side] = {"exp": float(np.polyfit(x, y, 1)[0]), "n": len(pts)}
                else:
                    row[side] = None
            pl, pr = Pf(zs - 1e-10), Pf(zs + 1e-10)
            row["jump_1e-10"] = abs(pr - pl) if (pl is not None and pr is not None) else None
            out["c"].append(row)
            print("  u=%.2f border %d: jump %s  left %s  right %s  keys %s | %s"
                  % (u, border, ("%.1e" % row["jump_1e-10"]) if row["jump_1e-10"] is not None else None,
                     ("exp %.3f" % row["left"]["exp"]) if row["left"] else None,
                     ("exp %.3f" % row["right"]["exp"]) if row["right"] else None,
                     row["key_left"], row["key_right"]))
    json.dump(out, open(os.path.join(HERE, "d110c_d28_arbitration_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n-> d110c_d28_arbitration_2026-09-13.json")


if __name__ == "__main__":
    main()
