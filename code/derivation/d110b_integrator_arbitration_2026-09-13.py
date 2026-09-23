#!/usr/bin/env python3
"""
d110b -- arbitrate the old and fixed integrators, and re-measure the borders.

d110 left two things unresolved:

  (1) test (a) was supposed to compare the two integrators only where the
      short-excursion defect cannot act, yet they differed by up to 1.57.  Either
      the premise was wrong (the defect acts there too) or the fixed integrator has
      a defect of its own.  Every disagreeing case is listed here with both event
      sequences and a brute-force time-stepping run that decides which is right.

  (2) with the fixed integrator, border 1 (end of (+)) is continuous to ~1e-3 at a
      2e-7 separation -- a square-root (grazing) law would give exactly that -- and
      border 2 (birth of (+,-,+)) is continuous at u = 0.3 and 0.71 but still jumps
      by 0.18 at u = 0.5, where fixed and brute-force also disagree by 0.18.  The
      switching points are re-located WITH THE FIXED INTEGRATOR and the local
      exponent on each side is measured; the u = 0.5 case is arbitrated by
      brute force.

Usage:  python d110b_integrator_arbitration_2026-09-13.py
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


h = _load("d107h", "d107h_piece_map_2026-09-13.py")
fx = _load("d110", "d110_orbit_integrator_fix_2026-09-13.py")
d26 = _load("d26", "d26_word_pmp_2026-08-29.py")


def w(o):
    return "".join("+" if s > 0 else "-" for _, s in o["events"][1:]) if o else None


def ev(o):
    return ", ".join("%.5f%s" % (t, "+" if s > 0 else "-") for t, s in o["events"]) if o else None


def q_bc(u):
    return (1 + math.sqrt(u)) ** 2


def q_lo1(u):
    return 1 + math.sqrt(1 - (2 * u - 1) ** 2 + u * u / 4)


def part1():
    print("(1) where old and fixed integrators disagree, and who is right")
    rows = []
    cases = []
    for u in (0.3, 0.5, 0.71):
        for q in (1.5, 1.9, 2.2):
            for z in (D / 2, D / 2 + 0.01, D / 2 - 0.01):
                cases.append((u, q, z, "plus-grid"))
        for q in (2.15, 2.65):
            s = d26.solve_word(u, 2 * q / D, D)
            if s:
                cases.append((u, q, s["z0"], "pmp-fixed-point"))
    for u, q, z, tag in cases:
        k = 2 * q / D
        th = u / k
        o1 = h.orbit_events(z, u, k, D, th)
        o2 = fx.orbit_events_fixed(z, u, k, D, th)
        if not (o1 and o2):
            continue
        d = abs(o1["z_next"] - o2["z_next"])
        if d < 1e-9:
            continue
        ob = fx.orbit_bruteforce(z, u, k, D)
        verdict = None
        if ob:
            e1, e2 = abs(o1["z_next"] - ob["z_next"]), abs(o2["z_next"] - ob["z_next"])
            verdict = "fixed" if e2 < e1 else "old"
        row = {"u": u, "q": q, "z": z, "tag": tag, "diff": d,
               "old": {"z_next": o1["z_next"], "events": ev(o1)},
               "fixed": {"z_next": o2["z_next"], "events": ev(o2)},
               "brute": {"z_next": ob["z_next"] if ob else None, "events": ev(ob)},
               "closer": verdict}
        rows.append(row)
        print("  u=%.2f q=%.3f z=%.5f [%s] diff %.3e -> closer to brute force: %s" % (u, q, z, tag, d, verdict))
        print("     old   %s" % row["old"]["events"])
        print("     fixed %s" % row["fixed"]["events"])
        print("     brute %s" % row["brute"]["events"])
    return rows


def local_exponents(Pf, zs):
    res = {}
    for side, sg in (("left", -1), ("right", +1)):
        P0 = Pf(zs + sg * 1e-13)
        ss = np.logspace(-10, -4, 19)
        pts = [(s, Pf(zs + sg * s)) for s in ss]
        pts = [(s, v) for s, v in pts if v is not None and P0 is not None and abs(v - P0) > 1e-15]
        if len(pts) < 6:
            res[side] = None
            continue
        x = np.log([s for s, _ in pts])
        y = np.log([abs(v - P0) for _, v in pts])
        # exponent over the smallest decades (where the local law holds)
        e_small = float(np.polyfit(x[: len(x) // 2], y[: len(y) // 2], 1)[0])
        e_all = float(np.polyfit(x, y, 1)[0])
        res[side] = {"exp_small": e_small, "exp_all": e_all}
    return res


def part2():
    print("\n(2) borders re-located and re-measured with the fixed integrator")
    rows = []
    for u in (0.3, 0.5, 0.71):
        for border in (1, 2):
            if border == 1:
                q = q_bc(u) - 3e-3
                zstar = D / 2
            else:
                q = q_lo1(u) + 3e-3
                s = d26.solve_word(u, 2 * q / D, D)
                if not s:
                    print("  u=%.2f border 2: no (+,-,+) orbit" % u)
                    continue
                zstar = s["z0"]
            k = 2 * q / D
            th = u / k
            Pf = lambda z, u=u, k=k, th=th: (lambda o: o["z_next"] if o else None)(fx.orbit_events_fixed(z, u, k, D, th))
            W = lambda z, u=u, k=k, th=th: w(fx.orbit_events_fixed(z, u, k, D, th))

            def key(z):
                o = fx.orbit_events_fixed(z, u, k, D, th)
                if not o:
                    return None
                if border == 1:
                    return w(o)
                e = o["events"]
                return (w(o), (e[2][0] - e[1][0]) > th) if len(e) >= 3 else (w(o), None)

            k0 = key(zstar)
            zs = None
            step = D / 4000
            for direction in (+1, -1):
                z = zstar
                for _ in range(8000):
                    z2 = z + direction * step
                    if key(z2) != k0:
                        a, b = z, z2
                        for _ in range(70):
                            mdl = 0.5 * (a + b)
                            if key(mdl) == k0:
                                a = mdl
                            else:
                                b = mdl
                        c = 0.5 * (a + b)
                        if zs is None or abs(c - zstar) < abs(zs - zstar):
                            zs = c
                        break
                    z = z2
            if zs is None:
                print("  u=%.2f border %d: no switch near the fixed point" % (u, border))
                continue
            pr, pl = Pf(zs + 1e-12), Pf(zs - 1e-12)
            if pr is None or pl is None:
                # the integrator returns None on a tangential touch or when it runs
                # out of events; step a little further out before measuring
                pr, pl = Pf(zs + 1e-9), Pf(zs - 1e-9)
            jump = abs(pr - pl) if (pr is not None and pl is not None) else float("nan")
            ex = local_exponents(Pf, zs)
            row = {"u": u, "border": border, "q": q, "z_star": zstar, "z_s": zs, "jump_1e-12": jump,
                   "key_left": str(key(zs - 1e-9)), "key_right": str(key(zs + 1e-9)), **ex}
            if border == 2 and abs(u - 0.5) < 1e-9:
                arb = []
                for dz in (-1e-4, -1e-6, 1e-6, 1e-4):
                    ob = fx.orbit_bruteforce(zs + dz, u, k, D)
                    arb.append({"dz": dz, "fixed": Pf(zs + dz), "brute": ob["z_next"] if ob else None})
                row["bruteforce_arbitration"] = arb
            rows.append(row)
            print("  u=%.2f border %d: jump %.1e  left %s  right %s  keys %s | %s"
                  % (u, border, jump,
                     ("exp %.3f (all %.3f)" % (ex["left"]["exp_small"], ex["left"]["exp_all"])) if ex["left"] else None,
                     ("exp %.3f (all %.3f)" % (ex["right"]["exp_small"], ex["right"]["exp_all"])) if ex["right"] else None,
                     row["key_left"], row["key_right"]))
            if "bruteforce_arbitration" in row:
                for a in row["bruteforce_arbitration"]:
                    print("       dz=%+.0e fixed %s brute %s" % (a["dz"], a["fixed"], a["brute"]))
    return rows


def main():
    out = {"disagreements": part1(), "borders": part2()}
    json.dump(out, open(os.path.join(HERE, "d110b_integrator_arbitration_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n-> d110b_integrator_arbitration_2026-09-13.json")


if __name__ == "__main__":
    main()
