#!/usr/bin/env python3
"""
d115 -- the orbit structure of the quantized loop, recomputed end to end with the
validated integrator v3 (d113, d113b: 329 in-domain points against brute force,
none off after dt refinement; d28 wrong or undefined on 16.4% of the same points).

Everything d107b/c/h, d108 and d110c computed with d27 or d28, and the basin
crossing d30 computed with d28, is redone here:

  (1) (+) orbit: multiplier from the map vs (1-u-q)/(1-u+q); the end of the stable
      (+) fixed point (z* = Delta/2) bisected in q vs q_bc(u) = (1+sqrt u)^2.
  (2) (+,-,+) orbit across the band: d26 closure solution confirmed as a v3 fixed
      point, piece (word, min_gap > theta), same-piece multiplier, stability,
      agreement with d107g's closed form where it is valid, section validity
      (no crossing of the period still inside the window at the section).
  (3) attractor census: 60 initial in-domain states per (u, q).
  (4) basin crossing: shares of the (+) and (+,-,+) basins along q (d29's
      basin_map and d28's scan with v3 substituted), the q where the (+,-,+) basin
      overtakes, against the flow's measured switch (d30).
  (5) local exponents at both borders.

Usage:  python d115_orbit_structure_v3_2026-09-13.py
"""
import importlib.util
import json
import math
import os
import time
from collections import Counter

import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
D = math.radians(45.0)


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


v3m = _load("d113", "d113_integrator_v3_2026-09-13.py")
d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
g = _load("d107g", "d107g_pmp_multiplier_closed_2026-09-13.py")
d29 = _load("d29", "d29_basin_and_tongues_2026-08-29.py")
d107e = _load("d107e", "d107e_lower_edge_closed_form_2026-09-13.py")


def orb(z, u, q):
    k = 2 * q / D
    return v3m.orbit_v3(z, u, k, D, u / k)


def orbit_as_d28(z0, u, k, delta, th, max_ev=16):
    o = v3m.orbit_v3(z0, u, k, delta, th)
    return None if o is None else {"z_next": o["z_next"], "period": o["period"], "word": o["word"],
                                   "fold": o["fold"], "min_gap": o["min_gap"]}


d29.d28.orbit = orbit_as_d28


def q_bc(u):
    return (1 + math.sqrt(u)) ** 2


def q_lo(u):
    b2 = d107e.q_lo2(u)
    return float(np.nanmax([d107e.q_lo1(u), b2[0] if b2 else float("nan")]))


def key(o, u, q):
    th = u / (2 * q / D)
    return None if o is None else (o["word"], bool(o["min_gap"] > th))


def mult_same_piece(z, u, q):
    base = orb(z, u, q)
    if base is None:
        return None, float("nan"), float("nan")
    kb = key(base, u, q)
    est = []
    for hh in (1e-6, 3e-7, 1e-7, 3e-8):
        op, om = orb(z + hh, u, q), orb(z - hh, u, q)
        sp, sm = key(op, u, q) == kb, key(om, u, q) == kb
        if sp and sm:
            est.append((op["z_next"] - om["z_next"]) / (2 * hh))
        elif sp:
            est.append((op["z_next"] - base["z_next"]) / hh)
        elif sm:
            est.append((base["z_next"] - om["z_next"]) / hh)
    if not est:
        return kb, float("nan"), float("nan")
    return kb, float(np.median(est)), float(np.ptp(est))


def section_valid(o, u, q):
    th = u / (2 * q / D)
    return not any(t + th > o["period"] + 1e-12 for t, s in o["events"][:-1])


def part1():
    print("(1) (+) orbit")
    errs = []
    for u in np.arange(0.1, 0.96, 0.1):
        for q in np.linspace(1.05, q_bc(u) - 0.02, 6):
            o = orb(D / 2, float(u), float(q))
            if o is None or o["word"] != "+" or abs(o["z_next"] - D / 2) > 1e-9:
                continue
            _, m, spr = mult_same_piece(D / 2, float(u), float(q))
            errs.append(abs(m - (1 - u - q) / (1 - u + q)))
    ends = []
    for u in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.71, 0.8, 0.9):
        alive = lambda q: (lambda o: o is not None and o["word"] == "+" and abs(o["z_next"] - D / 2) < 1e-9)(orb(D / 2, u, q))
        lo, hi = 1.05, q_bc(u) + 0.5
        if not alive(lo) or alive(hi):
            ends.append({"u": u, "q_end": None}); continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if alive(mid) else (lo, mid)
        ends.append({"u": u, "q_end": 0.5 * (lo + hi), "q_bc": q_bc(u), "err": 0.5 * (lo + hi) - q_bc(u)})
    print("    multiplier: %d points, max |map - closed| %.2e" % (len(errs), max(errs)))
    print("    end of (+): max |q_end - q_bc| %.2e over %d u"
          % (max(abs(e["err"]) for e in ends if e["q_end"]), sum(1 for e in ends if e["q_end"])))
    return {"mult_n": len(errs), "mult_max_err": max(errs), "ends": ends}


def part2():
    print("(2) (+,-,+) across the band")
    res = {}
    for u in (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        stats = Counter()
        agree = []
        p = None
        for q in np.arange(q_lo(u) + 0.003, q_bc(u), 0.01):
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
                stats["no closure solution"] += 1
                continue
            p = list(sol)
            o = orb(sol[0], u, q)
            if o is None or o["word"] not in ("-++",) or abs(o["z_next"] - sol[0]) > 1e-7:
                stats["closure solution not a v3 fixed point"] += 1
                continue
            stats["fixed point"] += 1
            if not section_valid(o, u, q):
                stats["section not 1-D"] += 1
                continue
            kb, m, spr = mult_same_piece(sol[0], u, q)
            if not np.isfinite(m) or spr > 1e-5:
                stats["multiplier untrusted"] += 1
                continue
            stats["trusted"] += 1
            stats["stable" if abs(m) < 1 else "UNSTABLE"] += 1
            stats["piece gap>th" if kb[1] else "piece gap<=th"] += 1
            t1, t2 = sol[1] / D, sol[2] / D
            if g.valid(u, q, t1, t2):
                agree.append(abs(g.mult_closed(u, q, t1, t2) - m))
        res[str(u)] = {"stats": dict(stats), "closed_n": len(agree), "closed_max_err": max(agree) if agree else None}
        print("    u=%.1f %s  closed-form %d, max %.1e" % (u, dict(stats), len(agree), max(agree) if agree else float("nan")))
    return res


def part3():
    print("(3) attractor census")
    rows = []
    for u in (0.3, 0.5, 0.71, 0.9):
        for q in np.round(np.arange(1.6, 4.01, 0.2), 2):
            k = 2 * q / D
            c, wd = Counter(), Counter()
            zmax = D + (1 - u) / k
            for z0 in np.linspace(-0.2 * D, zmax - 1e-6, 60):
                z = z0
                kind = None
                for _ in range(300):
                    o = orb(z, u, q)
                    if o is None:
                        kind = "failed"; break
                    if not section_valid(o, u, q):
                        kind = "leaves 1-D section"; break
                    z = o["z_next"]
                if kind is None:
                    rec, words = [], []
                    for _ in range(64):
                        rec.append(z)
                        o = orb(z, u, q)
                        if o is None or not section_valid(o, u, q):
                            kind = "leaves 1-D section" if o else "failed"; break
                        words.append(o["word"]); z = o["z_next"]
                if kind is None:
                    rec = np.array(rec)
                    for per in range(1, 9):
                        if np.all(np.abs(rec[per:] - rec[:-per]) < 1e-9):
                            kind = "p%d" % per; wd["p%d:%s" % (per, "".join(words[:per]))] += 1; break
                    if kind is None:
                        kind = "aperiodic"
                c[kind] += 1
            rows.append({"u": u, "q": float(q), "census": dict(c), "words": dict(wd)})
            print("    u=%.2f q=%.2f %s %s" % (u, q, dict(c), dict(wd.most_common(2))))
    return rows


def part4():
    print("(4) basin crossing vs the flow's switch")
    meas = {round(m["u"], 2): m for m in json.load(open(os.path.join(HERE, "d30_results_2026-08-29.json"), encoding="utf-8"))["B_measured"]}
    out = []
    for u in (0.3, 0.5, 0.71, 0.9):
        rows = []
        for q in np.arange(max(q_lo(u), 1.2) + 0.01, q_bc(u) - 0.005, 0.05):
            k = 2 * q / D
            fps = [f for f in d29.d28.scan(u, k, D) if abs(f["mult"]) < 1.0]
            plus = [j for j, f in enumerate(fps) if f["word"] == "+"]
            pmp = [j for j, f in enumerate(fps) if f["word"] in ("-++", "+-+")]
            if not fps:
                continue
            zs, lab = d29.basin_map(u, k, D, fps)
            sp = float(sum(np.mean(lab == j) for j in plus))
            sm = float(sum(np.mean(lab == j) for j in pmp))
            rows.append({"q": float(q), "share_plus": sp, "share_pmp": sm, "unlabelled": float(np.mean(lab == -1))})
        cross = None
        for a, b in zip(rows, rows[1:]):
            da, db = a["share_pmp"] - a["share_plus"], b["share_pmp"] - b["share_plus"]
            if da < 0 <= db:
                cross = a["q"] + (b["q"] - a["q"]) * (-da) / (db - da)
                break
        m = meas[round(u, 2)]
        out.append({"u": u, "rows": rows, "basin_crossing": cross, "switch": m["mid"],
                    "switch_bracket": [m["last_plus"], m["first_pmp"]]})
        print("    u=%.2f  basin crossing %s  flow switch %.3f [%.3f, %.3f]"
              % (u, ("%.3f" % cross) if cross else None, m["mid"], m["last_plus"], m["first_pmp"]))
    return out


def part5():
    print("(5) local exponents at both borders")
    res = []
    for u in (0.3, 0.5, 0.71):
        for border in (1, 2):
            if border == 1:
                q = q_bc(u) - 3e-3; zstar = D / 2
            else:
                q = d107e.q_lo1(u) + 3e-3
                s = d26.solve_word(u, 2 * q / D, D)
                if not s:
                    continue
                zstar = s["z0"]
            K = lambda z: key(orb(z, u, q), u, q)
            Pf = lambda z: (lambda o: o["z_next"] if o else None)(orb(z, u, q))
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
                            a, b = (mdl, b) if K(mdl) == k0 else (a, mdl)
                        c = 0.5 * (a + b)
                        if zs is None or abs(c - zstar) < abs(zs - zstar):
                            zs = c
                        break
                    z = z2
            if zs is None:
                continue
            row = {"u": u, "border": border, "q": q, "z_s": zs}
            for side, sg in (("left", -1), ("right", +1)):
                P0 = Pf(zs + sg * 1e-13)
                ss = np.logspace(-11, -6, 16)
                pts = [(s_, Pf(zs + sg * s_)) for s_ in ss]
                pts = [(s_, v) for s_, v in pts if v is not None and P0 is not None and abs(v - P0) > 1e-15]
                row[side] = float(np.polyfit(np.log([p[0] for p in pts]), np.log([abs(p[1] - P0) for p in pts]), 1)[0]) if len(pts) >= 6 else None
            pl, pr = Pf(zs - 1e-10), Pf(zs + 1e-10)
            row["jump_1e-10"] = abs(pr - pl) if (pl is not None and pr is not None) else None
            res.append(row)
            print("    u=%.2f border %d: jump %s  exponent left %s right %s" % (u, border, row["jump_1e-10"], row["left"], row["right"]))
    return res


def main():
    t0 = time.time()
    out = {"plus": part1(), "pmp_band": part2(), "census": part3(), "basin_crossing": part4(), "borders": part5()}
    json.dump(out, open(os.path.join(HERE, "d115_orbit_structure_v3_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n%.0f s -> d115_orbit_structure_v3_2026-09-13.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
