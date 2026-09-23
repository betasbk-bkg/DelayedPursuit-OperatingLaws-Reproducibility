#!/usr/bin/env python3
"""
d110 -- the section-map integrator mislabels short excursions; fixed and tested.

THE DEFECT (found by d109).  d27.orbit (and its copy in d107h) decides the
direction of a boundary crossing by WHICH boundary was hit: the upper one is
'+', the lower one '-'.  Right after a crossing, m sits exactly on that boundary,
the cell is not re-derived (m is not below lo - 1e-12), and if the parabola dips
below the boundary and comes back up BEFORE the next delay-window breakpoint,
the upward return also hits 'lo' and is recorded as a second BACKWARD crossing.
z then jumps by -Delta instead of +Delta.  At u = 0.5, q = q_bc - 0.003 the
right-hand side of the switch showed events 0.325465-, 0.325516- : two backward
crossings 5e-5 apart, which m cannot do.  Orbits whose excursion outlasts the
delay window (t2 - t1 > theta) were never affected, because the breakpoint came
first and re-derived the cell.

THE FIX.  Track the cell index explicitly; set a crossing's direction from the
sign of m' at the crossing; move the cell by one in that direction.

TESTS.
  (a) agreement with the old integrator on orbits the defect cannot reach (the
      (+) fixed point, and the (+,-,+) orbit on its main piece);
  (b) agreement with a brute-force time-stepping integrator of the same hybrid
      system (no event algebra at all) on both sides of both borders;
  (c) continuity of the corrected map across the switching points d109 reported
      as jumps.

Usage:  python d110_orbit_integrator_fix_2026-09-13.py
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


def orbit_events_fixed(z0, u, k, delta, th, max_ev=40):
    """Exact piecewise integration from a forward crossing (m = -delta/2) to the
    next forward crossing of the boundary +delta/2 (one cell up, net)."""
    cell = 0                                  # current cell: m in [-d/2 + cell*d, d/2 + cell*d]
    m = -delta / 2.0
    z = z0
    T = 0.0
    events = [(0.0, +1)]
    for _ in range(max_ev):
        cand = [te + th for te, _ in events if te + th > T + 1e-14]
        T_bp = min(cand) if cand else float("inf")
        J = sum(s for te, s in events if T - th < te <= T)
        A = (1.0 - u) - k * (z + T) + k * delta * J   # m'(T') = A + k T'
        lo = -delta / 2.0 + cell * delta
        hi = lo + delta
        t_hit, bnd_hit = None, None
        for bnd in (hi, lo):
            a = k / 2.0
            b = A
            c = m - bnd - A * T - k * T * T / 2.0
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            r = math.sqrt(disc)
            for root in ((-b - r) / (2 * a), (-b + r) / (2 * a)):
                if root > T + 1e-12 and (t_hit is None or root < t_hit):
                    t_hit, bnd_hit = root, bnd
        if t_hit is None or t_hit > T_bp:
            if not np.isfinite(T_bp):
                return None
            Tn = T_bp
            m = m + A * (Tn - T) + k * (Tn * Tn - T * T) / 2.0
            z -= (Tn - T)
            T = Tn
            continue
        Tn = t_hit
        slope = A + k * Tn
        if abs(slope) < 1e-14:
            return None                       # tangential touch: undefined, reported
        which = +1 if slope > 0 else -1
        m = bnd_hit
        z -= (Tn - T)
        T = Tn
        z += which * delta
        events.append((T, which))
        cell += which if (which > 0 and bnd_hit == hi) or (which < 0 and bnd_hit == lo) else 0
        if which > 0 and cell == 1:
            return {"z_next": z, "period": T, "events": events}
    return None


def orbit_bruteforce(z0, u, k, delta, dt=1e-6, T_max=3.0):
    """Time-stepping of the same hybrid system, crossings detected by sign change
    of (m - boundary); no quadratic roots, no event algebra."""
    th = u / k
    m = -delta / 2.0 + 1e-15
    z = z0
    T = 0.0
    cell = 0
    events = [(0.0, +1)]
    n = int(T_max / dt)
    for _ in range(n):
        J = sum(s for te, s in events if T - th < te <= T)
        dm = (1.0 - u) - k * z + k * delta * J
        m_new = m + dt * dm
        z -= dt
        T += dt
        lo = -delta / 2.0 + cell * delta
        hi = lo + delta
        if m_new >= hi:
            cell += 1
            z += delta
            events.append((T, +1))
            if cell == 1:
                return {"z_next": z, "period": T, "events": events}
        elif m_new < lo:
            cell -= 1
            z -= delta
            events.append((T, -1))
        m = m_new
    return None


def main():
    h = _load("d107h", "d107h_piece_map_2026-09-13.py")
    d26 = _load("d26", "d26_word_pmp_2026-08-29.py")
    j109 = json.load(open(os.path.join(HERE, "d109_border_normal_form_2026-09-13.json"), encoding="utf-8"))
    out = {"a": [], "b": [], "c": []}

    print("(a) old vs fixed integrator on orbits the defect cannot reach")
    worst = 0.0
    for u in (0.3, 0.5, 0.71):
        for q in (1.5, 1.9, 2.2):
            k = 2 * q / D
            for z in (D / 2, D / 2 + 0.01, D / 2 - 0.01):
                o1 = h.orbit_events(z, u, k, D, u / k)
                o2 = orbit_events_fixed(z, u, k, D, u / k)
                if o1 and o2 and len(o1["events"]) == 2:
                    worst = max(worst, abs(o1["z_next"] - o2["z_next"]))
        for q in (q_ + 0.05 for q_ in (2.1, 2.6)):
            k = 2 * q / D
            s = d26.solve_word(u, k, D)
            if s:
                o1 = h.orbit_events(s["z0"], u, k, D, u / k)
                o2 = orbit_events_fixed(s["z0"], u, k, D, u / k)
                if o1 and o2:
                    worst = max(worst, abs(o1["z_next"] - o2["z_next"]))
    out["a"] = worst
    print("    max |old - fixed| = %.2e" % worst)

    print("\n(b)(c) at the switching points d109 reported as jumps")
    for key in ("border1_plus_end", "border2_pmp_birth"):
        for r in j109[key]:
            if "error" in r or abs(r["dq"] - 3e-3) > 1e-12:
                continue
            u, q, zs = r["u"], r["q"], r["z_s"]
            k = 2 * q / D
            th = u / k
            row = {"border": key, "u": u, "q": q, "z_s": zs}
            vals = {}
            for dz in (-1e-4, -1e-7, 1e-7, 1e-4):
                o_old = h.orbit_events(zs + dz, u, k, D, th)
                o_fix = orbit_events_fixed(zs + dz, u, k, D, th)
                o_bf = orbit_bruteforce(zs + dz, u, k, D)
                vals[dz] = (o_old["z_next"] if o_old else None,
                            o_fix["z_next"] if o_fix else None,
                            o_bf["z_next"] if o_bf else None,
                            "".join("+" if s > 0 else "-" for _, s in o_fix["events"][1:]) if o_fix else None)
            jump_old = abs(vals[1e-7][0] - vals[-1e-7][0]) if vals[1e-7][0] is not None and vals[-1e-7][0] is not None else None
            jump_fix = abs(vals[1e-7][1] - vals[-1e-7][1]) if vals[1e-7][1] is not None and vals[-1e-7][1] is not None else None
            bf_err = max(abs(v[1] - v[2]) for v in vals.values() if v[1] is not None and v[2] is not None)
            row.update({"jump_old": jump_old, "jump_fixed": jump_fix, "fixed_vs_bruteforce": bf_err,
                        "words_fixed": {str(k_): v[3] for k_, v in vals.items()}})
            out["c"].append(row)
            print("  %-18s u=%.2f  jump old %s  jump fixed %s  |fixed - bruteforce| max %.1e  words %s"
                  % (key, u, ("%.2e" % jump_old) if jump_old is not None else None,
                     ("%.2e" % jump_fix) if jump_fix is not None else None, bf_err,
                     [vals[d][3] for d in (-1e-4, -1e-7, 1e-7, 1e-4)]))
    json.dump(out, open(os.path.join(HERE, "d110_orbit_integrator_fix_2026-09-13.json"), "w"), indent=1, default=float)
    print("\n-> d110_orbit_integrator_fix_2026-09-13.json")


if __name__ == "__main__":
    main()
