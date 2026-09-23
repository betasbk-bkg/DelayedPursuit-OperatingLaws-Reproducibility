# -*- coding: utf-8 -*-
"""
Derivation note -- D18: where the edge law c = k*Delta/2 + a actually holds,
and whether its breakdown is numerical, geometric, or dynamical.

D16/D17 left an unsatisfactory statement: c is linear in k for k = 1.5...5, the
published geometry sits at k = 5, and the pre-registered extrapolation failed at
k = 6.5 (c = -0.04) and k = 8 (c = 2.39).  Calling that "an interpolation, not a
law" is not good enough -- if the published geometry sits just inside a boundary
we have to know where the boundary is and what sets it.

Three questions, each with a control:

  A. NUMERICS.  Is the k = 6.5 breakdown a time-step artefact?  The delay buffer
     holds nd = (u/k)/dt steps, which thins as k grows.  Repeat k = 5 and 6.5 at
     dt = 2e-3 (default), 1e-3 and 5e-4.  If the non-monotone m_s(u) survives,
     it is dynamics, not discretisation.

  B. BOUNDARY.  Scan k on a fine grid with an explicit well-definedness test:
     the edge law is only meaningful where m_s(u) is monotone in u and the linear
     fit is tight.  Report c ONLY where both hold, and locate the boundary.

  C. ORGANISING VARIABLE.  Is the boundary at a critical k, or at a critical
     k*Delta/2 (the chattering number of Sec. 10.4)?  Repeat the scan with a
     16-direction grid (Delta = 22.5 deg, m_c = Delta/2 - 3 = 8.25 deg) and
     compare boundaries in k and in k*Delta/2.

Outputs: d18_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


d9 = _load("d9", "d9_mc_law_2026-08-29.py")
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")

US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]


def b_table(n_dirs=8, n=2001, a_acc=3.0):
    delta = 360.0/n_dirs
    ms = np.linspace(-delta/2, delta/2, n)
    bs = np.array([d9.b_closed(m, a_acc, n_dirs) for m in ms])
    return np.radians(ms), np.radians(bs), math.radians(delta)


def edge_curve(k, grid, dt=2.0e-3, T=900.0, us=US):
    ms, r2s = [], []
    for u in us:
        M, _ = d10.flow(u, k=k, T=T, dt=dt, mgrid=grid)
        ctr, rho = d10.density(M, nbin=180)
        e, r2, pk = d10.caustic_edge(ctr, rho)
        ms.append(e); r2s.append(r2)
    return np.array(ms), np.array(r2s)


def law_fit(us, ms, k, delta_rad):
    """c and a well-definedness verdict."""
    good = ~np.isnan(ms)
    if good.sum() < 5:
        return dict(ok=False, why="too few caustics")
    u = np.array(us)[good]; m = ms[good]
    d = np.diff(m)
    mono = bool(np.all(d > 0))
    cf = np.polyfit(u, m, 1)
    res = m - np.polyval(cf, u)
    rms = float(np.sqrt(np.mean(res**2)))
    c = float(cf[0]/(math.degrees(1.0)/k))
    return dict(ok=bool(mono and rms < 1.0), monotone=mono, rms=rms, c=c,
                intercept=float(cf[1]), a=float(c - k*delta_rad/2),
                why="" if (mono and rms < 1.0) else
                    ("non-monotone" if not mono else "scatter %.2f deg" % rms))


if __name__ == "__main__":
    out = {}
    g8 = b_table(8)
    D8 = g8[2]

    print("[A] numerical control: does the k = 6.5 breakdown survive dt refinement?")
    print(f"{'k':>5} {'dt':>8} " + " ".join(f"{u:>7.2f}" for u in US) + f" {'monotone':>9}")
    rowsA = []
    for k in [5.0, 6.5]:
        for dt in [2.0e-3, 1.0e-3, 5.0e-4]:
            ms, r2s = edge_curve(k, g8, dt=dt, T=600.0)
            mono = bool(np.all(np.diff(ms) > 0))
            rowsA.append(dict(k=k, dt=dt, m_s=ms.tolist(), monotone=mono))
            print(f"{k:5.1f} {dt:8.1e} " + " ".join(f"{v:7.2f}" for v in ms) +
                  f" {str(mono):>9}")
    out["A_dt_control"] = rowsA

    print("\n[B] fine k scan, 8 directions  (Delta = 45 deg)")
    print(f"{'k':>5} {'k*D/2':>7} {'c':>7} {'a':>7} {'rms':>6} {'mono':>6} {'verdict':>16}")
    rowsB = []
    for k in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]:
        ms, r2s = edge_curve(k, g8)
        f = law_fit(US, ms, k, D8)
        f.update(k=k, kD2=k*D8/2, m_s=ms.tolist())
        rowsB.append(f)
        if "c" in f:
            print(f"{k:5.1f} {k*D8/2:7.2f} {f['c']:7.2f} {f['a']:7.3f} {f['rms']:6.2f} "
                  f"{str(f['monotone']):>6} {('OK' if f['ok'] else f['why']):>16}")
        else:
            print(f"{k:5.1f} {k*D8/2:7.2f} {'--':>7} {'--':>7} {'--':>6} {'--':>6} "
                  f"{f['why']:>16}")
    out["B_scan_8dir"] = rowsB
    ok = [r for r in rowsB if r.get("ok")]
    if ok:
        print(f"    law holds for k in [{min(r['k'] for r in ok)}, "
              f"{max(r['k'] for r in ok)}]  i.e. k*Delta/2 in "
              f"[{min(r['kD2'] for r in ok):.2f}, {max(r['kD2'] for r in ok):.2f}]")

    print("\n[C] same scan on a 16-direction grid  (Delta = 22.5 deg)")
    g16 = b_table(16)
    D16 = g16[2]
    print(f"{'k':>5} {'k*D/2':>7} {'c':>7} {'a':>7} {'rms':>6} {'mono':>6} {'verdict':>16}")
    rowsC = []
    for k in [6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0]:
        ms, r2s = edge_curve(k, g16)
        f = law_fit(US, ms, k, D16)
        f.update(k=k, kD2=k*D16/2, m_s=ms.tolist())
        rowsC.append(f)
        if "c" in f:
            print(f"{k:5.1f} {k*D16/2:7.2f} {f['c']:7.2f} {f['a']:7.3f} {f['rms']:6.2f} "
                  f"{str(f['monotone']):>6} {('OK' if f['ok'] else f['why']):>16}")
        else:
            print(f"{k:5.1f} {k*D16/2:7.2f} {'--':>7} {'--':>7} {'--':>6} {'--':>6} "
                  f"{f['why']:>16}")
    out["C_scan_16dir"] = rowsC
    ok16 = [r for r in rowsC if r.get("ok")]
    if ok16:
        print(f"    law holds for k in [{min(r['k'] for r in ok16)}, "
              f"{max(r['k'] for r in ok16)}]  i.e. k*Delta/2 in "
              f"[{min(r['kD2'] for r in ok16):.2f}, {max(r['kD2'] for r in ok16):.2f}]")
        print(f"    a on 16 directions: "
              + ", ".join(f"{r['a']:.3f}" for r in ok16))

    json.dump(out, open(os.path.join(HERE, "d18_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d18_results_2026-08-29.json")
