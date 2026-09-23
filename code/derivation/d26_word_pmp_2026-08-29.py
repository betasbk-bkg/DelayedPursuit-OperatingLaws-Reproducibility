# -*- coding: utf-8 -*-
"""
Derivation note -- D26: the edge law beyond the window, for the word (+,-,+).

Instrumenting the faithful flow (w as state) gives a clean symbolic picture:

    k*Delta/2 < 2.55   ->  word "+"     one forward crossing per period
    k*Delta/2 > 2.55   ->  word "+-+"   two forward, one backward
    in BOTH cases the period is exactly T_p = Delta

so Eq. (12) is the "+"-word solution and the region beyond the window is the
"+-+"-word solution of the same hybrid system.  This closes that word.

Set-up.  Crossings at T = 0 (forward), t1 (backward), t2 (forward), period Delta.
With z' = -1 and jumps +Delta, -Delta, +Delta,

    z(T) = z0 - T           on [0, t1)
           z0 - T - Delta   on [t1, t2)
           z0 - T           on [t2, Delta)

Requiring <z> = 0 over the period (otherwise the lateral error drifts) gives

    z0 = Delta/2 + (t2 - t1)                                           (26.1)

and with it the net advance is automatically Delta, as in the "+" word.  The
phase velocity carries the in-flight boost of Eq. (11),

    m' = (1-u) - k z(T) + k Delta J(T),   J = net jumps in (T - theta_d, T]

so m is piecewise quadratic with breakpoints {0, t1, t2} and {theta_d, t1+theta_d,
t2+theta_d}.  Three crossing conditions

    m(t1) = -Delta/2 ,   m(t2) = +Delta/2 (approached from -Delta/2 after t1),
    m(Delta) = +Delta/2

close the map for (z0, t1, t2); the caustic edge is then the value of m at the
sign change of m' in the free stretch.  Everything here is exact piecewise
polynomial arithmetic plus a 3-variable root find -- no simulation.

Outputs: d26_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0)


def zval(T, z0, t1, t2, delta):
    z = z0 - T
    if t1 <= T < t2:
        z -= delta
    return z


def jnet(T, t1, t2, delta, th):
    """net jumps inside (T - th, T]: crossings at 0 (+), t1 (-), t2 (+), and
    the periodic images at Delta (+)."""
    j = 0
    for tc, s in ((0.0, +1), (t1, -1), (t2, +1), (delta, +1),
                  (-delta + t2, +1), (-delta + t1, -1)):
        if T - th < tc <= T:
            j += s
    return j


def m_profile(z0, t1, t2, u, k, delta, th, n=4000):
    """exact piecewise-quadratic integration of m' on [0, Delta]."""
    brk = sorted(set([0.0, delta, t1, t2, th, t1+th, t2+th]
                     + [x for x in (t1, t2) if 0 < x < delta]))
    brk = [b for b in brk if 0.0 <= b <= delta]
    if brk[0] > 0:
        brk = [0.0] + brk
    if brk[-1] < delta:
        brk = brk + [delta]
    Ts, Ms = [0.0], [-delta/2.0]
    m = -delta/2.0
    for a, b in zip(brk[:-1], brk[1:]):
        if b - a < 1e-12:
            continue
        mid = 0.5*(a+b)
        J = jnet(mid, t1, t2, delta, th)
        # m' = (1-u) - k (z0 - T [- delta]) + k delta J  = A + k T
        off = delta if (t1 <= mid < t2) else 0.0
        A = (1.0-u) - k*(z0 - off) + k*delta*J
        m = m + A*(b-a) + k*(b*b - a*a)/2.0
        Ts.append(b); Ms.append(m)
    return np.array(Ts), np.array(Ms), brk


def residuals(p, u, k, delta, th):
    z0, t1, t2 = p
    Ts, Ms, _ = m_profile(z0, t1, t2, u, k, delta, th)
    m_at = lambda t: float(np.interp(t, Ts, Ms))
    # unwrapped picture: m leaves -Delta/2 upward (boost), falls back through
    # -Delta/2 at t1 (backward crossing), folds below it, rises through -Delta/2
    # at t2 (forward crossing) and reaches +Delta/2 at the end of the period.
    r1 = m_at(t1) + delta/2.0
    r2 = m_at(t2) + delta/2.0
    r3 = m_at(delta) - delta/2.0
    return [r1, r2, r3]


def solve_word(u, k, delta):
    th = u/k
    best = None
    for f1 in (0.25, 0.35, 0.5):
        for f2 in (0.55, 0.7, 0.85):
            p0 = [delta/2 + delta*(f2-f1), delta*f1, delta*f2]
            try:
                sol, info, ier, msg = fsolve(residuals, p0, args=(u, k, delta, th),
                                             full_output=True)
                if ier == 1 and 0 < sol[1] < sol[2] < delta:
                    r = max(abs(x) for x in residuals(sol, u, k, delta, th))
                    if best is None or r < best[1]:
                        best = (sol, r)
            except Exception:
                pass
    if best is None:
        return None
    z0, t1, t2 = best[0]
    Ts, Ms, _ = m_profile(z0, t1, t2, u, k, delta, th)
    # the fold sits in the excursion below -Delta/2; in the wrapped cell it
    # appears at m_s + Delta
    m_fold = float(Ms.min())
    wrapped = m_fold + delta if m_fold < -delta/2 else m_fold
    return dict(z0=z0, t1=t1, t2=t2, resid=best[1],
                z0_check=float(z0 - (delta/2.0 + (t2-t1))),
                m_s=wrapped*DEG, m_fold_unwrapped=m_fold*DEG)


if __name__ == "__main__":
    out = []
    D = math.radians(45.0)
    US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]
    print("word (+,-,+) map versus the flow, Delta = 45 deg")
    for k in (6.5, 8.0):
        print(f"\n  k = {k}  (k*Delta/2 = {k*D/2:.2f})")
        print(f"    {'u':>5} {'m_s map':>9} {'m_s flow':>10} {'diff':>7} "
              f"{'t1/D':>6} {'t2/D':>6} {'resid':>9}")
        ms_map, ms_flow, us = [], [], []
        for u in US:
            s = solve_word(u, k, D)
            M, _ = d10.flow(u, k=k, T=900.0, mgrid=d19.sawtooth(45.0))
            ctr, rho = d10.density(M, nbin=360)
            i = int(np.argmax(rho))
            sel = (ctr > ctr[i]) & (ctr <= ctr[i]+1.5) & (rho > 0)
            fl = float("nan")
            if sel.sum() >= 4:
                cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
                fl = float(-cf[1]/cf[0])
            if s:
                us.append(u); ms_map.append(s["m_s"]); ms_flow.append(fl)
                print(f"    {u:5.2f} {s['m_s']:9.2f} {fl:10.2f} {s['m_s']-fl:+7.2f} "
                      f"{s['t1']/D:6.3f} {s['t2']/D:6.3f} {s['resid']:9.1e}"
                      f"  <z>=0 check {s['z0_check']:+.3f}")
                out.append(dict(k=k, u=u, m_s_map=s["m_s"], m_s_flow=fl,
                                t1=s["t1"], t2=s["t2"], resid=s["resid"]))
            else:
                print(f"    {u:5.2f}   no root")
        if len(us) >= 3:
            d = np.array(ms_map) - np.array(ms_flow)
            print(f"    mean offset {np.nanmean(d):+.2f} deg, spread {np.nanstd(d):.2f}")
    json.dump(out, open(os.path.join(HERE, "d26_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d26_results_2026-08-29.json")
