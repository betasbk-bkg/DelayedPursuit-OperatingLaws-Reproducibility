# -*- coding: utf-8 -*-
"""
Derivation note -- D27: which periodic branch does the flow select?

D26 closed the (+,-,+) map algebraically but found several periodic solutions and
no way to pick the one the flow realises.  The missing ingredient is stability.

Poincare section.  Take the instant of a forward boundary crossing.  There m is
pinned to the boundary, so the only state left is z -- provided no earlier
crossing still sits inside the delay window, which is checked per case.  The
return map is therefore ONE-DIMENSIONAL,

    z  |-->  P(z),

and the branch the flow selects is the fixed point with |P'(z*)| < 1.

P is evaluated by exact piecewise integration of the hybrid system

    z' = -1,   m' = (1-u) - k z(T) + k*Delta*J(T),   J = net jumps in (T-theta_d, T]

with boundaries at -Delta/2 + n*Delta, z -> z +- Delta at each crossing, and
events located by solving the segment quadratics.  No time stepping, no
simulation of the loop.

Reported per (k, u): every fixed point of P, its multiplier P'(z*), the word it
realises, its caustic edge, and the edge measured from the flow.

Outputs: d27_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
DEG = math.degrees(1.0)


def orbit(z0, u, k, delta, th, max_ev=12):
    """integrate from a forward crossing (m = -delta/2, z = z0) to the next
    forward crossing one cell up.  Exact between events."""
    m = -delta/2.0
    z = z0
    T = 0.0
    events = [(0.0, +1)]          # the crossing that defines the section
    word = "+"
    m_min = None
    guard = 0
    while guard < max_ev:
        guard += 1
        # next breakpoint from the delay window (an event leaving it)
        cand = [te + th for te, _ in events if te + th > T + 1e-14]
        T_bp = min(cand) if cand else float("inf")
        J = sum(s for te, s in events if T - th < te <= T)
        A = (1.0 - u) - k*(z + T) + k*delta*J      # z(T') = z - (T' - T) -> z + T - T'
        # m'(T') = A + k*T'   with z(T') = z0' - T', z0' = z + T
        # solve m(T') = boundary for the two boundaries adjacent to m
        lo, hi = -delta/2.0, delta/2.0
        while m > hi + 1e-12:
            lo += delta; hi += delta
        while m < lo - 1e-12:
            lo -= delta; hi -= delta
        t_hit, which = None, 0
        for bnd, w in ((hi, +1), (lo, -1)):
            # m(T') = m + A(T'-T) + k(T'^2 - T^2)/2 = bnd
            a = k/2.0
            b = A
            c = m - bnd - A*T - k*T*T/2.0
            disc = b*b - 4*a*c
            if disc < 0:
                continue
            r = math.sqrt(disc)
            for root in ((-b - r)/(2*a), (-b + r)/(2*a)):
                if root > T + 1e-12 and (t_hit is None or root < t_hit):
                    t_hit, which = root, w
        if t_hit is None or t_hit > T_bp:
            # advance to the breakpoint, no event
            if not np.isfinite(T_bp):
                return None
            Tn = T_bp
            mn = m + A*(Tn - T) + k*(Tn*Tn - T*T)/2.0
            # fold inside this stretch?
            tf = -A/k
            if T < tf < Tn:
                mf = m + A*(tf-T) + k*(tf*tf - T*T)/2.0
                m_min = mf if m_min is None else min(m_min, mf)
            z -= (Tn - T); m = mn; T = Tn
            continue
        Tn = t_hit
        tf = -A/k
        if T < tf < Tn:
            mf = m + A*(tf-T) + k*(tf*tf - T*T)/2.0
            m_min = mf if m_min is None else min(m_min, mf)
        m = m + A*(Tn - T) + k*(Tn*Tn - T*T)/2.0
        z -= (Tn - T)
        T = Tn
        z += which*delta
        events.append((T, which))
        word += "+" if which > 0 else "-"
        if which > 0 and m > -delta/2.0 + delta - 1e-9:
            gaps = [events[i+1][0]-events[i][0] for i in range(len(events)-1)]
            return dict(z_next=z, period=T, word=word[1:], m_min=m_min,
                        min_gap=min(gaps) if gaps else float("inf"), th=th)
    return None


def fixed_points(u, k, delta, n=1200):
    th = u/k
    zs = np.linspace(-0.2*delta, 1.4*delta, n)
    vals = []
    for z in zs:
        o = orbit(z, u, k, delta, th)
        vals.append(o["z_next"] - z if o else float("nan"))
    vals = np.array(vals)
    out = []
    for i in range(len(zs)-1):
        a, b = vals[i], vals[i+1]
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0 or np.sign(a) == np.sign(b):
            continue
        zl, zr = zs[i], zs[i+1]
        for _ in range(80):
            zm = 0.5*(zl+zr)
            o = orbit(zm, u, k, delta, th)
            if o is None:
                break
            f = o["z_next"] - zm
            if np.sign(f) == np.sign(a):
                zl, a = zm, f
            else:
                zr = zm
        zst = 0.5*(zl+zr)
        o = orbit(zst, u, k, delta, th)
        if o is None:
            continue
        h = 1e-6
        op, om = orbit(zst+h, u, k, delta, th), orbit(zst-h, u, k, delta, th)
        if op is None or om is None:
            continue
        mult = (op["z_next"] - om["z_next"])/(2*h)
        if o["m_min"] is None:
            continue
        wrapped = o["m_min"]
        while wrapped < -delta/2:
            wrapped += delta
        while wrapped > delta/2:
            wrapped -= delta
        out.append(dict(z=zst, mult=float(mult), word=o["word"],
                        period_ratio=o["period"]/delta, m_s=wrapped*DEG,
                        gap_ok=bool(o["min_gap"] > o["th"])))
    return out


def flow_edge(u, k, delta_deg=45.0):
    M, _ = d10.flow(u, k=k, T=900.0, mgrid=d19.sawtooth(delta_deg))
    ctr, rho = d10.density(M, nbin=360)
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+1.5) & (rho > 0)
    if sel.sum() < 4:
        return float("nan")
    cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
    return float(-cf[1]/cf[0])


if __name__ == "__main__":
    D = math.radians(45.0)
    out = []
    print("fixed points of the 1-D return map, and which one is stable")
    for k in (5.0, 6.5, 8.0):
        print(f"\n  k = {k}  (k*Delta/2 = {k*D/2:.2f})")
        print(f"    {'u':>5} {'word':>6} {'mult':>8} {'stable':>7} {'m_s map':>9} "
              f"{'m_s flow':>9} {'diff':>7} {'gap>th':>7}")
        for u in (0.30, 0.50, 0.71, 0.90):
            fps = fixed_points(u, k, D)
            fl = flow_edge(u, k)
            if not fps:
                print(f"    {u:5.2f}   no fixed point")
                continue
            fps = [f for f in fps if abs(f["mult"]) < 5.0 and f["word"] in ("+", "-++", "+-+")]
            for fp in fps:
                st = abs(fp["mult"]) < 1.0
                mark = "  <-- " if st else "      "
                print(f"    {u:5.2f} {fp['word']:>6} {fp['mult']:8.3f} "
                      f"{str(st):>7} {fp['m_s']:9.2f} {fl:9.2f} "
                      f"{fp['m_s']-fl:+7.2f} {str(fp['gap_ok']):>7}{mark}")
                out.append(dict(k=k, u=u, flow_edge=fl, **fp))
    st = [r for r in out if abs(r["mult"]) < 1.0 and np.isfinite(r["flow_edge"])]
    if st:
        d = [r["m_s"]-r["flow_edge"] for r in st]
        print(f"\nstable branches only: n = {len(st)}, mean diff {np.mean(d):+.2f} deg, "
              f"spread {np.std(d):.2f} deg")
    json.dump(out, open(os.path.join(HERE, "d27_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d27_results_2026-08-29.json")
