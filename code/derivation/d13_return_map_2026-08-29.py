# -*- coding: utf-8 -*-
"""
Derivation note -- D13: the caustic-edge coefficient c from an exact
return map  (NEXT_c_and_mc plan, target 2, route B -- the analytic side).

D11 pinned c = 2.32 numerically on the linear branch but could not derive it:
the first-order perturbation of the stall condition gave slope -2 against a
measured -1.41.  This script replaces the perturbation with an EXACT treatment
of the snap region plus a piecewise-linear caricature of the return region, and
closes the orbit with a return map.

Exact result used throughout (no perturbation).  In the snap region b'(m) = -1
exactly, so

    z' = -w'(T - th) + b' m' = -k z(T-th) + (-1)(1 - k z(T-th)) = -1     EXACT

hence z(T - th) = z(T) + th exactly, and therefore

    m' = 1 - k z(T - th) = (1 - u) - k z ,        k*th = u                (S1)

so the snap-region orbit is the parabola

    m(z) = C - (1-u) z + (k/2) z^2 ,   C = m0 + (1-u) z0 - (k/2) z0^2     (S2)

with a fold at z* = (1-u)/k and edge

    m_s = C - (1-u)^2 / (2k) .                                            (S3)

The delay therefore enters twice: explicitly through (1-u)^2 in (S3), and
implicitly through C, which the return map fixes.  Caricature of the return
region (m_c < |m| < Delta/2): b linear with slope s = (m_c - b_e)/(Delta/2 - m_c),
where b_e = |b| at the cell boundary.  While the transit is shorter than the
delay -- true here -- the delayed argument still lies in the snap region where
z' = -1, so the return region integrates in closed form too:

    m(T) = m_c + (1 - u - k z_e) T + (k/2) T^2
    z(T) = z_e + s T - k(1+s) [ (z_e + th) T - T^2/2 ]

Closing the loop (boundary jump z -> z + 2 b_e, m -> m - Delta) gives a scalar
fixed point in z_e, hence C(u), hence m_s(u) and c = k * d m_s / du.

Outputs: d13_results_2026-08-29.json
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEG = math.degrees(1.0)
K = 5.0
DELTA = math.radians(45.0)
M_C = math.radians(19.5)          # D9 closed form: Delta/2 - a_acc


def return_transit(z_e, u, k, m_c, delta, s, th):
    """closed-form transit of the return region; returns (T_r, z_out, ok)."""
    A = (k/2.0)
    B = (1.0 - u - k*z_e)
    C = m_c - delta/2.0
    disc = B*B - 4*A*C
    if disc < 0:
        return None, None, False
    T_r = (-B + math.sqrt(disc))/(2*A)
    if T_r <= 0:
        T_r = (-B - math.sqrt(disc))/(2*A)
    if T_r <= 0:
        return None, None, False
    z_out = z_e + s*T_r - k*(1.0+s)*((z_e + th)*T_r - T_r*T_r/2.0)
    return T_r, z_out, (T_r <= th*1.5)


def snap_traverse(z_in, u, k, m0, m_end):
    """parabola (S2) from (m0, z_in) to m_end on the post-fold branch."""
    C = m0 + (1.0-u)*z_in - (k/2.0)*z_in*z_in
    disc = (1.0-u)**2 - 2*k*(C - m_end)
    if disc < 0:
        return None, C
    z_out = ((1.0-u) - math.sqrt(disc))/k
    return z_out, C


def orbit(u, k=K, m_c=M_C, delta=DELTA, b_e=0.0, iters=400, tol=1e-12):
    """fixed point of the one-cell return map; returns m_s and diagnostics."""
    s = (m_c - b_e)/(delta/2.0 - m_c)
    th = u/k
    z_e = 0.0
    for it in range(iters):
        T_r, z_out, ok = return_transit(z_e, u, k, m_c, delta, s, th)
        if T_r is None:
            return None
        z_next = z_out + 2.0*b_e
        z_new, C = snap_traverse(z_next, u, k, -delta/2.0, m_c)
        if z_new is None:
            return None
        if abs(z_new - z_e) < tol:
            z_e = z_new
            break
        z_e = 0.5*z_e + 0.5*z_new          # damped iteration
    T_r, z_out, ok = return_transit(z_e, u, k, m_c, delta, s, th)
    z_next = z_out + 2.0*b_e
    _, C = snap_traverse(z_next, u, k, -delta/2.0, m_c)
    m_s = C - (1.0-u)**2/(2.0*k)
    return dict(u=u, z_e=z_e, C=C, m_s_deg=m_s*DEG, T_r=T_r, theta_d=th,
                short_transit=bool(ok), s_ret=s)


def slope_c(us, ms, k=K):
    cf = np.polyfit(us, ms, 1)
    return float(cf[0]/(DEG/k)), float(cf[1]), float(cf[0])


if __name__ == "__main__":
    out = {"k": K, "m_c_deg": M_C*DEG}
    d10 = json.load(open(os.path.join(HERE, "d10_results_2026-08-29.json")))
    flow = {round(r["u"], 2): r["m_s"] for r in d10["rows"]}

    print("[A] exact snap-region result: stall at k z = 1 - u  (not the")
    print("    perturbative 1 - 2u).  Checked against D10's turning points:")
    for r in d10["rows"]:
        if r.get("n_turns"):
            print(f"    u={r['u']:4.2f}   k z0 measured {r['kz0_turn']:+7.3f}   "
                  f"1-u = {1-r['u']:+7.3f}   1-2u = {1-2*r['u']:+7.3f}")
    S = [(r["u"], r["kz0_turn"]) for r in d10["rows"] if r.get("n_turns") and r["u"] >= 0.3]
    U = np.array([x[0] for x in S]); Z = np.array([x[1] for x in S])
    cf = np.polyfit(U, Z, 1)
    print(f"    measured slope {cf[0]:+.2f}   exact (S1) -1   perturbation -2")
    out["stall_slope_measured"] = float(cf[0])

    print("\n[B] return-map orbit vs the numerical flow")
    for b_e_deg in [0.0, 2.5, 5.0]:
        b_e = math.radians(b_e_deg)
        us, ms, fl = [], [], []
        print(f"  b_e = {b_e_deg:4.1f} deg   (return slope s = "
              f"{(M_C-b_e)/(DELTA/2-M_C):.2f})")
        for u in [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]:
            o = orbit(u, b_e=b_e)
            if o is None:
                print(f"    u={u:4.2f}  no closed orbit")
                continue
            us.append(u); ms.append(o["m_s_deg"]); fl.append(flow.get(round(u, 2)))
            print(f"    u={u:4.2f}  m_s = {o['m_s_deg']:+7.2f} deg   "
                  f"flow {flow.get(round(u,2)):+7.2f}   diff "
                  f"{o['m_s_deg']-flow.get(round(u,2)):+6.2f}   "
                  f"T_r/theta_d = {o['T_r']/o['theta_d']:.2f}")
        if len(us) >= 3:
            c, A, B = slope_c(np.array(us), np.array(ms))
            cflow, _, _ = slope_c(np.array(us), np.array([f for f in fl]))
            print(f"    -> c(return map) = {c:.2f}    c(flow) = {cflow:.2f}")
            out.setdefault("branches", []).append(
                dict(b_e_deg=b_e_deg, c_map=c, c_flow=cflow,
                     m_s=ms, u=us, flow=fl))

    print("\n[C] where the u-dependence lives")
    b_e = 0.0
    for u in [0.30, 0.60, 0.90]:
        o = orbit(u, b_e=b_e)
        expl = -(1.0-u)**2/(2.0*K)*DEG
        print(f"    u={u:4.2f}   C = {o['C']*DEG:+7.2f} deg   "
              f"explicit -(1-u)^2/2k = {expl:+6.2f} deg   m_s = {o['m_s_deg']:+7.2f}")
    o1, o2 = orbit(0.3, b_e=b_e), orbit(0.9, b_e=b_e)
    dC = (o2["C"]-o1["C"])*DEG/0.6
    dexp = ((-(1-0.9)**2 + (1-0.3)**2)/(2*K))*DEG/0.6
    print(f"    dC/du = {dC:+.2f} deg   d(explicit)/du = {dexp:+.2f} deg  "
          f"-> c_C = {dC/(DEG/K):.2f}, c_explicit = {dexp/(DEG/K):.2f}")
    out["split"] = dict(dC_du_deg=float(dC), d_explicit_du_deg=float(dexp))

    json.dump(out, open(os.path.join(HERE, "d13_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d13_results_2026-08-29.json")
