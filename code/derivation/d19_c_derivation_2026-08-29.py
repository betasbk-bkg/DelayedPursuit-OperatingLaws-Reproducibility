# -*- coding: utf-8 -*-
"""
Derivation note -- D19: closed-form derivation of the caustic-edge law,
including the coefficient a.

The failed return map of D13 dropped one term.  With the pure sawtooth (b = -m
on the whole cell, justified empirically by D14/D15: c is identical for the
sawtooth and for the true b) the loop is a piecewise-linear HYBRID system:

    z' = -1                     exactly, everywhere (b' = -1)
    m' = 1 - k z(T - theta_d)
    at m = +Delta/2:  m -> m - Delta,  z -> z + Delta        (forward crossing)
    at m = -Delta/2:  m -> m + Delta,  z -> z - Delta        (backward crossing)

The term D13 missed: because z is read DELAYED, a crossing that has happened
within the last theta_d is still "in flight" -- the jump has hit z but not yet
the m equation.  Writing J(T) for the net number of jumps inside (T-theta_d, T],

    z(T - theta_d) = z(T) + theta_d - Delta * J(T)
    ==>  m' = (1 - u) - k z(T) + k Delta J(T) ,      u = k theta_d          (*)

so for a time theta_d after every crossing the phase velocity is boosted by
k*Delta.  That boost is the entire delay dependence.

DERIVATION (one forward crossing per period; period T_p = Delta exactly, since
z falls at rate 1 and regains Delta per net crossing).  Put T = 0 just after a
crossing, so m(0) = -Delta/2, z(0) = z0, z(T) = z0 - T.

  advance over one period must equal Delta:
      int_0^Delta [(1-u) - k(z0 - T)] dT  +  k*Delta*theta_d  =  Delta
      (1-u) - k z0 + k Delta/2 + u = 1        ==>       z0 = Delta/2         (1)

  the fold is m' = 0.  While the jump is in flight m' >= kDelta - ... > 0, so the
  fold lies in the free stretch:
      T* = z0 - (1-u)/k = Delta/2 - (1-u)/k ,   which needs  T* > theta_d
      <=>  k Delta / 2 > 1                                                  (2)

  evaluating m at T*:
      m_s(u) = -k Delta^2 / 8  -  (1-u)^2 / (2k)  +  u Delta / 2            (3)

  hence the edge law and the coefficient:
      dm_s/du = Delta/2 + (1-u)/k      ==>      c = k Delta/2 + (1 - u)     (4)

So "a" is not a constant: a = 1 - u.  Fitting a straight line over a window of u
returns a = 1 - <u>, which is why D16/D18 measured 0.30-0.40 on a window with
<u> = 0.60.

  validity:  lower  k Delta/2 > 1                      [fold after the boost]
             upper  m_s(u_min) > -Delta/2              [no backward crossing]   (5)

Both boundaries are tested against the measured domain of D18.

Outputs: d19_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d10", os.path.join(HERE, "d10_delayed_flow_2026-08-29.py"))
d10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d10)
DEG = math.degrees(1.0)


def sawtooth(delta_deg, n=4001):
    ms = np.linspace(-delta_deg/2, delta_deg/2, n)
    return np.radians(ms), np.radians(-ms), math.radians(delta_deg)


def m_s_closed(u, k, delta):
    """equation (3), radians"""
    return -k*delta*delta/8.0 - (1.0-u)**2/(2.0*k) + u*delta/2.0


def c_closed(u, k, delta):
    """equation (4)"""
    return k*delta/2.0 + (1.0 - u)


if __name__ == "__main__":
    out = {}
    US = [0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]

    print("[A] closed form (3) vs the pure-sawtooth flow -- same model, no fitting")
    rows = []
    for delta_deg, ks in [(45.0, [3.0, 4.0, 5.0, 5.5]), (22.5, [6.0, 8.0, 10.0, 12.0])]:
        grid = sawtooth(delta_deg)
        D = math.radians(delta_deg)
        for k in ks:
            print(f"\n  Delta = {delta_deg} deg,  k = {k},  k*Delta/2 = {k*D/2:.2f}")
            print(f"    {'u':>5} {'m_s flow':>10} {'m_s eq(3)':>10} {'diff':>7}")
            fl, cl = [], []
            for u in US:
                M, _ = d10.flow(u, k=k, T=900.0, mgrid=grid)
                ctr, rho = d10.density(M, nbin=180)
                e, r2, pk = d10.caustic_edge(ctr, rho)
                pr = m_s_closed(u, k, D)*DEG
                fl.append(e); cl.append(pr)
                print(f"    {u:5.2f} {e:10.2f} {pr:10.2f} {e-pr:+7.2f}")
            fl, cl = np.array(fl), np.array(cl)
            sl_f = np.polyfit(US, fl, 1)[0]
            sl_c = np.polyfit(US, cl, 1)[0]
            off = float(np.mean(fl-cl))
            rows.append(dict(delta=delta_deg, k=k, kD2=k*D/2,
                             slope_flow=float(sl_f), slope_closed=float(sl_c),
                             c_flow=float(sl_f*k/DEG), c_closed=float(sl_c*k/DEG),
                             offset=off, spread=float(np.std(fl-cl))))
            print(f"    slope: flow {sl_f:7.2f}  eq(3) {sl_c:7.2f}   "
                  f"-> c flow {sl_f*k/DEG:.3f}  c eq(4) {sl_c*k/DEG:.3f}   "
                  f"({100*(sl_f-sl_c)/sl_c:+.1f}%)")
            print(f"    constant offset {off:+.2f} deg (spread {np.std(fl-cl):.2f})")
    out["A_curves"] = rows
    err = [abs(r["c_flow"]-r["c_closed"])/r["c_closed"] for r in rows]
    print(f"\n  c: closed form vs flow, mean |err| = {100*np.mean(err):.1f}%, "
          f"max {100*max(err):.1f}%")

    print("\n[B] the fitted constant a is 1 - <u>")
    ub = float(np.mean(US))
    print(f"    window mean <u> = {ub:.4f}  ->  a_predicted = 1 - <u> = {1-ub:.4f}")
    print(f"    measured a (D18): 8 dirs 0.357-0.395, 16 dirs 0.246-0.392")
    out["a_pred"] = 1-ub

    print("\n[C] validity boundaries from (5) vs the measured domain of D18")
    print(f"    lower bound: k*Delta/2 > 1   (D18 lowest tested valid point 1.18)")
    print(f"    {'Delta':>7} {'k':>6} {'k*D/2':>7} {'m_s(u=0.3)':>11} {'-Delta/2':>9} "
          f"{'predicted':>10}")
    rowsC = []
    for delta_deg, ks in [(45.0, [4.0, 5.0, 5.5, 6.0, 6.5]),
                          (22.5, [10.0, 11.0, 12.0, 13.0, 14.0])]:
        D = math.radians(delta_deg)
        for k in ks:
            ms = m_s_closed(0.30, k, D)*DEG
            ok = ms > -delta_deg/2
            rowsC.append(dict(delta=delta_deg, k=k, kD2=k*D/2, m_s=ms,
                              bound=-delta_deg/2, predicted_ok=bool(ok)))
            print(f"    {delta_deg:7.1f} {k:6.1f} {k*D/2:7.2f} {ms:11.2f} "
                  f"{-delta_deg/2:9.2f} {'OK' if ok else 'BREAK':>10}")
    out["C_boundary"] = rowsC
    print("    D18 measured: 8 dirs last valid k = 5.5, first broken 6.0;")
    print("                  16 dirs last valid k = 12.0, first broken 14.0")

    json.dump(out, open(os.path.join(HERE, "d19_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d19_results_2026-08-29.json")
