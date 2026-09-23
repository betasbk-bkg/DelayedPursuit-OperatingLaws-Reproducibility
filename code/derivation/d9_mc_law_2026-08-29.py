# -*- coding: utf-8 -*-
"""
Derivation note -- D9: closed form for the vote transfer function b(m),
and the snap half-width m_c  (NEXT_c_and_mc plan, targets 1 and 5).

Hypothesis under test:      m_c = Delta/2 - a_acc
(measured 19.5 deg = 22.5 - 3, where a_acc is the accurate agents' uniform noise
half-width).  m_c is the intercept of the caustic-edge law m_s(u) = -m_c + c u/k,
so pinning it removes one of the two remaining free quantities in rho(m).

Route: the aggregate is the direction of a vector mean over 150 agents, so for
large N its expectation is exactly computable -- no Monte Carlo needed.  Each
agent class contributes the probability mass its noise distribution puts in each
grid cell:

    E[vote vector] = sum_c w_c * sum_k p_k^c * e_k ,     p_k^c = P(angle in cell k)

with accurate ~ U(m-a, m+a), slow -> delta at m (static probe: prev = ideal),
other ~ U(m-30, m+30), troll uniform over the grid (contributes zero).  Then

    b(m) = arg( E[vote vector] ) - m .

Checks:
  A. the closed form reproduces the engine's Monte-Carlo b(m) (d3 measurement)
  B. m_c(a_acc) over a sweep of the accurate half-width      -> slope -1?
  C. m_c(Delta) over n_dirs in {4, 6, 8, 12, 16}             -> intercept Delta/2?
  D. the analytic peak condition  dphi/dm = 1  at onset

Outputs: d9_results_2026-08-29.json
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# engine composition (simulation_main.gen_votes, N=150, troll=0.05)
N_AG, TROLL = 150, 0.05
A_ACC_DEF, A_OTHER = 3.0, 30.0


def class_counts(N=N_AG, troll=TROLL):
    n_troll = round(N*troll)
    rem = N - n_troll
    n_acc = round(rem*0.7368)
    n_slow = round(rem*0.2105)
    n_other = rem - n_acc - n_slow
    return n_acc, n_slow, n_other, n_troll


def uniform_cell_mass(m, a, delta, kmax=6):
    """mass of U(m-a, m+a) in each grid cell k (cell k spans k*delta +- delta/2)."""
    ks = np.arange(-kmax, kmax+1)
    lo, hi = m-a, m+a
    cl, ch = ks*delta - delta/2, ks*delta + delta/2
    ov = np.clip(np.minimum(hi, ch) - np.maximum(lo, cl), 0, None)
    return ks, ov/(2.0*a)


def b_closed(m, a_acc=A_ACC_DEF, n_dirs=8, N=N_AG, troll=TROLL, a_other=A_OTHER):
    """closed-form snap characteristic, degrees."""
    delta = 360.0/n_dirs
    n_acc, n_slow, n_other, n_troll = class_counts(N, troll)
    vx = vy = 0.0
    # accurate
    ks, p = uniform_cell_mass(m, a_acc, delta)
    ang = np.radians(ks*delta)
    vx += n_acc*float(np.sum(p*np.cos(ang))); vy += n_acc*float(np.sum(p*np.sin(ang)))
    # slow: delta mass at m (static probe) -> nearest cell
    kk = round(m/delta)
    vx += n_slow*math.cos(math.radians(kk*delta)); vy += n_slow*math.sin(math.radians(kk*delta))
    # other
    ks, p = uniform_cell_mass(m, a_other, delta)
    ang = np.radians(ks*delta)
    vx += n_other*float(np.sum(p*np.cos(ang))); vy += n_other*float(np.sum(p*np.sin(ang)))
    # troll: isotropic over the grid -> zero mean
    return (math.degrees(math.atan2(vy, vx)) - m + 180.0) % 360.0 - 180.0


def m_c_of(a_acc=A_ACC_DEF, n_dirs=8, step=0.005):
    """snap half-width = phase of maximal |b| on the positive half cell."""
    delta = 360.0/n_dirs
    ms = np.arange(0.0, delta/2 + 1e-9, step)
    b = np.array([b_closed(m, a_acc, n_dirs) for m in ms])
    i = int(np.argmax(np.abs(b)))
    return float(ms[i]), float(b[i])


if __name__ == "__main__":
    out = {}
    print("[A] closed form vs the engine's Monte-Carlo transfer function")
    d3 = json.load(open(os.path.join(HERE, "d3_results_2026-08-29.json")))["transfer"]
    ms, bm = np.array(d3["m"]), np.array(d3["b"])
    bc = np.array([b_closed(m) for m in ms])
    d = bc - bm
    print(f"    grid points {len(ms)}   max |diff| = {np.abs(d).max():.3f} deg   "
          f"rms = {np.sqrt(np.mean(d**2)):.3f} deg")
    for probe in [5.0, 15.0, 18.0, 20.25, 22.5]:
        print(f"      m={probe:6.2f}   MC {np.interp(probe, ms, bm):+7.2f}   "
              f"closed {b_closed(probe):+7.2f}")
    out["validation"] = dict(max_abs_diff=float(np.abs(d).max()),
                             rms=float(np.sqrt(np.mean(d**2))))

    print("\n[B] m_c versus the accurate agents' half-width a   (n_dirs = 8, Delta/2 = 22.5)")
    print(f"{'a':>6} {'m_c':>8} {'22.5 - a':>10} {'diff':>7} {'|b| at peak':>12}")
    rows = []
    for a in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        mc, bpk = m_c_of(a)
        rows.append(dict(a=a, m_c=mc, pred=22.5-a, b_peak=bpk))
        print(f"{a:6.1f} {mc:8.3f} {22.5-a:10.3f} {mc-(22.5-a):+7.3f} {bpk:12.2f}")
    out["a_sweep"] = rows
    sel = [r for r in rows if r["a"] <= 10.0]
    A = np.array([r["a"] for r in sel]); M = np.array([r["m_c"] for r in sel])
    cf = np.polyfit(A, M, 1)
    print(f"    fit: m_c = {cf[1]:.3f} {cf[0]:+.3f}*a      (hypothesis: 22.500 - 1.000*a)")
    out["a_fit"] = dict(slope=float(cf[0]), intercept=float(cf[1]))

    print("\n[C] m_c versus the command grid   (a = 3 deg)")
    print(f"{'n_dirs':>7} {'Delta':>7} {'m_c':>8} {'Delta/2 - 3':>12} {'diff':>7}")
    rows = []
    for n in [4, 6, 8, 12, 16]:
        mc, bpk = m_c_of(A_ACC_DEF, n)
        pred = 180.0/n - A_ACC_DEF
        rows.append(dict(n_dirs=n, delta=360.0/n, m_c=mc, pred=pred))
        print(f"{n:7d} {360.0/n:7.2f} {mc:8.3f} {pred:12.3f} {mc-pred:+7.3f}")
    out["grid_sweep"] = rows

    print("\n[D] why the peak sits at the onset: dphi/dm at first crossing")
    delta = 45.0
    n_acc, n_slow, n_other, n_troll = class_counts()
    w_acc = n_acc/N_AG
    for a in [3.0, 8.0, 14.0, 20.0]:
        # phi ~ sin(Delta)*F for small F;  F' = w_acc/(2a) at onset
        slope = math.degrees(math.sin(math.radians(delta)))*w_acc/(2*a)
        print(f"    a={a:5.1f}:  dphi/dm at onset = {slope:6.2f}   "
              f"{'peak pinned at onset' if slope > 1 else 'peak smears past onset'}")
    out["onset_slope"] = {str(a): math.degrees(math.sin(math.radians(delta)))*w_acc/(2*a)
                          for a in [3.0, 8.0, 14.0, 20.0]}
    print(f"    -> the law holds while a < {math.degrees(math.sin(math.radians(delta)))*w_acc/2:.1f} deg")

    json.dump(out, open(os.path.join(HERE, "d9_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d9_results_2026-08-29.json")
