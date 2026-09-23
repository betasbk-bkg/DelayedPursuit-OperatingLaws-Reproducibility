# -*- coding: utf-8 -*-
"""
Derivation note -- D10: the delayed cylinder flow in continuous time, and
the caustic-edge coefficient c  (NEXT_c_and_mc plan, target 2, route A).

D8 left m_s(u) = -m_c + c*u/k with c bracketed only to [1.3, 2.4], because the
u = 0 point came from a continuous-time flow while the u > 0 points came from a
ZOH-sampled stochastic surrogate: two different estimators.  This script removes
that mismatch by integrating the DELAYED flow in continuous time, so every u --
including u = 0 -- is measured the same way.

Derivation of the delayed flow.  Keeping the delay in both channels of the aim
angle (the delayed arc, which sets the path direction, and the delayed lateral
error, which sets the feedback):

    w'(t)  = a z(t),      z(t) = c0 - w(t - tau_tot) + b(m(t))
    m'(t)  = w0 - a z(t - tau_tot)

Non-dimensionalising time by w0 = v/R, with k = R/L, the delay becomes
theta_d = w0 tau_tot = u/k and the Theorem-1 offset becomes c0 = (1/2 - u)/k, so

    dw/dT = k z(T),       z(T) = (1/2 - u)/k - w(T - u/k) + b(m(T))
    dm/dT = 1 - k z(T - u/k)

and the only parameters left are k and u -- the similarity law of Sec. IV-F,
now visible directly in the equations.  At u = 0 this reduces to the undelayed
flow of D7.

b(m) is taken from the CLOSED FORM of D9 (validated against the engine to
0.025 deg rms), so the flow contains no measured curve and no fitted constant.

Outputs: d10_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d9", os.path.join(HERE, "d9_mc_law_2026-08-29.py"))
d9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d9)

DELTA = 45.0
K = 5.0                      # R/L for the published circle (R = 10, L = 2)
M_C = DELTA/2 - 3.0          # D9 closed form: Delta/2 - a_acc


def b_table(n=2001, a_acc=3.0, n_dirs=8):
    """closed-form b on a fine grid, radians in and out."""
    delta = 360.0/n_dirs
    ms = np.linspace(-delta/2, delta/2, n)
    bs = np.array([d9.b_closed(m, a_acc, n_dirs) for m in ms])
    return np.radians(ms), np.radians(bs), math.radians(delta)


def flow(u, k=K, T=1500.0, dt=2.0e-3, burn=0.35, mgrid=None):
    """integrate the delayed cylinder flow (method of steps, full history).
    Returns the sampled phase in degrees and the turning points (m, z0, z),
    where z0 = c0 - w(T) + b(m(T)) is the UNDELAYED tracking error.
    Inner loop uses direct index arithmetic on a uniform b-table (no np.interp)."""
    mm, bb, D = mgrid
    nb = len(mm)
    step = (mm[-1] - mm[0])/(nb - 1)
    m0 = mm[0]
    bl = bb.tolist()
    theta_d = u/k
    nd = max(0, int(round(theta_d/dt)))
    n = int(T/dt)
    W = [0.0]*(n+1); Z = [0.0]*(n+1)
    M = np.empty(n+1)
    m = 0.0; w = 0.0
    c0 = (0.5 - u)/k
    prev_dm = None
    turns = []
    half = D/2
    deg = 180.0/math.pi
    for i in range(n):
        mw = ((m + half) % D) - half
        x = (mw - m0)/step
        j = int(x)
        if j < 0: j = 0
        elif j > nb-2: j = nb-2
        f = x - j
        bv = bl[j]*(1.0-f) + bl[j+1]*f
        wd = W[i-nd] if i >= nd else 0.0
        z = c0 - wd + bv
        z0 = c0 - w + bv
        zd = Z[i-nd] if i >= nd else 0.0
        dm = 1.0 - k*zd
        if prev_dm is not None and prev_dm < 0.0 <= dm and i > 0.4*n:
            turns.append((mw*deg, z0, z))
        prev_dm = dm
        w += dt*k*z
        m += dt*dm
        W[i+1] = w; Z[i+1] = z
        M[i+1] = (((m + half) % D) - half)*deg
    keep = int(burn*n)
    return M[keep:], turns


def density(M, nbin=180):
    e = np.linspace(-DELTA/2, DELTA/2, nbin+1)
    c, _ = np.histogram(M, bins=e)
    return 0.5*(e[1:]+e[:-1]), c/max(c.sum(), 1)


def caustic_edge(ctr, rho, width=8.0):
    """fold law: rho^-2 is linear in m on the forward branch; its zero is m_s."""
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+width) & (rho > 0)
    if sel.sum() < 5:
        return float('nan'), float('nan'), float(ctr[i])
    y = rho[sel]**-2.0
    cf = np.polyfit(ctr[sel], y, 1)
    r2 = 1 - np.sum((y-np.polyval(cf, ctr[sel]))**2)/np.sum((y-y.mean())**2)
    return float(-cf[1]/cf[0]), float(r2), float(ctr[i])


if __name__ == "__main__":
    grid = b_table()
    print(f"k = {K},  m_c (D9 closed form) = {M_C:.2f} deg")
    print("\ncontinuous-time delayed flow: caustic edge versus delay margin")
    print(f"{'u':>6} {'theta_d':>8} {'peak':>8} {'m_s':>8} {'R^2':>7} "
          f"{'-m_c + 2u/k':>12} {'c implied':>10}")
    rows = []
    for u in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.71, 0.80, 0.90]:
        M, turns = flow(u, mgrid=grid)
        ctr, rho = density(M)
        ms_, r2, pk = caustic_edge(ctr, rho)
        pred2 = -M_C + math.degrees(2*u/K)
        c_imp = (ms_ + M_C)/(u/K)/math.degrees(1.0) if u > 0 else float('nan')
        kz0 = K*float(np.mean([t[1] for t in turns])) if turns else float("nan")
        rows.append(dict(u=u, theta_d=u/K, peak=pk, m_s=ms_, r2=r2,
                         pred_c2=pred2, c_implied=c_imp, kz0_turn=kz0,
                         kz0_pred=1-2*u, n_turns=len(turns)))
        print(f"{u:6.2f} {u/K:8.4f} {pk:8.2f} {ms_:8.2f} {r2:7.4f} {pred2:12.2f} "
              f"{c_imp:10.3f}   kz0(turn)={kz0:+.3f} vs 1-2u={1-2*u:+.3f} "
              f"[n={len(turns)}]")

    ok = [r for r in rows if r["r2"] > 0.9 and not math.isnan(r["m_s"])]
    if len(ok) >= 3:
        U = np.array([r["u"] for r in ok]); MS = np.array([r["m_s"] for r in ok])
        cf = np.polyfit(U, MS, 1)
        c_fit = cf[0]/math.degrees(1.0/K)
        print(f"\nlinear fit over {len(ok)} usable points (R^2 > 0.9):")
        print(f"  m_s(u) = {cf[1]:+.3f} {cf[0]:+.3f}*u  [deg]")
        print(f"  intercept vs -m_c = {-M_C:+.3f}   diff {cf[1]+M_C:+.3f} deg")
        print(f"  slope -> c = {c_fit:.3f}   (two-channel argument: 2)")
    else:
        cf = [float('nan')]*2; c_fit = float('nan')
        print("\nnot enough clean caustics for a fit")

    json.dump(dict(k=K, m_c=M_C, rows=rows,
                   fit_intercept=float(cf[1]), fit_slope=float(cf[0]),
                   c_fit=float(c_fit)),
              open(os.path.join(HERE, "d10_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d10_results_2026-08-29.json")
