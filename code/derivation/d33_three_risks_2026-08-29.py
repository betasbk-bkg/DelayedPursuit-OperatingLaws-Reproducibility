# -*- coding: utf-8 -*-
"""
Derivation note -- D33: the three residual risks of Sec. 19.3.

RISK 1 -- the +0.37 deg intercept offset (Sec. 14.1).  Decisive test: build the
density ANALYTICALLY from the exact orbit of Sec. 11.2, where the caustic edge is
known exactly, and run the SAME estimator on it.  If the estimator returns
m_s + 0.37 on a density whose edge is known, the offset is estimator bias and
Eq. (12) is exact; if it returns m_s, the offset is a real term the derivation
is missing.

RISK 2 -- sigma_theta's 7-8% error, attributed to the orbit being noise-free so
its caustic is sharper than the real one.  The broadening is not free: the
residual vote spread s perturbs the aggregate by s, and quasi-statically a
persistent angular error q costs a lateral offset y = L q, so the aim angle -- and
hence the in-cell phase -- is displaced by y/L = s.  So the phase is smeared by
sigma_m = s_bar, a DERIVED width (1.06 deg), with nothing fitted.  Convolve and
re-evaluate.

RISK 3 -- everything is checked against one engine.  Implementation independence
is testable: a from-scratch simulator of the same model (exact 2-D circle,
midpoint integration instead of Euler, its own delay buffer and its own vote
aggregation) must reproduce sigma_theta(u) and the density peak.  Model-versus-
reality is out of reach here and stays the paper's stated limitation.

Outputs: d33_results_2026-08-29.json
"""
import importlib.util, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
d12 = _load("d12", "d12_sigma_closed_2026-08-29.py")
d19 = _load("d19", "d19_c_derivation_2026-08-29.py")
d20 = _load("d20", "d20_sigma_analytic_2026-08-29.py")
d32 = _load("d32", "d32_independence_audit_2026-08-29.py")

DEG = math.degrees(1.0)
K, DELTA_DEG = 5.0, 45.0
D = math.radians(DELTA_DEG)
US = (0.40, 0.50, 0.60, 0.71, 0.80, 0.90)


def analytic_samples(u, k, delta, n=400000):
    """m(T) sampled uniformly in time over one period -- the exact orbit."""
    cA, cB, th = d20.orbit_polys(u, k, delta)
    nA = max(int(n*th/delta), 200)
    TA = np.linspace(0.0, th, nA)
    TB = np.linspace(th, delta, n-nA)
    mA = cA[0] + cA[1]*TA + cA[2]*TA**2
    mB = cB[0] + cB[1]*TB + cB[2]*TB**2
    m = np.degrees(np.concatenate([mA, mB]))
    return ((m + delta*DEG/2) % (delta*DEG)) - delta*DEG/2


# --------------------------------------------------------------------- RISK 1
def risk1():
    print("[RISK 1] estimator bias: same estimator on an EXACT density")
    print(f"    {'u':>5} {'m_s exact':>10} {'estimator':>10} {'bias':>7} "
          f"{'flow-eq(12)':>12}")
    rows = []
    for u in (0.30, 0.50, 0.71, 0.90):
        ms = analytic_samples(u, K, D)
        ctr, rho = d10.density(ms, nbin=360)
        i = int(np.argmax(rho))
        sel = (ctr > ctr[i]) & (ctr <= ctr[i]+1.5) & (rho > 0)
        est = float('nan')
        if sel.sum() >= 4:
            cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
            est = float(-cf[1]/cf[0])
        exact = d19.m_s_closed(u, K, D)*DEG
        M, _ = d10.flow(u, k=K, T=900.0, mgrid=d19.sawtooth(DELTA_DEG))
        c2, r2 = d10.density(M, nbin=360)
        j = int(np.argmax(r2))
        s2 = (c2 > c2[j]) & (c2 <= c2[j]+1.5) & (r2 > 0)
        fl = float('nan')
        if s2.sum() >= 4:
            cf2 = np.polyfit(c2[s2], r2[s2]**-2.0, 1)
            fl = float(-cf2[1]/cf2[0])
        rows.append(dict(u=u, exact=exact, estimator=est, bias=est-exact,
                         flow_minus_eq=fl-exact))
        print(f"    {u:5.2f} {exact:10.2f} {est:10.2f} {est-exact:+7.2f} "
              f"{fl-exact:+12.2f}")
    b = [r["bias"] for r in rows if np.isfinite(r["bias"])]
    f = [r["flow_minus_eq"] for r in rows if np.isfinite(r["flow_minus_eq"])]
    print(f"    mean estimator bias {np.mean(b):+.2f} deg vs "
          f"mean flow-minus-Eq(12) {np.mean(f):+.2f} deg")
    print(f"    -> {'ESTIMATOR BIAS explains the offset' if abs(np.mean(b)-np.mean(f)) < 0.15 else 'residual remains: a real term'}")
    return rows


# --------------------------------------------------------------------- RISK 2
def risk2():
    print("\n[RISK 2] noise broadening of the caustic, width derived (sigma_m = s_bar)")
    grid = np.linspace(-DELTA_DEG/2, DELTA_DEG/2, 361)
    s_bar = float(np.sqrt(np.mean([d12.b_s_closed(v)[1]**2 for v in grid])))
    d8 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d8_results_2026-08-29.json")))["sigma_vs_u"]}
    d21 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d21_results_2026-08-29.json")))["rows"]}
    print(f"    derived width sigma_m = s_bar = {s_bar:.2f} deg")
    print(f"    {'u':>5} {'measured':>9} {'sharp':>8} {'err':>7} {'broadened':>10} {'err':>7}")
    rng = np.random.default_rng(5)
    rows = []
    for u in US:
        ms = analytic_samples(u, K, D, n=200000)
        sm = ms + rng.normal(0.0, s_bar, ms.size)
        sm = ((sm + DELTA_DEG/2) % DELTA_DEG) - DELTA_DEG/2
        vals = np.array([sum(x*x for x in d12.b_s_closed(v)) for v in grid])
        f = lambda arr: math.sqrt(float(np.mean(np.interp(arr, grid, vals))))
        sig_b = f(sm)
        meas = d8[u]["sigma"]
        sharp = d21[u]["two_piece"]
        rows.append(dict(u=u, measured=meas, sharp=sharp, broadened=sig_b,
                         rel_sharp=(sharp-meas)/meas, rel_broad=(sig_b-meas)/meas))
        print(f"    {u:5.2f} {meas:9.3f} {sharp:8.3f} {100*(sharp-meas)/meas:+6.1f}% "
              f"{sig_b:10.3f} {100*(sig_b-meas)/meas:+6.1f}%")
    es = 100*np.mean([abs(r["rel_sharp"]) for r in rows])
    eb = 100*np.mean([abs(r["rel_broad"]) for r in rows])
    print(f"    mean |err|: sharp {es:.1f}%  ->  broadened {eb:.1f}%")
    return dict(s_bar=s_bar, rows=rows, mean_sharp=es, mean_broad=eb)


# --------------------------------------------------------------------- RISK 3
def independent_sim(u, k=K, delta_deg=DELTA_DEG, R=10.0, seeds=4, dur=90.0):
    """From-scratch simulator of the same model.  Differences from the engine:
    exact parametric circle (no polyline), midpoint (RK2) integration, own ring
    buffer for the delay, own vote aggregation, heading-angle state."""
    L = 2.0
    tau_tot = 0.6663
    WIN, T_s, DT = 0.3, 1.0/12.0, 1.0/120.0      # half the engine's step
    v = u*L/tau_tot
    tau = tau_tot - WIN/2 - T_s
    out = []
    for sd in range(seeds):
        rng = np.random.default_rng(900+sd)
        nb = int(round(tau/DT)) + 1
        buf = np.zeros((nb, 2)); bi = 0
        phi = 0.0                                  # angle on the circle
        pos = np.array([R, 0.0])
        th = math.pi/2                             # heading
        cmd = th
        buf[:] = pos
        errs = []
        nsteps = int(dur/DT)
        vote_every = max(1, int(round(WIN/DT)))
        for i in range(nsteps):
            if i % vote_every == 0:
                pd = buf[(bi - nb + 1) % nb]
                ang = math.atan2(pd[1], pd[0])
                # look-ahead point on the exact circle
                arc = ang + L/R
                tgt = np.array([R*math.cos(arc), R*math.sin(arc)])
                d = tgt - pd
                ideal = math.degrees(math.atan2(d[1], d[0]))
                err = d32.gen_votes_param(ideal, ideal, rng, n_dirs=int(360/delta_deg))
                errs.append(err)
                cmd = math.radians(ideal + err)
            # midpoint step on heading + position
            k1 = (cmd - th)/T_s
            th_mid = th + 0.5*DT*k1
            k2 = (cmd - th_mid)/T_s
            th = th + DT*k2
            vel = np.array([math.cos(th), math.sin(th)])*v
            pos = pos + DT*vel
            bi = (bi + 1) % nb
            buf[bi] = pos
        errs = np.array(errs[len(errs)//3:])
        out.append(float(np.std(errs)))
    return float(np.mean(out)), float(np.std(out))


def risk3():
    print("\n[RISK 3] implementation independence: a from-scratch simulator")
    d8 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d8_results_2026-08-29.json")))["sigma_vs_u"]}
    print(f"    {'u':>5} {'engine':>8} {'independent':>12} {'diff':>7}")
    rows = []
    for u in (0.50, 0.71, 0.90):
        s, sd = independent_sim(u)
        m = d8[u]["sigma"]
        rows.append(dict(u=u, engine=m, independent=s, spread=sd, rel=(s-m)/m))
        print(f"    {u:5.2f} {m:8.3f} {s:12.3f} {100*(s-m)/m:+6.1f}%")
    e = 100*np.mean([abs(r["rel"]) for r in rows])
    print(f"    mean |diff| = {e:.1f}%")
    return rows


if __name__ == "__main__":
    out = {}
    out["risk1"] = risk1()
    out["risk2"] = risk2()
    out["risk3"] = risk3()
    json.dump(out, open(os.path.join(HERE, "d33_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d33_results_2026-08-29.json")
