# -*- coding: utf-8 -*-
"""
Derivation note -- D7: the analytic structure of rho(m), and closing
open problems (ii) and (iii) with no measured closed-loop input.

Three claims, each tested:

CLAIM 1 (similarity).  Written in the cylinder variables the quasi-static flow is
        m' = w0 (1 - k z),      z' = w0 [ b'(m) - k z (1 + b'(m)) ],   k = R/L
so w0 = v/R is a pure time rescaling: the undelayed attractor -- and hence
rho(m) -- cannot depend on speed at all.  The speed dependence observed in D3/D6
must therefore be a DELAY effect, and can enter only through the delay margin
u = v tau_tot / L.  Prediction: rho(m) and sigma_theta depend on (k, u) only, so
different (v, tau) pairs with the same u must collapse.

CLAIM 2 (fold caustic).  Inside the snap region b'(m) = -1 exactly, so z' = -w0
and the orbit in the (m, z) plane is the exact parabola
        m(z) = C - z + (k/2) z^2                            (speed-free)
whose residence density is
        rho(m)  =  N / sqrt( 1 - 2k (C - m) )   ~  1/sqrt(m - m_s)
with the caustic edge at the stall z = 1/k = L/R, i.e. m_s = C - 1/(2k).
rho is a fold caustic, not a smoothed sweep.

CLAIM 3 (closure).  sigma_theta(v, tau) computed from the derived rho -- i.e.
from the 1-DOF surrogate, whose only inputs are the open-loop vote transfer
function and the loop primitives -- reproduces the engine, and substituted into
the D5 margin equation reproduces the published circle optima.  Then (ii) and
(iii) contain no measured closed-loop quantity.

Outputs: d7_results_2026-08-29.json
"""
import importlib.util, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d6", os.path.join(HERE, "d6_rho_derivation_2026-08-29.py"))
d6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d6)
L, R, WIN, T_S, DELTA = d6.L, d6.R, d6.WIN, d6.T_S, d6.DELTA


def sigma_and_rho(v, tau, ms, b, s, seeds=6):
    M, E = [], []
    for sd in range(seeds):
        m, e, z, w = d6.surrogate(v, tau, ms, b, s, seed=700+sd)
        M.append(m); E.append(e)
    M = np.concatenate(M); E = np.concatenate(E)
    ctr, rho = d6.density(M)
    return float(E.std()), ctr, rho


def flow_orbit(ms, b, k=R/L, n=400000, dt=2e-4, m0=0.0, z0=0.0):
    """undelayed cylinder flow, w0 factored out (time in units of 1/w0)."""
    mb = np.radians(b); mm = np.radians(ms)
    bp = np.gradient(mb, mm)
    m, z = math.radians(m0), z0
    D = math.radians(DELTA)
    M = np.empty(n); Z = np.empty(n)
    for i in range(n):
        mw = ((m + D/2) % D) - D/2
        bpv = float(np.interp(mw, mm, bp))
        dm = (1 - k*z)
        dz = bpv - k*z*(1 + bpv)
        m += dt*dm; z += dt*dz
        M[i] = ((m + D/2) % D) - D/2; Z[i] = z
    keep = n//2
    return np.degrees(M[keep:]), Z[keep:]


if __name__ == "__main__":
    t0 = time.time()
    ms, b, s = d6.load_transfer()
    out = {}

    print("[CLAIM 1] similarity: does rho depend only on u = v*tau_tot/L ?")
    print(f"{'u':>6} {'tau':>6} {'v':>6} {'sigma':>8} {'peak m':>8} {'rho peak share':>15}")
    groups = {}
    for u in [0.50, 0.71, 0.90]:
        for tau in [0.2, 0.433, 0.8]:
            tt = tau + WIN/2 + T_S
            v = u*L/tt
            sig, ctr, rho = sigma_and_rho(v, tau, ms, b, s)
            cg = d6.coarse(ctr, rho)
            pk = max(cg, key=lambda p: p[1])
            groups.setdefault(u, []).append(dict(tau=tau, v=v, sigma=sig,
                                                 peak_m=pk[0], peak_share=pk[1]))
            print(f"{u:6.2f} {tau:6.3f} {v:6.3f} {sig:8.3f} {pk[0]:8.1f} {100*pk[1]:14.1f}%")
        sp = [g["sigma"] for g in groups[u]]
        print(f"       -> sigma spread across tau at fixed u: "
              f"{100*(max(sp)-min(sp))/np.mean(sp):.1f}%")
    out["similarity"] = groups

    print("\n[CLAIM 2] fold caustic of the undelayed flow (w0 scaled out)")
    M, Z = flow_orbit(ms, b)
    ctr, rho = d6.density(M, nbin=45)
    k = R/L
    stall = math.degrees(1.0/k)
    inside = (np.abs(M) < 19.0)
    print(f"  stall z = 1/k = {1.0/k:.3f} rad = {stall:.2f} deg;  "
          f"orbit z range [{Z.min():+.3f}, {Z.max():+.3f}] rad")
    i = int(np.argmax(rho))
    print(f"  invariant density peak at m = {ctr[i]:+.2f} deg, share {100*rho[i]:.1f}% "
          f"(uniform 2.2%)")
    # fold test: rho^-2 should be linear in m on the caustic side
    sel = (ctr > ctr[i]) & (ctr < ctr[i]+12) & (rho > 0)
    if sel.sum() > 4:
        cf = np.polyfit(ctr[sel], rho[sel]**-2.0, 1)
        pred = np.polyval(cf, ctr[sel])
        r2 = 1 - np.sum((rho[sel]**-2-pred)**2)/np.sum((rho[sel]**-2-np.mean(rho[sel]**-2))**2)
        print(f"  fold law test: rho^-2 linear in m on the forward branch, R^2 = {r2:.4f} "
              f"(edge at m_s = {-cf[1]/cf[0]:+.2f} deg)")
        out["fold"] = dict(R2=float(r2), m_s=float(-cf[1]/cf[0]), peak=float(ctr[i]))

    print("\n[CLAIM 3] margin equation driven by the DERIVED sigma_theta (no engine)")
    d5 = json.load(open(os.path.join(HERE, "d5_results_2026-08-29.json")))
    OBS = {r["tau"]: r["v_obs"] for r in d5["rows"]}
    speeds = np.array([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5])
    print(f"{'tau':>6} {'v*_obs':>8} {'v*_engine':>10} {'v*_derived':>11} {'err_der':>9}")
    rows = []
    for j, tau in enumerate([0.1, 0.433, 0.6, 1.0, 1.5]):
        sig = np.array([sigma_and_rho(v, tau, ms, b, s, seeds=3)[0] for v in speeds])
        cf = np.polyfit(speeds, np.radians(sig), 3)
        f = lambda v, c=cf: float(np.polyval(c, v))
        tt = tau + WIN/2 + T_S
        vg = np.linspace(speeds[0], speeds[-1], 400)
        J = np.array([(L/R)**2*(L/2 - v*tt)**2 + L**2*f(v)**2 for v in vg])
        vp = float(vg[int(np.argmin(J))])
        veng = d5["rows"][j]["v_pred"]
        rows.append(dict(tau=tau, v_obs=OBS[tau], v_engine=veng, v_derived=vp,
                         rel=(vp-OBS[tau])/OBS[tau], sigma=sig.tolist()))
        print(f"{tau:6.3f} {OBS[tau]:8.3f} {veng:10.3f} {vp:11.3f} "
              f"{100*(vp-OBS[tau])/OBS[tau]:+8.1f}%")
    out["margin_from_derived_sigma"] = rows
    e = [abs(r["rel"]) for r in rows]
    print(f"  mean |err| = {100*np.mean(e):.1f}%  (engine-driven version: 8.1%)")

    json.dump(out, open("d7_results_2026-08-29.json", "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> d7_results_2026-08-29.json")
