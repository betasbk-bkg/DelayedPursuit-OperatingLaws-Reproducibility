# -*- coding: utf-8 -*-
"""
Derivation note -- D8: the caustic-edge law m_s(u), completing rho(m).

D7 established: rho(m) = rho(m; k, u), k = R/L, u = v*tau_tot/L, and that the
undelayed flow's density is an exact fold caustic rho ~ 1/sqrt(m - m_s).
What is left is where the edge sits, i.e. m_s(u).

Derivation.  With the loop delay the stall condition ceases to be evaluated at
the current phase: m'(t) = w0 - a z(t - tau_tot), so m' = 0 at z(t-tau_tot)=1/k.
Between t-tau_tot and t the phase advances by (to leading order) w0*tau_tot per
delayed channel, and the delay appears in BOTH channels of the aim angle -- the
delayed arc (which sets theta_p) and the delayed lateral error (which sets the
feedback).  Hence the edge is predicted to shift forward by

        Delta m_s = 2 * w0 * tau_tot = 2 u / k        [rad]

    ==>  m_s(u) = -m_c + 2u/k ,

with m_c the snap-region half-width of the measured b(m) (the phase at which the
mixture stops snapping, |b| maximal).  m_c is a property of the vote model, not
a fitted constant.

Test: extract the caustic edge by the fold law itself (rho^-2 is linear in m on
the forward branch, so its zero crossing locates m_s far more precisely than a
histogram peak), over a range of u, and check slope = 2/k and intercept = -m_c.

Outputs: d8_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("d6", os.path.join(HERE, "d6_rho_derivation_2026-08-29.py"))
d6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d6)
L, R, WIN, T_S, DELTA = d6.L, d6.R, d6.WIN, d6.T_S, d6.DELTA
K = R/L


def caustic_edge(ctr, rho, width=10.0):
    """locate the fold edge: on the forward branch rho^-2 is linear in m."""
    i = int(np.argmax(rho))
    sel = (ctr > ctr[i]) & (ctr <= ctr[i]+width) & (rho > 0)
    if sel.sum() < 4:
        return float('nan'), float('nan'), float('nan')
    y = rho[sel]**-2.0
    cf = np.polyfit(ctr[sel], y, 1)
    r2 = 1 - np.sum((y-np.polyval(cf, ctr[sel]))**2)/np.sum((y-y.mean())**2)
    return float(-cf[1]/cf[0]), float(r2), float(ctr[i])


def rho_at(u, tau, ms, b, s, seeds=6, nbin=90, noise=True):
    """deterministic by default: the caustic is a property of the flow, and
    additive vote noise only broadens it."""
    tt = tau + WIN/2 + T_S
    v = u*L/tt
    M = []
    for sd in range(seeds):
        M.append(d6.surrogate(v, tau, ms, b, s, seed=3000+sd, noise=noise,
                              dur=400.0)[0])
    M = np.concatenate(M)
    return d6.density(M, nbin=nbin) + (v,)


if __name__ == "__main__":
    ms, b, s = d6.load_transfer()
    m_c = float(ms[int(np.argmin(b))])            # phase of maximal snap error
    print(f"snap-region half-width from the measured transfer: m_c = {m_c:.2f} deg "
          f"(b min = {b.min():.2f} deg)")
    print(f"predicted law:  m_s(u) = {-m_c:.2f} + {math.degrees(2/K):.2f}*u   [deg]\n")

    print(f"{'u':>6} {'tau':>6} {'v':>6} {'m_s fit':>9} {'R^2':>7} {'m_s pred':>9} {'diff':>7}")
    rows = []
    for u in [0.40, 0.50, 0.60, 0.71, 0.80, 0.90, 1.00]:
        for tau in [0.433, 0.8]:
            ctr, rho, v = rho_at(u, tau, ms, b, s)
            edge, r2, pk = caustic_edge(ctr, rho)
            pred = -m_c + math.degrees(2*u/K)
            rows.append(dict(u=u, tau=tau, v=v, m_s=edge, r2=r2, peak=pk, m_s_pred=pred))
            print(f"{u:6.2f} {tau:6.3f} {v:6.3f} {edge:9.2f} {r2:7.4f} {pred:9.2f} "
                  f"{edge-pred:+7.2f}")

    ok = [r for r in rows if r["u"] <= 0.8 and not math.isnan(r["m_s"])]
    U = np.array([r["u"] for r in ok]); MS = np.array([r["m_s"] for r in ok])
    cf = np.polyfit(U, MS, 1)
    print(f"\nfitted   m_s(u) = {cf[1]:+.2f} + {cf[0]:.2f}*u   [deg]   (n={len(ok)})")
    print(f"predicted        = {-m_c:+.2f} + {math.degrees(2/K):.2f}*u")
    print(f"slope  ratio fit/pred = {cf[0]/math.degrees(2/K):.3f}")
    print(f"intercept diff        = {cf[1]+m_c:+.2f} deg")
    # anchored version: continuous-flow edge at u = 0 (D7) = -19.73 deg
    anchored = [(r["u"], (r["m_s"]+19.73)/r["u"]*K/math.degrees(1.0)) for r in ok]
    cs = [a[1] for a in anchored]
    print(f"coefficient c in m_s = -m_c + c*u/k, anchored at the u=0 flow edge: "
          f"c = {np.mean(cs):.2f} (range {min(cs):.2f}-{max(cs):.2f})")

    # sigma_theta(u): the object (ii) and (iii) needed
    print(f"\nsigma_theta as a function of the delay margin u  "
          f"(open-loop uniform sweep = 10.95 deg)")
    print(f"{'u':>6} {'sigma':>8} {'attenuation':>12}")
    sig_u = []
    for u in [0.40, 0.50, 0.60, 0.71, 0.80, 0.90, 1.00, 1.20]:
        tt = 0.433 + WIN/2 + T_S
        v = u*L/tt
        E = np.concatenate([d6.surrogate(v, 0.433, ms, b, s, seed=4000+i)[1]
                            for i in range(8)])
        sig_u.append(dict(u=u, sigma=float(E.std()), atten=float(E.std())/10.9456))
        print(f"{u:6.2f} {E.std():8.3f} {E.std()/10.9456:12.3f}")

    json.dump(dict(m_c=m_c, rows=rows, fit_slope=float(cf[0]), fit_intercept=float(cf[1]),
                   pred_slope=math.degrees(2/K), sigma_vs_u=sig_u),
              open("d8_results_2026-08-29.json", "w"), indent=2, default=float)
    print("\n-> d8_results_2026-08-29.json")
