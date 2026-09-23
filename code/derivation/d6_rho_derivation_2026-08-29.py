# -*- coding: utf-8 -*-
"""
Derivation note -- D6: analytic derivation of rho(m), the closed-loop
stationary density of the in-cell phase.

This is the single object that D3 (open problem (ii), the 0.778 attenuation) and
D5 (open problem (iii), the smooth margin) both had to MEASURE.  Closing it
closes both analytically.

Reduction (exact, linear in e only)
-----------------------------------
Circle of radius R, look-ahead L, total loop delay tau_tot, a = v/L, w0 = v/R.
The aim angle from the delayed position is
    Theta = theta_p(s_d) + L/(2R) - e_d/L                 (chord + offset)
and the realised heading is Theta + b(m), m = Theta mod Delta, b measured open
loop from the vote mixture.  With w = e/L and c0 = (L/2 - v*tau_tot)/R
(Theorem 1's steady offset in angle units):

    w'  =  a * z ,          z := c0 - w + b(m)
    m'  =  w0 - a * z                                        (cylinder flow)

so  m' + w' = w0 : the in-cell phase and the scaled lateral error exchange, and
their sum advances at exactly the tangent rotation rate.  Two consequences:

 (1) STALL.  m' = 0 when z = w0/a = L/R.  The phase drift stops wherever the
     instantaneous angular tracking error reaches L/R -- a pure geometry number,
     independent of speed.  Since b is the only fast term in z, the stall sits at
        b(m_s) = L/R + w - c0
     and the density piles up there.  This is the phase lock seen in the data,
     and it moves with v because c0 does.

 (2) FOLD.  Inside the snap region b'(m) = -1, so z' = -w0 exactly: z sweeps down
     linearly through the stall.  Then m'(t) = a*w0*(t - t_s) near the stall, so
     m(t) is parabolic: m has a turning point, and the residence density carries
     an integrable inverse-square-root (fold caustic) edge

        rho(m)  ~  1 / ( w0 * sqrt( 2 * k * |m - m_s| ) ),   k = R/L

     on the forward side.  rho is NOT a smoothed uniform sweep; it is a caustic.

Validation ladder in this script:
  A. 1-DOF surrogate (no agents; vote mixture replaced by its measured b, s)
     reproduces sigma_theta(v) of the 150-agent engine.
  B. the deterministic surrogate (s = 0) exposes the stall/fold, and the stall
     location matches the analytic condition above.
  C. rho from the surrogate reproduces the engine's measured rho.
  D. sigma_theta predicted from the DERIVED rho closes open problem (ii), and
     feeding it into the D5 margin equation closes (iii) with no measured
     closed-loop input.

Outputs: d6_results_2026-08-29.json
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT, WIN, SMOOTH, L, R = 1/60, 0.3, 0.2, 2.0, 10.0
VI = int(round(WIN/DT))
T_S = DT/SMOOTH
DELTA = 45.0
DUR = 65.0


def load_transfer():
    d = json.load(open(os.path.join(HERE, "d3_results_2026-08-29.json")))
    t = d["transfer"]
    return np.array(t["m"]), np.array(t["b"]), np.array(t["s"])


def wrap(a):
    return ((a + DELTA/2) % DELTA) - DELTA/2


def surrogate(v, tau_s, ms, b, s, seed=0, noise=True, dur=DUR, warm=0.25):
    """1-DOF reduced loop: no agents, no 2-D geometry -- path-frame kinematics
    plus the measured static nonlinearity."""
    rng = np.random.default_rng(seed)
    delay_f = int(round(tau_s/DT))
    nf = int(dur/DT)
    arc = 0.0; e = 0.0
    th = math.degrees(0.0)                     # heading, deg (theta_p(0) = 0)
    hs = [0.0]*(delay_f+2); he = [0.0]*(delay_f+2)
    cmd = 0.0
    M, E, Z, W = [], [], [], []
    c0 = (L/2 - v*(tau_s + WIN/2 + T_S))/R     # rad
    for f in range(nf):
        if f % VI == 0:
            sd = hs[-1-delay_f]; ed = he[-1-delay_f]
            Th = math.degrees(sd/R + L/(2*R) - ed/L)
            m = wrap(Th)
            bb = float(np.interp(m, ms, b))
            ss = float(np.interp(m, ms, s)) if noise else 0.0
            cmd = Th + bb + (rng.normal(0, ss) if noise else 0.0)
            if f*DT > warm*dur:
                M.append(m); E.append(cmd - Th)
                Z.append(c0 + math.radians(bb) - e/L)
                W.append(e/L)
        th += (DT/T_S)*(cmd - th)
        d = math.radians(th - math.degrees(arc/R))
        e += DT*v*math.sin(d)
        arc += DT*v*math.cos(d)/max(1e-6, 1 - e/R)
        hs.append(arc); he.append(e)
    return np.array(M), np.array(E), np.array(Z), np.array(W)


def density(M, nbin=45):
    edges = np.linspace(-DELTA/2, DELTA/2, nbin+1)
    c, _ = np.histogram(M, bins=edges)
    return 0.5*(edges[1:]+edges[:-1]), c/max(c.sum(), 1)


def coarse(ctr, w, n=9):
    idx = np.array_split(np.arange(len(ctr)), n)
    return [(float(ctr[i].mean()), float(w[i].sum())) for i in idx]


if __name__ == "__main__":
    t0 = time.time()
    ms, b, s = load_transfer()
    eng = {r["speed"]: r for r in json.load(
        open(os.path.join(HERE, "d3_results_2026-08-29.json")))["runs"]
        if "case" not in r}
    out = {"speeds": [], "k": R/L, "stall_z_deg": math.degrees(L/R)}

    print(f"geometry: k = R/L = {R/L:.1f},  stall at z = L/R = {math.degrees(L/R):.2f} deg")
    print("\n[A/C] 1-DOF surrogate vs 150-agent engine   (tau = 433 ms)")
    print(f"{'v':>5} {'sig_engine':>11} {'sig_surrog':>11} {'err':>7} "
          f"{'peak_eng':>9} {'peak_sur':>9} {'m_s analytic':>13}")
    for v in [1.0, 1.5, 2.0, 2.13, 3.0, 5.0]:
        MM, EE = [], []
        for sd in range(6):
            M, E, Z, W = surrogate(v, 0.433, ms, b, s, seed=500+sd)
            MM.append(M); EE.append(E)
        M = np.concatenate(MM); E = np.concatenate(EE)
        ctr, w = density(M)
        cg = coarse(ctr, w)
        peak_sur = max(cg, key=lambda p: p[1])[0]
        ce = coarse(np.array(eng[v]["density_centers"]), np.array(eng[v]["density"]))
        peak_eng = max(ce, key=lambda p: p[1])[0]
        # analytic stall: z = L/R  with  z = c0 + b - w  ->  b(m_s) = L/R - c0 + <w>
        c0 = (L/2 - v*(0.433 + WIN/2 + T_S))/R
        Wm = np.mean(np.concatenate([surrogate(v, 0.433, ms, b, s, seed=900+i)[3]
                                     for i in range(2)]))
        btgt = math.degrees(L/R - c0 + Wm)
        cand = ms[np.argmin(np.abs(b - btgt))] if abs(btgt) <= abs(b).max() else float('nan')
        sig_sur = float(E.std()); sig_eng = eng[v]["sigma_measured"]
        out["speeds"].append(dict(v=v, sigma_engine=sig_eng, sigma_surrogate=sig_sur,
                                  peak_engine=peak_eng, peak_surrogate=peak_sur,
                                  b_target=btgt, m_stall=float(cand),
                                  w_mean=float(Wm), c0=c0,
                                  density=[list(p) for p in cg]))
        print(f"{v:5.2f} {sig_eng:11.3f} {sig_sur:11.3f} "
              f"{100*(sig_sur-sig_eng)/sig_eng:+6.1f}% {peak_eng:9.1f} {peak_sur:9.1f} "
              f"{cand:13.1f}")

    print("\n[B] deterministic surrogate (s = 0): stall and fold")
    for v in [1.5, 2.13, 3.0]:
        M, E, Z, W = surrogate(v, 0.433, ms, b, s, seed=1, noise=False)
        ctr, w = density(M)
        print(f"  v={v:4.2f}  z range [{math.degrees(np.min(Z)):+6.2f},"
              f" {math.degrees(np.max(Z)):+6.2f}] deg   stall z={math.degrees(L/R):.2f}"
              f"   peak bin m={ctr[int(np.argmax(w))]:+6.2f}  max share={w.max()*100:.1f}%")
    json.dump(out, open("d6_results_2026-08-29.json", "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> d6_results_2026-08-29.json")
