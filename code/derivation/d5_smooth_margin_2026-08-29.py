# -*- coding: utf-8 -*-
"""
Derivation note -- D5: an ANALYTIC form for the smooth-geometry margin
(open problem (iii)).

Manuscript status: the smooth (circle/ellipse) margin is only *computationally*
closed -- a reduced model reproduces x* = 1.40/1.41 against 1.347 observed and
the ellipse margin to +-4%, but "an analytic -- as opposed to computational --
form for the smooth-geometry margin" is listed as open.  The 8-direction circle
sits at u* = 0.65-0.75, well above the continuous-limit 1/2 of Theorem 1, and
the closure report records u_max = 0.75 as "characterized, not solved".

Derivation
----------
On a smooth path two channels add:

    RMSE^2(v) = (L/R)^2 (L/2 - v*tau_tot)^2       [Theorem 1 sagitta bias]
              + L^2 * sigma_th^2(v, tau)          [quantisation channel: a
                                                   persistent angular error
                                                   q costs y = L*q  (Sec. IV-C)]

D3 showed that sigma_th is NOT a constant of the vote model: it is the closed-
loop in-cell phase density folded through the measured snap characteristic, and
that density tightens as the loop runs faster (10.8 deg at v = 1 -> 8.2 deg at
v = 3 -> back up at v = 5).  Stationarity of the sum therefore gives

    (L/R)^2 (v tau_tot - L/2) tau_tot + L^2 sigma_th sigma_th' = 0

    ==>   u* = 1/2 - (R^2 / (L*tau_tot)) * sigma_th(v*) * dsigma_th/dv     (**)

which is the analytic smooth margin.  It says exactly why the quantised circle
sits above the continuous limit: sigma_th' < 0 over the operating range, so the
quantisation channel pays for extra speed, and the margin is lifted above 1/2 by
an amount set by the curvature radius, the delay budget, and the slope of the
phase-locking curve.  For R -> infinity or a vanishing grid the correction dies
and u* -> 1/2, recovering Theorem 1.

Test: measure sigma_th(v, tau) with the engine's vote mixture in closed loop
(zero fitted constants), then solve (**) and compare with the published
8-direction circle optima.

Outputs: d5_results_2026-08-29.json
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.environ.get("ENGINE_DIR") or next(
    p for p in [os.path.join(HERE, "..", "..", "engine", "crowd_control_engine"),
                os.path.join(HERE, "..", "submission", "repo",
                             "engine", "crowd_control_engine")]
    if os.path.isdir(p))
sys.path.insert(0, ENGINE)
import simulation_main as sm
from simulation_main import Circle, DIRS, gen_votes, DT, VOTE_INT, LOOK, SMOOTH

L, R = 2.0, 10.0
WIN = 0.3
T_S = DT/SMOOTH
N_AG, TROLL = 150, 0.05
DUR = 65.0
FRAMES = int(DUR/DT)

# published 8-direction circle optima (closure report Sec. 3.4)
OBS = {0.1: 3.93, 0.433: 2.13, 0.6: 1.76, 1.0: 1.218, 1.5: 0.869}


def sigma_theta(speed, tau_s, seeds=4):
    """closed-loop std of (aggregate - ideal), degrees."""
    delay_f = int(round(tau_s/DT))
    traj = Circle(R)
    allerr = []
    for sd in range(seeds):
        rng = np.random.default_rng(4000+sd)
        pos = traj.start(); vel = np.zeros(2); hist = [pos.copy()]
        prev = 0.0; cur = np.array([1.0, 0.0]); errs = []
        for f in range(FRAMES):
            if f % VOTE_INT == 0:
                dp = hist[max(0, len(hist)-1-delay_f)]
                _, arc = traj.closest(dp)
                d = traj.at(arc + LOOK) - dp
                n = np.linalg.norm(d)
                if n > 1e-10:
                    d = d/n
                ia = math.degrees(math.atan2(d[1], d[0]))
                votes = gen_votes(ia, prev, TROLL, N_AG, rng); prev = ia
                bl = DIRS[votes].mean(axis=0); g = np.linalg.norm(bl)
                if g > 1e-10:
                    cur = bl/g
                aa = math.degrees(math.atan2(cur[1], cur[0]))
                errs.append((aa - ia + 180) % 360 - 180)
            vel = vel + SMOOTH*(cur*speed - vel)
            pos = pos + vel*DT
            hist.append(pos.copy())
        allerr.append(np.std(errs))
    return float(np.mean(allerr)), float(np.std(allerr))


def margin(tau_s, vgrid, sig, dsig=None):
    """minimise the two-channel cost
         J(v) = (L/R)^2 (L/2 - v tau_tot)^2 + L^2 sigma_th(v)^2
    (whose stationarity condition is (**); minimising is the robust form, since
    (**) can have spurious roots where sigma_th(v) is flat)."""
    tau_tot = tau_s + WIN/2 + T_S
    J = np.array([(L/R)**2*(L/2 - v*tau_tot)**2 + L**2*sig(v)**2 for v in vgrid])
    i = int(np.argmin(J))
    v = float(vgrid[i])
    if 1 <= i <= len(vgrid)-2:
        c = np.polyfit(vgrid[i-1:i+2], J[i-1:i+2], 2)
        if c[0] > 0:
            vv = -c[1]/(2*c[0])
            if vgrid[i-1] <= vv <= vgrid[i+1]:
                v = vv
    return v, v*tau_tot/L


if __name__ == "__main__":
    t0 = time.time()
    speeds = np.array([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5])
    taus = [0.1, 0.433, 0.6, 1.0, 1.5]
    print("measuring sigma_theta(v, tau) in closed loop  [deg]")
    hdr = "  v \\ tau " + "".join(f"{t:>9.3f}" for t in taus)
    print(hdr)
    S = np.zeros((len(speeds), len(taus)))
    for i, v in enumerate(speeds):
        row = []
        for j, t in enumerate(taus):
            s, _ = sigma_theta(v, t)
            S[i, j] = s; row.append(s)
        print(f"{v:9.2f} " + "".join(f"{x:9.3f}" for x in row), flush=True)

    out = dict(speeds=speeds.tolist(), taus=taus, sigma_deg=S.tolist(), rows=[])
    print(f"\nanalytic margin (**) vs published optima   [tau_tot = tau + {WIN/2+T_S:.4f}]")
    print(f"{'tau':>6} {'v*_obs':>8} {'v*_pred':>8} {'u*_obs':>8} {'u*_pred':>8} {'err':>8}")
    vg = np.linspace(speeds[0], speeds[-1], 400)
    for j, t in enumerate(taus):
        # smooth sigma(v) with a local quadratic, then differentiate analytically
        cf = np.polyfit(speeds, np.radians(S[:, j]), 3)
        sig = lambda v, c=cf: float(np.polyval(c, v))
        dsig = lambda v, c=cf: float(np.polyval(np.polyder(c), v))
        vp, up = margin(t, vg, sig, dsig)
        tau_tot = t + WIN/2 + T_S
        vo = OBS[t]; uo = vo*tau_tot/L
        out["rows"].append(dict(tau=t, v_obs=vo, v_pred=vp, u_obs=uo, u_pred=up,
                                rel=(vp-vo)/vo))
        print(f"{t:6.3f} {vo:8.3f} {vp:8.3f} {uo:8.3f} {up:8.3f} {100*(vp-vo)/vo:+7.1f}%")

    json.dump(out, open("d5_results_2026-08-29.json", "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> d5_results_2026-08-29.json")
