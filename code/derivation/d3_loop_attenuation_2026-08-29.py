# -*- coding: utf-8 -*-
"""
Derivation note -- D3: the 0.778 loop attenuation (open problem (ii)).

Manuscript status: sigma_th = 8.60 deg (measured in closed loop) versus 11.06 deg
(computable open-loop swept transfer error of the vote mixture); the ratio 0.778
is unexplained, "a first-order sensitivity estimate gives ~0.90".

Hypothesis tested here
----------------------
0.778 is NOT a loop gain.  It is the closed-loop stationary density of the
in-cell phase.  Write the vote-mixture transfer function measured open loop as

    aggregate = ideal + b(m) + eta(m),   m = ideal mod Delta, signed in [-D/2,D/2]

with b, s = std(eta) measured directly from gen_votes (phase14's reduction).
The open-loop number assumes m is swept UNIFORMLY:

    sigma_open^2 = E_unif[ b(m)^2 + s(m)^2 ]                        (~11.06 deg)

But in closed loop m is not uniform.  Near the cell centre the snap b(m) ~ -m
steers the vehicle off the path; the loop answers by rotating the aim direction
the same way, which pushes m AWAY from the centre (the centre is a repeller).
The only place the drive balances is the cell boundary, where the mixture splits
between two grid directions and |b| collapses (measured: |b| = 14.25 deg at
m = 15 deg but 4.99 deg at m = 22.5 deg).  So the stationary density piles up at
the boundary, exactly where the effective quantisation error is smallest:

    sigma_closed^2 = E_rho[ b(m)^2 + s(m)^2 ] < sigma_open^2

Prediction with zero fitted constants: E_rho[.] evaluated with the MEASURED
rho(m) from a closed-loop run and the MEASURED b, s reproduces 8.60 deg.

Outputs: d3_results_2026-08-29.json
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
from simulation_main import (Circle, Square, DIRS, gen_votes,
                             DT, VOTE_INT, DELAY_F, LOOK, FRAMES, SMOOTH)

DELTA = 45.0
N_AG, TROLL = 150, 0.05


def signed_offset(ang):
    return ((ang + DELTA/2) % DELTA) - DELTA/2


# ---------------- 1. open-loop transfer function of the vote mixture ----------
def transfer(nm=61, draws=1500, seed=11):
    """b(m), s(m) from gen_votes alone -- no simulation, no v* data."""
    rng = np.random.default_rng(seed)
    ms = np.linspace(-DELTA/2, DELTA/2, nm)
    b = np.zeros(nm); s = np.zeros(nm)
    for i, m in enumerate(ms):
        errs = np.empty(draws)
        for d in range(draws):
            # prev_angle = m  -> slow agents contribute no lag term (static probe)
            v = gen_votes(m, m, TROLL, N_AG, rng)
            bl = DIRS[v].mean(axis=0)
            errs[d] = (math.degrees(math.atan2(bl[1], bl[0])) - m + 180) % 360 - 180
        b[i] = errs.mean(); s[i] = errs.std()
    return ms, b, s


# ---------------- 2. closed-loop run: record ideal angle + realised error -----
def closed_loop(traj, speed, seed, n_dirs_full=True):
    rng = np.random.default_rng(seed)
    pos = traj.start(); vel = np.zeros(2)
    hist = [pos.copy()]; prev = 0.0
    cur = np.array([1.0, 0.0])
    ideals, errs = [], []
    for f in range(FRAMES):
        if f % VOTE_INT == 0:
            dp = hist[max(0, len(hist)-1-DELAY_F)]
            _, arc = traj.closest(dp)
            d = traj.at(arc + LOOK) - dp
            n = np.linalg.norm(d)
            if n > 1e-10:
                d = d/n
            ia = math.degrees(math.atan2(d[1], d[0]))
            votes = gen_votes(ia, prev, TROLL, N_AG, rng)
            prev = ia
            bl = DIRS[votes].mean(axis=0)
            g = np.linalg.norm(bl)
            if g > 1e-10:
                cur = bl/g
            aa = math.degrees(math.atan2(cur[1], cur[0]))
            ideals.append(ia)
            errs.append((aa - ia + 180) % 360 - 180)
        vel = vel + SMOOTH*(cur*speed - vel)
        pos = pos + vel*DT
        hist.append(pos.copy())
    return np.array(ideals), np.array(errs)


def predict_from_density(ms, b, s, m_samples, nbin=45):
    """E_rho[b^2 + s^2] with rho estimated from the closed-loop m samples."""
    edges = np.linspace(-DELTA/2, DELTA/2, nbin+1)
    cnt, _ = np.histogram(m_samples, bins=edges)
    w = cnt/max(cnt.sum(), 1)
    ctr = 0.5*(edges[1:]+edges[:-1])
    bi = np.interp(ctr, ms, b); si = np.interp(ctr, ms, s)
    return float(np.sqrt(np.sum(w*(bi**2 + si**2)))), ctr, w


if __name__ == "__main__":
    t0 = time.time()
    print("[1] measuring the open-loop vote transfer function b(m), s(m) ...", flush=True)
    ms, b, s = transfer()
    sig_unif = float(np.sqrt(np.mean(b**2 + s**2)))
    print(f"    E_unif[b^2+s^2]^(1/2) = {sig_unif:.3f} deg   (manuscript: 11.06)")
    print(f"    |b| at m=15 deg: {abs(np.interp(15.0, ms, b)):.2f},  "
          f"at m=22.5: {abs(np.interp(22.4, ms, b)):.2f}   "
          f"s at 22.5: {np.interp(22.4, ms, s):.2f}")

    out = dict(sigma_open_uniform=sig_unif,
               transfer=dict(m=ms.tolist(), b=b.tolist(), s=s.tolist()),
               runs=[])

    print("\n[2] closed-loop runs on the circle (R=10): density of the in-cell phase")
    print(f"{'speed':>6} {'sig_meas':>9} {'sig_pred':>9} {'ratio_meas':>11} "
          f"{'ratio_pred':>11} {'|m|_rms':>8} {'unif=12.99':>10}")
    for speed in [1.0, 1.5, 2.0, 2.13, 3.0, 5.0]:
        IA, ER = [], []
        for sd in range(6):
            ia, er = closed_loop(Circle(10), speed, seed=1000+sd)
            IA.append(ia); ER.append(er)
        IA = np.concatenate(IA); ER = np.concatenate(ER)
        m = signed_offset(IA)
        sig_meas = float(ER.std())
        sig_pred, ctr, w = predict_from_density(ms, b, s, m)
        row = dict(speed=speed, sigma_measured=sig_meas, sigma_pred_density=sig_pred,
                   ratio_measured=sig_meas/sig_unif, ratio_pred=sig_pred/sig_unif,
                   m_rms=float(np.sqrt(np.mean(m**2))),
                   density_centers=ctr.tolist(), density=w.tolist())
        out["runs"].append(row)
        print(f"{speed:6.2f} {sig_meas:9.3f} {sig_pred:9.3f} {sig_meas/sig_unif:11.3f} "
              f"{sig_pred/sig_unif:11.3f} {np.sqrt(np.mean(m**2)):8.2f} {12.99:10.2f}")

    # square for contrast: segment directions sit ON the grid -> m pinned near 0
    print("\n[3] contrast: square (grid-aligned sides)")
    for speed in [0.6, 1.0]:
        ia, er = closed_loop(Square(10), speed, seed=77)
        m = signed_offset(ia)
        sp, _, _ = predict_from_density(ms, b, s, m)
        print(f"  v={speed}: sigma_meas={er.std():6.3f}  sigma_pred={sp:6.3f}  "
              f"|m|_rms={np.sqrt(np.mean(m**2)):5.2f}")
        out["runs"].append(dict(case="square", speed=speed,
                                sigma_measured=float(er.std()),
                                sigma_pred_density=sp,
                                m_rms=float(np.sqrt(np.mean(m**2)))))

    json.dump(out, open("d3_results_2026-08-29.json", "w"), indent=2, default=float)
    print(f"\ndone in {time.time()-t0:.0f}s -> d3_results_2026-08-29.json")
