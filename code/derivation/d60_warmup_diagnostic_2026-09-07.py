#!/usr/bin/env python3
"""
d60 -- Why Gate A's shape fit failed: is it the missing warm-up discard?

d59 fitted d38's heading-servo J(u) to the reference engine's RMSE(v) curves and got 11-25%
residuals, against 2.4-5.2% for the same estimator on Nav2.  Three candidates were
listed; this script tests the first one, which is the only one the manuscript
states outright:

    "RMSE = ... computed over the full T = 3900-frame (65 s) run with no warmup
     discard"                                    (SciRep manuscript, System Model)

The Nav2 harness discards settle_laps = 1.0 and a 3.5 m tail.  A start transient is
v-dependent, so including it adds a term the model does not have.

Two other things are fixed at the same time, because they cost nothing here:
  * the speed grid.  the reference engine used 13 speeds with 1.0 m/s gaps above 2.5, and the
    Circle optimum at tau = 100 ms sits at v = 4.0 -- inside the coarse region, and
    at tau = 100/200 ms the reported v* is the LAST grid point (5.0), i.e. the
    optimum was never bracketed.  Here the grid is uniform and extended.
  * MC is raised, since a run costs 0.13 s.

The engine is imported unchanged from the repository; only the measurement window
and the sweep grid differ.  If the residual drops to the Nav2 level, Gate A's
failure was a protocol difference and the two experiments are directly comparable
after all.  If it does not, the heading-servo J genuinely does not describe this
engine and the merge needs more than a finite-alpha derivation.

Usage:  python3 d60_warmup_diagnostic_2026-09-07.py [--mc 15] [--quick]
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import least_squares

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402  (engine, unchanged)

LOOK = sm.LOOK
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH
TAUS_MS = [100, 200, 300, 433, 600, 1000]

_p = HERE / 'd38_corner_energy_two_plants_2026-08-29.py'
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def make_J(plant, umax=1.2):
    us = np.linspace(0.04, umax, 120)
    Js = np.array([d38.transit_energy(float(u), plant, h=1.0e-3) for u in us])
    ok = np.isfinite(Js) & (Js > 0)
    us, Js = us[ok], Js[ok]
    lg = np.log(Js)

    def J(u):
        return np.exp(np.interp(np.clip(u, us[0], us[-1]), us, lg))
    return J


def simulate_windowed(traj, N, troll, seed, V, delay_f, warm_frac):
    """The repository's simulate() loop, with the error series kept so the same
    run yields both the full-run RMSE and a warm-up-discarded RMSE."""
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    pos_hist = [pos.copy()]
    prev_angle = 0.0
    cur_dir = np.array([1., 0.])
    errors = np.empty(sm.FRAMES)

    for f in range(sm.FRAMES):
        if f % sm.VOTE_INT == 0:
            di = max(0, len(pos_hist) - 1 - delay_f)
            dp = pos_hist[di]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + LOOK) - dp
            n = np.linalg.norm(ideal)
            if n > 1e-10:
                ideal = ideal / n
            ia = np.degrees(np.arctan2(ideal[1], ideal[0]))
            votes = sm.gen_votes(ia, prev_angle, troll, N, rng)
            prev_angle = ia
            blend = sm.DIRS[votes].mean(axis=0)
            g = np.linalg.norm(blend)
            cur_dir = blend / g if g > 1e-10 else np.array([1., 0.])
        vel += sm.SMOOTH * (cur_dir * V - vel)
        pos = pos + vel * sm.DT
        pos_hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        errors[f] = np.linalg.norm(pos - cp)

    k = int(sm.FRAMES * warm_frac)
    return (float(np.sqrt(np.mean(errors ** 2))),
            float(np.sqrt(np.mean(errors[k:] ** 2))))


def sweep(traj, speeds, mc, warm_frac, seed_base=31, seed_off=150):
    out = {}
    for tms in TAUS_MS:
        delay_f = int(round(tms / 1000.0 / sm.DT))
        full, warm = [], []
        for v in speeds:
            fs, ws = [], []
            for i in range(mc):
                a, b = simulate_windowed(traj, 150, 0.05, i * seed_base + seed_off,
                                         v, delay_f, warm_frac)
                fs.append(a); ws.append(b)
            full.append(float(np.mean(fs)))
            warm.append(float(np.mean(ws)))
        out[tms] = {'full': full, 'warm': warm}
    return out


def shapefit(res, speeds, key, J):
    taus = np.array([t / 1000.0 for t in TAUS_MS])
    curves = [np.array(res[t][key], float) for t in TAUS_MS]
    k = len(taus)

    def resid(p):
        c = p[0]
        out = []
        for i, rs in enumerate(curves):
            A, B = p[1 + i], p[1 + k + i]
            u = speeds * (taus[i] + c) / LOOK
            pred = np.sqrt(np.maximum(A * J(u), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    p0 = [TAU_EXTRA] + [c.min() ** 2 for c in curves] + [c.min() * .5 for c in curves]
    lb = [0.0] + [0.0] * (2 * k)
    ub = [2.0] + [np.inf] * (2 * k)
    s = least_squares(resid, p0, bounds=(lb, ub))
    return (float(s.x[0]),
            float(100 * np.sqrt(np.mean(resid(s.x) ** 2))))


def vstar(speeds, rmses):
    r = np.asarray(rmses)
    i = int(np.argmin(r))
    if i in (0, len(r) - 1):
        return float(speeds[i]), 'EDGE'
    c = np.polyfit(speeds[i - 1:i + 2], r[i - 1:i + 2], 2)
    v = -c[1] / (2 * c[0]) if c[0] > 0 else speeds[i]
    return (float(v), 'parabola') if speeds[i - 1] <= v <= speeds[i + 1] \
        else (float(speeds[i]), 'grid')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=15)
    ap.add_argument('--warm', type=float, default=0.25,
                    help='fraction of the run discarded as start transient')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    speeds = np.arange(0.25, 6.01, 0.25) if not a.quick else np.arange(0.5, 6.01, 0.5)
    traj = sm.Circle()
    n = len(TAUS_MS) * len(speeds) * a.mc
    print('=' * 78)
    print('d60 -- warm-up discard diagnostic on the reference engine\'s own engine (Circle)')
    print('=' * 78)
    print('  engine imported unchanged from the repository')
    print('  speeds %.2f..%.2f step %.2f (%d)   vs the paper\'s 13-point grid'
          % (speeds[0], speeds[-1], speeds[1] - speeds[0], len(speeds)))
    print('  MC = %d, warm-up discard = %.0f%% of the run' % (a.mc, 100 * a.warm))
    print('  %d runs, about %.1f min' % (n, n * 0.13 / 60))

    t0 = time.time()
    res = sweep(traj, speeds, a.mc, a.warm)
    print('  done in %.1f min' % ((time.time() - t0) / 60))

    Jh = make_J('heading_servo')
    print()
    print('  %-22s %10s %10s' % ('', 'full run', 'warm-up cut'))
    c_f, r_f = shapefit(res, speeds, 'full', Jh)
    c_w, r_w = shapefit(res, speeds, 'warm', Jh)
    print('  %-22s %9.4f %10.4f' % ('composite c (s)', c_f, c_w))
    print('  %-22s %9.2f%% %9.2f%%' % ('shape residual', r_f, r_w))
    print('  a-priori c = %.4f s;  Nav2 residual with the same estimator 2.4-5.2%%'
          % TAU_EXTRA)

    print()
    print('  v* per tau (parabola on the fine grid):')
    print('  %8s %14s %14s %12s' % ('tau (ms)', 'full', 'warm-cut', 'paper (13pt)'))
    paper = {100: 3.931, 200: 3.231, 300: 2.577, 433: 2.134, 600: 1.758, 1000: 1.218}
    rows = []
    for t in TAUS_MS:
        vf, hf = vstar(speeds, res[t]['full'])
        vw, hw = vstar(speeds, res[t]['warm'])
        rows.append({'tau_ms': t, 'v_full': vf, 'how_full': hf,
                     'v_warm': vw, 'how_warm': hw, 'v_paper': paper[t]})
        print('  %8d %9.4f %-4s %9.4f %-4s %12.3f'
              % (t, vf, hf[:4], vw, hw[:4], paper[t]))

    out = {'speeds': speeds.tolist(), 'mc': a.mc, 'warm_frac': a.warm,
           'fit_full': {'c_s': c_f, 'rms_pct': r_f},
           'fit_warm': {'c_s': c_w, 'rms_pct': r_w},
           'vstar': rows, 'curves': res}
    json.dump(out, open(HERE / 'd60_warmup_diagnostic_2026-09-07.json', 'w'), indent=2)
    print('\n-> d60_warmup_diagnostic_2026-09-07.json')


if __name__ == '__main__':
    main()
