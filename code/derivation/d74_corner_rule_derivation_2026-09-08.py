#!/usr/bin/env python3
"""
d74 -- derive the corner-coverage threshold instead of picking one.

the path-following note, section 6 (not distributed) recommends retaining a grid speed only if the agent
traverses at least K corners inside the fixed horizon, and section 9 admits that
K = 4 was a placeholder with nothing behind it.  A rule with an invented constant
is not better than no rule; it just moves the arbitrariness somewhere less
visible.  So K gets measured.

THE QUANTITY.  What low-speed truncation damages is the estimate of the
geometry's steady-state tracking error: with a fixed horizon, a slow agent scores
a short, unrepresentative arc plus a start transient that has not decayed.  So
the bias is

    bias(v) = RMSE(65 s) / RMSE(long) - 1

where the long run is taken as converged.  That bias is a function of how much
geometry was sampled, i.e. of the corner count v*DUR/perimeter*n_corners.

THE THRESHOLD.  A bias only matters when it exceeds the noise the experiment
already carries, so K is the smallest corner count at which |bias| drops below
the engine's own seed-to-seed scatter of RMSE at the same point.  That makes the
rule's constant a measured denominator rather than a choice -- the same
discipline the Nav2 side used when it replaced a 0.7% invented tolerance with a
measured 1.269% reproducibility floor.

Stated before running: the bias should be large and positive at low v (a short
arc dominated by the start transient), fall as more geometry is covered, and
cross the scatter somewhere in the low single digits of corners.

Usage:  python3 d74_corner_rule_derivation_2026-09-08.py [--mc 5] [--long 800]
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402


def run(traj, V, tau_ms, dur, seed, N=150, troll=0.05):
    rng = np.random.default_rng(seed)
    frames = int(dur / sm.DT)
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    gam = 0.5
    err = np.empty(frames)
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + sm.LOOK) - dp
            n = float(np.linalg.norm(ideal))
            if n > 1e-10:
                ideal = ideal / n
            ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
            votes = sm.gen_votes(ia, prev, troll, N, rng)
            prev = ia
            b = sm.DIRS[votes].mean(axis=0)
            gam = float(np.linalg.norm(b))
            if gam > 1e-10:
                cur = b / gam
            tgt = cur * math.sqrt(gam) * V
        else:
            tgt = cur * V
        vel += sm.SMOOTH * (tgt - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        err[f] = float(np.linalg.norm(pos - cp))
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=5)
    ap.add_argument('--long', type=float, default=800.0)
    ap.add_argument('--tau', type=int, default=433)
    ap.add_argument('--traj', default='square',
                    choices=['square', 'circle'],
                    help='the path-following note (not distributed) assumes a smooth path needs no '
                         'coverage rule because it is homogeneous in '
                         'arclength.  That is an assumption, not a '
                         'measurement -- --traj circle tests it.')
    a = ap.parse_args()

    if a.traj == 'circle':
        traj = sm.Circle(10)
        ncorner = 0                  # smooth: no corners to miss
    else:
        traj = sm.Square()
        ncorner = 4                  # four turns; the five segments are an
                                     # artefact of the start point sitting mid-edge
    perim = float(traj.circ)
    speeds = np.arange(0.25, 5.01, 0.25)

    print('=' * 78)
    print('d74 -- deriving the corner-coverage threshold on the Square')
    print('=' * 78)
    print('  %s: perimeter %.1f m, %d corners, horizon %.0f s, reference %.0f s'
          % (a.traj, perim, ncorner, sm.DUR, a.long))
    print('  tau %d ms, MC %d seeds' % (a.tau, a.mc))
    print()
    print('  %6s %7s %8s %10s %10s %9s %9s'
          % ('v', 'laps', 'corners', 'RMSE 65s', 'RMSE ref', 'bias', 'scatter'))

    t0 = time.time()
    rows = []
    for V in speeds:
        short = [run(traj, V, a.tau, sm.DUR, k * 31 + 150) for k in range(a.mc)]
        longr = [run(traj, V, a.tau, a.long, k * 31 + 150) for k in range(a.mc)]
        ms, ml = float(np.mean(short)), float(np.mean(longr))
        scatter = float(np.std(short, ddof=1) / ms)
        bias = ms / ml - 1
        corners = V * sm.DUR / perim * ncorner if ncorner else 0.0
        laps = V * sm.DUR / perim
        rows.append({'v': float(V), 'laps': laps, 'corners': corners,
                     'rmse_short': ms, 'rmse_long': ml, 'bias': bias,
                     'scatter': scatter})
        print('  %6.2f %7.2f %8.2f %10.4f %10.4f %8.1f%% %8.1f%%'
              % (V, laps, corners, ms, ml, 100 * bias, 100 * scatter))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    # Where does the SYSTEMATIC part of the bias end?  d79 showed that comparing
    # |bias| against the per-run scatter is the wrong test -- the scatter does
    # not shrink with more seeds, so that comparison never converges.  What the
    # data actually shows is a sign change: the bias is systematically negative
    # below about one lap and scatters about zero above it.  So report the lap
    # count at which the systematic run of negative bias ends, and let the reader
    # see the table rather than a manufactured constant.
    print()
    neg = [r for r in rows if r['bias'] < -0.05]
    if neg:
        last = max(r['laps'] for r in neg)
        print('  bias below -5%% occurs up to %.2f laps (v = %.2f);'
              % (last, max(r['v'] for r in neg)))
        run_end = None
        for r in rows:
            if r['bias'] >= -0.05:
                run_end = r['laps']
                break
        print('  the unbroken run of negative bias ends at %.2f laps.'
              % (run_end if run_end else float('nan')))
    else:
        print('  no point on this grid has a bias below -5%%.')
    print('  -> the coverage rule is in LAPS of the repeating unit, not corners;')
    print('     see the path-following note, section 6 (not distributed).')

    json.dump({'traj': a.traj, 'perimeter': perim, 'n_corners': ncorner,
               'horizon_s': sm.DUR,
               'reference_s': a.long, 'tau_ms': a.tau, 'mc': a.mc,
               'rows': rows, 'K_corners': None},
              open(HERE / ('d74_corner_rule_%s_2026-09-08.json' % a.traj),
                   'w'), indent=2)
    print('\n-> d74_corner_rule_2026-09-08.json')


if __name__ == '__main__':
    main()
