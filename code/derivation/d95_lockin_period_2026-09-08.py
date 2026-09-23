#!/usr/bin/env python3
"""
d95 -- turn the reference engine's autocorrelation coincidence into a law, on its own engine.

the reference engine measured the autocorrelation of the direction error to justify an
independence assumption.  The assumption failed (the file records
independence: False) and a secondary peak at lag 13 was noted in passing:

    "The period k x WIN = 13 x 0.3 s = 3.9 s coincides with the time for the
     Circle trajectory to rotate 45 deg at v = 2.0 m/s"

The word doing the work there is "coincides".  d89 measured, on a different
axis, that the vote layer shifts u* in proportion to the fraction of the circuit
turned at a CONSTANT rate -- 3.3%, 6.3%, 28.4%, 39.5% as the arc fraction goes
0.386, 0.653, 0.834, 1.000 -- and that an ellipse, which turns at a varying rate,
barely moves.  That is a lock-in between the heading's rotation and the 45 deg
vote grid, and if it is real the autocorrelation period is not a coincidence but
a measurement of it:

    k_peak * WIN = (pi/4) R / v      ->      k_peak = (pi/4) R / (v WIN)

At R = 10, v = 2.0, WIN = 0.3 that is 13.09, against the observed 13.  One point
cannot distinguish a law from a coincidence, so this sweeps the speed: the peak
must move as 1/v, and doubling v must halve the lag.

Stated before running:
    v = 1.0 -> k = 26.2      v = 3.0 -> k = 8.7
    v = 2.0 -> k = 13.1      v = 4.0 -> k = 6.5

A peak that sits at 13 regardless of speed would refute the lock-in outright.

Usage:  python3 d95_lockin_period_2026-09-08.py
"""
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2] / "engine" / "crowd_control_engine"   # the engine as it ships in this package
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402


def theta_err_series(traj, speed, N=150, troll=0.05, seed=7, warm=0.25):
    """The engine's own loop, recording aggregate-vs-ideal direction error."""
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    prev = 0.0
    cur = np.array([1.0, 0.0])
    errs = []
    for f in range(sm.FRAMES):
        if f % sm.VOTE_INT == 0:
            di = max(0, len(hist) - 1 - sm.DELAY_F)
            dp = hist[di]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + sm.LOOK) - dp
            n = float(np.linalg.norm(ideal))
            if n > 1e-10:
                ideal = ideal / n
            ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
            votes = sm.gen_votes(ia, prev, troll, N, rng)
            prev = ia
            b = sm.DIRS[votes].mean(axis=0)
            g = float(np.linalg.norm(b))
            if g > 1e-10:
                cur = b / g
            aa = float(np.degrees(np.arctan2(cur[1], cur[0])))
            errs.append((aa - ia + 180.0) % 360.0 - 180.0)
        vel += sm.SMOOTH * (cur * speed - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
    e = np.array(errs)
    return e[int(len(e) * warm):]


def acf(x, nlags):
    x = x - x.mean()
    d = float(x @ x)
    return [1.0] + [float(x[:-k] @ x[k:]) / d for k in range(1, nlags + 1)]


def first_peak(a, lo=4):
    """First interior local maximum above lag `lo`."""
    best = None
    for k in range(lo, len(a) - 1):
        if a[k] > a[k - 1] and a[k] >= a[k + 1] and a[k] > 0.1:
            if best is None or a[k] > a[best]:
                best = k
            break
    return best


def main():
    traj = sm.Circle()
    R = getattr(traj, 'R', 10.0)
    print('=' * 78)
    print('d95 -- does the autocorrelation peak track 1/v, as a lock-in requires?')
    print('=' * 78)
    print('  engine defaults: WIN %.2f s, VOTE_INT %d, R %.1f m, N 150, troll 5%%'
          % (sm.WIN, sm.VOTE_INT, R))
    print('  predicted peak lag = (pi/4) R / (v WIN)')
    print()
    print('  %6s %12s %12s %10s %10s'
          % ('v', 'predicted k', 'observed k', 'acf peak', 'ratio'))

    out = {'R': R, 'win': sm.WIN, 'rows': []}
    for v in (1.0, 1.5, 2.0, 3.0, 4.0):
        e = theta_err_series(traj, v)
        nl = min(60, len(e) // 4)
        a = acf(e, nl)
        k = first_peak(a)
        pred = (math.pi / 4) * R / (v * sm.WIN)
        out['rows'].append({'v': v, 'k_pred': pred, 'k_obs': k,
                            'acf_at_peak': (a[k] if k else None),
                            'acf': a[:40]})
        print('  %6.1f %12.2f %12s %10s %10s'
              % (v, pred, k if k else '-',
                 ('%.3f' % a[k]) if k else '-',
                 ('%.2f' % (k / pred)) if k else '-'))

    ok = [r for r in out['rows'] if r['k_obs']]
    print()
    if len(ok) >= 3:
        lv = np.log([r['v'] for r in ok])
        lk = np.log([r['k_obs'] for r in ok])
        slope = float(np.polyfit(lv, lk, 1)[0])
        rat = np.array([r['k_obs'] / r['k_pred'] for r in ok])
        print('  k_obs ~ v^%+.3f   (a lock-in requires -1)' % slope)
        print('  k_obs / k_pred: mean %.3f, spread %.1f%%'
              % (rat.mean(), 100 * rat.std(ddof=1) / rat.mean()))
        out['slope'] = slope
        out['ratio_mean'] = float(rat.mean())
        if abs(slope + 1.0) < 0.15:
            print()
            print('  -> the peak tracks 1/v.  The lag-13 observation was not a')
            print('     coincidence; it is one point on a law, and the law is the')
            print('     lock-in d89 measured on the geometry axis.')
        else:
            print()
            print('  -> the peak does NOT track 1/v.  The lock-in account fails on')
            print("     the main paper's own data, and d89's arc-fraction scaling")
            print('     needs a different explanation.')

    json.dump(out, open(HERE / 'd95_lockin_period_2026-09-08.json', 'w'), indent=2)
    print('\n-> d95_lockin_period_2026-09-08.json')


if __name__ == '__main__':
    main()
