#!/usr/bin/env python3
"""
d72 -- does the arc-monotonic scheme clear the lemniscate lobe capture, in the
       region where the capture actually happens?

d71 located the artefact: it is NOT everywhere.  Lobe capture appears at LOW
speed and SHORT latency and clears as either rises --

    v = 1.0, tau = 100   lobe balance 0.000    captured
    v = 1.0, tau = 433   lobe balance 0.000    captured
    v = 1.0, tau = 1000  lobe balance 0.440    fine
    v = 5.0, tau = 100   lobe balance 0.491    fine

-- so d70's first pass missed it by testing at v = 2.0, tau = 433, which is
outside the capture region.  Any claim that a fix works has to be made where the
thing being fixed occurs.

Written down before running:
  * arc-monotonic should clear the capture, because capture is a projection
    ambiguity at the self-crossing and the monotone window cannot see the other
    branch;
  * on the conditions that were ALREADY fine, monotone should change nothing.

The metric is lobe occupancy, not arc coverage.  d70 used coverage and got 100%
even under capture, because the global projection maps a position on one lobe
onto both branches near the crossing -- the coverage statistic is contaminated by
the very ambiguity it is trying to detect.

Usage:  python3 d72_lemniscate_fix_2026-09-08.py
"""
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402

_s = importlib.util.spec_from_file_location(
    'd70', HERE / 'd70_arc_monotonic_2026-09-08.py')
d70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d70)

TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH      # this study's composition


def run(V, tau_ms, scheme, seed=181, N=150, troll=0.05):
    traj = sm.Lemniscate()
    mono = d70.ArcMonotone(traj)
    mono.reset(0.0)
    mono.fwd = max(3.0, 4.0 * V * sm.WIN)
    rng = np.random.default_rng(seed)
    frames = sm.FRAMES
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    gam = 0.5
    err = np.empty(frames)
    xs = np.empty(frames)
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            if scheme == 'monotone':
                _, arc = mono.closest_ctrl(dp)
            else:
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
        xs[f] = float(pos[0])
    r = float(np.mean(xs > 0.3))
    l = float(np.mean(xs < -0.3))
    t = r + l
    return {'rmse': float(np.sqrt(np.mean(err ** 2))),
            'lobe_balance': float(min(r, l) / t) if t > 0 else 0.0,
            'frac_left': l, 'frac_right': r}


def main():
    print('=' * 78)
    print('d72 -- arc-monotonic vs the shipped projection, IN the capture region')
    print('=' * 78)
    print('  lobe balance 0.50 = both lobes equally;  0.00 = one lobe only')
    print('  u = v * tau_tot / LOOK, with tau_tot = tau + WIN/2 + DT/SMOOTH')
    print()
    print('  %5s %6s %7s | %11s %11s | %9s %9s'
          % ('v', 'tau', 'u', 'bal global', 'bal mono', 'RMSE gl', 'RMSE mo'))
    cases = [(1.0, 100), (1.0, 200), (1.0, 433), (1.0, 1000),
             (2.0, 100), (2.0, 433), (3.0, 100), (3.0, 433),
             (5.0, 100), (5.0, 433)]
    out = {'rows': []}
    for V, tms in cases:
        u = V * (tms / 1000.0 + TAU_EXTRA) / sm.LOOK
        g = run(V, tms, 'global')
        m = run(V, tms, 'monotone')
        out['rows'].append({'v': V, 'tau_ms': tms, 'u': u,
                            'global': g, 'monotone': m})
        flag = ''
        if g['lobe_balance'] < 0.10:
            flag = '   CAPTURED -> %s' % ('CLEARED' if m['lobe_balance'] > 0.30
                                          else 'STILL CAPTURED')
        print('  %5.1f %6d %7.3f | %11.3f %11.3f | %9.4f %9.4f%s'
              % (V, tms, u, g['lobe_balance'], m['lobe_balance'],
                 g['rmse'], m['rmse'], flag))

    cap = [r for r in out['rows'] if r['global']['lobe_balance'] < 0.10]
    ok = [r for r in out['rows'] if r['global']['lobe_balance'] >= 0.10]
    print()
    if cap:
        cleared = sum(1 for r in cap if r['monotone']['lobe_balance'] > 0.30)
        print('  captured under the shipped scheme: %d of %d cases'
              % (len(cap), len(out['rows'])))
        print('  cleared by arc-monotonic:          %d of %d' % (cleared, len(cap)))
    if ok:
        d = [abs(r['monotone']['rmse'] / r['global']['rmse'] - 1) for r in ok]
        print('  where nothing was wrong, RMSE moves by %.2f%% on average, '
              '%.2f%% worst' % (100 * np.mean(d), 100 * np.max(d)))
        print('  (this is the no-op check: the scheme must not disturb the')
        print('   conditions that were already sound)')

    json.dump(out, open(HERE / 'd72_lemniscate_fix_2026-09-08.json', 'w'), indent=2)
    print('\n-> d72_lemniscate_fix_2026-09-08.json')


if __name__ == '__main__':
    main()
