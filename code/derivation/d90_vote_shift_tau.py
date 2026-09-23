#!/usr/bin/env python3
"""
d90 -- is the vote layer's shift of u* a property of the layer, or does it
       interact with the delay?

d85 measured the shift on the circle at ONE latency: u* goes 0.5164 -> 0.7206,
+39.5%, at tau = 433 ms.  That is now a headline result -- it is what separates
the derived 0.4995 (deterministic plant) from the published 0.7109 (crowd plant)
-- and it rests on a single point.

d89 established the mechanism: the shift scales with the fraction of the circuit
turned at a CONSTANT rate (arc fraction 0.386/0.653/0.834/1.000 -> shift
3.3/6.3/28.4/39.5%), and the ellipse, which turns at a varying rate, barely
moves.  If the mechanism is a lock-in between the 45 deg vote grid and the
heading's rotation, then the lock-in condition involves the turn rate per vote
interval, v*WIN/R, which changes with v -- and v* itself changes with tau.  So
the shift has no reason to be tau-invariant, and measuring it is the test.

Stated before running:
  * if the shift is flat in tau, it is a pure layer effect and u*(crowd) is a
    constant of the aggregation, quotable on its own;
  * if it varies, the crowd margin is not a constant at all and every published
    crowd u* has to carry its tau.

Circle only: it carries the largest signal, and d89 already mapped the geometry
axis.

Usage:  python3 d90_vote_shift_tau.py [--mc 8]
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2] / "engine" / "crowd_control_engine"   # the engine as it ships in this package
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402


def _l(f, n):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d84 = _l('d84_stadium_sweep_2026-09-08.py', 'd84')
d85 = _l('d85_crowd_ellipse_2026-09-08.py', 'd85')

LOOK = sm.LOOK
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=8)
    a = ap.parse_args()
    frames = int(240.0 / sm.DT)
    g = d85.Param(10.0, 10.0)

    print('=' * 78)
    print('d90 -- tau dependence of the vote layer shift (circle)')
    print('=' * 78)
    print('  DUR 240 s, MC %d, bowl estimator' % a.mc)
    print('  d85 at tau = 433: 0.5164 -> 0.7206  (+39.5%)')
    print()
    print('  %6s %10s %11s %11s %10s %10s %9s'
          % ('tau', 'tau_tot', 'u* det', 'u* crowd', 'shift', 'v* crowd',
             'turn/vote'))

    out = {'mc': a.mc, 'rows': []}
    t0 = time.time()
    for tms in (100, 200, 433, 1000):
        tt = tms / 1000.0 + TAU_EXTRA
        df = int(round(tms / 1000.0 / sm.DT))
        # centre the grid on the deterministic optimum but leave room above for
        # the crowd shift, which d85 showed can be +40%
        v0 = 0.4995 * LOOK / tt
        speeds = np.arange(max(0.2, 0.5 * v0), 2.2 * v0, 0.03 * v0)
        res = {}
        for layer, q, mc in (('det', False, 1), ('crowd', True, a.mc)):
            rr = []
            for v in speeds:
                vals = [d85.simulate(g, v, df, frames, 900 + 37 * k, q)
                        for k in range(mc)]
                rr.append(float(np.mean(vals)) if np.all(np.isfinite(vals))
                          else float('nan'))
            vv, how, n = d84.bowl_locate(speeds, rr, factor=4.0)
            res[layer] = (vv, how, n)
        ud = res['det'][0] * tt / LOOK if res['det'][0] else None
        uc = res['crowd'][0] * tt / LOOK if res['crowd'][0] else None
        sh = 100 * (uc / ud - 1) if (ud and uc) else float('nan')
        # heading turned per vote interval at the crowd optimum, in vote cells
        turn = (math.degrees(res['crowd'][0] * sm.WIN / 10.0) / 45.0
                if res['crowd'][0] else float('nan'))
        out['rows'].append({'tau_ms': tms, 'tau_tot': tt, 'u_det': ud,
                            'u_crowd': uc, 'shift_pct': sh,
                            'v_crowd': res['crowd'][0],
                            'turn_per_vote_cells': turn,
                            'how_det': res['det'][1], 'how_crowd': res['crowd'][1]})
        print('  %6d %10.4f %11.4f %11.4f %9.1f%% %10.4f %9.3f'
              % (tms, tt, ud or float('nan'), uc or float('nan'), sh,
                 res['crowd'][0] or float('nan'), turn))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    sh = [r['shift_pct'] for r in out['rows'] if np.isfinite(r['shift_pct'])]
    print()
    if len(sh) >= 3:
        print('  shift across tau: %.1f%% .. %.1f%%  (spread %.1f points)'
              % (min(sh), max(sh), max(sh) - min(sh)))
        if max(sh) - min(sh) < 5:
            print('  -> FLAT: the shift is a property of the aggregation layer,')
            print('     and u*(crowd) can be quoted without a latency.')
        else:
            print('  -> NOT flat: the crowd margin depends on tau, so every')
            print('     published crowd u* has to carry the latency it was')
            print('     measured at.  That includes the reference engine\'s.')

    json.dump(out, open(HERE / 'd90_vote_shift_tau_2026-09-08.json', 'w'), indent=2)
    print('\n-> d90_vote_shift_tau_2026-09-08.json')


if __name__ == '__main__':
    main()
