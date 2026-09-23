#!/usr/bin/env python3
"""
d52 -- What the clustered noise in E7 actually is: a partial corner in the window.

E7 ran the identical row five times and got u* = 0.3122, 0.3122, 0.3050, 0.3122,
0.3025 -- three values identical to four digits and two others, i.e. clustered,
not Gaussian (largest gap 5.6x the median).  That scatter, sd 1.52%, is the
denominator every null control and every coefficient interval is measured against,
so its mechanism is worth finding rather than absorbing.

The 25 raw point files are archived here.  They identify a REMOVABLE
sensitivity; they do not prove it is the whole of the 1.52%.  What they show:

  * every run whose RMSE differs also has a different SAMPLE COUNT and a longer
    ELAPSED time, by a small integer number of control periods (0.05 s at 20 Hz).
    The goal check fires once per control cycle, so the run length is quantised
    and which quantum you land in is a coin flip;
  * the extra samples land INSIDE the measurement window, not after it;
  * and the window spans 30.0 -> 74.5 m of a 20 m lap = 2.225 LAPS.

A window of 2.225 laps ends part-way through a corner cycle, so where it is cut
depends on a phase nobody controls.  An INTEGER number of laps has no partial
corner and cannot have this sensitivity: 30.0 -> 70.0 m is exactly 2 laps and
still leaves 8.0 m of margin to the 3.5 m tail exclusion.

BUT the correlation between run-length variation and RMSE spread is only r = 0.22,
and v = 0.8191 varies by 0.100 s of run length with an RMSE spread of just 0.31%.
So run length is a proxy, not the mechanism, and this is a hypothesis with a cheap
test rather than a finding.  run_point.py now reports rmse_m_intlap alongside
rmse_m; a replicate re-run decides it.  Nothing already reported is invalidated
either way -- the effect is inside the 1.52% reproducibility that every interval
was computed against.

Usage:  python3 d52_noise_mechanism_2026-09-06.py
"""
import collections
import glob
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
RAW = HERE / 'nav2_raw_points_replicate_2026-09-06'


def main():
    files = sorted(glob.glob(str(RAW / 'rep*_v*.json')))
    if not files:
        raise SystemExit('no archived replicate point files')
    by = collections.defaultdict(list)
    for f in files:
        name = pathlib.Path(f).stem
        rep, v = name.split('_v')[0], name.split('_v')[1]
        by[v].append((rep, json.load(open(f))))

    print('=' * 78)
    print('d52 -- the mechanism behind E7\'s clustered noise')
    print('=' * 78)
    d0 = by[sorted(by)[0]][0][1]
    lap, tot = d0['lap_length_m'], d0['total_length_m']
    settle, tail = 1.5, 3.5
    lo, hi = settle * lap, tot - tail
    n_int = int(math.floor((hi - lo) / lap + 1e-9))
    print('  lap %.1f m, total %.1f m, window %.1f -> %.1f m = %.4f LAPS'
          % (lap, tot, lo, hi, (hi - lo) / lap))
    print('  integer-lap window would be %.1f -> %.1f m (%d laps), '
          'leaving %.1f m to the stub' % (lo, lo + n_int * lap, n_int,
                                          tot - (lo + n_int * lap)))
    print('  -> the present window ends %.2f lap into a corner cycle.'
          % ((hi - lo) / lap - n_int))

    print()
    print('  per speed: does RMSE move together with run length and sample count?')
    print('  %-10s %10s %12s %10s %12s' %
          ('v', 'rmse spread', 'elapsed span', 'd_samples', 'ctrl periods'))
    rows = []
    for v in sorted(by):
        recs = [r[1] for r in by[v]]
        rm = np.array([r['rmse_m'] for r in recs])
        el = np.array([r['elapsed_s'] for r in recs])
        ns = np.array([r['n_samples_total'] for r in recs])
        spread = 100 * (rm.max() - rm.min()) / rm.mean()
        span = el.max() - el.min()
        rows.append({'v': float(v), 'rmse_spread_pct': spread,
                     'elapsed_span_s': float(span),
                     'sample_span': int(ns.max() - ns.min()),
                     'ctrl_periods': float(span / 0.05)})
        print('  %-10s %9.2f%% %11.4fs %10d %12.2f'
              % (v, spread, span, ns.max() - ns.min(), span / 0.05))

    sp = np.array([r['rmse_spread_pct'] for r in rows])
    el = np.array([r['elapsed_span_s'] for r in rows])
    if len(sp) > 2 and el.std() > 0:
        rho = float(np.corrcoef(sp, el)[0, 1])
        print()
        print('  correlation between RMSE spread and run-length span: r = %+.3f' % rho)
        print('  This is WEAK, and there is a clear counterexample: v = 0.8191 varies')
        print('  by 0.100 s of run length yet its RMSE spreads only 0.31%.  So run')
        print('  length is a PROXY, not the mechanism.  What the data does support is')
        print('  the one-way implication: the speed with no run-length variation at')
        print('  all (v = 1.0239, 0.007 s) also has essentially no RMSE spread.')
        print('  A run can be longer without shifting the phase AT THE WINDOW EDGE if')
        print('  the extra time is spent after the window closes -- which is exactly')
        print('  what a partial-corner sensitivity predicts, and why the fix is to')
        print('  remove the partial corner rather than to stabilise the run length.')

    # the one row where nothing varied
    quiet = min(rows, key=lambda r: r['elapsed_span_s'])
    loud = max(rows, key=lambda r: r['elapsed_span_s'])
    print()
    print('  quietest speed v=%.4f: run length varies %.4f s -> RMSE spread %.2f%%'
          % (quiet['v'], quiet['elapsed_span_s'], quiet['rmse_spread_pct']))
    print('  loudest  speed v=%.4f: run length varies %.4f s -> RMSE spread %.2f%%'
          % (loud['v'], loud['elapsed_span_s'], loud['rmse_spread_pct']))

    print()
    print('  WHAT IS ESTABLISHED, AND WHAT IS NOT')
    print('  ESTABLISHED (structural, no statistics needed): the measurement window')
    print('    spans 2.225 laps, so it ends 0.23 lap into a corner cycle.  Where that')
    print('    cut falls depends on the trajectory phase, and the phase is not')
    print('    controlled -- the run length is quantised by the control period.')
    print('  ESTABLISHED (data): runs that differ in RMSE differ in sample count and')
    print('    elapsed time by whole control periods, and the extra samples fall')
    print('    INSIDE the window; the one speed with no length variation has no')
    print('    RMSE spread.')
    print('  NOT ESTABLISHED: that run length predicts the spread.  r = %+.3f is weak'
          % rho)
    print('    and v = 0.8191 contradicts it.  So this identifies a real and')
    print('    removable sensitivity, but does not yet prove it is the whole of the')
    print('    1.52%% sd.')
    print()
    print('  NEXT STEP, not a claim: run_point.py now reports rmse_m_intlap over a')
    print('  whole-lap window alongside the original.  Re-running the replicate block')
    print('  decides it -- if the integer-lap sd is materially below 1.52%%, adopt it;')
    print('  if not, the mechanism is elsewhere and this was a dead end worth ruling')
    print('  out.  Nothing already reported is invalidated either way: the effect is')
    print('  inside the 1.52%% reproducibility every interval was computed against.')

    json.dump({'lap_m': lap, 'total_m': tot, 'window_laps': (hi - lo) / lap,
               'window_laps_int': n_int, 'per_speed': rows},
              open(HERE / 'd52_noise_mechanism_2026-09-06.json', 'w'), indent=2)
    print('\n-> d52_noise_mechanism_2026-09-06.json')


if __name__ == '__main__':
    main()
