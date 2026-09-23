#!/usr/bin/env python3
"""
d63 -- Does the whole-lap measurement window remove E7's clustered scatter?

d52 found the measurement window spans 2.225 laps, so it ends 0.23 lap into a
corner cycle; where that cut falls depends on a trajectory phase nobody controls,
because the run length is quantised by the control period.  That was offered as a
HYPOTHESIS with a cheap test, not a finding -- the correlation with run length was
only r = 0.22 and one speed contradicted it.

run_point.py now reports rmse_m_intlap alongside rmse_m: the same run, scored over
an INTEGER number of laps (30.0 -> 70.0 m, exactly 2 laps, still 8.0 m clear of the
exit stub) instead of 30.0 -> 74.5 m.  E7' re-ran the five identical rows with both
windows recorded, so this is a paired comparison on the same runs -- no run-to-run
noise enters the difference at all.

Pass condition, stated before looking: the integer-lap window must reduce the sd of
u* materially below the 1.52% the original window gave.  If it does not, the
partial-corner hypothesis is wrong and d52's candidate is ruled out.

Usage:  python3 d63_intlap_window_test_2026-09-07.py
"""
import glob
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
L = 0.6
TAU_BOOK = 0.20 + 0.030          # tau_inject + T_ctrl/2 + T_sim/2


def locate(vs, rs):
    o = np.argsort(vs)
    vs, rs = np.asarray(vs)[o], np.asarray(rs)[o]
    i = int(np.argmin(rs))
    if i in (0, len(rs) - 1):
        return None
    x0, x1, x2 = vs[i - 1], vs[i], vs[i + 1]
    y0, y1, y2 = rs[i - 1], rs[i], rs[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-15:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if a <= 0:
        return None
    v = -b / (2 * a)
    return float(v) if x0 <= v <= x2 else None


def stats(us):
    us = np.array([u for u in us if u is not None])
    return us, float(100 * us.std(ddof=1) / us.mean()), \
        float(100 * (us.max() - us.min()) / us.mean())


def main():
    f = sorted(glob.glob(str(HERE / 'nav2_block_replicate2_*.json')))[-1]
    blk = json.load(open(f))
    rows = [r for r in blk['rows'] if r.get('curve')]

    print('=' * 78)
    print("d63 -- integer-lap window vs the original, paired on E7' runs")
    print('=' * 78)
    print('  %s, %d identical rows, %d points'
          % (pathlib.Path(f).name, len(rows), sum(len(r['curve']) for r in rows)))
    print('  original window 30.0-74.5 m = 2.225 laps;  integer 30.0-70.0 m = 2 laps')

    u_full, u_int, per_v = [], [], {}
    for r in rows:
        vs = np.array([c['v'] for c in r['curve']])
        rf = np.array([c['rmse'] for c in r['curve']])
        ri = np.array([c.get('rmse_intlap', np.nan) for c in r['curve']], float)
        vf, vi = locate(vs, rf), locate(vs, ri)
        u_full.append(vf * TAU_BOOK / L if vf else None)
        u_int.append(vi * TAU_BOOK / L if vi else None)
        for v, a, b in zip(vs, rf, ri):
            per_v.setdefault(round(float(v), 4), []).append((a, b))

    uf, sdf, rgf = stats(u_full)
    ui, sdi, rgi = stats(u_int)
    print()
    print('  %-22s %s' % ('u* original', ', '.join('%.6f' % x for x in uf)))
    print('  %-22s %s' % ('u* integer-lap', ', '.join('%.6f' % x for x in ui)))
    print()
    print('  %-22s %8s %8s' % ('', 'sd %', 'range %'))
    print('  %-22s %8.3f %8.3f' % ('original window', sdf, rgf))
    print('  %-22s %8.3f %8.3f' % ('integer-lap window', sdi, rgi))
    print('  %-22s %8.2fx %7.2fx' % ('reduction factor', sdf / sdi if sdi else np.inf,
                                     rgf / rgi if rgi else np.inf))

    print()
    print('  per-speed RMSE scatter across the five identical runs:')
    print('  %10s %14s %14s %10s' % ('v', 'orig spread%', 'intlap spread%', 'ratio'))
    ratios = []
    for v in sorted(per_v):
        a = np.array([x[0] for x in per_v[v]])
        b = np.array([x[1] for x in per_v[v]])
        sa = 100 * (a.max() - a.min()) / a.mean()
        sb = 100 * (b.max() - b.min()) / b.mean()
        ratios.append(sa / sb if sb > 0 else np.nan)
        print('  %10.4f %13.3f%% %13.3f%% %10.2f' % (v, sa, sb, ratios[-1]))

    print()
    verdict = ('CONFIRMED' if sdi < 0.7 * sdf else
               ('partial' if sdi < 0.95 * sdf else 'RULED OUT'))
    print('  VERDICT: %s' % verdict)
    if verdict == 'CONFIRMED':
        print('    The partial corner was the dominant noise source.  Adopt the')
        print('    integer-lap window; every interval computed against the 1.52%%')
        print('    reproducibility can be tightened to %.2f%%.' % sdi)
    elif verdict == 'partial':
        print('    The integer-lap window helps but does not account for the scatter.')
        print('    Keep it (it costs nothing and removes a known sensitivity) but the')
        print('    reproducibility floor stays at the measured %.2f%%.' % sdi)
    else:
        print("    d52's partial-corner hypothesis is RULED OUT: scoring over whole")
        print('    laps does not reduce the scatter.  The mechanism is elsewhere and')
        print('    the 1.52%% floor stands as measured.')

    json.dump({'file': pathlib.Path(f).name,
               'u_full': [float(x) for x in uf], 'u_int': [float(x) for x in ui],
               'sd_full_pct': sdf, 'sd_int_pct': sdi,
               'range_full_pct': rgf, 'range_int_pct': rgi,
               'verdict': verdict},
              open(HERE / 'd63_intlap_window_test_2026-09-07.json', 'w'), indent=2)
    print('\n-> d63_intlap_window_test_2026-09-07.json')


if __name__ == '__main__':
    main()
