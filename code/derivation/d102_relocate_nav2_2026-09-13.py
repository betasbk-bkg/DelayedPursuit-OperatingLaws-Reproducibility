#!/usr/bin/env python3
"""
d102 -- re-locate every Nav2 optimum with the estimators that superseded the
        one the headline constant was measured with.

WHY.  The paper's third-party constant, u* = 0.3029 pooled over nine blocks, was
located by a three-point parabola through the grid minimum and its two
neighbours (`experiment.locate`).  Two later results say that estimator is not
the one to use:

  * d78 showed it moves with the grid and replaced it with a whole-bowl fit for
    the crowd-engine geometry work;
  * d100 showed the whole-bowl fit is itself biased on steeply asymmetric curves
    (it returned 0.0503 against a grid minimum of 0.0903) and replaced it with a
    window symmetric in u about the grid minimum.

Neither correction was ever propagated to the Nav2 layer, and the manuscript now
carries that as an open problem.  This closes it: every stored RMSE(v) curve is
re-located under all three estimators and the pooled constant is recomputed.

THE SECOND THING THIS MEASURES.  d56 pools only rows whose stored verdict is
`how == 'parabola'`, i.e. it silently drops the rows where the estimator failed
to find an interior optimum.  That is selection on the quantity being measured.
The pooled constant is therefore also recomputed WITHOUT that filter, so the
size of the selection effect is visible rather than assumed small.

SELF-TEST FIRST.  Re-running the three-point estimator on a stored curve must
reproduce that row's stored v_obs.  If it does not, this script is reading the
curves differently from the code that produced them and nothing below means
anything.

Usage:  python3 d102_relocate_nav2_2026-09-13.py
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
DERIVED = 0.306034

# the nine blocks d56 pools, pinned by name (d56's own PINNED table)
BLOCKS = [
    ('nav2_block_freq_2026-09-05.json', 'E2 freq'),
    ('nav2_block_tsim_2026-09-05.json', 'E3 tsim'),
    ('nav2_block_lookahead_2026-09-05.json', 'E4 lookahead'),
    ('nav2_block_alpha_2026-09-06.json', 'E5 alpha (a<=90)'),
    ('nav2_block_nulls_2026-09-06.json', 'E6 nulls'),
    ('nav2_block_replicate_2026-09-06.json', 'E7 replicate'),
    ('nav2_block_freq_iso_2026-09-06.json', 'E8 freq_iso t=.10'),
    ('nav2_block_freq_iso_tau005_2026-09-06.json', 'E11a freq_iso t=.05'),
    ('nav2_block_freq_iso_tau020_2026-09-06.json', 'E11b freq_iso t=.20'),
]


# ------------------------------------------------------------- the estimators
def three_point(vs, rs):
    """experiment.locate: parabola through the grid minimum and its neighbours."""
    i = int(np.argmin(rs))
    if i in (0, len(rs) - 1):
        return vs[i], 'EDGE'
    (x0, y0), (x1, y1), (x2, y2) = (vs[i - 1], rs[i - 1]), (vs[i], rs[i]), (vs[i + 1], rs[i + 1])
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-15:
        return vs[i], 'grid'
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if a <= 0:
        return vs[i], 'grid'
    v = -b / (2 * a)
    return (v, 'parabola') if x0 <= v <= x2 else (vs[i], 'grid')


def bowl(vs, rs, factor=4.0):
    """d78/d84: one parabola through every point within factor x min."""
    lo = min(rs) * factor
    sel = [k for k in range(len(rs)) if rs[k] <= lo]
    if len(sel) < 3:
        return None, 'too few'
    c = np.polyfit([vs[k] for k in sel], [rs[k] for k in sel], 2)
    if c[0] <= 0:
        return None, 'convex'
    v = -c[1] / (2 * c[0])
    return (float(v), 'bowl') if vs[0] <= v <= vs[-1] else (None, 'outside')


def symmetric(vs, rs):
    """d100: widest window symmetric in v about the grid minimum, m = 2 primary."""
    i = int(np.argmin(rs))
    if i in (0, len(rs) - 1):
        return None, 'EDGE', None
    ests = {}
    for m in (1, 2, 3):
        if i - m < 0 or i + m >= len(rs):
            continue
        x = np.array(vs[i - m:i + m + 1])
        y = np.array(rs[i - m:i + m + 1])
        c = np.polyfit(x, y, 2)
        if c[0] <= 0:
            continue
        v = -c[1] / (2 * c[0])
        if x[0] <= v <= x[-1]:
            ests[m] = float(v)
    if not ests:
        return None, 'no fit', None
    return ests.get(2, ests[max(ests)]), 'symmetric', (min(ests.values()), max(ests.values()))


def main():
    print('=' * 78)
    print('d102 -- the Nav2 constant under the estimator that superseded its own')
    print('=' * 78)

    rows = []
    worst_self = 0.0
    n_self = 0
    for fn, label in BLOCKS:
        p = HERE / fn
        if not p.exists():
            print('  NOT READ: %s' % fn)
            continue
        blk = json.load(open(p, encoding='utf-8'))
        for r in blk['rows']:
            cur = r.get('curve')
            if not cur or not r.get('u_obs') or not r.get('v_obs'):
                continue
            if label.startswith('E5') and r.get('alpha_deg', 0) > 90:
                continue
            pts = sorted((c['v'], c['rmse']) for c in cur
                         if c.get('rmse') is not None and c['rmse'] == c['rmse'])
            if len(pts) < 5:
                continue
            vs = [p_[0] for p_ in pts]
            rs = [p_[1] for p_ in pts]
            # u = v * (tau_tot / L); the stored pair gives that ratio exactly
            k = r['u_obs'] / r['v_obs']

            v3, how3 = three_point(vs, rs)
            if how3 == 'parabola' and r.get('how') == 'parabola':
                worst_self = max(worst_self, abs(v3 / r['v_obs'] - 1))
                n_self += 1
            vb, howb = bowl(vs, rs)
            vsym, hows, band = symmetric(vs, rs)
            rows.append({
                'block': label, 'stored_how': r.get('how'),
                'k': k,
                'u_stored': r['u_obs'],
                'u_3pt': v3 * k if v3 else None, 'how_3pt': how3,
                'u_bowl': vb * k if vb else None, 'how_bowl': howb,
                'u_sym': vsym * k if vsym else None, 'how_sym': hows,
                'u_sym_band': [band[0] * k, band[1] * k] if band else None,
            })

    print('\n  self-test: three-point estimator re-run on the stored curves')
    print('    max relative deviation from the stored v_obs: %.2e  (n = %d)'
          % (worst_self, n_self))
    if worst_self > 1e-6:
        print('    -> FAILED. This script reads the curves differently from the')
        print('       code that produced them; nothing below is meaningful.')
        return 1
    print('    -> passed; the curves are being read the same way.')

    def pool(key, only_parabola):
        vals = [r[key] for r in rows
                if r[key] is not None
                and (not only_parabola or r['stored_how'] == 'parabola')]
        if len(vals) < 3:
            return None
        a = np.array(vals)
        return (len(a), float(a.mean()), float(100 * a.std(ddof=1) / a.mean()),
                100 * (a.mean() / DERIVED - 1))

    print('\n  pooled constant, by estimator and by whether d56\'s filter is applied')
    print('  %-26s %6s %10s %9s %11s' % ('', 'n', 'mean u*', 'sd %', 'vs derived'))
    out = {'derived': DERIVED, 'self_test': worst_self, 'pooled': {}}
    for key, name in (('u_stored', 'stored (3-point)'),
                      ('u_3pt', 're-run 3-point'),
                      ('u_bowl', 'whole-bowl (d78)'),
                      ('u_sym', 'symmetric window (d100)')):
        for filt, ftag in ((True, "d56 filter: how=='parabola'"), (False, 'all locatable rows')):
            r = pool(key, filt)
            if not r:
                print('  %-26s %6s' % ('%s / %s' % (name, ftag), 'n<3'))
                continue
            n, m, sd, dev = r
            out['pooled']['%s|%s' % (key, 'filtered' if filt else 'all')] = {
                'n': n, 'mean': m, 'sd_pct': sd, 'dev_pct': dev}
            print('  %-26s %6d %10.4f %8.1f%% %+10.1f%%'
                  % ('%s%s' % (name, '' if filt else ' (unfiltered)'), n, m, sd, dev))

    # what the change actually is
    a = out['pooled'].get('u_stored|filtered')
    b = out['pooled'].get('u_sym|filtered')
    c = out['pooled'].get('u_stored|all')
    print()
    if a and b:
        print('  estimator change, same rows: %.4f -> %.4f  (%+.1f%%)'
              % (a['mean'], b['mean'], 100 * (b['mean'] / a['mean'] - 1)))
    if a and c:
        print('  selection change, same estimator: %.4f (n=%d) -> %.4f (n=%d)  (%+.1f%%)'
              % (a['mean'], a['n'], c['mean'], c['n'],
                 100 * (c['mean'] / a['mean'] - 1)))
    print()
    print('  rows the d56 filter drops, by stored verdict:')
    drops = {}
    for r in rows:
        if r['stored_how'] != 'parabola':
            drops[r['stored_how']] = drops.get(r['stored_how'], 0) + 1
    print('    %s  (of %d rows with a curve)' % (drops if drops else 'none', len(rows)))

    out['rows'] = rows
    json.dump(out, open(HERE / 'd102_relocate_nav2_2026-09-13.json', 'w'), indent=2)
    print('\n-> d102_relocate_nav2_2026-09-13.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
