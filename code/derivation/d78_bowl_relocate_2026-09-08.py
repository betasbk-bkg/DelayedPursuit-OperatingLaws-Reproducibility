#!/usr/bin/env python3
"""
d78 -- re-locate u*(alpha) with a whole-bowl fit instead of a three-point
       parabola, on the data d65 and d66 already produced.

d66 ran the crowd leg under d65's protocol and the fit came out unusable:
A cos^q(alpha/2) with r2 = 0.42 and rms 21.4%, against the clean leg's 0.9974
and 1.04%.  The culprit is visible in the located optima --

    alpha    120     90     60     45     30     15
    u*     0.2872 0.4077 0.4219 0.3042 0.4704 0.6073
                                 ^^^^^^ off-trend by a mile

-- one polygon whose located minimum sits where the trend says it cannot.  With
the vote layer on and MC = 5, the RMSE curve is noisy, and `locate()` takes a
parabola through the three points around the grid minimum.  Three points is
exactly the estimator this project already retired once: CRITERIA_AUDIT records
that the three-point parabola is biased and that the whole-curve shape fit is the
primary estimator.  d65 and d66 never got that upgrade.

So: same data, better estimator.  Fit a parabola to EVERY point inside the bowl
(all speeds whose RMSE is within a stated factor of the grid minimum) rather than
to the three nearest.  It uses ten to twenty points instead of three, so a single
noisy cell cannot carry the answer, and it makes no assumption about the shape of
J(u) -- which matters here, because the shape of J is what the alpha law is
about.

The clean leg is re-located the same way, so the two remain comparable.  Any
change in the clean numbers is itself information: if the clean leg barely moves
and the crowd leg moves a lot, the difference between them was estimator noise,
not quantisation.

Usage:  python3 d78_bowl_relocate_2026-09-08.py [--factor 2.0]
"""
import argparse
import glob
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def bowl_locate(speeds, rmse, factor=2.0, min_pts=5):
    """Parabola through every point inside the bowl, not just three.

    The bowl is every speed whose RMSE is within `factor` of the grid minimum.
    Returns (v_star, how, n_points_used).
    """
    s = np.asarray(speeds, float)
    r = np.asarray([np.nan if x is None else x for x in rmse], float)
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() < min_pts:
        return None, 'too-few', int(ok.sum())
    s, r = s[ok], r[ok]
    i = int(np.argmin(r))
    sel = r <= factor * r[i]
    # keep the contiguous run containing the minimum, so a far-away dip cannot
    # drag the fit
    j = int(np.argmin(r))
    lo = j
    while lo > 0 and sel[lo - 1]:
        lo -= 1
    hi = j
    while hi < len(r) - 1 and sel[hi + 1]:
        hi += 1
    ss, rr = s[lo:hi + 1], r[lo:hi + 1]
    if len(ss) < min_pts:
        return float(s[i]), 'narrow-bowl', len(ss)
    c = np.polyfit(ss, rr, 2)
    if c[0] <= 0:
        return float(s[i]), 'not-convex', len(ss)
    v = -c[1] / (2 * c[0])
    if not (ss[0] <= v <= ss[-1]):
        return float(s[i]), 'outside', len(ss)
    return float(v), 'bowl', len(ss)


def fit_law(alphas, us):
    cs = np.array([math.cos(math.radians(a) / 2) for a in alphas])
    us = np.asarray(us, float)
    M = np.vstack([np.log(cs), np.ones_like(cs)]).T
    cf, *_ = np.linalg.lstsq(M, np.log(us), rcond=None)
    q, A = float(cf[0]), float(np.exp(cf[1]))
    yh = M @ cf
    r2 = float(1 - ((np.log(us) - yh) ** 2).sum()
               / ((np.log(us) - np.log(us).mean()) ** 2).sum())
    rms = float(100 * np.sqrt(np.mean((np.exp(yh) / us - 1) ** 2)))
    qs = []
    for i in range(len(us)):
        k = [j for j in range(len(us)) if j != i]
        c2, *_ = np.linalg.lstsq(M[k], np.log(us[k]), rcond=None)
        qs.append(float(c2[0]))
    return {'q': q, 'A': A, 'r2': r2, 'rms_pct': rms,
            'q_loo_sd': float(np.std(qs, ddof=1)), 'n': len(us)}


def leg(path, factor):
    d = json.load(open(path))
    speeds = d['speeds']
    tt, look = d['tau_tot'], 2.0
    out = []
    for r in d['rows']:
        v_old, how_old = r['v_star'], r['how']
        v_new, how_new, npts = bowl_locate(speeds, r['rmse'], factor)
        out.append({'alpha_deg': r['alpha_deg'],
                    'u_old': r['u_star'], 'how_old': how_old,
                    'u_new': (v_new * tt / look) if v_new else None,
                    'how_new': how_new, 'n_pts': npts})
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor', type=float, default=2.0)
    a = ap.parse_args()

    legs = {}
    for name, pat in (('clean', 'd65_heading_servo_alpha_law_side10_tau433_*.json'),
                      ('crowd', 'd66_crowd_alpha_law_side10_tau433_*.json')):
        hits = sorted(glob.glob(str(HERE / pat)))
        if not hits:
            print('  !! no file for %s (%s)' % (name, pat))
            continue
        legs[name] = leg(hits[-1], a.factor) + (pathlib.Path(hits[-1]).name,)

    print('=' * 78)
    print('d78 -- u*(alpha) re-located with a whole-bowl fit')
    print('=' * 78)
    print('  bowl = every speed whose RMSE is within %.1fx the grid minimum,'
          % a.factor)
    print('  contiguous around it.  Three-point parabola -> many-point parabola.')
    print()

    res = {'factor': a.factor, 'legs': {}}
    for name in ('clean', 'crowd'):
        if name not in legs:
            continue
        d, rows, fn = legs[name]
        print('  %s   (%s)' % (name.upper(), fn))
        print('    %7s %10s %10s %10s %8s %8s'
              % ('alpha', 'u* 3-point', 'u* bowl', 'change', 'how', 'n pts'))
        for r in rows:
            ch = ('%+.1f%%' % (100 * (r['u_new'] / r['u_old'] - 1))
                  if r['u_new'] else '-')
            print('    %7.0f %10.4f %10s %10s %8s %8d'
                  % (r['alpha_deg'], r['u_old'],
                     ('%.4f' % r['u_new']) if r['u_new'] else '-', ch,
                     r['how_new'], r['n_pts']))
        good = [r for r in rows if r['u_new'] and r['how_new'] == 'bowl']
        if len(good) >= 3:
            f_new = fit_law([r['alpha_deg'] for r in good],
                            [r['u_new'] for r in good])
            f_old = fit_law([r['alpha_deg'] for r in rows],
                            [r['u_old'] for r in rows])
            print('      3-point:  A %.4f  q %.3f  r2 %.4f  rms %5.2f%%'
                  % (f_old['A'], f_old['q'], f_old['r2'], f_old['rms_pct']))
            print('      bowl   :  A %.4f  q %.3f  r2 %.4f  rms %5.2f%%  '
                  '(LOO sd %.3f, n %d)'
                  % (f_new['A'], f_new['q'], f_new['r2'], f_new['rms_pct'],
                     f_new['q_loo_sd'], f_new['n']))
            res['legs'][name] = {'file': fn, 'rows': rows,
                                 'fit_3point': f_old, 'fit_bowl': f_new}
        print()

    if 'clean' in res['legs'] and 'crowd' in res['legs']:
        c, w = res['legs']['clean']['fit_bowl'], res['legs']['crowd']['fit_bowl']
        print('  SAME PROTOCOL, SAME ESTIMATOR, one line different:')
        print('    q  clean %.3f -> crowd %.3f   (%+.1f%%)'
              % (c['q'], w['q'], 100 * (w['q'] / c['q'] - 1)))
        print('    A  clean %.4f -> crowd %.4f  (%+.1f%%)'
              % (c['A'], w['A'], 100 * (w['A'] / c['A'] - 1)))
        sep = abs(w['q'] - c['q']) / max(w['q_loo_sd'], 1e-9)
        print('    separation in q: %.2f of the crowd leg\'s LOO sd' % sep)
        if w['r2'] < 0.9:
            print('    !! the crowd fit is still poor (r2 %.2f) -- the estimator'
                  % w['r2'])
            print('       was not the whole problem, and q must not be quoted')
        res['comparison'] = {'q_clean': c['q'], 'q_crowd': w['q'],
                             'separation_loo_sd': sep}

    json.dump(res, open(HERE / 'd78_bowl_relocate_2026-09-08.json', 'w'), indent=2)
    print('\n-> d78_bowl_relocate_2026-09-08.json')


if __name__ == '__main__':
    main()
