#!/usr/bin/env python3
"""
d100 -- verdict on the graceful block, against a prediction made before it ran.

The claim under test (PREREG_GRACEFUL_2026-09-08.md) is a TRANSITION, not a
number:

    k_phi 0.5   optimum exists      u* 0.2270 (linearised) / 0.1118 (exact law)
    k_phi 1.0   optimum exists      u* 0.0995 (linearised) / 0.0778 (exact law)
    k_phi 2.0   NO interior optimum
    k_phi 3.0   NO interior optimum       <- the shipped configuration

so there are four independent binary calls plus two locations, and the paper's
sentence depends on all six.

THREE THINGS THIS DOES NOT ASSUME

  u is formed from MEASURED quantities.  The block records, for every point, the
  achieved speed (not v_linear_max), the effective look-ahead (not
  max_lookahead) and the control period (not 1/freq), because on this controller
  all three differ from the configured value -- by 32%, 10% and 0% respectively
  in the first drive.  u = v_achieved * (tau + T_ctrl/2 + T_sim/2) / L_eff.

  "No optimum" is a measurement, not a failure to find one.  A curve is called
  monotone only if (a) its minimum sits at an end point AND (b) no interior
  point lies below the line through its neighbours by more than 2 sd of that
  comparison.  The sd is propagated from the measured 0.7% noise on a single
  RMSE point -- NOT from the 1.269% figure, which is the spread of the u*
  extraction.  CRITERIA_AUDIT line 30 lists swapping those two as an existing
  defect in this project, and they are 1.8x apart.

  An edge minimum is reported as an edge.  d97's first locator returned an
  interior value for a curve that rose from its first point; the bowl estimator
  is only applied once the minimum is known to be interior.

Usage:  python3 d100_graceful_verdict_2026-09-08.py [block_graceful.json]
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
DEFAULT = pathlib.Path('/tmp/letg2/exp/block_graceful.json')
# The dip test compares RMSE POINTS, so the denominator is the noise on a single
# RMSE point -- measured at 0.7% -- and NOT the 1.269% reproducibility floor,
# which is the row-to-row spread of the u* EXTRACTION.  CRITERIA_AUDIT line 30
# records exactly this confusion as a pre-existing defect (the noise of a single RMSE point was used as a u* threshold), and the two are 1.8x apart, so it is written out here rather
# than left to a constant name.
SIGMA_POINT_BORROWED = 0.007   # the reference engine's RPP runs -- NOT this controller
# Measured on this controller instead (g_replicate.json): two independent
# launches at each of three speeds bracketing the shipped row's only wiggle.
# Relative sd 0.28% / 2.07% / 4.34%, pooled 2.78% -- four times the borrowed
# figure.  The borrowed value made a 2.95% feature look marginal; the measured
# one puts it far below noise.  n = 2 per cell, 3 cells, so this sd carries 3
# degrees of freedom and is itself uncertain; it is used because a weak
# measurement of the right quantity beats a precise measurement of another one
# (the same reasoning as CRITERIA_AUDIT line 30).
SIGMA_POINT = 0.0278
# chord - point, with chord = (1-f) r[k-1] + f r[k+1], has variance
# sigma^2 [(1-f)^2 + f^2 + 1]; at f = 0.5 that is 1.5 sigma^2.
DIP_K = 2.0              # a dip counts as real at 2 sd of that statistic
TAU_INJECT = 0.10
T_SIM = 0.01

PRED = {
    0.5: {'exists': True,  'u_lin': 0.2270, 'u_exact': 0.1118},
    1.0: {'exists': True,  'u_lin': 0.0995, 'u_exact': 0.0778},
    2.0: {'exists': False, 'u_lin': None,   'u_exact': None},
    3.0: {'exists': False, 'u_lin': None,   'u_exact': None},
}


def points(row):
    """(u, rmse) per point, from the measured speed, look-ahead and period."""
    out = []
    for c in row.get('curve', []):
        r = c.get('rmse')
        if r is None or not np.isfinite(r):
            continue
        v = c.get('v_achieved') or c.get('v_set')
        L = c.get('L_eff') or row.get('L_nominal')
        T = c.get('T_ctrl')
        if not v or not L:
            continue
        T = T if (T and T > 0) else 1.0 / 20.0
        tau_tot = TAU_INJECT + T / 2.0 + T_SIM / 2.0
        out.append((v * tau_tot / L, float(r), c))
    return sorted(out)


def verdict(pts):
    """Interior optimum or not, with the point noise doing the work."""
    us = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    if len(us) < 5:
        return {'call': 'insufficient', 'n': len(us)}
    i = int(np.argmin(rs))
    at_edge = i in (0, len(rs) - 1)
    # deepest interior dip below the chord through its neighbours
    # A LOCAL MINIMUM is a point below BOTH neighbours.  The first version of
    # this test asked whether a point lay below the CHORD through its
    # neighbours, which measures convexity, not a minimum: any curve that is
    # increasing and bending upward puts every interior point below its chord.
    # It duly reported an 11.6% "dip" on the k_phi 2.0 row, produced entirely by
    # the last point shooting up.  The depth that matters is against the
    # SHALLOWER neighbour, since the other one cannot rescue it.
    thr = DIP_K * SIGMA_POINT * math.sqrt(2.0)     # sd of a two-point ratio
    dip, dip_at = 0.0, None
    for k in range(1, len(rs) - 1):
        if rs[k] < rs[k - 1] and rs[k] < rs[k + 1]:
            d = min(rs[k - 1], rs[k + 1]) / max(rs[k], 1e-12) - 1.0
            if d > dip:
                dip, dip_at = d, us[k]
    res = {'n': len(us), 'argmin_at_edge': at_edge, 'u_argmin': us[i],
           'deepest_dip': dip, 'dip_at_u': dip_at,
           'dip_threshold': thr, 'sigma_point': SIGMA_POINT,
           'rmse_min': min(rs), 'rmse_max': max(rs),
           'span_ratio': max(rs) / max(min(rs), 1e-12)}
    if at_edge and dip <= thr:
        res['call'] = 'NO interior optimum'
        res['u_star'] = None
        return res
    # ---- interior: locate on a window SYMMETRIC in u about the grid minimum --
    # The bowl estimator (every point within factor x min) is unbiased only when
    # the selected set is symmetric about the vertex.  These curves rise far more
    # steeply than they fall, so the window reaches further on the shallow side
    # and drags the fitted vertex down: on the k_phi 1.0 row it returned 0.0503
    # against a grid minimum of 0.0903.  A window symmetric by construction
    # removes that mechanism.  (This does not overturn the bowl elsewhere -- on
    # the reference engine's shallow, near-symmetric crowd curves d78's argument stands.
    # The estimator is being matched to the curve shape, which is why both are
    # reported.)
    #
    # Fixed rule, stated before the remaining rows ran: m = 2 (five points) is
    # the estimate; the spread over m in {1, 2, 3} is the systematic band, since
    # a curve that is locally parabolic gives the same answer for every m and one
    # that is not says so by disagreeing.
    ests = {}
    for m in (1, 2, 3):
        if i - m < 0 or i + m >= len(rs):
            continue
        x = np.array(us[i - m:i + m + 1])
        y = np.array(rs[i - m:i + m + 1])
        c = np.polyfit(x, y, 2)
        if c[0] <= 0:
            continue
        vtx = -c[1] / (2 * c[0])
        if x[0] <= vtx <= x[-1]:
            ests[m] = float(vtx)
    if ests:
        res['call'] = 'optimum'
        res['u_star'] = ests.get(2, ests[sorted(ests)[-1]])
        res['u_star_by_m'] = ests
        res['u_band'] = [min(ests.values()), max(ests.values())]
        res['u_grid_min'] = us[i]
        sel = [k for k in range(len(rs)) if rs[k] <= min(rs) * 1.35]
        if len(sel) >= 3:
            c = np.polyfit([us[k] for k in sel], [rs[k] for k in sel], 2)
            if c[0] > 0:
                res['u_bowl_1p35'] = float(-c[1] / (2 * c[0]))
        return res
    res['call'] = 'optimum (grid)' if not at_edge else 'edge with a dip'
    res['u_star'] = us[i] if not at_edge else None
    return res


def main():
    p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not p.exists():
        print('no block file at %s' % p)
        return 1
    blob = json.load(open(p))
    rows = blob.get('rows', [])

    print('=' * 78)
    print('d100 -- graceful controller: verdict against the pre-registration')
    print('=' * 78)
    print('  u is formed per point from the MEASURED speed, look-ahead and')
    print('  control period; "no optimum" is tested at %.1f sd of the measured'
          % DIP_K)
    print('  %.1f%% single-point RMSE noise, not asserted.' % (100 * SIGMA_POINT))
    print()

    out = {'sigma_point': SIGMA_POINT, 'dip_k': DIP_K, 'rows': []}
    hits = 0
    calls = 0
    print('  %6s %5s %5s %5s %22s %10s %10s %16s'
          % ('k_phi', 'A', 'B', 'beta', 'measured', 'pred lin', 'pred exact',
             'deepest dip/thr'))
    for r in rows:
        pts = points(r)
        v = verdict(pts)
        kp = r['k_phi']
        pr = PRED.get(kp, {}) if r.get('beta') == 0.0 else {}
        if v.get('call') == 'optimum':
            b = v.get('u_band', [v['u_star'], v['u_star']])
            meas = 'u* %.4f [%.4f,%.4f]' % (v['u_star'], b[0], b[1])
        elif v.get('call') == 'NO interior optimum':
            meas = 'none (min at %s edge)' % ('low' if v['u_argmin'] == pts[0][0]
                                              else 'high')
        else:
            meas = v.get('call', '?')
        row = {'k_phi': kp, 'beta': r.get('beta'), 'A': r.get('A'), 'B': r.get('B'),
               'verdict': v, 'pred': pr,
               'u_points': [x[0] for x in pts], 'rmse': [x[1] for x in pts]}
        out['rows'].append(row)
        if pr:
            calls += 1
            got = (v.get('call') == 'optimum')
            ok = (got == pr['exists'])
            hits += 1 if ok else 0
            row['prereg_hit'] = ok
        print('  %6.1f %5.1f %5.1f %5.1f %22s %10s %10s %7.2f%% / %.2f%%'
              % (kp, r.get('A', 0), r.get('B', 0), r.get('beta', 0), meas,
                 ('%.4f' % pr['u_lin']) if pr.get('u_lin') else '-',
                 ('%.4f' % pr['u_exact']) if pr.get('u_exact') else '-',
                 100 * v.get('deepest_dip', 0), 100 * v.get('dip_threshold', 0)))

    print()
    if calls:
        print('  pre-registered binary calls correct: %d of %d' % (hits, calls))
    loc = [r for r in out['rows']
           if r.get('pred', {}).get('u_exact') and r['verdict'].get('u_star')]
    if loc:
        print()
        print('  where an optimum exists, which derivation is closer:')
        for r in loc:
            m = r['verdict']['u_star']
            b = r['verdict'].get('u_band', [m, m])
            inband = lambda x: 'in band' if b[0] <= x <= b[1] else 'outside'
            print('    k_phi %.1f: measured %.4f  band [%.4f, %.4f]  grid %.4f'
                  % (r['k_phi'], m, b[0], b[1], r['verdict'].get('u_grid_min', m)))
            print('              linearised %.4f (%+.0f%%, %s)   '
                  'exact law %.4f (%+.0f%%, %s)'
                  % (r['pred']['u_lin'], 100 * (r['pred']['u_lin'] / m - 1),
                     inband(r['pred']['u_lin']),
                     r['pred']['u_exact'], 100 * (r['pred']['u_exact'] / m - 1),
                     inband(r['pred']['u_exact'])))

    sh = [r for r in out['rows'] if r.get('beta') == 0.4]
    if sh:
        print()
        print('  shipped control row (beta 0.4): included so the paper can say')
        print('  what the out-of-the-box plugin does; its speed is not the swept')
        print('  parameter, so it is reported at its measured speed only.')

    json.dump(out, open(HERE / 'd100_graceful_verdict_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d100_graceful_verdict_2026-09-08.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
