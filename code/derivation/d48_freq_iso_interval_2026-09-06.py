#!/usr/bin/env python3
"""
d48 -- The T_ctrl/2 coefficient on the isolated geometry (E8), with an interval.

Three numbers now exist for the same quantity, and they must be reconciled:

    1.0921   E2, seg/L = 8.3, all 54 points          -- 20 of 54 break corner isolation
    1.2378   E2, seg/L = 8.3, isolation violators dropped (34 points)
    1.081    E8, seg/L = 16.7, sweep defined in u    -- nothing violates, nothing dropped

The middle number was never a measurement of the coefficient: dropping the top of
every curve removes the shape information the fit uses, and biases what is left.
E8 was run precisely so that no point has to be dropped.  This script gives the
E8 number an interval, by three routes that fail differently:

  (a) profile in s     -- refit everything else at fixed s, walk s until the
                          residual rises by 10% (the criterion d43 already uses
                          for its c profile, kept identical so the two are
                          comparable)
  (b) leave-one-row-out -- refit six times, each without one frequency.  This is
                          the honest test of whether one bad row is carrying it.
  (c) E7 noise floor    -- 5 identical rows gave u* sd 1.52%; propagate that as a
                          row-level perturbation of the fitted tau_tot and see
                          what s does.

Theorem 2 predicts s = 1.000 exactly, with no free parameter.

Usage:  python3 d48_freq_iso_interval_2026-09-06.py
"""
import argparse
import glob
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).parent
L = 0.6
U_MIN, U_MAX, U_N = 0.04, 0.50, 47
# Row-to-row sd of u*, relative.  E7 alone gave 1.522% on 5 rows; E7, E7'
# and E7" pool to 1.269% on 15 rows (dof 12, 95% CI [0.910%, 2.095%]), and
# the pooled value is the right one -- it is the measured denominator every
# tolerance here rests on.  Overridable so the earlier intervals can still
# be reproduced with --u-sd 0.01522.
U_SD_ROW = 0.01269

_p = HERE / 'd38_corner_energy_two_plants_2026-08-29.py'
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def make_J():
    us = np.linspace(U_MIN, U_MAX, U_N)
    Js = np.array([d38.transit_energy(float(u), 'curvature_command', h=1.0e-3) for u in us])
    logJ = np.log(Js)

    def J(u):
        return np.exp(np.interp(np.clip(u, U_MIN, U_MAX), us, logJ))
    return J


def fit(data, terms, tau, J, s_fixed=None, s_free=True):
    k = len(data)

    def resid(p):
        c = p[0]
        if s_fixed is not None:
            s = s_fixed
            off = 1
        elif s_free:
            s = p[1]
            off = 2
        else:
            s = 1.0
            off = 1
        out = []
        for i, (vs, rs) in enumerate(data):
            A, B = p[off + i], p[off + k + i]
            tt = tau + s * terms[i] + c
            pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    free_s = (s_fixed is None) and s_free
    p0 = [0.010] + ([1.0] if free_s else [])
    lb = [-0.05] + ([0.0] if free_s else [])
    ub = [0.10] + ([4.0] if free_s else [])
    for vs, rs in data:
        p0.append(max(rs.min() ** 2 / max(J(vs * (tau + 0.01) / L).min(), 1e-12), 1e-9))
        lb.append(0.0); ub.append(np.inf)
    for vs, rs in data:
        p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
    sol = least_squares(resid, p0, bounds=(lb, ub))
    rms = float(100 * np.sqrt(np.mean(resid(sol.x) ** 2)))
    s_out = s_fixed if s_fixed is not None else (float(sol.x[1]) if free_s else 1.0)
    return {'s': s_out, 'c_ms': 1000 * float(sol.x[0]), 'rms_pct': rms}


def load(path, term_key, label_key):
    blk = json.load(open(path))
    rows = [r for r in blk['rows'] if r.get('curve')]
    tau = rows[0].get('tau_inject', blk['meta'].get('tau'))
    data = [(np.array([c['v'] for c in r['curve']]),
             np.array([c['rmse'] for c in r['curve']])) for r in rows]
    terms = np.array([r[term_key] for r in rows])
    labels = [r[label_key] for r in rows]
    return data, terms, tau, labels, blk


def main():
    ap = argparse.ArgumentParser()
    # PINNED (2026-09-13).  The default used to be the glob
    # 'nav2_block_freq_iso_*.json', taking the last name in sorted order.  When
    # this script produced the reported 1.0814 that was the tau = 0.10 block (its
    # JSON records 'block': 'nav2_block_freq_iso_2026-09-06.json'); the tau005 and
    # tau020 blocks written later that day sort after it, so the same default now
    # silently selects tau = 0.20 -- the d56 glob drift again.  Pinned by name.
    ap.add_argument('--glob', default='nav2_block_freq_iso_2026-09-06.json')
    ap.add_argument('--term', default='T_ctrl_half')
    ap.add_argument('--label', default='freq')
    ap.add_argument('--out', default='d48_freq_iso_interval_2026-09-06.json')
    ap.add_argument('--u-sd', type=float, default=U_SD_ROW, dest='u_sd',
                    help='row-level u* sd for leg (c); default is the '
                         'pooled E7/E7prime/E7second floor, 1.269%%')
    a = ap.parse_args()
    J = make_J()
    path = sorted(glob.glob(str(HERE / a.glob)))[-1]
    data, terms, tau, freqs, blk = load(path, a.term, a.label)
    n_pts = sum(len(v) for v, _ in data)

    print('=' * 78)
    print('d48 -- T_ctrl/2 coefficient on the isolated geometry')
    print('=' * 78)
    print('  block %s' % pathlib.Path(path).name)
    side = blk['meta'].get('side', 5.0)
    print('  term %s, side %.1f m, seg/L %.1f, %d rows, %d points, tau_inject %.2f'
          % (a.term, side, side / L, len(data), n_pts, tau))
    print('  isolated by construction: %s'
          % ('YES (u-sweep capped at u_isolated)' if 'iso' in path
             else 'NO -- this block predates the fix, see CRITERIA_AUDIT section 3'))

    best = fit(data, terms, tau, J)
    pinned = fit(data, terms, tau, J, s_free=False)
    print()
    print('  coefficient free   s = %.4f   c = %+.1f ms   rms %.2f%%'
          % (best['s'], best['c_ms'], best['rms_pct']))
    print('  coefficient pinned s = 1.0000   c = %+.1f ms   rms %.2f%%'
          % (pinned['c_ms'], pinned['rms_pct']))
    print('  Theorem 2 predicts s = 1.000 with no free parameter.')

    out = {'block': pathlib.Path(path).name, 'n_rows': len(data), 'n_points': n_pts,
           's_free': best, 's_pinned': pinned}

    # ---- (a) profile in s ---------------------------------------------------
    print()
    print('  (a) profile in s -- refit everything else at fixed s')
    thr = best['rms_pct'] * 1.10
    grid = np.arange(0.70, 1.61, 0.02)
    prof = [(float(s), fit(data, terms, tau, J, s_fixed=float(s))['rms_pct'])
            for s in grid]
    inside = [s for s, r in prof if r <= thr]
    print('      best rms %.3f%%, threshold (+10%%) %.3f%%' % (best['rms_pct'], thr))
    print('      s within threshold: [%.3f, %.3f]' % (min(inside), max(inside)))
    print('      s = 1.000 %s' % ('IS inside this interval'
                                  if min(inside) <= 1.0 <= max(inside)
                                  else 'is OUTSIDE this interval'))
    out['profile_s'] = {'threshold_rms_pct': thr,
                        'interval': [min(inside), max(inside)],
                        'contains_one': bool(min(inside) <= 1.0 <= max(inside)),
                        'curve': prof}

    # ---- (b) leave one row out ---------------------------------------------
    print()
    print('  (b) leave-one-row-out')
    loo = []
    for i in range(len(data)):
        d2 = [d for j, d in enumerate(data) if j != i]
        t2 = np.array([t for j, t in enumerate(terms) if j != i])
        r = fit(d2, t2, tau, J)
        loo.append({'dropped_row': freqs[i], 's': r['s'], 'c_ms': r['c_ms'],
                    'rms_pct': r['rms_pct']})
        print('      without %-9s:  s = %.4f   c = %+.1f ms   rms %.2f%%'
              % (str(freqs[i]), r['s'], r['c_ms'], r['rms_pct']))
    ss = np.array([x['s'] for x in loo])
    print('      s over the six refits: %.4f to %.4f  (sd %.4f)'
          % (ss.min(), ss.max(), ss.std(ddof=1)))
    out['leave_one_out'] = loo

    # ---- (c) E7 noise floor propagated -------------------------------------
    print()
    print('  (c) row noise (u* sd %.3f%%) propagated into the fit'
          % (100 * a.u_sd))
    rng = np.random.default_rng(20260906)
    sims = []
    for _ in range(200):
        d2 = [(vs * (1.0 + rng.normal(0, a.u_sd)), rs) for vs, rs in data]
        sims.append(fit(d2, terms, tau, J)['s'])
    sims = np.array(sims)
    print('      s = %.4f +- %.4f   (2.5-97.5 pct: %.4f to %.4f)'
          % (sims.mean(), sims.std(ddof=1),
             np.percentile(sims, 2.5), np.percentile(sims, 97.5)))
    print('      s = 1.000 %s'
          % ('IS inside the 95%% band' if np.percentile(sims, 2.5) <= 1.0
             <= np.percentile(sims, 97.5) else 'is OUTSIDE the 95%% band'))
    out['u_sd_used'] = float(a.u_sd)
    out['noise_propagated'] = {'mean': float(sims.mean()), 'sd': float(sims.std(ddof=1)),
                               'p2_5': float(np.percentile(sims, 2.5)),
                               'p97_5': float(np.percentile(sims, 97.5))}

    # ---- how well determined is s, really -----------------------------------
    print()
    print('=' * 78)
    ss = np.array([x['s'] for x in loo])
    lever = float(terms.max() / terms.min())
    print('  s = %.4f   profile [%.3f, %.3f]   95%% [%.3f, %.3f]   LOO sd %.4f'
          % (best['s'], out['profile_s']['interval'][0], out['profile_s']['interval'][1],
             out['noise_propagated']['p2_5'], out['noise_propagated']['p97_5'],
             ss.std(ddof=1)))
    print('  term swept %g -> %g (%.0fx lever arm)' % (terms.min(), terms.max(), lever))
    excl = not (out['profile_s']['contains_one']
                and out['noise_propagated']['p2_5'] <= 1.0 <= out['noise_propagated']['p97_5'])
    print('  Theorem 2 requires s = 1.000 exactly: %s'
          % ('EXCLUDED by both the profile and the 95% band' if excl
             else 'consistent with the data'))
    if ss.std(ddof=1) > 0.05:
        worst = min(loo, key=lambda x: x['s']) if abs(min(x['s'] for x in loo) - best['s']) \
            > abs(max(x['s'] for x in loo) - best['s']) else max(loo, key=lambda x: x['s'])
        print('  WARNING: leave-one-out sd is %.3f -- s is carried by single rows.'
              % ss.std(ddof=1))
        print('           dropping row %s alone moves s to %.4f.'
              % (str(worst['dropped_row']), worst['s']))
        print('           the interval above understates the real uncertainty; the')
        print('           sweep needs a longer lever arm before s can be quoted.')
    out['lever_arm'] = lever
    out['loo_sd'] = float(ss.std(ddof=1))
    out['excludes_one'] = bool(excl)

    json.dump(out, open(HERE / a.out, 'w'), indent=2)
    print('\n-> %s' % a.out)


if __name__ == '__main__':
    main()
