#!/usr/bin/env python3
"""
d62 -- The coefficient on tau_inject, measured on the fixed harness (E1').

d50 fitted tau_tot = s_tau * tau_inject + c on the PRELIMINARY tau sweep and got
s_tau = 0.9713, 95% [0.9490, 0.9995].  That number carries weight out of
proportion to its dataset: it is below 1 while both ZOH coefficients come out
above 1, and that ASYMMETRY is the whole argument that rejected the "J(u) is off
by a global multiplicative factor" hypothesis (d55's H2', rejected at p = 0.019).
An argument that load-bearing should not rest on the one block still labelled
preliminary -- exit stub inside the measurement window, lap count changing inside
a row, 5 of 36 points breaking corner isolation.

E1' re-ran it: side 10 m (seg/L = 16.7), u-space sweep capped at u_isolated,
tau_inject 0.05 -> 0.80 (16x), 5 rows x 9 points, every row an interior minimum.

Fit, same estimator as everywhere else -- whole-curve shape against d38's
curvature-command J(u), per-row scale A and floor B, shared s_tau and c:

    RMSE(v)^2 = A * J(v * (s_tau * tau + c) / L) + B^2

Usage:  python3 d62_tau_iso_coefficient_2026-09-07.py
"""
import glob
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).parent
L = 0.6
U_MIN, U_MAX, U_N = 0.04, 0.50, 47
U_SD_ROW = 0.0152                    # E7 row-to-row reproducibility

_p = HERE / 'd38_corner_energy_two_plants_2026-08-29.py'
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def make_J():
    us = np.linspace(U_MIN, U_MAX, U_N)
    Js = np.array([d38.transit_energy(float(u), 'curvature_command', h=1.0e-3)
                   for u in us])
    lg = np.log(Js)

    def J(u):
        return np.exp(np.interp(np.clip(u, U_MIN, U_MAX), us, lg))
    return J


def fit(rows, J, s_fixed=None):
    k = len(rows)
    free = s_fixed is None

    def resid(p):
        c = p[0]
        s = p[1] if free else s_fixed
        off = 2 if free else 1
        out = []
        for i, (tau, vs, rs) in enumerate(rows):
            A, B = p[off + i], p[off + k + i]
            tt = s * tau + c
            pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    p0 = [0.030] + ([1.0] if free else [])
    lb = [-0.05] + ([0.3] if free else [])
    ub = [0.20] + ([3.0] if free else [])
    for tau, vs, rs in rows:
        p0.append(rs.min() ** 2); lb.append(0.0); ub.append(np.inf)
    for tau, vs, rs in rows:
        p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
    sol = least_squares(resid, p0, bounds=(lb, ub))
    return {'c_ms': 1000 * float(sol.x[0]),
            's_tau': float(sol.x[1]) if free else s_fixed,
            'rms_pct': float(100 * np.sqrt(np.mean(resid(sol.x) ** 2)))}


def main():
    f = sorted(glob.glob(str(HERE / 'nav2_block_tau_iso_*.json')))[-1]
    blk = json.load(open(f))
    rows = [(r['tau_inject'],
             np.array([c['v'] for c in r['curve']]),
             np.array([c['rmse'] for c in r['curve']]))
            for r in blk['rows'] if r.get('curve')]
    J = make_J()
    npts = sum(len(r[1]) for r in rows)

    print('=' * 78)
    print("d62 -- coefficient on tau_inject, fixed harness (E1')")
    print('=' * 78)
    print('  %s' % pathlib.Path(f).name)
    print('  side %.1f m, seg/L %.1f, %d rows, %d points, tau %s'
          % (blk['meta']['side'], blk['meta']['side'] / L, len(rows), npts,
             [r[0] for r in rows]))

    us = [r['u_obs'] for r in blk['rows'] if r.get('u_obs')]
    print()
    print('  u* per row: %s' % ', '.join('%.5f' % u for u in us))
    print('  spread %.2f%% over a %.0fx tau range; E7 reproducibility is 1.52%% sd'
          % (100 * (max(us) - min(us)) / np.mean(us),
             max(r[0] for r in rows) / min(r[0] for r in rows)))
    print('  mean %.4f vs the derived 0.306034  ->  %+.2f%%'
          % (np.mean(us), 100 * (np.mean(us) / 0.306034 - 1)))

    best = fit(rows, J)
    pinned = fit(rows, J, s_fixed=1.0)
    print()
    print('  s_tau free    = %.4f   c = %+.2f ms   rms %.2f%%'
          % (best['s_tau'], best['c_ms'], best['rms_pct']))
    print('  s_tau pinned 1: c = %+.2f ms   rms %.2f%%'
          % (pinned['c_ms'], pinned['rms_pct']))
    print('  a-priori c = T_ctrl/2 + T_sim/2 = 30.0 ms')

    thr = best['rms_pct'] * 1.10
    grid = np.arange(0.70, 1.41, 0.01)
    inside = [float(x) for x in grid if fit(rows, J, s_fixed=float(x))['rms_pct'] <= thr]
    lo, hi = (min(inside), max(inside)) if inside else (float('nan'),) * 2

    loo = []
    for i in range(len(rows)):
        r = fit([x for j, x in enumerate(rows) if j != i], J)
        loo.append(r['s_tau'])
        print('    without tau = %.2f:  s_tau = %.4f   c = %+.2f ms'
              % (rows[i][0], r['s_tau'], r['c_ms']))
    loo = np.array(loo)

    rng = np.random.default_rng(20260907)
    sims = []
    for _ in range(200):
        pert = [(t, vs * (1.0 + rng.normal(0, U_SD_ROW)), rs) for t, vs, rs in rows]
        sims.append(fit(pert, J)['s_tau'])
    sims = np.array(sims)
    p25, p975 = float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))

    print()
    print('  profile (+10%% rms): [%.3f, %.3f]' % (lo, hi))
    print('  leave-one-row-out : %.4f to %.4f (sd %.4f)'
          % (loo.min(), loo.max(), loo.std(ddof=1)))
    print('  E7 noise propagated: %.4f +- %.4f, 95%% [%.4f, %.4f]'
          % (sims.mean(), sims.std(ddof=1), p25, p975))
    print()
    print('  %-28s %10s %12s' % ('', 'value', 'contains 1?'))
    print('  %-28s %10.4f %12s' % ('E1\' (this, fixed harness)', best['s_tau'],
                                   'YES' if lo <= 1.0 <= hi else 'NO'))
    print('  %-28s %10.4f %12s' % ('d50 (preliminary sweep)', 0.9713, 'NO'))
    print()
    print('  the asymmetry that rejected H2\':')
    print('    T_ctrl/2 coefficient (E8/E11) 1.0814   [1.048, 1.131]')
    print('    T_sim/2  coefficient (E10)    1.2626   [1.177, 1.544]')
    print('    tau_inject coefficient (here) %.4f   [%.3f, %.3f]'
          % (best['s_tau'], p25, p975))
    if best['s_tau'] < 1.0 < 1.0814:
        print('    -> the asymmetry SURVIVES on the fixed harness.')
    elif abs(best['s_tau'] - 1.0) < 0.05:
        print('    -> tau_inject now sits AT 1, while both ZOH terms exceed it.')
        print('       That is a cleaner statement than the preliminary 0.971:')
        print('       the injected pure delay enters with weight exactly 1 and the')
        print('       two sampled terms do not.')
    else:
        print('    -> the asymmetry does NOT survive; d55\'s H2\' rejection weakens.')

    json.dump({'file': pathlib.Path(f).name, 'n_rows': len(rows), 'n_points': npts,
               'u_star': us, 'best': best, 'pinned': pinned,
               'profile': [lo, hi], 'loo': loo.tolist(),
               'noise_ci': [p25, p975], 'noise_mean': float(sims.mean())},
              open(HERE / 'd62_tau_iso_coefficient_2026-09-07.json', 'w'), indent=2)
    print('\n-> d62_tau_iso_coefficient_2026-09-07.json')


if __name__ == '__main__':
    main()
