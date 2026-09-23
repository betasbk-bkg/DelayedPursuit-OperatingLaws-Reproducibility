#!/usr/bin/env python3
"""
d54 -- The pre-registered E11 discriminator, with the error bar computed instead
       of assumed.  This corrects two sigma values I quoted earlier today.

E11 pre-registered c(tau_inject) as the discriminator:

    H1   an extra loop delay proportional to T_ctrl   -> c independent of tau
    H2'  J(u) off by a multiplicative factor k        -> dc/dtau = k - 1 = 0.081

Three blocks now exist, all on the isolating geometry, all 6 rows x 9 points:

    tau = 0.05   s = 1.0641   c = +2.56 ms
    tau = 0.10   s = 1.0814   c = +5.18 ms
    tau = 0.20   s = 1.0866   c = -5.39 ms

WHAT WAS WRONG.  I put an error bar on c by taking E7's row-to-row sd of a
SINGLE-ROW tau_tot fit (2.11 ms) and dividing by sqrt(6).  That is not the
uncertainty of c in a BLOCK fit: c is strongly correlated with s there, and d43
had already reported its own c profile as spanning [+6, +18] ms -- about +-6 ms,
seven times the number I used.  Every sigma I quoted from that (2.15, 1.19, then
7.59 and 17.80) is therefore wrong.

So profile c directly, the same way d48 profiles s: scan c on a grid, refit
everything else (s, and each row's A and B) at fixed c, and take the range where
the residual stays within 10% of its best value.  Then propagate THOSE widths
into the slope.

Usage:  python3 d54_c_vs_tau_2026-09-06.py
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


def load(path):
    blk = json.load(open(path))
    rows = [r for r in blk['rows'] if r.get('curve')]
    tau = blk['meta']['tau']
    data = [(np.array([c['v'] for c in r['curve']]),
             np.array([c['rmse'] for c in r['curve']])) for r in rows]
    terms = np.array([r['T_ctrl_half'] for r in rows])
    return data, terms, tau


def fit(data, terms, tau, J, c_fixed=None):
    k = len(data)
    free_c = c_fixed is None

    def resid(p):
        c = p[0] if free_c else c_fixed
        s = p[1] if free_c else p[0]
        off = 2 if free_c else 1
        out = []
        for i, (vs, rs) in enumerate(data):
            A, B = p[off + i], p[off + k + i]
            tt = tau + s * terms[i] + c
            pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    p0 = ([0.005] if free_c else []) + [1.08]
    lb = ([-0.08] if free_c else []) + [0.0]
    ub = ([0.10] if free_c else []) + [4.0]
    for vs, rs in data:
        p0.append(max(rs.min() ** 2 / max(J(vs * (tau + 0.01) / L).min(), 1e-12), 1e-9))
        lb.append(0.0); ub.append(np.inf)
    for vs, rs in data:
        p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
    sol = least_squares(resid, p0, bounds=(lb, ub))
    rms = float(100 * np.sqrt(np.mean(resid(sol.x) ** 2)))
    c_out = float(sol.x[0]) if free_c else c_fixed
    s_out = float(sol.x[1]) if free_c else float(sol.x[0])
    return {'c_ms': 1000 * c_out, 's': s_out, 'rms_pct': rms}


def main():
    J = make_J()
    specs = [(0.05, 'nav2_block_freq_iso_tau005_*.json'),
             (0.10, 'nav2_block_freq_iso_2026-09-06.json'),
             (0.20, 'nav2_block_freq_iso_tau020_*.json')]

    print('=' * 78)
    print('d54 -- c(tau_inject), with c profiled rather than assumed')
    print('=' * 78)
    print('  %8s %9s %11s %22s %8s' %
          ('tau', 's', 'c (ms)', 'c profile (+10% rms)', 'rms %'))

    pts = []
    for tau_want, pat in specs:
        f = sorted(glob.glob(str(HERE / pat)))
        if not f:
            print('  tau=%.2f  MISSING (%s)' % (tau_want, pat))
            continue
        data, terms, tau = load(f[-1])
        best = fit(data, terms, tau, J)
        thr = best['rms_pct'] * 1.10
        grid = np.arange(-0.060, 0.0601, 0.002)
        inside = [float(c) for c in grid
                  if fit(data, terms, tau, J, c_fixed=float(c))['rms_pct'] <= thr]
        lo, hi = (min(inside), max(inside)) if inside else (float('nan'),) * 2
        half = 1000 * (hi - lo) / 2
        pts.append({'tau': tau, 'c_ms': best['c_ms'], 's': best['s'],
                    'c_lo_ms': 1000 * lo, 'c_hi_ms': 1000 * hi,
                    'c_half_ms': half, 'rms_pct': best['rms_pct'],
                    'n_points': int(sum(len(v) for v, _ in data))})
        print('  %8.2f %9.4f %+11.2f   [%+7.1f, %+7.1f] ms %8.2f'
              % (tau, best['s'], best['c_ms'], 1000 * lo, 1000 * hi, best['rms_pct']))

    if len(pts) < 3:
        raise SystemExit('need all three blocks')

    print()
    print('  the half-width of the c profile is %.1f-%.1f ms, against the %.2f ms'
          % (min(p['c_half_ms'] for p in pts), max(p['c_half_ms'] for p in pts),
             2.11 / 6 ** 0.5))
    print('  I used earlier.  Every sigma quoted from that number today was wrong.')

    x = np.array([p['tau'] for p in pts])
    y = np.array([p['c_ms'] for p in pts])
    w = np.array([p['c_half_ms'] for p in pts])
    # weighted straight-line fit, weights 1/half-width^2
    W = 1.0 / w ** 2
    Sw, Sx, Sy = W.sum(), (W * x).sum(), (W * y).sum()
    Sxx, Sxy = (W * x * x).sum(), (W * x * y).sum()
    D = Sw * Sxx - Sx * Sx
    slope = (Sw * Sxy - Sx * Sy) / D
    icept = (Sxx * Sy - Sx * Sxy) / D
    sd_slope = float(np.sqrt(Sw / D))

    print()
    print('  weighted fit of c vs tau:  slope = %.1f +- %.1f ms/s, intercept %+.2f ms'
          % (slope, sd_slope, icept))
    print()
    verdicts = {}
    for name, pred in [('H1   c independent of tau', 0.0),
                       ("H2'  dc/dtau = k-1 = 0.081", 81.4)]:
        z = abs(slope - pred) / sd_slope
        verdicts[name] = z
        print('    %-30s predicts %6.1f ms/s  ->  %.2f sigma  %s'
              % (name, pred, z, 'EXCLUDED' if z > 3 else
                 ('tension' if z > 2 else 'consistent')))

    print()
    print('  and the thing that did NOT move:')
    print('    s = %s  -- three independent blocks, three tau values'
          % ', '.join('%.4f' % p['s'] for p in pts))
    print('    every one of them excludes s = 1.000.')
    print()
    print('  c, by contrast, is +2.6, +5.2, -5.4 ms: NOT monotone, so a straight')
    print('  line through it is a summary, not a model.  With the real profile')
    print('  widths the three points are %s a common value.'
          % ('consistent with' if abs(slope) / sd_slope < 2 else 'not consistent with'))

    json.dump({'points': pts, 'slope_ms_per_s': float(slope),
               'slope_sd_ms_per_s': sd_slope, 'intercept_ms': float(icept),
               'sigma_H1': verdicts.get('H1   c independent of tau'),
               'sigma_H2': verdicts.get("H2'  dc/dtau = k-1 = 0.081")},
              open(HERE / 'd54_c_vs_tau_2026-09-06.json', 'w'), indent=2)
    print('\n-> d54_c_vs_tau_2026-09-06.json')


if __name__ == '__main__':
    main()
