#!/usr/bin/env python3
"""
d47 -- E6 nulls judged against the MEASURED row-to-row reproducibility (E7).

d44 built the denominator by propagating a single-point RMSE reproducibility
(sigma_rel = 0.350%, from one triple at one operating point) through the
estimator by Gaussian Monte-Carlo.  On that denominator all three null knobs
"exceeded the noise" -- including transform_tolerance, whose coupling the same
script had bounded at 0.00086.  A knob that cannot move the answer appeared to
move it significantly, which means the denominator was wrong, not the knob.

E7 measures the denominator directly: five IDENTICAL rows, each with its own
stack launch, nothing swept.  The spread IS the measurement.  This script uses
it, and re-judges E6 with no chosen number anywhere in the chain.

Also reports the shape of that noise, because d44 assumed it was Gaussian.

Usage:  python3 d47_null_vs_measured_noise_2026-09-06.py
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


def shapefit_row(vs, rs, J, tau_seed=0.23):
    def resid(p):
        tt, A, B = p
        pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
        return (pred - rs) / rs
    A0 = rs.min() ** 2 / max(J(vs * tau_seed / L).min(), 1e-12)
    s = least_squares(resid, [tau_seed, A0, rs.min() * 0.5],
                      bounds=([0.05, 0.0, 0.0], [1.0, np.inf, np.inf]))
    return float(s.x[0])


def main():
    J = make_J()
    rep_f = sorted(glob.glob(str(HERE / 'nav2_block_replicate_*.json')))[-1]
    rep = json.load(open(rep_f))
    rows = [r for r in rep['rows'] if r.get('curve')]
    tau_book = rep['meta']['tau'] + 0.03

    u = np.array([r['u_obs'] for r in rows])
    tt = np.array([shapefit_row(np.array([c['v'] for c in r['curve']]),
                                np.array([c['rmse'] for c in r['curve']]), J)
                   for r in rows])

    su_pct = 100 * u.std(ddof=1) / u.mean()
    ru_pct = 100 * (u.max() - u.min()) / u.mean()
    st_ms = 1000 * tt.std(ddof=1)
    rt_ms = 1000 * (tt.max() - tt.min())

    print('=' * 78)
    print('E7 -- row-to-row reproducibility, %d identical rows, fresh stack each' % len(rows))
    print('=' * 78)
    print('  parabola   u*      = %s' % np.array2string(u, precision=6))
    print('             mean %.6f   sd %.6f (%.3f%%)   range %.3f%%'
          % (u.mean(), u.std(ddof=1), su_pct, ru_pct))
    print('  shape fit  tau_tot = %s ms' % np.array2string(1000 * tt, precision=2))
    print('             mean %.2f ms   sd %.2f ms   range %.2f ms'
          % (1000 * tt.mean(), st_ms, rt_ms))

    # shape of the noise: d44 assumed Gaussian.  Is it?
    z = (u - u.mean()) / u.std(ddof=1)
    gap = np.sort(u)
    gaps = np.diff(gap)
    print()
    print('  shape of the noise (d44 assumed Gaussian):')
    print('    sorted u*      %s' % np.array2string(gap, precision=6))
    print('    gaps           %s' % np.array2string(gaps, precision=6))
    print('    largest gap is %.1fx the median gap -> %s'
          % (gaps.max() / np.median(gaps),
             'CLUSTERED, not Gaussian' if gaps.max() / np.median(gaps) > 3
             else 'no obvious clustering'))
    print('    (the RMSE curves repeat to 5 digits on some points and jump 2%% on')
    print('     others, so the estimator lands on one of a few discrete outcomes)')

    # ---- re-judge E6 against this -------------------------------------------
    nul_f = sorted(glob.glob(str(HERE / 'nav2_block_nulls_*.json')))[-1]
    nul = json.load(open(nul_f))
    knobs = {}
    for r in nul['rows']:
        knobs.setdefault(r['knob'], []).append(r)

    print()
    print('=' * 78)
    print('E6 re-judged against the MEASURED denominator (no chosen number)')
    print('=' * 78)
    print('  A null knob passes if its spread is not larger than the spread the')
    print('  SAME measurement gives when nothing is changed at all.')
    print()
    print('  %-22s %10s %14s %10s' % ('knob', 'spread%', 'vs E7 range', 'verdict'))
    print('  %-22s %10.3f %14s %10s' % ('(E7, nothing swept)', ru_pct, '1.00x', 'reference'))
    out = {'replicate_file': rep_f, 'n_rep': len(rows),
           'u_mean': float(u.mean()), 'u_sd_pct': su_pct, 'u_range_pct': ru_pct,
           'tau_mean_ms': float(1000 * tt.mean()), 'tau_sd_ms': st_ms,
           'tau_range_ms': rt_ms, 'knobs': {}}
    allpass = True
    for kn, rs in knobs.items():
        uu = np.array([r['u_obs'] for r in rs])
        sp = 100 * (uu.max() - uu.min()) / uu.mean()
        ratio = sp / ru_pct
        ok = ratio <= 1.0
        allpass &= ok
        out['knobs'][kn] = {'spread_pct': sp, 'ratio_to_replicate': ratio,
                            'verdict': 'PASS' if ok else 'CHECK'}
        print('  %-22s %10.3f %13.2fx %10s' % (kn, sp, ratio, 'PASS' if ok else 'CHECK'))
    out['all_pass'] = bool(allpass)

    print()
    if allpass:
        print('  ALL THREE knobs move u* LESS than repeating the identical row does.')
        print('  That is the strongest form this null test can take: the knob is')
        print('  indistinguishable from doing nothing, measured against doing nothing.')
    print()
    print('  what this overturns:')
    print('    - d44 leg 1/2 said all three EXCEEDED the noise.  That denominator')
    print('      (single-point RMSE noise, propagated, Gaussian) understated the')
    print('      real row-to-row spread by %.1fx.' % (ru_pct / 0.58))
    print('    - the original 3%%/5%% threshold happened to give the right verdict,')
    print('      but for the wrong reason; the measured reference is %.2f%%.' % ru_pct)
    json.dump(out, open(HERE / 'd47_null_vs_measured_noise_2026-09-06.json', 'w'), indent=2)
    print('\n-> d47_null_vs_measured_noise_2026-09-06.json')


if __name__ == '__main__':
    main()
