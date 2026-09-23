#!/usr/bin/env python3
"""
d58 -- the reference engine's corner-frequency experiment is a corner-ISOLATION sweep, and it
       was never read that way.

the reference engine ran 11,700 simulations sweeping the square's half-width h over
{10, 7, 5, 3.5, 2.5} at a FIXED corner angle of 90 degrees, six latencies and two
troll levels, and concluded that "corner frequency is collinear with alpha
(r = +0.94) and thus adds no independent explanatory power".

But with LOOK = 2.0 m the side length is 2h, so that sweep is

    seg / L  =  2h / LOOK  =  10, 7, 5, 3.5, 2.5

which is exactly the corner-isolation variable d45 derived from the loop's
characteristic equation.  d45 found the transient needs 3.95 L to decay at
u = 0.306 and 15.85 L at u = 0.44, so h = 2.5 (seg/L = 2.5) cannot possibly
isolate its corners and h = 10 only barely can.

That matters for the one remaining inconsistency between the two papers.  this study's
d38 predicts u* is independent of the corner angle; Nav2 confirms it (u* moves
1.05x over alpha = 30-90 deg) but the reference engine's own data appears to contradict it
(u* moves 2.4x over alpha = 0-120 deg).  If u* also moves with seg/L at FIXED
alpha, then part of what the reference engine reads as an angle effect is a corner-density
effect, and the two papers may not disagree about the angle at all.

This script asks the reference engine's corner-frequency data that question directly.

Usage:  python3 d58_corner_freq_isolation_2026-09-07.py
"""
import json
import pathlib
import zipfile

import numpy as np
from scipy.optimize import least_squares

ZIP = pathlib.Path(r'C:\Users\betas\Downloads\crowd_control_engine (4).zip')
HERE = pathlib.Path(__file__).parent
LOOK = 2.0
WIN, DT, SMOOTH = 0.3, 1.0 / 60.0, 0.2
TAU_EXTRA = WIN / 2 + DT / SMOOTH


def fit_lag(tau, v):
    """this study lag law: v* = u* L / (tau + c), both free."""
    def resid(p):
        return (p[0] * LOOK / (tau + p[1]) - v) / v
    s = least_squares(resid, [0.5, TAU_EXTRA], bounds=([0.01, 0.0], [3.0, 2.0]))
    pred = s.x[0] * LOOK / (tau + s.x[1])
    return (float(s.x[0]), float(s.x[1]),
            float(100 * np.sqrt(np.mean(((pred - v) / v) ** 2))))


def fit_power(tau, v):
    x, y = np.log(tau), np.log(v)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(-coef[0]), float(np.exp(coef[1]))


def main():
    z = zipfile.ZipFile(ZIP)
    cf = json.loads(z.read('crowd_control_engine/corner_freq_results.json'))
    uni = json.loads(z.read('crowd_control_engine/unified_vstar_results.json'))

    print('=' * 78)
    print("d58 -- the reference engine's corner-frequency sweep, read as a corner-isolation sweep")
    print('=' * 78)
    print('  ScaledSquare(h): side = 2h, corner angle fixed at 90 deg')
    print('  LOOK = %.1f m  ->  seg/L = 2h / LOOK' % LOOK)
    print()
    print('  %6s %8s %8s %10s %10s %8s %8s %9s' %
          ('h', 'side', 'seg/L', 'u* (lag)', 'c (s)', 'rms%', 'beta', 'x*=C'))

    rows = []
    for h in cf['h_values']:
        key = str(int(h)) if float(h).is_integer() else str(h)
        blob = cf['results'].get(key) or cf['results'].get(str(h))
        if not blob:
            print('  %6s  MISSING' % h)
            continue
        taus, vst = [], []
        for tms, trd in sorted(blob.items(), key=lambda kv: int(kv[0])):
            e = trd.get('0.05') or list(trd.values())[0]
            taus.append(int(tms) / 1000.0)
            vst.append(e['vstar'])
        tau, v = np.array(taus), np.array(vst)
        u, c, rms = fit_lag(tau, v)
        beta, C = fit_power(tau, v)
        segL = 2 * float(h) / LOOK
        rows.append({'h': float(h), 'side': 2 * float(h), 'seg_over_L': segL,
                     'u_star': u, 'c_s': c, 'rms_pct': rms,
                     'beta': beta, 'C': C, 'n_tau': len(tau)})
        print('  %6s %8.1f %8.2f %10.4f %10.4f %7.2f%% %8.4f %9.4f'
              % (h, 2 * float(h), segL, u, c, rms, beta, C))

    us = np.array([r['u_star'] for r in rows])
    sg = np.array([r['seg_over_L'] for r in rows])
    print()
    print('  u* over the seg/L sweep at FIXED alpha = 90 deg:')
    print('     range %.4f - %.4f   ->  %.2fx  (spread %.1f%%)'
          % (us.min(), us.max(), us.max() / us.min(),
             100 * (us.max() - us.min()) / us.mean()))

    # compare with the angle sweep from the same repo
    ang = {}
    for geom, blob in uni['results'].items():
        taus, vst = [], []
        for tms, trd in sorted(blob['data'].items(), key=lambda kv: int(kv[0])):
            e = trd.get('0.05') or list(trd.values())[0]
            taus.append(int(tms) / 1000.0)
            vst.append(e['vstar'])
        if len(taus) >= 3:
            u, c, rms = fit_lag(np.array(taus), np.array(vst))
            ang[geom] = {'alpha': blob['alpha'], 'u_star': u,
                         'seg_over_L': (2 * 10.0 / LOOK) if geom == 'square' else None}
    au = np.array([a['u_star'] for a in ang.values()])
    print('  u* over the ANGLE sweep (alpha 0-120 deg, seg/L fixed):')
    print('     range %.4f - %.4f   ->  %.2fx  (spread %.1f%%)'
          % (au.min(), au.max(), au.max() / au.min(),
             100 * (au.max() - au.min()) / au.mean()))

    print()
    print('  VERDICT')
    r_seg = us.max() / us.min()
    r_ang = au.max() / au.min()
    if r_seg > 1.3:
        print('    u* moves %.2fx with seg/L at CONSTANT angle.  Corner density is')
        print('    NOT a nuisance variable here -- it is comparable to the %.2fx that')
        print('    the angle sweep produces, so the two are confounded in this data.'
              % r_ang)
        print("    the reference engine's own conclusion -- that corner frequency 'adds no")
        print("    independent explanatory power' -- does not follow from this.")
    else:
        print('    u* moves only %.2fx with seg/L at constant angle, against %.2fx'
              % (r_seg, r_ang))
        print('    for the angle sweep, so the angle effect is genuine and not a')
        print('    corner-density artefact.  The disagreement with d38 stands.')

    print()
    print('  isolation limits derived in d45 (curvature-command loop, indicative):')
    print('    u = 0.306 needs 3.95 L      u = 0.44 needs 15.85 L')
    print('    so h = 2.5 (seg/L 2.50) and h = 3.5 (seg/L 3.50) cannot isolate at all,')
    print('    and h = 10 (seg/L 10.0) is the only setting with real margin.')

    out = {'seg_sweep': rows, 'angle_sweep': ang,
           'ratio_seg': float(r_seg), 'ratio_angle': float(r_ang)}
    json.dump(out, open(HERE / 'd58_corner_freq_isolation_2026-09-07.json', 'w'),
              indent=2, default=float)
    print('\n-> d58_corner_freq_isolation_2026-09-07.json')


if __name__ == '__main__':
    main()
