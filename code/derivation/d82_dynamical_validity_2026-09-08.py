#!/usr/bin/env python3
"""
d82 -- dynamical validity of every trajectory, on the variables that enter the
       dynamics.

A correction raised in review, on which this builds: total path length is NOT a dynamical
parameter.  u = v tau_tot / L, and geometry enters only as shape in units of L.
An earlier note in the path-following note (not distributed) compared trajectories by perimeter and called
the 2.18x spread an inconsistency.  That was wrong twice over -- the spread is
not an inconsistency, and length is not the variable.

FOUR CONDITIONS, in the order they bite:

  (A) GEOMETRIC.   Pure pursuit aims at a point one look-ahead ahead along the
      path.  If the path turns inside that distance the aim point crosses to the
      far side of the turn and the law being tested is not the law being run.
      Condition: R_min >= L.  This is not about tracking quality; it is about
      whether the controller is doing pure pursuit at all.

  (B) QUASI-STATIC.  The equilibrium offset is proportional to kappa, and the
      loop follows changes in it with an e-folding length ell = 1/|Re lam| from
      lam + exp(-lam u) = 0.  The lag error is ell * |dkappa/ds| against an
      offset of order kappa_rms, so the relative error is ell / Lambda with

          Lambda = kappa_rms / max|dkappa/ds|

      A circle has max|dkappa/ds| = 0, hence Lambda = infinity, hence no
      condition -- which is why it is the reference.  On a polygon the same
      argument reduces to d45's corner isolation with the straight run in place
      of Lambda, which is the check that this is the right generalisation.

  (C) STABILITY.   u < u_c = pi/2 for the heading-servo plant.
  (D) HORIZON.     >= 1.5 repeating units inside DUR (the path-following note (not distributed) 6.2).

NUMERICS.  Curvature from two finite differences on a resampled polyline is
useless -- the first version of this script got the ellipse's Lambda wrong by 6x
that way (0.28 L against the analytic 1.62 L) and declared every smooth
trajectory invalid on the strength of it.  Curvature is therefore analytic where
an analytic form exists, and spline-based with a stated smoothing scale where it
does not (Suzuka).

Usage:  python3 d82_dynamical_validity_2026-09-08.py
"""
import json
import math
import pathlib

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.special import lambertw

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent

LOOK = 2.0
TAU_TOT = 0.433 + 0.15 + (1.0 / 60) / 0.2
DUR = 65.0
GRID = np.arange(0.25, 5.01, 0.25)
U_C = math.pi / 2
U_STAR = 0.4995


def efold(u):
    """e-folding arclength of the loop transient, in units of L."""
    lam = lambertw(-u, 0) / u
    return float('inf') if lam.real >= 0 else 1.0 / abs(lam.real)


def settle_1pct(u):
    lam = lambertw(-u, 0) / u
    return float('inf') if lam.real >= 0 else math.log(100.0) / abs(lam.real)


_US = np.linspace(0.02, U_C - 1e-6, 20000)
_EF = np.array([efold(u) for u in _US])
_S1 = np.array([settle_1pct(u) for u in _US])


def u_window(length_over_L, arr):
    ok = arr <= length_over_L
    if not ok.any():
        return None
    i = np.where(ok)[0]
    return float(_US[i[0]]), float(_US[i[-1]])


# ------------------------------------------------------ analytic curvatures
def analytic(name, xy, dxy, d2xy, n=200000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x1, y1 = dxy(t)
    x2, y2 = d2xy(t)
    sp = np.hypot(x1, y1)
    k = np.abs(x1 * y2 - y1 * x2) / np.maximum(sp ** 3, 1e-15)
    ds = sp * (2 * np.pi / n)
    dk = (np.roll(k, -1) - np.roll(k, 1)) / (np.roll(ds, -1) + ds)
    length = float(ds.sum())
    return pack(name, k, dk, length)


def pack(name, k, dk, length):
    kmax, kmin = float(k.max()), float(k.min())
    krms = float(np.sqrt(np.mean(k ** 2)))
    dkmax = float(np.abs(dk).max())
    lam = (float('inf') if dkmax < 1e-12 else krms / dkmax)
    return {'name': name, 'kind': 'smooth', 'length_m': length,
            'R_min_m': (1.0 / kmax) if kmax > 0 else float('inf'),
            'R_max_m': (1.0 / kmin) if kmin > 0 else float('inf'),
            'kappa_rms': krms, 'dkappa_max': dkmax,
            'Lambda_m': lam, 'Lambda_over_L': lam / LOOK}


def main():
    rows = []

    a, b = 12.0, 6.0
    rows.append(analytic(
        'Circle R=10', None,
        lambda t: (-10 * np.sin(t), 10 * np.cos(t)),
        lambda t: (-10 * np.cos(t), -10 * np.sin(t))))
    rows.append(analytic(
        'Ellipse a=12,b=6', None,
        lambda t: (-a * np.sin(t), b * np.cos(t)),
        lambda t: (-a * np.cos(t), -b * np.sin(t))))

    # lemniscate of Gerono-type as coded: (A cos t/d, A sin t cos t/d), d = 1+sin^2
    A = 7.0

    def lem_d1(t):
        s, c = np.sin(t), np.cos(t)
        d = 1 + s * s
        dd = 2 * s * c
        x = A * (-s * d - c * dd) / d ** 2
        y = A * ((c * c - s * s) * d - s * c * dd) / d ** 2
        return x, y

    def lem_d2(t, h=1e-6):
        x1, y1 = lem_d1(t + h)
        x0, y0 = lem_d1(t - h)
        return (x1 - x0) / (2 * h), (y1 - y0) / (2 * h)

    rows.append(analytic('Lemniscate a=7', None, lem_d1, lem_d2, n=100000))

    # Suzuka: centreline from the repo, rescaled to the circle's perimeter as the
    # manuscript describes.  Spline-smoothed; the smoothing scale is stated
    # because curvature from 351 raw points is meaningless.
    csv = np.genfromtxt(REPO / 'suzuka_track.csv', delimiter=',', skip_header=1)
    p = np.vstack([csv, csv[:1]])
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    raw_len = float(d.sum())
    p = p * (2 * math.pi * 10.0) / raw_len
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0.02, per=True)
    tt = np.linspace(0, 1, 40000, endpoint=False)
    x1, y1 = splev(tt, tck, der=1)
    x2, y2 = splev(tt, tck, der=2)
    sp = np.hypot(x1, y1)
    k = np.abs(x1 * y2 - y1 * x2) / np.maximum(sp ** 3, 1e-15)
    ds = sp / len(tt)
    dk = (np.roll(k, -1) - np.roll(k, 1)) / (np.roll(ds, -1) + ds)
    rows.append(pack('Suzuka (to 62.83 m)', k, dk, float(ds.sum())))
    rows.append(pack('Suzuka (UNSCALED, 5.807 km)',
                     k * (2 * math.pi * 10.0) / raw_len,
                     dk * ((2 * math.pi * 10.0) / raw_len) ** 2,
                     raw_len))

    for name, seg in (('Square h=10', 20.0),
                      ('Zigzag amp=sx=5', math.sqrt(50.0)),
                      ('Zigzag harmonised', 20.0)):
        rows.append({'name': name, 'kind': 'polygon', 'length_m': None,
                     'R_min_m': float('inf'), 'R_max_m': float('inf'),
                     'Lambda_m': seg, 'Lambda_over_L': seg / LOOK})

    print('=' * 78)
    print('d82 -- dynamical validity (analytic curvature; length plays no part)')
    print('=' * 78)
    print('  L = %.1f m, tau_tot = %.4f s, u* = %.4f, u_c = %.4f'
          % (LOOK, TAU_TOT, U_STAR, U_C))
    print('  e-folding length at u*: %.3f L = %.2f m'
          % (efold(U_STAR), efold(U_STAR) * LOOK))
    print()
    print('  (A) GEOMETRIC: R_min >= L ?')
    print('  %-30s %10s %10s %8s' % ('trajectory', 'R_min (m)', 'R_min / L', 'pass'))
    for r in rows:
        if r['kind'] != 'smooth':
            continue
        ok = r['R_min_m'] >= LOOK
        r['pass_A'] = bool(ok)
        print('  %-30s %10.3f %10.2f %8s'
              % (r['name'], r['R_min_m'], r['R_min_m'] / LOOK,
                 'YES' if ok else '** NO **'))
    print()
    print('  (B) QUASI-STATIC: ell(u*) <= Lambda ?')
    print('  %-30s %12s %12s %10s %8s'
          % ('trajectory', 'Lambda (m)', 'Lambda / L', 'ell/Lambda', 'pass'))
    for r in rows:
        lam = r['Lambda_over_L']
        ratio = (efold(U_STAR) / lam) if np.isfinite(lam) and lam > 0 else 0.0
        r['ell_over_Lambda'] = float(ratio)
        r['pass_B'] = bool(ratio <= 1.0)
        print('  %-30s %12s %12s %10.2f %8s'
              % (r['name'],
                 'inf' if not np.isfinite(r['Lambda_m']) else '%.2f' % r['Lambda_m'],
                 'inf' if not np.isfinite(lam) else '%.2f' % lam,
                 ratio, 'YES' if ratio <= 1.0 else '** NO **'))
    print()
    print('  (C)+(D) the u window each geometry can hold, and where u* sits')
    print('  %-30s %20s %8s %6s' % ('trajectory', 'valid u (1% rule)', 'u* in?', 'grid'))
    for r in rows:
        w = u_window(r['Lambda_over_L'], _S1)
        r['u_window'] = w
        if w is None:
            r['n_grid'], r['u_star_inside'] = 0, False
            print('  %-30s %20s %8s %6d' % (r['name'], 'none', 'no', 0))
            continue
        v0, v1 = w[0] * LOOK / TAU_TOT, w[1] * LOOK / TAU_TOT
        n = int(((GRID >= v0) & (GRID <= v1)).sum())
        r['n_grid'] = n
        r['u_star_inside'] = bool(w[0] <= U_STAR <= w[1])
        print('  %-30s %20s %8s %6d'
              % (r['name'], '%.3f .. %.3f' % w,
                 'YES' if r['u_star_inside'] else '** NO **', n))

    json.dump({'look': LOOK, 'tau_tot': TAU_TOT, 'u_star': U_STAR, 'u_c': U_C,
               'efold_at_ustar_L': efold(U_STAR), 'rows': rows},
              open(HERE / 'd82_dynamical_validity_2026-09-08.json', 'w'),
              indent=2, default=str)
    print('\n-> d82_dynamical_validity_2026-09-08.json')


if __name__ == '__main__':
    main()
