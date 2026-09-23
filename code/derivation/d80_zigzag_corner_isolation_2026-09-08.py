#!/usr/bin/env python3
"""
d80 -- the zigzag's THIRD axis: is the tooth spacing large enough to isolate its
       corners?

Raised in review.  the path-following note (not distributed) closes two axes -- topology (the look-ahead wrap,
fixed by extending the path) and horizon (low-speed truncation, fixed by the
>=1.5 repeating-unit rule).  Neither touches spacing, and spacing is what decides
whether a corner is entered with the previous corner still ringing.

THE LENGTH.  d45 derived this for the Nav2 side: the corner transient decays with
the dominant root of the loop's characteristic equation, and the audit's
convention is the arclength to reach 1%, log(100)/|Re lambda|, in units of L.
The crowd engine is the OTHER plant class, so the equation is d38's heading-servo
one,

    lam + exp(-lam u) = 0

and it has a closed form.  Substituting x = lam*u gives x*exp(x) = -u, so

    lam = W_0(-u) / u

with W the Lambert W function.  Checks: u -> 0 gives lam = -1, and u = pi/2 gives
Re lam = 0 exactly, which is the known heading-servo stability limit.  The first
version of this script scanned the complex plane with Newton from a grid, which
was both slow and wrong -- its filter discarded the real root and it reported "no
decaying root" at every u, including u -> 0 where lam = -1 by inspection.

WHAT CAME OUT, and it was not what was expected.  The decay length is NOT
monotonic in u: it falls to a minimum of 1.95 L near u = 0.40 and rises on both
sides.  So corner isolation is a WINDOW in u, not a ceiling -- which also means
d45's u_isolated(), a bisection for "the largest u that fits", is only valid for
the curvature-command plant whose decay length is monotone.  It must not be
reused on this one.

Usage:  python3 d80_zigzag_corner_isolation_2026-09-08.py
"""
import json
import math
import pathlib

import numpy as np
from scipy.special import lambertw

HERE = pathlib.Path(__file__).parent
LOOK = 2.0
TAU_TOT = 0.433 + 0.15 + (1.0 / 60) / 0.2     # tau + WIN/2 + DT/SMOOTH
DUR = 65.0
GRID = np.arange(0.25, 5.01, 0.25)
U_C = math.pi / 2                              # heading-servo stability limit


def settle(u):
    """Arclength in units of L for the corner transient to reach 1% (d45's rule)."""
    lam = lambertw(-u, 0) / u
    return float('inf') if lam.real >= 0 else math.log(100.0) / abs(lam.real)


_US = np.linspace(0.02, U_C - 1e-6, 20000)
_SET = np.array([settle(u) for u in _US])


def window(seg_over_L):
    """The u interval a given straight run can isolate.  An interval, not a cap."""
    ok = _SET <= seg_over_L
    if not ok.any():
        return None
    i = np.where(ok)[0]
    return float(_US[i[0]]), float(_US[i[-1]])


def main():
    tooth = math.sqrt(50.0)
    seg = tooth / LOOK
    print('=' * 78)
    print('d80 -- zigzag tooth spacing against the heading-servo decay length')
    print('=' * 78)
    print('  lam = W_0(-u)/u   (closed form);  settle = log(100)/|Re lam|, in L')
    print('  Zigzag(amp=5, sx=5): tooth %.2f m, LOOK %.1f m -> seg/L = %.3f'
          % (tooth, LOOK, seg))
    print('  tau_tot %.4f s, horizon %.0f s' % (TAU_TOT, DUR))
    print()
    print('  %7s %12s %12s' % ('u', 'settle (L)', 'settle (m)'))
    prof = []
    for u in (0.05, 0.10, 0.20, 0.30, 0.40, 0.4995, 0.55, 0.70, 1.00, 1.40):
        st = settle(u)
        prof.append({'u': u, 'settle_L': st, 'settle_m': st * LOOK})
        print('  %7.3f %12.2f %12.2f' % (u, st, st * LOOK))
    print('  -> NOT monotone: minimum near u = 0.40, rising on both sides.')
    print('     Isolation is a WINDOW in u, so d45\'s u_isolated() bisection')
    print('     (valid for curvature-command) must not be reused here.')
    print()

    lo, hi = window(seg)
    v_lo, v_hi = lo * LOOK / TAU_TOT, hi * LOOK / TAU_TOT
    ins = (GRID >= v_lo) & (GRID <= v_hi)
    print('  shipped geometry: isolated for %.4f <= u <= %.4f' % (lo, hi))
    print('                    i.e. %.2f <= v <= %.2f m/s' % (v_lo, v_hi))
    print('  published grid (20 speeds): %d isolated -> %s'
          % (ins.sum(), ', '.join('%.2f' % v for v in GRID[ins])))
    print('  stability limit u_c = pi/2 -> v = %.2f m/s; %d grid speeds exceed it'
          % (U_C * LOOK / TAU_TOT, int((GRID > U_C * LOOK / TAU_TOT).sum())))
    print()

    print('  would a bigger tooth help?  The two axes pull opposite ways: a longer')
    print('  tooth isolates more of the grid, but it lengthens the repeating unit,')
    print('  so the >=1.5-unit horizon rule bites at a higher speed.  The usable')
    print('  grid is the INTERSECTION.')
    print()
    print('  %7s %8s %18s %12s %18s %5s'
          % ('sx (m)', 'seg/L', 'isolated v', 'lap v >=', 'usable v', 'pts'))
    rows = []
    for sx in (5, 6, 8, 10, 12, 16, 20, 30, 50):
        t = math.sqrt(2) * sx
        sl = t / LOOK
        w = window(sl)
        if w is None:
            print('  %7d %8.2f   never isolated' % (sx, sl))
            continue
        a0, b0 = w[0] * LOOK / TAU_TOT, w[1] * LOOK / TAU_TOT
        v_lap = 1.5 * (2 * t) / DUR              # repeating unit = two teeth
        a, b = max(a0, v_lap), b0
        n = int(((GRID >= a) & (GRID <= b)).sum()) if a <= b else 0
        rows.append({'sx': sx, 'tooth_m': t, 'seg_over_L': sl,
                     'v_iso': [a0, b0], 'v_lap_min': v_lap,
                     'v_usable': [a, b] if a <= b else None, 'n_grid': n})
        print('  %7d %8.2f %18s %12.2f %18s %5d'
              % (sx, sl, '%.2f .. %.2f' % (a0, b0), v_lap,
                 ('%.2f .. %.2f' % (a, b)) if a <= b else 'EMPTY', n))
    best = max(rows, key=lambda r: r['n_grid'])
    print()
    print('  best of these: sx = %d m gives %d of 20 usable (v %.2f..%.2f);'
          % (best['sx'], best['n_grid'], best['v_usable'][0], best['v_usable'][1]))
    print('  the shipped sx = 5 m gives %d.  No size recovers the whole grid --'
          % [r for r in rows if r['sx'] == 5][0]['n_grid'])
    print('  the top of the grid runs into u_c, where the decay length diverges.')

    json.dump({'tooth_m': tooth, 'look_m': LOOK, 'seg_over_L': seg,
               'tau_tot': TAU_TOT, 'horizon_s': DUR, 'u_c': U_C,
               'profile': prof, 'shipped_window_u': [lo, hi],
               'shipped_window_v': [v_lo, v_hi],
               'shipped_grid_isolated': int(ins.sum()),
               'size_sweep': rows},
              open(HERE / 'd80_zigzag_isolation_2026-09-08.json', 'w'), indent=2)
    print('\n-> d80_zigzag_isolation_2026-09-08.json')


if __name__ == '__main__':
    main()
