#!/usr/bin/env python3
"""
d81 -- size the zigzag from the OTHER trajectories instead of choosing a number.

Decision taken: take option 1 from d80 (enlarge the tooth), but set the size by
harmonising the zigzag's parameters with the other trajectories rather than by
picking whatever maximises usable grid points.  That is the stronger move -- a
size chosen to maximise a count is a free parameter; a size inherited from the
existing convention is not.

WHAT CONVENTION ALREADY EXISTS.  The manuscript states one: Suzuka was
"normalized to 62.83 m for size comparison with the Circle reference", i.e.
equal path length to the circle.  The polygons carry a second, implicit one:
the Square is h = 10, so its side is 20 m and its straight run per look-ahead is
seg/L = 10.

Those two conventions do not agree for an open zigzag, because an extended
zigzag has no perimeter to equalise -- what it has is a REPEATING UNIT (two
teeth).  So the candidates are enumerated and priced, and the one that inherits
the most from the existing design wins.

Each candidate is scored on the three axes d80/the path-following note (not distributed) identified:
  isolation  -- the u window the straight run can isolate (d80, Lambert W)
  horizon    -- the >=1.5 repeating-unit rule (the path-following note (not distributed) 6.2)
  extent     -- ns must be large enough that v_max*DUR never reaches the end

Usage:  python3 d81_zigzag_harmonised_2026-09-08.py
"""
import json
import math
import pathlib

import numpy as np
from scipy.special import lambertw

HERE = pathlib.Path(__file__).parent
LOOK = 2.0
TAU_TOT = 0.433 + 0.15 + (1.0 / 60) / 0.2
DUR = 65.0
V_MAX = 5.0
GRID = np.arange(0.25, 5.01, 0.25)
U_C = math.pi / 2

SQUARE_SIDE = 20.0          # Square(h=10) -> sides of 20 m
SQUARE_PERIM = 80.0
CIRCLE_PERIM = 2 * math.pi * 10.0


def settle(u):
    lam = lambertw(-u, 0) / u
    return float('inf') if lam.real >= 0 else math.log(100.0) / abs(lam.real)


_US = np.linspace(0.02, U_C - 1e-6, 20000)
_SET = np.array([settle(u) for u in _US])


def window(seg_over_L):
    ok = _SET <= seg_over_L
    if not ok.any():
        return None
    i = np.where(ok)[0]
    return float(_US[i[0]]), float(_US[i[-1]])


def score(name, tooth, note):
    """tooth = straight run between corners, in metres (amp = sx = tooth/sqrt2)."""
    sx = tooth / math.sqrt(2.0)
    seg = tooth / LOOK
    unit = 2 * tooth                       # repeating unit: two teeth
    w = window(seg)
    v_iso = (w[0] * LOOK / TAU_TOT, w[1] * LOOK / TAU_TOT) if w else None
    v_lap = 1.5 * unit / DUR
    if v_iso is None:
        usable, n = None, 0
    else:
        a, b = max(v_iso[0], v_lap), v_iso[1]
        usable = (a, b) if a <= b else None
        n = int(((GRID >= a) & (GRID <= b)).sum()) if usable else 0
    ns_min = math.ceil(V_MAX * DUR / tooth)
    return {'name': name, 'note': note, 'tooth_m': tooth, 'sx_m': sx,
            'amp_m': sx, 'seg_over_L': seg, 'unit_m': unit,
            'v_iso': v_iso, 'v_lap_min': v_lap, 'v_usable': usable,
            'n_grid': n, 'ns_min': ns_min,
            'corners_per_100m': 100.0 / tooth}


def main():
    print('=' * 78)
    print('d81 -- sizing the zigzag from the existing conventions')
    print('=' * 78)
    print('  reference geometries:')
    print('    Circle  R = 10      perimeter %.2f m   (the manuscript\'s size'
          % CIRCLE_PERIM)
    print('                                            normalisation target)')
    print('    Square  h = 10      side %.1f m, perimeter %.1f m, seg/L = %.1f'
          % (SQUARE_SIDE, SQUARE_PERIM, SQUARE_SIDE / LOOK))
    print('    Zigzag  amp=sx=5    tooth %.2f m, seg/L = %.2f   (as shipped)'
          % (math.sqrt(50), math.sqrt(50) / LOOK))
    print()

    cands = [
        score('as shipped', math.sqrt(50.0),
              'amp = sx = 5; inherits nothing'),
        score('tooth = Square side', SQUARE_SIDE,
              'same straight run, same seg/L, same corner angle'),
        score('unit = Square perimeter', SQUARE_PERIM / 2,
              'same repeating unit, so the same horizon rule'),
        score('unit = Circle perimeter', CIRCLE_PERIM / 2,
              "the manuscript's length normalisation, applied to the unit"),
        score('d80 count-optimal', math.sqrt(2.0) * 10.0,
              'sx = 10 chosen to maximise usable points -- a free parameter'),
    ]

    print('  %-24s %8s %8s %8s %16s %8s %6s %6s'
          % ('candidate', 'tooth', 'sx=amp', 'seg/L', 'usable v', 'lap v>=',
             'pts', 'ns>='))
    for c in cands:
        print('  %-24s %8.2f %8.2f %8.2f %16s %8.2f %6d %6d'
              % (c['name'], c['tooth_m'], c['sx_m'], c['seg_over_L'],
                 ('%.2f..%.2f' % c['v_usable']) if c['v_usable'] else 'EMPTY',
                 c['v_lap_min'], c['n_grid'], c['ns_min']))

    print()
    print('  what each candidate inherits, and what it costs:')
    for c in cands:
        print('    %-24s %s' % (c['name'], c['note']))
    print()

    sq = [c for c in cands if c['name'] == 'tooth = Square side'][0]
    print('  RECOMMENDATION: tooth = the Square\'s side.')
    print('    amp = sx = %.3f m  (tooth %.1f m, seg/L %.1f -- identical to the'
          % (sq['sx_m'], sq['tooth_m'], sq['seg_over_L']))
    print('    Square), corner angle 90 deg (identical), corners per 100 m %.1f'
          % sq['corners_per_100m'])
    print('    against the Square\'s %.1f -- the same corner DENSITY too.'
          % (100.0 / SQUARE_SIDE))
    print('    ns >= %d so the agent never reaches the end at v_max = %.1f;'
          % (sq['ns_min'], V_MAX))
    print('    ns = %d gives %.0f m, a factor %.2f of margin.'
          % (24, 24 * sq['tooth_m'], 24 * sq['tooth_m'] / (V_MAX * DUR)))
    print('    usable grid %s -> %d of 20 speeds.'
          % ('%.2f..%.2f' % sq['v_usable'], sq['n_grid']))
    print()
    print('    Every parameter is inherited: the straight run and seg/L from the')
    print('    Square, the corner angle from the shipped zigzag, ns from the')
    print('    horizon bound.  Nothing is chosen to make a number come out.')
    print('    The repeating unit is HALF the Square\'s perimeter, so the horizon')
    print('    rule bites at %.2f m/s against the Square\'s %.2f -- cheaper, not'
          % (sq['v_lap_min'], 1.5 * SQUARE_PERIM / DUR))
    print('    harsher.')

    json.dump({'look': LOOK, 'tau_tot': TAU_TOT, 'horizon_s': DUR,
               'v_max': V_MAX, 'square_side': SQUARE_SIDE,
               'circle_perimeter': CIRCLE_PERIM, 'candidates': cands,
               'recommended': sq},
              open(HERE / 'd81_zigzag_harmonised_2026-09-08.json', 'w'), indent=2)
    print('\n-> d81_zigzag_harmonised_2026-09-08.json')


if __name__ == '__main__':
    main()
