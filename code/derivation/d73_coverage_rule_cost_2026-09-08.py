#!/usr/bin/env python3
"""
d73 -- what an a-priori coverage rule costs, geometry by geometry.

The merged paper fixes the traversal artefacts by making the paths periodic and
the arc progression monotone, keeping the fixed 65 s horizon.  That closes the
PROJECTION axis (lobe capture, look-ahead wrap).  It does not close the HORIZON
axis: a fixed time budget on a fixed geometry cannot cover both ends of a wide
speed grid, and d69 showed the two ends fail in mirror-image ways --

    low  v : the agent does not get round the geometry (d65: zero corners on a
             triangle at v <= 0.20)
    high v : on an OPEN path the agent runs out of path and the rest of the
             window is not path-following at all (d69: 51-70% of the window)

Closing the paths removes the high-v failure outright.  The low-v failure has to
be handled by a rule, and the rule must be computable from DESIGN CONSTANTS --
geometry, speed grid, horizon -- never from the measured curve, or the evaluation
window becomes endogenous and the whole methodological advantage is lost.

This script prices three candidate rules.  For each geometry it reports the speed
floor the rule implies and how many grid points are lost, so the rule can be
chosen with its cost visible instead of after seeing which one is convenient.

Usage:  python3 d73_coverage_rule_cost_2026-09-08.py
"""
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402

_s = importlib.util.spec_from_file_location(
    'd70', HERE / 'd70_arc_monotonic_2026-09-08.py')
d70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d70)

GRID = np.arange(0.25, 5.01, 0.25)      # the speed grid Paper 1 used


def corners_of(t, tol_deg=1.0):
    """Corners per lap, counted as DIRECTION CHANGES, not as segments.

    Counting segments is wrong for the engine's Square: its start point sits at
    the middle of the right edge, so the perimeter is cut into five segments
    (10, 20, 20, 20, 10) around four corners.  Any rule phrased in corners has to
    count turns.
    """
    segs = getattr(t, 'segs', None)
    if segs is None:
        return None
    closed = bool(np.allclose(t.c[0], t.c[-1]))
    dirs = []
    for a, b in segs:
        v = np.asarray(b, float) - np.asarray(a, float)
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            dirs.append(v / n)
    k = len(dirs)
    pairs = range(k) if closed else range(k - 1)
    return sum(1 for i in pairs
               if math.degrees(math.acos(
                   float(np.clip(dirs[i] @ dirs[(i + 1) % k], -1, 1)))) > tol_deg)


def seg_over_look(t):
    """Mean straight run between corners, in look-ahead units."""
    nc = corners_of(t)
    if not nc:
        return None
    return float(t.circ) / nc / sm.LOOK


def main():
    geoms = [('circle', sm.Circle()),
             ('square', sm.Square()),
             ('lemniscate', sm.Lemniscate()),
             ('zigzag (open, shipped)', sm.Zigzag()),
             ('zigzag (closed)', d70.ZigzagClosed())]

    print('=' * 78)
    print('d73 -- the price of an a-priori coverage rule, at a fixed %.0f s horizon'
          % sm.DUR)
    print('=' * 78)
    print('  speed grid %.2f .. %.2f step %.2f  (%d points, as Paper 1 ran it)'
          % (GRID[0], GRID[-1], GRID[1] - GRID[0], len(GRID)))
    print()
    print('  %-24s %9s %8s %9s %9s %10s %10s'
          % ('geometry', 'arclen', 'corners', 'seg/LOOK', 'v(1 lap)', 'v(1.5 lap)',
             'v(4 corn)'))
    out = {'horizon_s': sm.DUR, 'grid': GRID.tolist(), 'geoms': {}}
    for name, t in geoms:
        circ = float(t.circ)
        nc = corners_of(t)
        v1 = circ / sm.DUR
        v15 = 1.5 * circ / sm.DUR
        v4 = (4.0 / nc) * circ / sm.DUR if nc else float('nan')
        sl = seg_over_look(t)
        out['geoms'][name] = {'arclen_m': circ, 'corners': nc, 'seg_over_look': sl,
                              'v_1lap': v1, 'v_1p5lap': v15, 'v_4corners': v4}
        print('  %-24s %9.2f %8s %9s %9.2f %10.2f %10s'
              % (name, circ, nc if nc else '-',
                 ('%.2f' % sl) if sl else '-', v1, v15,
                 ('%.2f' % v4) if nc else '-'))

    print()
    print('  grid points SURVIVING each rule (of %d):' % len(GRID))
    print('  %-24s %12s %12s %12s'
          % ('geometry', '>=1 lap', '>=1.5 laps', '>=4 corners'))
    for name, t in geoms:
        g = out['geoms'][name]
        k1 = int(np.sum(GRID >= g['v_1lap']))
        k15 = int(np.sum(GRID >= g['v_1p5lap']))
        k4 = int(np.sum(GRID >= g['v_4corners'])) if g['corners'] else None
        g.update(n_1lap=k1, n_1p5lap=k15, n_4corners=k4)
        print('  %-24s %12d %12d %12s'
              % (name, k1, k15, k4 if k4 is not None else '-'))

    print()
    print('  READING')
    print('  * >=1.5 laps is d65\'s rule, written for SMALL polygons (side 10 m,')
    print('    perimeter 40 m).  On these geometries it deletes most of the grid:')
    for name, t in geoms:
        g = out['geoms'][name]
        print('      %-24s keeps %d of %d' % (name, g['n_1p5lap'], len(GRID)))
    print('  * a CORNER-COUNT rule is the one that transfers, because what low-v')
    print('    truncation destroys is corner sampling, not lap count.  A smooth')
    print('    path (circle, lemniscate) has no corners and needs no such rule --')
    print('    it is homogeneous, so any arc of it is a fair sample.')
    print('  * that is also why closing the zigzag is cheap: it doubles the')
    print('    circuit but doubles the corners with it, so the corner rule bites')
    print('    at the same speed.  Compare the two zigzag rows above.')

    # the square is the case that matters, because Paper 1 reported it
    sq = out['geoms']['square']
    print()
    print('  THE SQUARE IS THE CASE THAT MATTERS -- Paper 1 reported it.')
    print('    perimeter %.1f m, %d corners, horizon %.0f s'
          % (sq['arclen_m'], sq['corners'], sm.DUR))
    for v in (0.25, 0.50, 0.75, 1.00):
        print('      v = %.2f -> %5.1f m -> %.2f laps -> %.1f corners'
              % (v, v * sm.DUR, v * sm.DUR / sq['arclen_m'],
                 v * sm.DUR / sq['arclen_m'] * sq['corners']))
    print('    so the published grid already contains points that traverse')
    print('    fewer than one corner.  Whatever rule the merged paper adopts,')
    print('    it has to be applied to the square too, not only to the new')
    print('    geometries -- otherwise the two archives differ in a second way.')

    json.dump(out, open(HERE / 'd73_coverage_rule_2026-09-08.json', 'w'), indent=2)
    print('\n-> d73_coverage_rule_2026-09-08.json')


if __name__ == '__main__':
    main()
