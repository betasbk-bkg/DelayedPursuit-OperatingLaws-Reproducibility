#!/usr/bin/env python3
"""
d85 -- where does the published ellipse margin u* = 0.562 come from?

The deterministic answer is settled: every smooth geometry with R/L >= 1 gives
u* = 0.51, and it does not move.

    circle    0.4995            (d38, derived)
    ellipse   0.511 +- 0.001    (fine grid, bowl factors 1.5-8 all agree)
    stadium   0.509 +- 0.004    (R/L 1-6, S/L 2.5-20; d84)

The published ellipse value is 0.562, and at that speed the deterministic RMSE is
2.2x its minimum -- so 0.562 is not the deterministic optimum.  The remaining
candidate is the vote layer, and this measures it.

THE DESIGN POINT.  Running the crowd ellipse alone would not settle it: if it
came out at 0.562 that could be the votes, or the votes interacting with the
shape, or the measurement.  So the CIRCLE is run under the identical protocol as
a control.  Then:

    crowd circle 0.50, crowd ellipse 0.56  ->  votes interact with SHAPE
    both shifted by the same amount        ->  votes shift the margin, shape-free
    both at 0.51                           ->  0.562 is neither; it is the
                                               measurement, and d66's lesson
                                               (MC = 5 cannot locate u*) applies

d66 is why the grid is fine and MC is 10 rather than 5: at MC = 5 the crowd RMSE
curve swings 30-50% between adjacent speeds and `locate` reads noise.  Here the
bowl estimator uses 20+ points, so a single noisy cell cannot carry the answer.

Usage:  python3 d85_crowd_ellipse_2026-09-08.py [--mc 10]
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402

_s = importlib.util.spec_from_file_location(
    'd84', HERE / 'd84_stadium_sweep_2026-09-08.py')
d84 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d84)

LOOK = sm.LOOK
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH
PUBLISHED_ELLIPSE = 0.562
DET_ELLIPSE = 0.511
DET_CIRCLE = 0.4995


class Param:
    """Circle or ellipse as a dense sampled closed curve."""

    def __init__(self, a, b, n=8000):
        self.name = 'circle' if abs(a - b) < 1e-9 else 'ellipse'
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
        self.pts = np.vstack([c, c[:1]])
        d = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.arcs = np.concatenate([[0.0], np.cumsum(d)])
        self.circ = float(self.arcs[-1])

    def closest(self, p):
        d = np.linalg.norm(self.pts - p, axis=1)
        i = int(np.argmin(d))
        return self.pts[i].copy(), float(self.arcs[i])

    def at(self, arc):
        a = arc % self.circ
        i = min(int(np.searchsorted(self.arcs, a)), len(self.pts) - 1)
        return self.pts[i].copy()

    def start(self):
        return self.pts[0].copy()


def simulate(traj, V, delay_f, frames, seed, quantised, N=150, troll=0.05):
    """The engine's loop.  quantised=False is d65's deterministic leg."""
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    gam = 0.5
    err = np.empty(frames)
    dist = 0.0
    lap1 = None
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + LOOK) - dp
            n = float(np.linalg.norm(ideal))
            if n > 1e-10:
                ideal = ideal / n
            if quantised:
                ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
                votes = sm.gen_votes(ia, prev, troll, N, rng)
                b = sm.DIRS[votes].mean(axis=0)
                gam = float(np.linalg.norm(b))
                if gam > 1e-10:
                    cur = b / gam
                prev = ia
                tgt = cur * math.sqrt(gam) * V
            else:
                cur = ideal
                tgt = cur * V
        else:
            tgt = cur * V
        vel += sm.SMOOTH * (tgt - vel)
        step = vel * sm.DT
        dist += float(np.linalg.norm(step))
        pos = pos + step
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        err[f] = float(np.linalg.norm(pos - cp))
        if lap1 is None and dist >= traj.circ:
            lap1 = f
    if dist / traj.circ < 1.5 or lap1 is None:
        return float('nan')
    return float(np.sqrt(np.mean(err[lap1:] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=8)
    ap.add_argument('--dur', type=float, default=240.0)
    ap.add_argument('--tau', type=int, default=433)
    a = ap.parse_args()
    frames = int(a.dur / sm.DT)
    df = int(round(a.tau / 1000.0 / sm.DT))
    tt = a.tau / 1000.0 + TAU_EXTRA
    # The first grid was 1.10..2.10, centred on the DETERMINISTIC optimum -- an
    # error, because whether the crowd optimum sits there is the question.  The
    # crowd circle came out at the top edge (v = 2.10, 'outside'), and the reference engine's
    # own circle value x*(433) = 1.404 implies v* = 2.134, i.e. just past it.
    speeds = np.arange(1.00, 3.501, 0.05)

    print('=' * 78)
    print('d85 -- crowd vs deterministic, circle AND ellipse, same protocol')
    print('=' * 78)
    print('  DUR %.0f s, tau %d ms -> tau_tot %.4f, MC %d, grid %.2f..%.2f step 0.04'
          % (a.dur, a.tau, tt, a.mc, speeds[0], speeds[-1]))
    print('  references: circle 0.4995 (derived), ellipse 0.511 (deterministic),')
    print('              published ellipse %.3f' % PUBLISHED_ELLIPSE)
    print('  the reference engine circle x*(433) = 1.404 -> v* = 2.134 -> u* = %.4f'
          % (1.404 / math.sqrt(0.433) * tt / LOOK))
    print()

    t0 = time.time()
    out = {'tau_tot': tt, 'mc': a.mc, 'speeds': speeds.tolist(), 'rows': []}
    print('  %-10s %-14s %10s %10s %9s %7s'
          % ('geometry', 'layer', 'v*', 'u*', 'how', 'n pts'))
    for gname, (aa, bb) in (('circle', (10.0, 10.0)), ('ellipse', (12.0, 6.0))):
        g = Param(aa, bb)
        for layer, q, mc in (('deterministic', False, 1), ('crowd', True, a.mc)):
            rr = []
            for v in speeds:
                vals = [simulate(g, v, df, frames, 900 + 37 * k, q)
                        for k in range(mc)]
                rr.append(float(np.mean(vals)) if np.all(np.isfinite(vals))
                          else float('nan'))
            v, how, npts = d84.bowl_locate(speeds, rr, factor=4.0)
            if v is None:
                # every speed failed the lap gate -- the horizon is too short for
                # this geometry.  Say so rather than crashing on the format.
                out['rows'].append({'geometry': gname, 'layer': layer, 'mc': mc,
                                    'v_star': None, 'u_star': None, 'how': how,
                                    'n_pts': npts, 'perimeter': g.circ})
                print('  %-10s %-14s %10s %10s %9s %7d'
                      % (gname, layer, '-', '-', how, npts))
                continue
            u = v * tt / LOOK
            out['rows'].append({'geometry': gname, 'layer': layer, 'mc': mc,
                                'v_star': v, 'u_star': u, 'how': how,
                                'n_pts': npts, 'perimeter': g.circ,
                                'rmse': [None if not np.isfinite(x) else float(x)
                                         for x in rr]})
            print('  %-10s %-14s %10.4f %10.4f %9s %7d'
                  % (gname, layer, v, u, how, npts))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    def get(gname, layer):
        for r in out['rows']:
            if r['geometry'] == gname and r['layer'] == layer:
                return r['u_star']
        return None

    cd, cc = get('circle', 'deterministic'), get('circle', 'crowd')
    ed, ec = get('ellipse', 'deterministic'), get('ellipse', 'crowd')
    if None in (cd, cc, ed, ec):
        print()
        print('  one or more cells could not be located; no comparison drawn.')
        json.dump(out, open(HERE / 'd85_crowd_ellipse_2026-09-08.json', 'w'), indent=2)
        return
    print()
    print('  %-24s %12s %12s %12s' % ('', 'circle', 'ellipse', 'ellipse-circle'))
    print('  %-24s %12.4f %12.4f %12.4f' % ('deterministic', cd, ed, ed - cd))
    print('  %-24s %12.4f %12.4f %12.4f' % ('crowd (MC %d)' % a.mc, cc, ec, ec - cc))
    print('  %-24s %12.4f %12.4f' % ('crowd - deterministic', cc - cd, ec - ed))
    print()
    if abs(ec - PUBLISHED_ELLIPSE) < 0.02:
        print('  the crowd ellipse REPRODUCES the published %.3f.' % PUBLISHED_ELLIPSE)
        if abs(cc - cd) < 0.02:
            print('  and the crowd circle does not move -> the vote layer shifts the')
            print('  margin only where the curvature VARIES, i.e. it interacts with')
            print('  shape.  That is a real effect and belongs in the paper.')
        else:
            print('  and the crowd circle moves too -> the vote layer shifts the')
            print('  margin generally, shape-free.')
    else:
        print('  the crowd ellipse gives %.4f, NOT the published %.3f.'
              % (ec, PUBLISHED_ELLIPSE))
        print('  So 0.562 is neither the deterministic nor the crowd optimum under')
        print('  this protocol, and the difference is in how it was measured --')
        print('  d66 showed a three-point parabola on a noisy crowd curve reads')
        print('  noise.  Report that, and quote %.3f.' % ed)

    json.dump(out, open(HERE / 'd85_crowd_ellipse_2026-09-08.json', 'w'), indent=2)
    print('\n-> d85_crowd_ellipse_2026-09-08.json')


if __name__ == '__main__':
    main()
