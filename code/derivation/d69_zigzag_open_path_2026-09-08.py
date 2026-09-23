#!/usr/bin/env python3
"""
d69 -- does the zigzag trajectory shuttle on its last segment?

Raised in review.  Zigzag is the only OPEN path in the reference engine's engine: Circle,
Square, Ellipse and Lemniscate all close, and their arclength wraps cleanly.
Zigzag runs (0,0) -> (50, ...) and stops, but `at()` still wraps:

    def at(self, arc):
        arc = arc % self.circ        # <- wraps an OPEN path

so at arc > circ - LOOK the look-ahead point jumps back to the START of the
polyline, roughly 50 m behind the agent.  The commanded direction reverses; the
agent turns around; its nearest point walks back down the last segment, arc drops
below circ - LOOK, and the look-ahead jumps forward again.  That is a shuttle,
and it is confined to the last segment because that is where the wrap lives.

This script measures it rather than argues it:
  * where the agent actually is over time (x extent, and time spent past the
    last vertex);
  * how many times it reverses direction;
  * what the RMSE would be with the wrap removed -- i.e. treating zigzag as the
    open path it is, by clamping the look-ahead to the end.

Usage:  python3 d69_zigzag_open_path_2026-09-08.py
"""
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2] / "engine" / "crowd_control_engine"   # the engine as it ships in this package
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402


class ZigzagClamped(sm.Zigzag):
    """Same geometry; the look-ahead stops at the end instead of wrapping."""

    def at(self, arc):
        arc = min(arc, self.circ)
        for i, (a, b) in enumerate(self.segs):
            if arc <= self.cum[i + 1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]


def run(traj, V, tau_ms, dur, seed=181, N=150, troll=0.05, quantised=True):
    rng = np.random.default_rng(seed)
    frames = int(dur / sm.DT)
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    xs, err, vx = [], [], []
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + sm.LOOK) - dp
            g = float(np.linalg.norm(ideal))
            if g > 1e-10:
                ideal = ideal / g
            if quantised:
                ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
                votes = sm.gen_votes(ia, prev, troll, N, rng)
                b = sm.DIRS[votes].mean(axis=0)
                gb = float(np.linalg.norm(b))
                cur = b / gb if gb > 1e-10 else cur
                prev = ia
            else:
                cur = ideal
        vel += sm.SMOOTH * (cur * V - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        xs.append(float(pos[0]))
        vx.append(float(vel[0]))
        err.append(float(np.linalg.norm(pos - cp)))
    xs = np.array(xs)
    vx = np.array(vx)
    err = np.array(err)
    sign = np.sign(vx)
    reversals = int(np.sum((sign[1:] * sign[:-1]) < 0))
    last_vertex_x = float(traj.c[-2][0])
    return {'x_min': float(xs.min()), 'x_max': float(xs.max()),
            'x_final': float(xs[-1]), 'reversals': reversals,
            'frac_past_last_vertex': float(np.mean(xs > last_vertex_x)),
            'frac_time_in_last_segment': float(np.mean(xs > last_vertex_x)),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'rmse_second_half': float(np.sqrt(np.mean(err[len(err) // 2:] ** 2))),
            'x_series_every_60': xs[::60].round(3).tolist()}


def main():
    z = sm.Zigzag()
    print('=' * 78)
    print('d69 -- zigzag is an OPEN path whose look-ahead wraps')
    print('=' * 78)
    print('  vertices %d, from (%.1f, %.1f) to (%.1f, %.1f), arclength %.2f m'
          % (len(z.c), z.c[0][0], z.c[0][1], z.c[-1][0], z.c[-1][1], z.circ))
    print('  LOOK = %.1f m, so the wrap starts at arc = %.2f m'
          % (sm.LOOK, z.circ - sm.LOOK))
    print('  last vertex is at x = %.1f m' % z.c[-2][0])
    print()
    print('  %6s %7s | %8s %8s %8s %9s %10s %9s'
          % ('v', 'wrap', 'x_min', 'x_max', 'x_final', 'reversals',
             'frac last', 'RMSE'))

    out = {'circ': z.circ, 'look': sm.LOOK, 'runs': {}}
    for V in (2.0, 3.0, 5.0):
        for name, traj in (('wrap', sm.Zigzag()), ('clamp', ZigzagClamped())):
            r = run(traj, V, 433, 65.0)
            out['runs']['%s_v%.1f' % (name, V)] = r
            print('  %6.1f %7s | %8.2f %8.2f %8.2f %9d %9.1f%% %9.4f'
                  % (V, name, r['x_min'], r['x_max'], r['x_final'],
                     r['reversals'], 100 * r['frac_time_in_last_segment'],
                     r['rmse']))
    print()
    print('  reading: the agent covers %.0f m of path; if it traverses the zigzag'
          % z.circ)
    print('  once at v m/s it needs %.0f s at v = 2, and the engine runs 65 s.'
          % (z.circ / 2.0))
    print('  So a run that ENDS with x_max near %.0f and a large "frac last" has'
          % z.c[-1][0])
    print('  spent its scoring window shuttling at the end, not tracking a path.')

    json.dump(out, open(HERE / 'd69_zigzag_open_path_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d69_zigzag_open_path_2026-09-08.json')


if __name__ == '__main__':
    main()
