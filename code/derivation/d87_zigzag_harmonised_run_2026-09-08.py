#!/usr/bin/env python3
"""
d87 -- run the harmonised zigzag, against the shipped one.

d81 fixed the parameters by inheritance rather than by choice:

    tooth 20.0 m  = the Square's side      -> seg/L = 10.00, as the Square
    corner 90 deg = the shipped zigzag                 corner density as the Square
    ns = 24       = from v_max*DUR = 325 m, with 1.48x margin

d80 is why: the shipped zigzag (tooth 7.07 m, seg/L 3.54) isolates its corners
only for 0.203 <= u <= 0.555, four of twenty grid speeds, while the harmonised
one isolates 0.020 <= u <= 0.881 like the Square.

PRE-REGISTERED.  The harmonised zigzag has the Square's straight run, the
Square's corner angle and the Square's corner density.  The alpha law measured on
regular polygons (d65/d78) gives

    u*(90 deg) = 0.511 cos^0.88(45 deg) = 0.3767

so if the alternation of turn direction does not matter, the harmonised zigzag
should land there.  The shipped zigzag should NOT, because most of its grid is
outside its own isolation window.

That is the whole point of harmonising: with seg/L, angle and density matched to
the Square, the ONLY remaining difference is that the corners alternate in sign
and the path does not close.  The measurement therefore tests exactly that.

Deterministic engine: the question is geometric, and d66 showed the vote layer
cannot resolve u* at feasible MC.

Usage:  python3 d87_zigzag_harmonised_run_2026-09-08.py
"""
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
U_SQUARE = 0.511 * math.cos(math.radians(45.0)) ** 0.88


class ZigzagOpen:
    """The shipped sawtooth, extended.  amp = sx so the corner is 90 deg; the
    look-ahead never wraps because ns is large enough that the end is never
    reached (d75: the hard bound is v_max*DUR)."""

    def __init__(self, sx, ns):
        self.name = 'zigzag'
        self.sx = float(sx)
        pts = [np.array([0.0, 0.0])]
        for i in range(int(ns)):
            pts.append(np.array([(i + 1) * sx, sx if i % 2 == 0 else 0.0]))
        self.c = np.array(pts)
        self.segs = [(self.c[i], self.c[i + 1]) for i in range(len(self.c) - 1)]
        self.lens = [float(np.linalg.norm(b - a)) for a, b in self.segs]
        self.tooth = self.lens[0]
        self.circ = float(sum(self.lens))
        self.cum = np.array([0.0] + list(np.cumsum(self.lens)))
        self._a = np.array([a for a, _ in self.segs])
        self._v = np.array([b - a for a, b in self.segs])
        self._l2 = np.einsum('ij,ij->i', self._v, self._v)

    def closest(self, p):
        # Vectorised over all segments.  The scalar loop cost 60 projections per
        # frame x 14400 frames x 71 speeds = 61M, which overran a 15-minute
        # budget on the first geometry alone.
        w = p - self._a
        t = np.clip((w * self._v).sum(1) / self._l2, 0.0, 1.0)
        q = self._a + t[:, None] * self._v
        d = np.linalg.norm(q - p, axis=1)
        i = int(np.argmin(d))
        return q[i].copy(), float(self.cum[i] + t[i] * self.lens[i])

    def at(self, arc):
        arc = min(max(arc, 0.0), self.circ)      # clamp: the path is open
        for i, (a, b) in enumerate(self.segs):
            if arc <= self.cum[i + 1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


def run(traj, V, delay_f, frames, unit):
    """RMSE scored after the first repeating unit (two teeth), which is this
    geometry's analogue of the first lap."""
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    err = np.empty(frames)
    dist = 0.0
    first = None
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + LOOK) - dp
            n = float(np.linalg.norm(ideal))
            if n > 1e-10:
                cur = ideal / n
        vel += sm.SMOOTH * (cur * V - vel)
        step = vel * sm.DT
        dist += float(np.linalg.norm(step))
        pos = pos + step
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        err[f] = float(np.linalg.norm(pos - cp))
        if first is None and dist >= unit:
            first = f
    if dist < 1.5 * unit or first is None or dist > 0.95 * traj.circ:
        return float('nan')            # too little geometry, or ran off the end
    return float(np.sqrt(np.mean(err[first:] ** 2)))


def main():
    dur, tau = 240.0, 433
    frames = int(dur / sm.DT)
    df = int(round(tau / 1000.0 / sm.DT))
    tt = tau / 1000.0 + TAU_EXTRA
    speeds = np.arange(0.20, 3.001, 0.04)

    print('=' * 78)
    print('d87 -- harmonised zigzag vs shipped')
    print('=' * 78)
    print('  deterministic engine, DUR %.0f s, tau %d ms -> tau_tot %.4f'
          % (dur, tau, tt))
    print('  PRE-REGISTERED: the harmonised zigzag inherits the Square\'s straight')
    print('  run, corner angle and corner density, so the alpha law predicts')
    print('  u* = 0.511 cos^0.88(45 deg) = %.4f' % U_SQUARE)
    print()
    print('  %-12s %7s %8s %8s %9s %9s %10s %8s'
          % ('geometry', 'sx', 'tooth', 'seg/L', 'length', 'v*', 'u*', 'how'))

    out = {'dur_s': dur, 'tau_ms': tau, 'tau_tot': tt,
           'u_square_pred': U_SQUARE, 'speeds': speeds.tolist(), 'rows': []}
    t0 = time.time()
    for label, sx, ns in (('shipped', 5.0, 60), ('harmonised', 20.0 / math.sqrt(2), 24)):
        g = ZigzagOpen(sx, ns)
        unit = 2 * g.tooth
        rr = [run(g, v, df, frames, unit) for v in speeds]
        v, how, npts = d84.bowl_locate(speeds, rr, factor=4.0)
        u = v * tt / LOOK if v else None
        out['rows'].append({'label': label, 'sx': sx, 'ns': ns,
                            'tooth_m': g.tooth, 'seg_over_L': g.tooth / LOOK,
                            'length_m': g.circ, 'v_star': v, 'u_star': u,
                            'how': how, 'n_pts': npts,
                            'rmse': [None if not np.isfinite(x) else float(x)
                                     for x in rr]})
        print('  %-12s %7.3f %8.2f %8.2f %9.1f %9.4f %10.4f %8s'
              % (label, sx, g.tooth, g.tooth / LOOK, g.circ,
                 v if v else float('nan'), u if u else float('nan'), how))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    h = [r for r in out['rows'] if r['label'] == 'harmonised'][0]
    s = [r for r in out['rows'] if r['label'] == 'shipped'][0]
    print()
    if h['u_star']:
        print('  harmonised u* = %.4f against the alpha-law prediction %.4f  (%+.1f%%)'
              % (h['u_star'], U_SQUARE, 100 * (h['u_star'] / U_SQUARE - 1)))
        if abs(h['u_star'] / U_SQUARE - 1) < 0.05:
            print('  -> the alternation of turn direction does not move the optimum;')
            print('     with seg/L, angle and density matched, the zigzag IS the')
            print('     square as far as the corner law is concerned.')
        else:
            print('  -> the alternation DOES move the optimum.  That is a result in')
            print('     its own right and it needs its own explanation.')
    if s['u_star'] and h['u_star']:
        print('  shipped u* = %.4f  ->  the size fix moves the answer by %+.1f%%'
              % (s['u_star'], 100 * (h['u_star'] / s['u_star'] - 1)))

    json.dump(out, open(HERE / 'd87_zigzag_harmonised_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d87_zigzag_harmonised_2026-09-08.json')


if __name__ == '__main__':
    main()
