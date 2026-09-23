#!/usr/bin/env python3
"""
d83 -- the ellipse's margin should converge to the circle's when it is scaled up.

d82 measured the ellipse's quasi-static ratio at the optimum:

    ell(u*) / Lambda = 1.26 m / 2.21 m = 0.57

i.e. the loop's memory is 57% of the length over which the ellipse's curvature
changes.  That is not a small number, so the ellipse is NOT in the quasi-static
regime the circle's margin formula assumes.  the reference engine lists exactly this as an
open problem -- "a margin formula for arbitrary smooth geometries (ellipse
u* = 0.562)" -- without a reason for the discrepancy.  d82 supplies one.

A reason that only explains is worth little; this one PREDICTS.  Scaling an
ellipse by m multiplies Lambda by m and leaves ell alone, because ell is set by u
and L, not by the path.  So

    u*(ellipse, scale m)  ->  u*(circle) = 0.4995   as m grows,

and the published 0.562 should be recovered at m = 1.  If instead u* stays at
0.562 for every scale, the quasi-static account is wrong and the ellipse's margin
is a genuine shape effect.

Run on the DETERMINISTIC engine (vote layer off).  d66 showed the crowd leg
cannot resolve u* to better than tens of percent at MC = 5, and this test needs
1-2%.  The deterministic plant is the same heading-servo loop without that noise,
and it is the leg on which the alpha law was established (d65/d78).

Stated before running:
  * m = 1   -> ell/Lambda 0.57, u* well above 0.4995 (the published 0.562)
  * m = 6   -> ell/Lambda 0.10, u* within a few % of 0.4995
  * monotone in between.

Usage:  python3 d83_ellipse_scaling_2026-09-08.py [--dur 240]
"""
import argparse
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

LOOK = sm.LOOK
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH


class Ellipse:
    """a = 12 m, b = 6 m as published, scaled by m."""

    def __init__(self, m=1.0, a=12.0, b=6.0, n=8000):
        self.name = 'ellipse'
        self.m, self.a, self.b = m, a * m, b * m
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = np.stack([self.a * np.cos(t), self.b * np.sin(t)], axis=1)
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

    def quasi_static_ratio(self, u):
        """ell(u)/Lambda, both in metres, analytic curvature."""
        from scipy.special import lambertw
        a, b = self.a, self.b
        t = np.linspace(0, 2 * np.pi, 200000, endpoint=False)
        w = np.sqrt(a * a * np.sin(t) ** 2 + b * b * np.cos(t) ** 2)
        k = a * b / w ** 3
        dk = -3 * a * b * (a * a - b * b) * np.sin(t) * np.cos(t) / w ** 6
        lam_len = float(np.sqrt(np.mean(k ** 2)) / np.abs(dk).max())
        lam = lambertw(-u, 0) / u
        ell = LOOK / abs(lam.real)
        return ell / lam_len, lam_len


def simulate(traj, V, delay_f, frames):
    """d65's deterministic loop: delay, WIN hold, first-order lag, no votes."""
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
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
                cur = ideal / n
        vel += sm.SMOOTH * (cur * V - vel)
        step = vel * sm.DT
        dist += float(np.linalg.norm(step))
        pos = pos + step
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        err[f] = float(np.linalg.norm(pos - cp))
        if lap1 is None and dist >= traj.circ:
            lap1 = f
    laps = dist / traj.circ
    if laps < 1.5 or lap1 is None:
        return float('nan'), laps
    return float(np.sqrt(np.mean(err[lap1:] ** 2))), laps


def bowl_locate(speeds, rmse, factor=2.0, min_pts=5):
    s = np.asarray(speeds, float)
    r = np.asarray([np.nan if x is None else x for x in rmse], float)
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() < min_pts:
        return None, 'too-few'
    s, r = s[ok], r[ok]
    j = int(np.argmin(r))
    sel = r <= factor * r[j]
    lo = hi = j
    while lo > 0 and sel[lo - 1]:
        lo -= 1
    while hi < len(r) - 1 and sel[hi + 1]:
        hi += 1
    ss, rr = s[lo:hi + 1], r[lo:hi + 1]
    if len(ss) < min_pts:
        return float(s[j]), 'narrow'
    c = np.polyfit(ss, rr, 2)
    if c[0] <= 0:
        return float(s[j]), 'not-convex'
    v = -c[1] / (2 * c[0])
    return (float(v), 'bowl') if ss[0] <= v <= ss[-1] else (float(s[j]), 'outside')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dur', type=float, default=240.0)
    ap.add_argument('--tau', type=int, default=433)
    ap.add_argument('--vmax', type=float, default=6.0)
    a = ap.parse_args()
    frames = int(a.dur / sm.DT)
    speeds = np.arange(0.20, a.vmax + 1e-9, 0.10)
    df = int(round(a.tau / 1000.0 / sm.DT))
    tt = a.tau / 1000.0 + TAU_EXTRA

    print('=' * 78)
    print('d83 -- does the ellipse margin converge to the circle when scaled up?')
    print('=' * 78)
    print('  deterministic heading-servo loop, DUR %.0f s, tau %d ms -> tau_tot %.4f'
          % (a.dur, a.tau, tt))
    print('  circle reference u* = 0.4995 (d38);  published ellipse u* = 0.562')
    print()
    print('  %6s %9s %9s %10s %10s %9s %9s'
          % ('scale', 'a (m)', 'perim', 'ell/Lambda', 'v*', 'how', 'u*'))

    t0 = time.time()
    rows = []
    for m in (1.0, 1.5, 2.0, 3.0, 4.5, 6.0):
        e = Ellipse(m)
        ratio, lam_len = e.quasi_static_ratio(0.4995)
        rr = []
        for v in speeds:
            val, _ = simulate(e, v, df, frames)
            rr.append(val)
        v, how = bowl_locate(speeds, rr)
        u = v * tt / LOOK if v else None
        rows.append({'scale': m, 'a_m': e.a, 'perimeter_m': e.circ,
                     'Lambda_m': lam_len, 'ell_over_Lambda': ratio,
                     'v_star': v, 'how': how, 'u_star': u,
                     'rmse': [None if not np.isfinite(x) else float(x) for x in rr]})
        print('  %6.1f %9.1f %9.1f %10.3f %10.4f %9s %9.4f'
              % (m, e.a, e.circ, ratio, v, how, u))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    good = [r for r in rows if r['u_star']]
    print()
    if len(good) >= 3:
        print('  u* against the circle value 0.4995:')
        for r in good:
            print('    scale %4.1f  ell/Lambda %6.3f  u* %.4f  (%+.1f%%)'
                  % (r['scale'], r['ell_over_Lambda'], r['u_star'],
                     100 * (r['u_star'] / 0.4995 - 1)))
        first, last = good[0], good[-1]
        print()
        if abs(last['u_star'] / 0.4995 - 1) < abs(first['u_star'] / 0.4995 - 1) / 2:
            print('  -> u* CONVERGES toward the circle as the ellipse is scaled up.')
            print('     The published ellipse margin is a quasi-static failure, not')
            print('     a shape effect, and the reference engine\'s open problem has an answer.')
        else:
            print('  -> u* does NOT converge.  The quasi-static account is wrong and')
            print('     the ellipse margin is a genuine shape effect.  Say so.')

    json.dump({'dur_s': a.dur, 'tau_ms': a.tau, 'tau_tot': tt,
               'speeds': speeds.tolist(), 'rows': rows},
              open(HERE / 'd83_ellipse_scaling_2026-09-08.json', 'w'), indent=2)
    print('\n-> d83_ellipse_scaling_2026-09-08.json')


if __name__ == '__main__':
    main()
