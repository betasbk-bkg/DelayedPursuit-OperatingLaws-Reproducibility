#!/usr/bin/env python3
"""
d94 -- a THIRD plant class, and the sharpest structural test available.

The two classes tested so far both aim at a look-ahead point:

    heading-servo     command the direction to the look-ahead point
    curvature-command kappa = 2y/d^2 from the look-ahead point

Both therefore carry a length L, and the whole theory is built on the
dimensionless group u = v tau_tot / L.  Stanley's law does not:

    delta = psi_error + atan(k e / v)

It uses the path TANGENT at the nearest point plus a cross-track correction, and
there is no look-ahead anywhere in it.  So u is not even defined for this
controller, and the question is not "is u* the same" but "does Theorem 1's
STRUCTURE survive a control law that has no L".

THREE OUTCOMES, all informative:

  (1) no interior optimum -- RMSE falls monotonically with v.  Then the optimum
      is a property of look-ahead pursuit, not of delayed tracking in general,
      and the paper must say so.
  (2) an interior optimum whose location follows k tau_tot -- the natural group
      when the only length is v/k.  Then Theorem 1 generalises with the length
      supplied by the control law rather than by the look-ahead.
  (3) an interior optimum at fixed v tau_tot / L for some emergent L -- would be
      surprising and would need explaining.

The sweep is over BOTH k and tau, because a single (k, tau) cell cannot
distinguish (2) from (3).

Deterministic engine, circle and square, same protocol as d65/d84.

Usage:  python3 d94_stanley_third_class_2026-09-08.py
"""
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2] / "engine" / "crowd_control_engine"   # the engine as it ships in this package
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402

_s = importlib.util.spec_from_file_location(
    'd84', HERE / 'd84_stadium_sweep_2026-09-08.py')
d84 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d84)

TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH


class Circle:
    def __init__(self, R=10.0, n=8000):
        self.name = 'circle'
        self.R = R
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = np.stack([R * np.cos(t), R * np.sin(t)], 1)
        self.pts = np.vstack([c, c[:1]])
        d = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.arcs = np.concatenate([[0.0], np.cumsum(d)])
        self.circ = float(self.arcs[-1])

    def nearest(self, p):
        """closest point, signed cross-track error, and unit tangent there."""
        d = self.pts - p
        i = int(np.argmin(np.einsum('ij,ij->i', d, d)))
        q = self.pts[i]
        t = self.pts[(i + 1) % len(self.pts)] - self.pts[i - 1]
        t = t / max(np.linalg.norm(t), 1e-12)
        n = np.array([-t[1], t[0]])          # left normal
        e = float((p - q) @ n)               # signed cross-track
        return q, e, t

    def start(self):
        return self.pts[0].copy()


def simulate(traj, V, k, delay_f, frames):
    """Stanley on the delayed pose.  No look-ahead anywhere."""
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = None
    err = np.empty(frames)
    dist = 0.0
    lap1 = None
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, e, t = traj.nearest(dp)
            # delta = psi_error + atan(k e / v); with the commanded direction
            # expressed directly, that is the tangent rotated by -atan(k e / v)
            phi = -math.atan2(k * e, max(V, 1e-6))
            c, s = math.cos(phi), math.sin(phi)
            cur = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]])
        if cur is None:
            cur = np.array([0.0, 1.0])
        vel += sm.SMOOTH * (cur * V - vel)
        step = vel * sm.DT
        dist += float(np.linalg.norm(step))
        pos = pos + step
        hist.append(pos.copy())
        _, e, _ = traj.nearest(pos)
        err[f] = abs(e)
        if lap1 is None and dist >= traj.circ:
            lap1 = f
    if dist / traj.circ < 1.5 or lap1 is None:
        return float('nan')
    return float(np.sqrt(np.mean(err[lap1:] ** 2)))


def main():
    dur = 240.0
    frames = int(dur / sm.DT)
    traj = Circle(10.0)
    speeds = np.arange(0.20, 5.001, 0.10)

    print('=' * 78)
    print('d94 -- Stanley: a control law with no look-ahead')
    print('=' * 78)
    print('  deterministic engine, circle R = 10, DUR %.0f s' % dur)
    print('  outcomes: (1) no interior optimum  (2) v* set by k*tau_tot')
    print('            (3) v* at fixed v tau_tot / L for some emergent L')
    print()
    print('  %6s %6s %9s %10s %10s %11s %11s'
          % ('k', 'tau', 'tau_tot', 'v*', 'how', 'k*tau_tot', 'v* tau_tot'))

    out = {'dur_s': dur, 'speeds': speeds.tolist(), 'rows': []}
    t0 = time.time()
    for k in (0.5, 1.0, 2.0):
        for tms in (200, 433, 800):
            tt = tms / 1000.0 + TAU_EXTRA
            df = int(round(tms / 1000.0 / sm.DT))
            rr = [simulate(traj, v, k, df, frames) for v in speeds]
            v, how, npts = d84.bowl_locate(speeds, rr, factor=4.0)
            row = {'k': k, 'tau_ms': tms, 'tau_tot': tt, 'v_star': v,
                   'how': how, 'n_pts': npts,
                   'k_tau_tot': k * tt,
                   'v_tau_tot': (v * tt) if v else None,
                   'rmse': [None if not np.isfinite(x) else float(x) for x in rr]}
            out['rows'].append(row)
            print('  %6.2f %6d %9.4f %10s %10s %11.4f %11s'
                  % (k, tms, tt, ('%.4f' % v) if v else '-', how, k * tt,
                     ('%.4f' % (v * tt)) if v else '-'))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    ok = [r for r in out['rows'] if r['v_star'] and r['how'] == 'bowl']
    print()
    if len(ok) < 4:
        print('  fewer than four located optima: outcome (1) is in play --')
        print('  the RMSE curve may have no interior minimum for this law.')
    else:
        vt = np.array([r['v_tau_tot'] for r in ok])
        kt = np.array([r['k_tau_tot'] for r in ok])
        vs = np.array([r['v_star'] for r in ok])
        print('  spread of v* tau_tot  (the look-ahead group, L absorbed): %.1f%%'
              % (100 * vt.std(ddof=1) / vt.mean()))
        print('  spread of v*          (raw):                              %.1f%%'
              % (100 * vs.std(ddof=1) / vs.mean()))
        # does v* scale as 1/tau_tot at fixed k, and how does k enter?
        for k in sorted(set(r['k'] for r in ok)):
            rs = [r for r in ok if r['k'] == k]
            if len(rs) >= 2:
                lt = np.log([r['tau_tot'] for r in rs])
                lv = np.log([r['v_star'] for r in rs])
                sl = float(np.polyfit(lt, lv, 1)[0])
                print('    k = %.2f: v* ~ tau_tot^%+.3f  (look-ahead pursuit gives -1)'
                      % (k, sl))
        for tms in sorted(set(r['tau_ms'] for r in ok)):
            rs = [r for r in ok if r['tau_ms'] == tms]
            if len(rs) >= 2:
                lk = np.log([r['k'] for r in rs])
                lv = np.log([r['v_star'] for r in rs])
                sl = float(np.polyfit(lk, lv, 1)[0])
                print('    tau = %4d: v* ~ k^%+.3f' % (tms, sl))

    json.dump(out, open(HERE / 'd94_stanley_2026-09-08.json', 'w'), indent=2)
    print('\n-> d94_stanley_2026-09-08.json')


if __name__ == '__main__':
    main()
