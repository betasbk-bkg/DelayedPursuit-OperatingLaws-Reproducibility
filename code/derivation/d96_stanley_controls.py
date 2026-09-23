#!/usr/bin/env python3
"""
d96 -- the two controls that decide what d94 actually found.

d94 found the Stanley law has no interior optimum: RMSE rises monotonically from
the lowest speed the lap gate admits.  Read literally that says the optimum is a
property of look-ahead pursuit rather than of delayed tracking in general -- a
large claim, and one that two conflated explanations could both produce:

    (i)  Stanley has no LOOK-AHEAD, so there is no sagitta to cancel
    (ii) Stanley's gain is k e / v, which schedules with speed and may remove the
         trade-off on its own

and a third possibility that has nothing to do with Stanley:

    (iii) the harness cannot find an optimum in this configuration at all

CONTROL 1 settles (iii).  Same loop, same circle, same gate, same estimator, but
the commanded direction is the look-ahead direction -- the heading-servo law.  It
must produce u* ~ 0.51; if it does not, d94 measured the harness.

CONTROL 2 separates (i) from (ii).  Stanley with the speed taken OUT of the gain,
delta = psi_e + atan(k e).  If an optimum appears, the 1/v schedule was doing the
work and the look-ahead story is wrong.  If none appears, the absence of a
look-ahead is what removes it.

Usage:  python3 d96_stanley_controls.py
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


def _l(f, n):
    s = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d84 = _l('d84_stadium_sweep_2026-09-08.py', 'd84')
d94 = _l('d94_stanley_third_class_2026-09-08.py', 'd94')
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH


def simulate(traj, V, k, delay_f, frames, law):
    """law: 'stanley' | 'stanley_fixed_gain' | 'lookahead'."""
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([0.0, 1.0])
    err = np.empty(frames)
    dist = 0.0
    lap1 = None
    pts, arcs, circ = traj.pts, traj.arcs, traj.circ
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            if law == 'lookahead':
                d = pts - dp
                i = int(np.argmin(np.einsum('ij,ij->i', d, d)))
                a = (arcs[i] + sm.LOOK) % circ
                j = min(int(np.searchsorted(arcs, a)), len(pts) - 1)
                v2 = pts[j] - dp
                n = float(np.linalg.norm(v2))
                if n > 1e-10:
                    cur = v2 / n
            else:
                _, e, t = traj.nearest(dp)
                arg = (k * e / max(V, 1e-6)) if law == 'stanley' else (k * e)
                phi = -math.atan2(arg, 1.0)
                c, s = math.cos(phi), math.sin(phi)
                cur = np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]])
        vel += sm.SMOOTH * (cur * V - vel)
        step = vel * sm.DT
        dist += float(np.linalg.norm(step))
        pos = pos + step
        hist.append(pos.copy())
        _, e, _ = traj.nearest(pos)
        err[f] = abs(e)
        if lap1 is None and dist >= circ:
            lap1 = f
    if dist / circ < 1.5 or lap1 is None:
        return float('nan')
    return float(np.sqrt(np.mean(err[lap1:] ** 2)))


def main():
    dur = 240.0
    frames = int(dur / sm.DT)
    traj = d94.Circle(10.0)
    speeds = np.arange(0.30, 5.001, 0.10)
    tms = 433
    tt = tms / 1000.0 + TAU_EXTRA
    df = int(round(tms / 1000.0 / sm.DT))

    print('=' * 78)
    print('d96 -- controls for d94')
    print('=' * 78)
    print('  circle R = 10, DUR %.0f s, tau %d ms -> tau_tot %.4f' % (dur, tms, tt))
    print('  heading-servo reference: u* = 0.51 (d85: 0.5164 on this geometry)')
    print()
    print('  %-22s %6s %10s %10s %10s %8s'
          % ('law', 'k', 'v*', 'how', 'u*', 'n pts'))

    out = {'tau_tot': tt, 'speeds': speeds.tolist(), 'rows': []}
    cases = [('lookahead', None), ('stanley', 1.0), ('stanley_fixed_gain', 1.0),
             ('stanley_fixed_gain', 0.3), ('stanley_fixed_gain', 3.0)]
    for law, k in cases:
        rr = [simulate(traj, v, k if k else 0.0, df, frames, law) for v in speeds]
        v, how, n = d84.bowl_locate(speeds, rr, factor=4.0)
        u = v * tt / sm.LOOK if v else None
        out['rows'].append({'law': law, 'k': k, 'v_star': v, 'how': how,
                            'u_star': u, 'n_pts': n,
                            'rmse': [None if not np.isfinite(x) else float(x)
                                     for x in rr]})
        print('  %-22s %6s %10s %10s %10s %8d'
              % (law, ('%.1f' % k) if k else '-',
                 ('%.4f' % v) if v else '-', how,
                 ('%.4f' % u) if u else '-', n))

    g = {(r['law'], r['k']): r for r in out['rows']}
    la = g[('lookahead', None)]
    print()
    if la['u_star'] and abs(la['u_star'] / 0.5164 - 1) < 0.08:
        print('  CONTROL 1 passes: the same harness finds u* = %.4f for the'
              % la['u_star'])
        print('  look-ahead law, against 0.5164 measured independently.  So d94')
        print('  measured the control law, not the harness.')
    else:
        print('  CONTROL 1 FAILS: the harness does not reproduce the look-ahead')
        print('  optimum either, so d94 says nothing about Stanley.')

    fixed = [r for r in out['rows'] if r['law'] == 'stanley_fixed_gain'
             and r['how'] == 'bowl']
    print()
    if fixed:
        print('  CONTROL 2: removing the 1/v gain schedule DOES produce an')
        print('  optimum (%s).  So it was the schedule, not the missing'
              % ', '.join('k=%.1f -> u*=%.3f' % (r['k'], r['u_star']) for r in fixed))
        print('  look-ahead, that removed it.  d94\'s reading must be corrected.')
    else:
        print('  CONTROL 2: with the 1/v schedule removed there is still no')
        print('  optimum, so the absence of a LOOK-AHEAD is what removes it --')
        print('  d94\'s reading stands.')

    json.dump(out, open(HERE / 'd96_stanley_controls_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d96_stanley_controls_2026-09-08.json')


if __name__ == '__main__':
    main()
