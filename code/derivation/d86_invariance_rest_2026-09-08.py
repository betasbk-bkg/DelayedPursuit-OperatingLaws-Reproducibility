#!/usr/bin/env python3
"""
d86 -- close the two invariance items the path-following note (not distributed) left open.

Section 5.4 reports the arc-monotonic scheme as a no-op on Circle and Square,
with 12 latency cells inside +-1.2%, and then says plainly:

    "Not yet run: the ellipse and the circuit centreline. Both use a global argmin, so the same no-op is expected — but an expectation is not a measurement."

Section 9 leaves a second: the lemniscate re-enters the analysis only if its v*
curve is actually measured under the monotone projection, and d72 measured lobe
occupancy, not v*.

Both are settled here.

  (1) ELLIPSE and SUZUKA under global vs monotone control projection.  Both use a
      global argmin over sampled points in the shipped code, so the windowed
      projection should contain it and change nothing -- the same argument that
      held for the Circle.  Suzuka is run at its published normalisation, where
      d82 showed R_min/L = 0.09; the point is not to validate Suzuka but to check
      that the projection scheme is a no-op even there.

  (2) LEMNISCATE v* under both projections.  d72 showed the monotone scheme
      clears lobe capture at v <= 1 and changes tracking by up to 20% elsewhere,
      so unlike the others it is NOT expected to be a no-op.  What matters is
      whether the located optimum moves, because that is what the paper reports.

Deterministic engine throughout: d66 established that the crowd layer cannot
resolve u* at feasible MC, and the question here is about a projection scheme,
which the vote layer only obscures.

Usage:  python3 d86_invariance_rest_2026-09-08.py
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


def _load(stem, name):
    s = importlib.util.spec_from_file_location(name, HERE / stem)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d70 = _load('d70_arc_monotonic_2026-09-08.py', 'd70')
d84 = _load('d84_stadium_sweep_2026-09-08.py', 'd84')

LOOK = sm.LOOK
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH


class Sampled:
    """A closed curve given by points, with the shipped global-argmin projection."""

    def __init__(self, name, pts):
        self.name = name
        p = np.asarray(pts, float)
        if not np.allclose(p[0], p[-1]):
            p = np.vstack([p, p[:1]])
        self.pts = p
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        self.arcs = np.concatenate([[0.0], np.cumsum(d)])
        self.circ = float(self.arcs[-1])

    def closest(self, q):
        d = np.linalg.norm(self.pts - q, axis=1)
        i = int(np.argmin(d))
        return self.pts[i].copy(), float(self.arcs[i])

    def at(self, arc):
        a = arc % self.circ
        i = min(int(np.searchsorted(self.arcs, a)), len(self.pts) - 1)
        return self.pts[i].copy()

    def start(self):
        return self.pts[0].copy()


def make(name):
    t = np.linspace(0, 2 * np.pi, 8000, endpoint=False)
    if name == 'ellipse':
        return Sampled(name, np.stack([12 * np.cos(t), 6 * np.sin(t)], 1))
    if name == 'lemniscate':
        s, c = np.sin(t), np.cos(t)
        d = 1 + s * s
        return Sampled(name, np.stack([7 * c / d, 7 * s * c / d], 1))
    if name == 'suzuka':
        csv = np.genfromtxt(REPO / 'suzuka_track.csv', delimiter=',', skip_header=1)
        p = np.vstack([csv, csv[:1]])
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        return Sampled(name, p * (2 * math.pi * 10.0) / float(d.sum()))
    raise ValueError(name)


def run(traj, V, delay_f, frames, scheme, mono):
    mono.reset(0.0)
    mono.fwd = max(3.0, 4.0 * V * sm.WIN)
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
            if scheme == 'monotone':
                _, arc = mono.closest_ctrl(dp)
            else:
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
    if dist / traj.circ < 1.5 or lap1 is None:
        return float('nan')
    return float(np.sqrt(np.mean(err[lap1:] ** 2)))


def main():
    dur, taus = 240.0, (100, 200, 300, 433, 600, 1000)
    frames = int(dur / sm.DT)
    speeds = np.arange(0.20, 4.001, 0.05)

    print('=' * 78)
    print('d86 -- arc-monotonic invariance on the geometries 5.4 left open')
    print('=' * 78)
    print('  deterministic engine, DUR %.0f s, speed grid %.2f..%.2f step 0.05'
          % (dur, speeds[0], speeds[-1]))
    print()

    out = {'dur_s': dur, 'taus': list(taus), 'speeds': speeds.tolist(), 'rows': []}
    t0 = time.time()
    for gname in ('ellipse', 'suzuka', 'lemniscate'):
        g = make(gname)
        mono = d70.ArcMonotone(g)
        print('  %s  (perimeter %.1f m)' % (gname.upper(), g.circ))
        print('    %6s %11s %11s %9s %11s'
              % ('tau', 'v* global', 'v* mono', 'diff', 'u* mono'))
        for tms in taus:
            df = int(round(tms / 1000.0 / sm.DT))
            tt = tms / 1000.0 + TAU_EXTRA
            rg = [run(g, v, df, frames, 'global', mono) for v in speeds]
            rm = [run(g, v, df, frames, 'monotone', mono) for v in speeds]
            vg, hg, ng = d84.bowl_locate(speeds, rg, factor=4.0)
            vm, hm, nm = d84.bowl_locate(speeds, rm, factor=4.0)
            row = {'geometry': gname, 'tau_ms': tms, 'tau_tot': tt,
                   'v_global': vg, 'how_global': hg,
                   'v_mono': vm, 'how_mono': hm,
                   'u_mono': (vm * tt / LOOK) if vm else None}
            out['rows'].append(row)
            if vg and vm:
                print('    %6d %11.4f %11.4f %8.2f%% %11.4f'
                      % (tms, vg, vm, 100 * (vm / vg - 1), vm * tt / LOOK))
            else:
                print('    %6d %11s %11s %9s %11s'
                      % (tms, vg or '-', vm or '-', '-', '-'))
        rs = [r for r in out['rows']
              if r['geometry'] == gname and r['v_global'] and r['v_mono']]
        if rs:
            d = np.array([r['v_mono'] / r['v_global'] - 1 for r in rs])
            us = np.array([r['u_mono'] for r in rs])
            print('    mean |dv*| %.2f%%, max %.2f%%;  u* mono %.4f +- %.4f'
                  % (100 * np.mean(np.abs(d)), 100 * np.max(np.abs(d)),
                     us.mean(), us.std(ddof=1) if len(us) > 1 else 0.0))
        print('    -> %.1f min' % ((time.time() - t0) / 60))
        print()

    json.dump(out, open(HERE / 'd86_invariance_rest_2026-09-08.json', 'w'), indent=2)
    print('-> d86_invariance_rest_2026-09-08.json')


if __name__ == '__main__':
    main()
