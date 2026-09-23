#!/usr/bin/env python3
"""
d75 -- is an O(1) neighbourhood projection EXACT on the sawtooth, at the grid
       extremes rather than at a convenient sample of cells?

Context.  The proposed fix for the zigzag is to extend the path so the agent
never reaches its end inside the horizon, in two implementations:

    F  the shipped class with more teeth, Zigzag(ns=60) -- full scan, O(n_seg)
    A  a periodic subclass that extends the sawtooth analytically and projects
       against only the neighbouring segments -- O(1)

and the reported evidence is that A and F agree to 0.00e+00 over nine cells.

WHAT THAT EVIDENCE IS, AND IS NOT.  Bit-identical output does not mean two
independent methods converged on the same answer; it means the same arithmetic
was performed.  A's window returned the same nearest segment as F's full scan in
every frame of those nine cells, so downstream everything -- votes, RNG draws,
positions -- stayed on one trajectory.  That is a real and valuable check of A's
windowing logic.  It is NOT evidence about the physics, and it carries no
information about cells that were not run.

WHERE IT COULD BREAK.  A's window is exact only while the globally nearest
segment stays adjacent to the currently held one.  The teeth are sx = 5 m apart
with segment length sqrt(sx^2 + amp^2) = 7.07 m, and d69 measured zigzag tracking
RMSE up to 1.94 m at v = 5 -- with excursions well above the mean.  An agent that
cuts a corner badly can be closer to a segment two or three teeth away than to
its neighbours, and there A silently disagrees with F.  That regime is at the
grid EXTREMES (high speed, long latency, high troll ratio), which is exactly
where a nine-cell sample is least likely to have looked.

THE TEST.  Run the shipped geometry with the full scan and record, per frame,
the globally nearest segment index.  If that index never moves by more than one
between consecutive control updates, an O(1) three-segment window is exact and A
is safe.  If it ever jumps further, the cross-check must include those cells or A
needs a guard.

Stated before running: jumps of 2+ should appear at the top of the speed grid.

Usage:  python3 d75_window_exactness_2026-09-08.py
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


def build(ns=60, amp=5.0, sx=5.0):
    pts = [np.array([0.0, 0.0])]
    for i in range(ns):
        pts.append(np.array([(i + 1) * sx, amp if i % 2 == 0 else 0.0]))
    c = np.array(pts)
    a = c[:-1]
    b = c[1:]
    v = b - a
    l2 = np.einsum('ij,ij->i', v, v)
    lens = np.sqrt(l2)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    return {'c': c, 'a': a, 'v': v, 'l2': l2, 'lens': lens, 'cum': cum,
            'circ': float(cum[-1]), 'ns': ns}


def project_all(g, p):
    """Global nearest over every segment, vectorised.  Returns (idx, arc, dist)."""
    w = p - g['a']
    t = np.clip(np.einsum('ij,ij->i', w, g['v']) / g['l2'], 0.0, 1.0)
    q = g['a'] + t[:, None] * g['v']
    d = np.linalg.norm(q - p, axis=1)
    i = int(np.argmin(d))
    return i, float(g['cum'][i] + t[i] * g['lens'][i]), float(d[i])


def at(g, arc):
    arc = min(max(arc, 0.0), g['circ'])
    i = int(np.searchsorted(g['cum'], arc, 'right')) - 1
    i = min(max(i, 0), g['ns'] - 1)
    t = (arc - g['cum'][i]) / g['lens'][i]
    return g['a'][i] + np.clip(t, 0, 1) * g['v'][i]


def run(g, V, tau_ms, troll, seed=181, N=150):
    rng = np.random.default_rng(seed)
    frames = sm.FRAMES
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    pos = g['c'][0].copy()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    gam = 0.5
    err = np.empty(frames)
    ctrl_idx = []          # nearest-segment index at each CONTROL update
    err_idx = []           # and at every frame, for the error projection
    max_arc = 0.0
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            i, arc, _ = project_all(g, dp)
            ctrl_idx.append(i)
            max_arc = max(max_arc, arc)
            ideal = at(g, arc + sm.LOOK) - dp
            n = float(np.linalg.norm(ideal))
            if n > 1e-10:
                ideal = ideal / n
            ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
            votes = sm.gen_votes(ia, prev, troll, N, rng)
            prev = ia
            b = sm.DIRS[votes].mean(axis=0)
            gam = float(np.linalg.norm(b))
            if gam > 1e-10:
                cur = b / gam
            tgt = cur * math.sqrt(gam) * V
        else:
            tgt = cur * V
        vel += sm.SMOOTH * (tgt - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
        j, _, d = project_all(g, pos)
        err_idx.append(j)
        err[f] = d
    ci = np.array(ctrl_idx)
    ei = np.array(err_idx)
    jc = np.abs(np.diff(ci)) if ci.size > 1 else np.array([0])
    je = np.abs(np.diff(ei)) if ei.size > 1 else np.array([0])
    return {'rmse': float(np.sqrt(np.mean(err ** 2))),
            'max_arc': max_arc, 'reach_frac': max_arc / g['circ'],
            'max_jump_ctrl': int(jc.max()), 'max_jump_err': int(je.max()),
            'n_jump_gt1_ctrl': int(np.sum(jc > 1)),
            'n_jump_gt1_err': int(np.sum(je > 1)),
            'max_err': float(err.max()), 'idx_max': int(ci.max())}


def main():
    g = build(ns=60)
    print('=' * 78)
    print('d75 -- is an O(1) neighbourhood projection exact on the sawtooth?')
    print('=' * 78)
    print('  Zigzag(ns=60): %d segments, %.2f m each, arclength %.1f m'
          % (g['ns'], g['lens'][0], g['circ']))
    print('  hard bound on distance travelled: v_max * DUR = %.1f m'
          % (5.0 * sm.DUR))
    print('  -> ns >= %d is needed for the agent never to reach the end;'
          % math.ceil(5.0 * sm.DUR / g['lens'][0]))
    print('     ns = 60 gives %.1f m, a factor %.2f of margin'
          % (g['circ'], g['circ'] / (5.0 * sm.DUR)))
    print()
    print('  a three-segment window is exact iff max_jump == 1 everywhere')
    print()
    print('  %5s %6s %6s | %9s %9s | %9s %9s %9s'
          % ('v', 'tau', 'troll', 'jump ctrl', 'jump err', 'n>1 ctrl',
             'n>1 err', 'reach'))
    out = {'ns': g['ns'], 'arclen': g['circ'], 'rows': []}
    worst = 0
    for V in (1.0, 3.0, 5.0):
        for tms in (100, 433, 1000):
            for tr in (0.05, 0.40):
                r = run(g, V, tms, tr)
                r.update(v=V, tau_ms=tms, troll=tr)
                out['rows'].append(r)
                worst = max(worst, r['max_jump_ctrl'], r['max_jump_err'])
                flag = ''
                if max(r['max_jump_ctrl'], r['max_jump_err']) > 1:
                    flag = '  <-- WINDOW OF 3 IS NOT EXACT HERE'
                print('  %5.1f %6d %6.2f | %9d %9d | %9d %9d %8.1f%%%s'
                      % (V, tms, tr, r['max_jump_ctrl'], r['max_jump_err'],
                         r['n_jump_gt1_ctrl'], r['n_jump_gt1_err'],
                         100 * r['reach_frac'], flag))
    print()
    reach = max(r['reach_frac'] for r in out['rows'])
    print('  furthest the agent ever got: %.1f%% of the path -- the end is never'
          % (100 * reach))
    print('  seen, so the wrap cannot fire.  That part of the plan holds.')
    print()
    if worst <= 1:
        print('  max index jump anywhere: %d  ->  a three-segment window is EXACT'
              % worst)
        print('  on this grid, and A is safe as specified.')
    else:
        need = worst + 1
        print('  max index jump anywhere: %d  ->  a three-segment window is NOT'
              % worst)
        print('  exact.  A needs a half-width of at least %d segments, or a guard:'
              % need)
        print('    compute the windowed minimum, and if its distance exceeds half')
        print('    the tooth pitch (%.2f m), fall back to a full scan.  That is'
              % (g['lens'][0] / 2))
        print('    O(1) in the common case and exact always.')
    out['worst_jump'] = worst
    json.dump(out, open(HERE / 'd75_window_exactness_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d75_window_exactness_2026-09-08.json')


if __name__ == '__main__':
    main()
