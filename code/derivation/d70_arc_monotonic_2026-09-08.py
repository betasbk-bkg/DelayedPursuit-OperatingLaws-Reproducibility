#!/usr/bin/env python3
"""
d70 -- the arc-monotonic path-following scheme, and the invariance check that has
       to accompany it.

Decision taken: the merged paper fixes the traversal artefacts by making the paths
periodic and the arc progression monotone, keeping the fixed 65 s horizon.  That
is methodologically the cleanest of the options -- no evaluation window chosen
from the data, and all geometries treated alike.

It is also, word for word, the fix the reference engine already named.  Limitation 6:

    "A modified path-following scheme with arc-monotonic constraints would be
     required, and an analysis of how self-intersection geometry interacts with
     the scaling structure is left for follow-up work."

So the merged paper is not applying a second, different fix to a shared bug.  It
is delivering the follow-up the first paper specified.  What a reviewer holding
both archives will still ask is the obvious one:

    does the new scheme change the numbers the first paper reported?

That question has to be answered with a measurement, not an assurance, and this
script is that measurement.

THE SCHEME.  Only the CONTROL projection becomes monotone -- the one Limitation 6
names, "the closest-point projection used to map agent position to target arc
length".  The ERROR projection stays global, because the tracking error is by
definition the distance to the nearest point of the path; making it monotone
would change what is being measured rather than how it is followed.

  control:  arc_{k+1} = argmin over arcs in [arc_k - back, arc_k + fwd]
  error:    unchanged, global

On a convex closed path the window contains the global minimum anyway, so the
scheme is expected to be a no-op there.  That expectation is the invariance
check, and it is stated here before running:

  * Circle, Square, Ellipse, Suzuka  ->  v* unchanged (these have no
    self-crossing and no endpoint, so global and windowed projection agree)
  * Lemniscate  ->  both lobes traversed instead of one, so it can re-enter the
    analysis that Paper 1 had to exclude it from
  * Zigzag      ->  closed (out-and-back) so the fixed horizon is legitimate,
    with no look-ahead wrap and no shuttle

Usage:  python3 d70_arc_monotonic_2026-09-08.py --part coverage
        python3 d70_arc_monotonic_2026-09-08.py --part invariance --mc 5
"""
import argparse
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


# --------------------------------------------------------------------------
class ArcMonotone:
    """Windowed closest-point projection with a carried arc-length state.

    `ds` is the resampling pitch; `fwd`/`back` are the window in metres.  The
    window has to be wide enough that the projection can keep up with the agent
    -- at v m/s and a vote every WIN seconds the arc advances at most v*WIN per
    control update, so fwd is set from the speed rather than fixed.
    """

    def __init__(self, base, ds=0.02, fwd=3.0, back=0.5):
        self.base = base
        self.name = getattr(base, 'name', base.__class__.__name__)
        self.circ = float(base.circ)
        n = max(16, int(round(self.circ / ds)))
        self.s = np.linspace(0.0, self.circ, n, endpoint=False)
        self.pts = np.array([np.asarray(base.at(float(a)), float) for a in self.s])
        self.ds = self.circ / n
        self.fwd, self.back = fwd, back
        self.arc = 0.0
        self.laps = 0.0

    def reset(self, arc0=0.0):
        self.arc = float(arc0)
        self.laps = 0.0

    def at(self, arc):
        return self.base.at(arc)

    def start(self):
        return self.base.start()

    def closest_global(self, p):
        d = np.linalg.norm(self.pts - np.asarray(p, float), axis=1)
        i = int(np.argmin(d))
        return self.pts[i].copy(), float(self.s[i])

    def closest_ctrl(self, p):
        """Monotone: search only a window ahead of (and just behind) the carried
        arc, and carry the winner forward.  Wraps through 0 like any periodic
        parameter, which is what makes 'periodic' and 'monotone' the same fix."""
        i0 = int(round((self.arc - self.back) / self.ds))
        i1 = int(round((self.arc + self.fwd) / self.ds))
        idx = np.arange(i0, i1 + 1) % len(self.s)
        d = np.linalg.norm(self.pts[idx] - np.asarray(p, float), axis=1)
        j = int(np.argmin(d))
        adv = ((i0 + j) - (self.arc / self.ds)) * self.ds
        self.arc = (self.arc + adv) % self.circ
        self.laps += max(adv, 0.0) / self.circ
        return self.pts[idx[j]].copy(), float(self.arc)


class ZigzagClosed:
    """The zigzag, closed by returning along itself: an out-and-back circuit.

    The shipped Zigzag is the engine's only OPEN path, and every consequence
    follows from that -- the look-ahead wraps to the start (d69), and a fixed
    65 s horizon outlives the geometry.  Closing it removes both without
    introducing a data-dependent evaluation window.
    """

    def __init__(self, amp=5, ns=10, sx=5):
        self.name = 'zigzag_closed'
        pts = [np.array([0., 0.])]
        for i in range(ns):
            pts.append(np.array([(i + 1) * sx, amp if i % 2 == 0 else 0.]))
        out = np.array(pts)
        # return leg: the same vertices in reverse, offset so the two legs do not
        # coincide (a coincident return would recreate a projection ambiguity of
        # exactly the kind that broke the lemniscate)
        back = out[::-1].copy()
        back[:, 1] = back[:, 1] - (amp + 2.0)
        self.c = np.vstack([out, back, out[:1]])
        self.segs = [(self.c[i], self.c[i + 1]) for i in range(len(self.c) - 1)]
        self.lens = [float(np.linalg.norm(b - a)) for a, b in self.segs]
        self.circ = float(sum(self.lens))
        self.cum = np.array([0.0] + list(np.cumsum(self.lens)))

    def closest(self, p):
        bd, bp, ba = 1e18, self.c[0], 0.0
        for i, (a, b) in enumerate(self.segs):
            v = b - a
            l2 = float(v @ v)
            if l2 < 1e-12:
                continue
            t = float(np.clip((p - a) @ v / l2, 0, 1))
            pt = a + t * v
            d = float(np.linalg.norm(p - pt))
            if d < bd:
                bd, bp, ba = d, pt, self.cum[i] + t * self.lens[i]
        return bp, ba

    def at(self, arc):
        arc = arc % self.circ
        for i, (a, b) in enumerate(self.segs):
            if arc <= self.cum[i + 1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


# --------------------------------------------------------------------------
def run(traj, V, tau_ms, dur=sm.DUR, scheme='global', seed=181, N=150,
        troll=0.05, quantised=True, cover_bins=200):
    """One run.  scheme='global' reproduces the shipped behaviour exactly;
    scheme='monotone' applies the arc-monotonic control projection."""
    mono = traj if isinstance(traj, ArcMonotone) else ArcMonotone(traj)
    mono.reset(0.0)
    rng = np.random.default_rng(seed)
    frames = int(dur / sm.DT)
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    # the window must outrun the agent: v*WIN of arc per control update
    mono.fwd = max(3.0, 4.0 * V * sm.WIN)
    pos = mono.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    err = np.empty(frames)
    seen = np.zeros(cover_bins, bool)
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            if scheme == 'monotone':
                _, arc = mono.closest_ctrl(dp)
            else:
                _, arc = traj.closest(dp)
            ideal = np.asarray(traj.at(arc + sm.LOOK), float) - dp
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
        # the error projection stays GLOBAL: tracking error is distance to the
        # path, and that definition must not change with the following scheme
        cp, arc_e = mono.closest_global(pos)
        err[f] = float(np.linalg.norm(pos - cp))
        seen[min(cover_bins - 1, int(arc_e / mono.circ * cover_bins))] = True
    return {'rmse': float(np.sqrt(np.mean(err ** 2))),
            'coverage': float(seen.mean()),
            'laps_ctrl': float(mono.laps)}


def vstar(traj, tau_ms, speeds, mc, scheme, dur=sm.DUR):
    rs = []
    for V in speeds:
        e = [run(traj, V, tau_ms, dur=dur, scheme=scheme, seed=k * 31 + 150)['rmse']
             for k in range(mc)]
        rs.append(float(np.mean(e)))
    rs = np.array(rs)
    i = int(np.argmin(rs))
    if i in (0, len(rs) - 1):
        return float(speeds[i]), 'EDGE', rs.tolist()
    c = np.polyfit(speeds[i - 1:i + 2], rs[i - 1:i + 2], 2)
    if c[0] > 0:
        v = -c[1] / (2 * c[0])
        if speeds[i - 1] <= v <= speeds[i + 1]:
            return float(v), 'parabola', rs.tolist()
    return float(speeds[i]), 'grid', rs.tolist()


# --------------------------------------------------------------------------
def part_coverage(a):
    print('=' * 78)
    print('d70 -- does the arc-monotonic scheme actually traverse the whole path?')
    print('=' * 78)
    print('  coverage = fraction of the arclength ever visited, in %d bins' % 200)
    print()
    print('  %-16s %8s %10s %10s %10s'
          % ('trajectory', 'scheme', 'coverage', 'laps(ctrl)', 'RMSE'))
    out = {}
    cases = [('lemniscate', sm.Lemniscate()),
             ('zigzag (open)', sm.Zigzag()),
             ('zigzag_closed', ZigzagClosed()),
             ('circle', sm.Circle()),
             ('square', sm.Square())]
    for name, tj in cases:
        m = ArcMonotone(tj)
        for scheme in ('global', 'monotone'):
            r = run(m if scheme == 'monotone' else tj, a.vel, a.tau, scheme=scheme)
            if scheme == 'monotone':
                r = run(m, a.vel, a.tau, scheme='monotone')
            out['%s/%s' % (name, scheme)] = r
            print('  %-16s %8s %9.1f%% %10.2f %10.4f'
                  % (name if scheme == 'global' else '', scheme,
                     100 * r['coverage'], r['laps_ctrl'], r['rmse']))
    json.dump(out, open(HERE / 'd70_coverage_2026-09-08.json', 'w'), indent=2)
    print('\n-> d70_coverage_2026-09-08.json')


def part_invariance(a):
    print('=' * 78)
    print('d70 -- does the arc-monotonic scheme move what Paper 1 reported?')
    print('=' * 78)
    print('  Paper 1 reported Circle and Square (and Ellipse, Suzuka).  Neither')
    print('  self-crosses and neither has an endpoint, so the windowed projection')
    print('  should contain the global one and the scheme should be a NO-OP.')
    print('  Stated before running: v* unchanged within the MC scatter.')
    print()
    speeds = np.arange(0.25, 5.01, 0.25)
    taus = [100, 200, 300, 433, 600, 1000]
    out = {'speeds': speeds.tolist(), 'taus': taus, 'mc': a.mc, 'rows': []}
    t0 = time.time()
    for name, tj in (('circle', sm.Circle()), ('square', sm.Square())):
        m = ArcMonotone(tj)
        print('  %s' % name)
        print('    %6s %10s %10s %9s %10s'
              % ('tau', 'v* global', 'v* mono', 'diff', 'x* mono'))
        for tms in taus:
            vg, hg, rg = vstar(tj, tms, speeds, a.mc, 'global')
            vm, hm, rm = vstar(m, tms, speeds, a.mc, 'monotone')
            xm = vm * math.sqrt(tms / 1000.0)
            out['rows'].append({'traj': name, 'tau_ms': tms,
                                'v_global': vg, 'how_global': hg,
                                'v_mono': vm, 'how_mono': hm,
                                'x_star_mono': xm,
                                'rmse_global': rg, 'rmse_mono': rm})
            print('    %6d %10.4f %10.4f %8.1f%% %10.4f'
                  % (tms, vg, vm, 100 * (vm / vg - 1), xm))
        print('    -> %.1f min' % ((time.time() - t0) / 60))

    for name in ('circle', 'square'):
        rs = [r for r in out['rows'] if r['traj'] == name]
        d = np.array([r['v_mono'] / r['v_global'] - 1 for r in rs])
        lt = np.log([r['tau_ms'] / 1000.0 for r in rs])
        bg = -float(np.polyfit(lt, np.log([r['v_global'] for r in rs]), 1)[0])
        bm = -float(np.polyfit(lt, np.log([r['v_mono'] for r in rs]), 1)[0])
        print('  %s: mean |dv*| %.2f%%, max %.2f%%;  beta %.4f -> %.4f (%+.4f)'
              % (name, 100 * np.mean(np.abs(d)), 100 * np.max(np.abs(d)),
                 bg, bm, bm - bg))
        out['%s_beta_global' % name] = bg
        out['%s_beta_mono' % name] = bm

    json.dump(out, open(HERE / 'd70_invariance_2026-09-08.json', 'w'), indent=2)
    print('\n-> d70_invariance_2026-09-08.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='coverage',
                    choices=['coverage', 'invariance'])
    ap.add_argument('--vel', type=float, default=2.0)
    ap.add_argument('--tau', type=int, default=433)
    ap.add_argument('--mc', type=int, default=5)
    a = ap.parse_args()
    (part_coverage if a.part == 'coverage' else part_invariance)(a)


if __name__ == '__main__':
    main()
