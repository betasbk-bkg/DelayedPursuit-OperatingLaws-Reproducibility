#!/usr/bin/env python3
"""
d84 -- the missing geometry: finite curve radius alternating with finite straight.

A point raised in review about why Suzuka is in the paper at all: it is there because curves
and straights alternate, and the claim being supported is that the finding holds
on an all-round trajectory.  That purpose is legitimate and it exposes a real gap
-- no other geometry in the set has BOTH a finite curve radius and a finite
straight:

    circle, ellipse, lemniscate   curvature everywhere, no straight
    square, zigzag                straight everywhere, corners of radius ZERO
    Suzuka                        both -- and the only one

But Suzuka is an uncontrolled sample of that structure: neither the radius nor
the straight can be varied, so it yields one number, and a single number can
agree or disagree without ever discriminating.  Worse, d82 showed that at the
published normalisation its tightest corner has R/L = 0.09, so pure pursuit is
not even being run there.

The stadium -- two straights of length S joined by two semicircles of radius R --
has the same structure with two free knobs, and it fits the validity conditions.

PRE-REGISTERED, because both endpoints are already measured independently:

    R/L -> 0 at fixed S   the arcs become 90 deg corners, i.e. the square,
                          whose heading-servo optimum is
                          u* = 0.511 cos^0.88(45 deg) = 0.377   (d65/d78)
    S/L -> 0 at fixed R   the straights vanish, i.e. the circle,
                          u* = 0.4995                            (d38)

    so u* should rise MONOTONICALLY from ~0.377 toward ~0.4995 as R/L grows at
    fixed S/L, and fall toward 0.4995 as S/L shrinks at fixed R/L.

If it does, the two laws this project established on separate geometries are the
two ends of one continuum, and "holds on an all-round trajectory" is shown by a
swept family instead of asserted from one track.

Usage:  python3 d84_stadium_sweep_2026-09-08.py [--dur 240]
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

LOOK = sm.LOOK
TAU_EXTRA = sm.WIN / 2 + sm.DT / sm.SMOOTH
U_SQUARE = 0.511 * math.cos(math.radians(45.0)) ** 0.88     # d65/d78 alpha law
U_CIRCLE = 0.4995                                            # d38


class Stadium:
    """Two straights of length S joined by two semicircles of radius R.

    Closest point and arclength are analytic -- no resampled polyline, so no
    projection noise and no dependence on a sample spacing.

    Layout: straights along +x (y = +R) and -x (y = -R), right arc centred at
    (S/2, 0), left arc at (-S/2, 0).  Arclength starts at (-S/2, +R) going +x.
    """

    def __init__(self, R, S):
        self.name = 'stadium'
        self.R, self.S = float(R), float(S)
        self.arc = math.pi * self.R
        self.circ = 2 * self.S + 2 * self.arc

    def at(self, s):
        s = s % self.circ
        R, S = self.R, self.S
        if s <= S:                                    # top straight, +x
            return np.array([-S / 2 + s, R])
        s -= S
        if s <= self.arc:                             # right arc, top -> bottom
            th = math.pi / 2 - s / R
            return np.array([S / 2 + R * math.cos(th), R * math.sin(th)])
        s -= self.arc
        if s <= S:                                    # bottom straight, -x
            return np.array([S / 2 - s, -R])
        s -= S
        th = -math.pi / 2 - s / R                     # left arc, bottom -> top
        return np.array([-S / 2 + R * math.cos(th), R * math.sin(th)])

    def closest(self, p):
        R, S = self.R, self.S
        x, y = float(p[0]), float(p[1])
        best = (1e18, None, 0.0)
        # top straight
        xc = min(max(x, -S / 2), S / 2)
        d = math.hypot(x - xc, y - R)
        if d < best[0]:
            best = (d, np.array([xc, R]), xc + S / 2)
        # bottom straight
        d = math.hypot(x - xc, y + R)
        if d < best[0]:
            best = (d, np.array([xc, -R]), S + self.arc + (S / 2 - xc))
        # right arc
        dx, dy = x - S / 2, y
        n = math.hypot(dx, dy)
        if n > 1e-12 and dx >= 0:
            px, py = S / 2 + R * dx / n, R * dy / n
            d = math.hypot(x - px, y - py)
            if d < best[0]:
                th = math.atan2(dy, dx)
                best = (d, np.array([px, py]), S + R * (math.pi / 2 - th))
        # left arc.  at() runs it as th = -pi/2 - (s - 2S - arc)/R, so the
        # arclength offset is phi = (-pi/2 - th) mod 2pi, which lands in [0, pi]
        # across the atan2 branch cut at th = +-pi.  Getting this backwards is
        # what the self-test below caught.
        dx, dy = x + S / 2, y
        n = math.hypot(dx, dy)
        if n > 1e-12 and dx <= 0:
            px, py = -S / 2 + R * dx / n, R * dy / n
            d = math.hypot(x - px, y - py)
            if d < best[0]:
                th = math.atan2(dy, dx)
                phi = (-math.pi / 2 - th) % (2 * math.pi)
                best = (d, np.array([px, py]), 2 * S + self.arc + R * phi)
        return best[1], float(best[2] % self.circ)

    def start(self):
        return self.at(0.0)


def self_test(R=4.0, S=20.0, n=4001, tol=1e-9):
    """at() and closest() must invert each other.  They did not, the first time:
    the left arc's arclength was computed with the wrong sign and the mismatch
    reached 25 m on a 65 m circuit.  A geometry that fails this test produces a
    plausible-looking sweep of meaningless numbers, so it runs before anything
    else and aborts on failure."""
    g = Stadium(R, S)
    worst_pos = worst_arc = 0.0
    for s in np.linspace(0.0, g.circ, n, endpoint=False):
        p = g.at(float(s))
        q, a = g.closest(p)
        worst_pos = max(worst_pos, float(np.linalg.norm(p - q)))
        d = abs(a - s)
        worst_arc = max(worst_arc, min(d, g.circ - d))
    assert worst_pos < 1e-9 and worst_arc < 1e-6, (
        'stadium self-test FAILED: position %.3e, arclength %.3e'
        % (worst_pos, worst_arc))
    return worst_pos, worst_arc


def simulate(traj, V, delay_f, frames):
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


def bowl_locate(speeds, rmse, factor=4.0, min_pts=5):
    """Wider bowl than d78's: the deterministic minima here are sharp, and a
    factor of 2 collapsed to the grid point in d83."""
    s = np.asarray(speeds, float)
    r = np.asarray([np.nan if x is None else x for x in rmse], float)
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() < min_pts:
        return None, 'too-few', 0
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
        return float(s[j]), 'narrow', len(ss)
    c = np.polyfit(ss, rr, 2)
    if c[0] <= 0:
        return float(s[j]), 'not-convex', len(ss)
    v = -c[1] / (2 * c[0])
    return ((float(v), 'bowl', len(ss)) if ss[0] <= v <= ss[-1]
            else (float(s[j]), 'outside', len(ss)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dur', type=float, default=240.0)
    ap.add_argument('--tau', type=int, default=433)
    ap.add_argument('--vmax', type=float, default=4.0)
    a = ap.parse_args()
    frames = int(a.dur / sm.DT)
    speeds = np.arange(0.20, a.vmax + 1e-9, 0.10)
    df = int(round(a.tau / 1000.0 / sm.DT))
    tt = a.tau / 1000.0 + TAU_EXTRA

    print('=' * 78)
    print('d84 -- stadium: finite curve radius alternating with finite straight')
    print('=' * 78)
    print('  deterministic heading-servo loop, DUR %.0f s, tau %d ms, tau_tot %.4f'
          % (a.dur, a.tau, tt))
    print('  PRE-REGISTERED endpoints, both measured independently:')
    print('    R/L -> 0  (90 deg corner, the square)  u* = %.4f   [d65/d78]'
          % U_SQUARE)
    print('    S/L -> 0  (the circle)                 u* = %.4f   [d38]'
          % U_CIRCLE)
    print('  so u* should rise monotonically with R/L at fixed S/L.')
    print()
    wp, wa = self_test()
    print('  self-test: at()/closest() invert to %.1e m, %.1e m of arclength'
          % (wp, wa))
    print()
    print('  %6s %6s %7s %7s %9s %9s %9s %8s %7s'
          % ('R', 'S', 'R/L', 'S/L', 'perim', 'v*', 'how', 'u*', 'n pts'))

    t0 = time.time()
    rows = []
    plan = ([('R', R, 20.0) for R in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)]
            + [('S', 4.0, S) for S in (5.0, 10.0, 40.0)])
    for kind, R, S in plan:
        g = Stadium(R, S)
        rr = []
        for v in speeds:
            val, _ = simulate(g, v, df, frames)
            rr.append(val)
        v, how, npts = bowl_locate(speeds, rr)
        u = v * tt / LOOK if v else None
        rows.append({'sweep': kind, 'R': R, 'S': S, 'R_over_L': R / LOOK,
                     'S_over_L': S / LOOK, 'perimeter': g.circ,
                     'v_star': v, 'how': how, 'n_pts': npts, 'u_star': u,
                     'rmse': [None if not np.isfinite(x) else float(x) for x in rr]})
        print('  %6.1f %6.1f %7.2f %7.2f %9.1f %9.4f %9s %8.4f %7d'
              % (R, S, R / LOOK, S / LOOK, g.circ, v, how, u, npts))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    rs = [r for r in rows if r['sweep'] == 'R' and r['u_star']]
    print()
    if len(rs) >= 3:
        us = [r['u_star'] for r in rs]
        mono = all(us[i] <= us[i + 1] + 1e-9 for i in range(len(us) - 1))
        print('  R/L sweep at S/L = 10:')
        for r in rs:
            print('    R/L %5.2f  u* %.4f   (square %.4f .. circle %.4f)'
                  % (r['R_over_L'], r['u_star'], U_SQUARE, U_CIRCLE))
        print('  monotone in R/L: %s' % ('YES' if mono else 'NO'))
        lo, hi = min(us), max(us)
        inside = U_SQUARE - 0.02 <= lo and hi <= U_CIRCLE + 0.02
        print('  spans %.4f .. %.4f;  inside the two endpoints: %s'
              % (lo, hi, 'YES' if inside else 'NO'))
        if mono and inside:
            print()
            print('  -> the square law and the circle law are the two ends of one')
            print('     continuum, and the all-round claim is carried by a swept')
            print('     family rather than by a single track.')

    json.dump({'dur_s': a.dur, 'tau_ms': a.tau, 'tau_tot': tt,
               'u_square_pred': U_SQUARE, 'u_circle_pred': U_CIRCLE,
               'speeds': speeds.tolist(), 'rows': rows},
              open(HERE / 'd84_stadium_sweep_2026-09-08.json', 'w'), indent=2)
    print('\n-> d84_stadium_sweep_2026-09-08.json')


if __name__ == '__main__':
    main()
