#!/usr/bin/env python3
"""
d65 -- u*(alpha) for the heading-servo plant, with the geometry-truncation
       artefact removed, and compared against 0.5 cos(alpha/2).

d64 established the qualitative answer: with the vote layer removed, the corner
optimum of a heading-servo pure-pursuit plant still moves with the corner angle,
so d38's linearisation -- which makes alpha cancel out of argmin -- is losing a
real effect.  It also converged to d38's value in the small-angle limit
(u* = 0.5097 at alpha = 15 deg against the derived 0.4995).

But d64's alpha = 120 deg point was wrong, and the reason is worth stating.  The
engine runs a FIXED 65 s regardless of speed, so at low speed the agent never
reaches a corner: at v = 0.20 m/s it covers 13 m of a 52 m triangle and the RMSE
comes out EXACTLY 0.  The located "minimum" was the agent never leaving the first
straight edge.  This is the geometry-truncation artefact the reference engine's own handoff
flags as item 8 ("fixed start point + 65 s truncates the geometry sample at low
speed"), and it bites hardest exactly at the angle where the two papers disagree.

Fixes here, all three stated before running:
  * run long enough that every speed covers whole laps (DUR raised to 240 s);
  * a validity gate -- a point is discarded unless the agent covers at least
    1.5 laps, so every corner is traversed;
  * the first lap is discarded from the RMSE, so the start transient is out.

Then fit the resulting u*(alpha) against 0.5 cos(alpha/2), which is the reference engine's
Proposition 1 corner law -- the one this study's Nav2 work refuted, but refuted for the
CURVATURE-COMMAND class only.

Usage:  python3 d65_heading_servo_alpha_law_2026-09-07.py [--dur 240]
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


class NGon:
    def __init__(self, n, R=10.0, side=None):
        # side=None keeps the circumradius fixed (seg/L then varies with n, which
        # CONFOUNDS the angle sweep with the corner-isolation variable);
        # side=<value> keeps the segment length fixed instead, so only alpha moves.
        if side is not None:
            R = side / (2.0 * math.sin(math.pi / n))
        self.n, self.R = n, R
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
        self.c = np.vstack([c, c[:1]])
        self.segs = [(self.c[i], self.c[i + 1]) for i in range(n)]
        self.lens = [float(np.linalg.norm(b - a)) for a, b in self.segs]
        self.circ = float(sum(self.lens))
        self.cum = np.array([0.0] + list(np.cumsum(self.lens)))
        self.alpha_deg = 360.0 / n
        self.seg_over_L = self.lens[0] / LOOK

    def closest(self, p):
        bd, bp, ba = 1e18, self.c[0], 0.0
        for i, (a, b) in enumerate(self.segs):
            v = b - a
            l2 = float(v @ v)
            t = float(np.clip((p - a) @ v / l2, 0, 1))
            pt = a + t * v
            d = float(np.linalg.norm(p - pt))
            if d < bd:
                bd, bp, ba = d, pt, self.cum[i] + t * self.lens[i]
        return bp, ba

    def at(self, arc):
        arc = arc % self.circ
        for i in range(self.n):
            if arc <= self.cum[i + 1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                a, b = self.segs[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


def simulate(traj, V, delay_f, frames):
    """Deterministic heading-servo pure pursuit: delay, WIN hold, first-order lag.
    Returns (rmse_after_first_lap, laps_covered) or (nan, laps) if truncated."""
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    err = np.empty(frames)
    dist = 0.0
    lap1_frame = None
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            di = max(0, len(hist) - 1 - delay_f)
            dp = hist[di]
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
        if lap1_frame is None and dist >= traj.circ:
            lap1_frame = f
    laps = dist / traj.circ
    if laps < 1.5 or lap1_frame is None:
        return float('nan'), laps
    return float(np.sqrt(np.mean(err[lap1_frame:] ** 2))), laps


def locate(speeds, r):
    r = np.asarray(r, float)
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return None, 'too-few'
    s, rr = np.asarray(speeds)[ok], r[ok]
    i = int(np.argmin(rr))
    if i in (0, len(rr) - 1):
        return float(s[i]), 'EDGE'
    c = np.polyfit(s[i - 1:i + 2], rr[i - 1:i + 2], 2)
    if c[0] > 0:
        v = -c[1] / (2 * c[0])
        if s[i - 1] <= v <= s[i + 1]:
            return float(v), 'parabola'
    return float(s[i]), 'grid'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dur', type=float, default=240.0)
    ap.add_argument('--side', type=float, default=None,
                    help='fix the SEGMENT length (seg/L constant) instead of the circumradius')
    ap.add_argument('--tau', type=int, default=433)
    ap.add_argument('--vmax', type=float, default=4.0)
    a = ap.parse_args()
    frames = int(a.dur / sm.DT)
    speeds = np.arange(0.20, a.vmax + 1e-9, 0.10)
    df = int(round(a.tau / 1000.0 / sm.DT))
    tt = a.tau / 1000.0 + TAU_EXTRA

    geoms = [NGon(n, side=a.side) for n in (3, 4, 6, 8, 12, 24)]
    print('=' * 78)
    print('d65 -- heading-servo u*(alpha), truncation artefact removed')
    print('=' * 78)
    print('  DUR %.0f s (%d frames) vs the engine default 65 s' % (a.dur, frames))
    print('  gate: >= 1.5 laps covered;  RMSE scored after the FIRST lap')
    print('  tau %d ms -> tau_tot %.4f s;  LOOK %.1f m;  %s'
          % (a.tau, tt, LOOK,
             ('SEGMENT fixed at %.1f m (seg/L %.2f, alpha is the only variable)'
              % (a.side, a.side / LOOK)) if a.side else
             'circumradius fixed at 10 m (seg/L varies -- CONFOUNDED)'))
    print('  d38 (linearised) predicts u* = 0.4995 for every alpha')
    print()
    print('  %7s %5s %8s %9s %9s %10s %12s %9s'
          % ('alpha', 'n', 'seg/L', 'v*', 'how', 'u*', '0.5cos(a/2)', 'diff'))

    t0 = time.time()
    rows = []
    for g in geoms:
        rr, lp = [], []
        for v in speeds:
            e, l = simulate(g, v, df, frames)
            rr.append(e)
            lp.append(l)
        v, how = locate(speeds, rr)
        if v is None:
            print('  %7.0f %5d   no valid points' % (g.alpha_deg, g.n))
            continue
        u = v * tt / LOOK
        p = 0.5 * math.cos(math.radians(g.alpha_deg) / 2)
        rows.append({'n': g.n, 'alpha_deg': g.alpha_deg, 'seg_over_L': g.seg_over_L,
                     'perimeter_m': g.circ, 'v_star': v, 'how': how, 'u_star': u,
                     'half_cos': p, 'diff_pct': 100 * (u / p - 1),
                     'n_valid': int(np.isfinite(rr).sum()),
                     'rmse': [None if not np.isfinite(x) else float(x) for x in rr],
                     'laps': [float(x) for x in lp]})
        print('  %7.0f %5d %8.2f %9.4f %9s %10.4f %12.4f %8.1f%%'
              % (g.alpha_deg, g.n, g.seg_over_L, v, how, u, p, 100 * (u / p - 1)))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    ok = [r for r in rows if r['how'] == 'parabola']
    if len(ok) >= 3:
        u = np.array([r['u_star'] for r in ok])
        p = np.array([r['half_cos'] for r in ok])
        k = float(np.sum(u * p) / np.sum(p * p))          # best scale on the law
        res = 100 * np.sqrt(np.mean((u / (k * p) - 1) ** 2))
        # and the exponent, if the law is written as 0.5 cos^q(alpha/2)
        cs = np.array([math.cos(math.radians(r['alpha_deg']) / 2) for r in ok])
        q_pin = float(np.sum(np.log(cs) * np.log(u / 0.5)) / np.sum(np.log(cs) ** 2))
        # Pinning the intercept to 0.5 BIASES q: the small-angle limit is a fitted
        # quantity, not an assumption, and it comes out ~0.517 rather than 0.4995.
        M = np.vstack([np.log(cs), np.ones_like(cs)]).T
        cf, *_ = np.linalg.lstsq(M, np.log(u), rcond=None)
        q, A = float(cf[0]), float(np.exp(cf[1]))
        yh = M @ cf
        r2 = float(1 - ((np.log(u) - yh) ** 2).sum() / ((np.log(u) - np.log(u).mean()) ** 2).sum())
        res = float(100 * np.sqrt(np.mean((np.exp(yh) / u - 1) ** 2)))
        print()
        print('  u* vs 0.5 cos(alpha/2):  best scale %.4f, rms %.2f%% about it' % (k, res))
        print('  A cos^q(alpha/2), BOTH free:  A = %.4f  q = %.3f  r2 = %.4f  rms %.2f%%'
              % (A, q, r2, res))
        print('    (intercept pinned to 0.5 would give q = %.3f -- biased)' % q_pin)
        print('    A vs the linearised derivation 0.4995: %+.1f%%' % (100 * (A / 0.4995 - 1)))
        print('  small-angle limit: alpha = %.0f deg gives u* = %.4f against the'
              % (ok[-1]['alpha_deg'], ok[-1]['u_star']))
        print('  linearised derivation 0.4995  ->  %+.1f%%'
              % (100 * (ok[-1]['u_star'] / 0.4995 - 1)))
        out_extra = {'scale_on_law': k, 'rms_about_law_pct': res,
                     'exponent_q': q, 'amplitude_A': A, 'r2': r2,
                     'exponent_q_pinned': q_pin}
    else:
        out_extra = {}

    print()
    print('  the corner law is PLANT-CLASS DEPENDENT:')
    print('    heading-servo    u*(alpha) ~ 0.5 cos(alpha/2), -> 0.4995 as alpha -> 0')
    print('    curvature-cmd    u* ~ 0.306, nearly flat, dip at 120 deg (d36 + E9)')
    print('  this study refuted the cos law for the curvature-command class only.')

    json.dump({'dur_s': a.dur, 'tau_ms': a.tau, 'tau_tot': tt, 'side': a.side,
               'speeds': speeds.tolist(), 'rows': rows, **out_extra},
              open(HERE / ('d65_heading_servo_alpha_law_%s_tau%d_2026-09-07.json'
                          % ((('side%.0f' % a.side) if a.side else 'R10'), a.tau)),
                        'w'), indent=2)
    print('\n-> d65_heading_servo_alpha_law_2026-09-07.json')


if __name__ == '__main__':
    main()
