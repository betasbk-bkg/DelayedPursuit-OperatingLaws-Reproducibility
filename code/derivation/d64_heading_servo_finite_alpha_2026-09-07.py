#!/usr/bin/env python3
"""
d64 -- The corner optimum of a heading-servo plant at FINITE corner angle.

THE GAP THIS CLOSES.  d38 derives u* = argmin J(u) for both plant classes, but it
linearises about the path: the corner enters as alpha * delta(x - x_c), alpha is a
multiplicative scale on a linear equation, and so it cancels out of argmin.  d38
therefore predicts u* INDEPENDENT of the corner angle for both classes.

For the curvature-command class that prediction is right, and d36 -- which does
not linearise, it simulates the actual RPP kernel on an n-gon -- reproduces the
small residual angle dependence including the dip at alpha = 120 deg that E9 then
measured on the real controller (predicted 0.2779, measured 0.2515).

For the heading-servo class there is no such nonlinear calculation, and the reference engine's
crowd engine -- a heading-servo pure-pursuit plant -- shows u* moving 2.43x over
alpha = 0..120 deg.  Either d38's linearisation is losing a real effect, or that
2.43x comes from the crowd's vote quantisation rather than from the corner geometry.
Nothing in either paper separates those two.

THE INSTRUMENT.  the reference engine's engine already IS the plant: arc-length lookahead from
a delayed position, a heading command, a hold of WIN, and a first-order lag.  The
only thing between it and a clean deterministic heading-servo simulator is the
vote layer.  So run it with the votes REMOVED -- the commanded direction is the
ideal direction, exactly, with delay and hold intact -- and sweep the corner angle.

    quantisation OFF, u* still moves with alpha  ->  d38's linearisation is
                                                     losing a real geometric effect
    quantisation OFF, u* flat at ~0.5            ->  the 2.43x is the quantisation
                                                     interacting with corners, and
                                                     d38 is right about the geometry

Everything else is held at the engine's own constants, and the same run is also
done WITH the standard 8-direction votes so the two are paired.

Usage:  python3 d64_heading_servo_finite_alpha_2026-09-07.py [--taus 433] [--mc 15]
"""
import argparse
import json
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
    """Regular n-gon, circumradius R, so every geometry has the same scale."""

    def __init__(self, n, R=10.0):
        self.n = n
        self.R = R
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
        self.c = np.vstack([c, c[:1]])
        self.segs = [(self.c[i], self.c[i + 1]) for i in range(n)]
        self.lens = [float(np.linalg.norm(b - a)) for a, b in self.segs]
        self.circ = float(sum(self.lens))
        self.cum = np.array([0.0] + list(np.cumsum(self.lens)))
        self.alpha_deg = 360.0 / n
        self.name = 'ngon%d' % n

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
        for i in range(self.n):
            if arc <= self.cum[i + 1] + 1e-9:
                t = (arc - self.cum[i]) / self.lens[i]
                a, b = self.segs[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return self.c[-1]

    def start(self):
        return self.c[0].copy()


def simulate(traj, V, delay_f, quantised, N=150, troll=0.05, seed=181,
             settle_frac=0.25):
    """The engine's loop.  quantised=False replaces the vote layer with the ideal
    direction, leaving delay, hold and first-order lag exactly as they are."""
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    prev = 0.0
    cur = np.array([1.0, 0.0])
    err = np.empty(sm.FRAMES)
    for f in range(sm.FRAMES):
        if f % sm.VOTE_INT == 0:
            di = max(0, len(hist) - 1 - delay_f)
            dp = hist[di]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + LOOK) - dp
            n = float(np.linalg.norm(ideal))
            if n > 1e-10:
                ideal = ideal / n
            ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
            if quantised:
                votes = sm.gen_votes(ia, prev, troll, N, rng)
                blend = sm.DIRS[votes].mean(axis=0)
                g = float(np.linalg.norm(blend))
                cur = blend / g if g > 1e-10 else np.array([1.0, 0.0])
            else:
                cur = ideal
            prev = ia
        vel += sm.SMOOTH * (cur * V - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        err[f] = float(np.linalg.norm(pos - cp))
    k = int(sm.FRAMES * settle_frac)
    return float(np.sqrt(np.mean(err[k:] ** 2)))


def locate(speeds, r):
    r = np.asarray(r)
    i = int(np.argmin(r))
    if i in (0, len(r) - 1):
        return float(speeds[i]), 'EDGE'
    c = np.polyfit(speeds[i - 1:i + 2], r[i - 1:i + 2], 2)
    if c[0] > 0:
        v = -c[1] / (2 * c[0])
        if speeds[i - 1] <= v <= speeds[i + 1]:
            return float(v), 'parabola'
    return float(speeds[i]), 'grid'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taus', type=int, nargs='+', default=[300, 433, 600])
    ap.add_argument('--mc', type=int, default=9)
    ap.add_argument('--vmax', type=float, default=6.0)
    a = ap.parse_args()
    speeds = np.arange(0.25, a.vmax + 1e-9, 0.25)

    geoms = [NGon(n) for n in (3, 4, 6, 8, 12, 24)]
    print('=' * 78)
    print('d64 -- heading-servo corner optimum at finite alpha')
    print('=' * 78)
    print('  engine from the the reference engine repository, circumradius 10 m, LOOK = %.1f m' % LOOK)
    print('  tau_tot = tau + WIN/2 + DT/SMOOTH = tau + %.4f s' % TAU_EXTRA)
    print('  d38 linearised prediction: u* = 0.4995 for EVERY alpha')
    print('  speeds %.2f..%.2f step %.2f, tau %s ms, MC %d (quantised only)'
          % (speeds[0], speeds[-1], speeds[1] - speeds[0], a.taus, a.mc))

    t0 = time.time()
    out = {'taus_ms': a.taus, 'speeds': speeds.tolist(), 'mc': a.mc, 'rows': []}
    print()
    print('  %8s %7s %8s | %9s %9s | %9s %9s'
          % ('alpha', 'n', 'tau', 'u* clean', 'how', 'u* quant', 'how'))
    for g in geoms:
        for tms in a.taus:
            df = int(round(tms / 1000.0 / sm.DT))
            tt = tms / 1000.0 + TAU_EXTRA
            rc = [simulate(g, v, df, quantised=False) for v in speeds]
            vc, hc = locate(speeds, rc)
            rq = [float(np.mean([simulate(g, v, df, True, seed=i * 31 + 150)
                                 for i in range(a.mc)])) for v in speeds]
            vq, hq = locate(speeds, rq)
            out['rows'].append({'n': g.n, 'alpha_deg': g.alpha_deg, 'tau_ms': tms,
                                'tau_tot': tt, 'v_clean': vc, 'how_clean': hc,
                                'u_clean': vc * tt / LOOK, 'how_quant': hq,
                                'v_quant': vq, 'u_quant': vq * tt / LOOK,
                                'rmse_clean': rc, 'rmse_quant': rq})
            print('  %8.0f %7d %8d | %9.4f %9s | %9.4f %9s'
                  % (g.alpha_deg, g.n, tms, vc * tt / LOOK, hc,
                     vq * tt / LOOK, hq))
        print('     -> %.1f min' % ((time.time() - t0) / 60))

    print()
    print('  u* averaged over tau, per geometry:')
    print('  %8s %12s %12s' % ('alpha', 'clean', 'quantised'))
    ac, aq = {}, {}
    for g in geoms:
        rs = [r for r in out['rows'] if r['n'] == g.n]
        ac[g.alpha_deg] = float(np.mean([r['u_clean'] for r in rs]))
        aq[g.alpha_deg] = float(np.mean([r['u_quant'] for r in rs]))
        print('  %8.0f %12.4f %12.4f' % (g.alpha_deg, ac[g.alpha_deg], aq[g.alpha_deg]))
    vc = np.array(list(ac.values()))
    vq = np.array(list(aq.values()))
    print()
    print('  clean     : %.4f - %.4f  ->  %.2fx  (spread %.1f%%)'
          % (vc.min(), vc.max(), vc.max() / vc.min(),
             100 * (vc.max() - vc.min()) / vc.mean()))
    print('  quantised : %.4f - %.4f  ->  %.2fx  (spread %.1f%%)'
          % (vq.min(), vq.max(), vq.max() / vq.min(),
             100 * (vq.max() - vq.min()) / vq.mean()))
    print('  the reference engine published sweep gave 2.43x; d38 predicts 1.00x at 0.4995')

    rc_ = vc.max() / vc.min()
    print()
    if rc_ > 1.5:
        print('  VERDICT: the angle dependence is DETERMINISTIC GEOMETRY.')
        print('  It survives with the vote layer removed, so d38\'s linearisation is')
        print('  losing a real effect and the finite-alpha heading-servo corner needs')
        print('  the nonlinear treatment d36 already gives the other plant class.')
    elif rc_ < 1.15:
        print('  VERDICT: with quantisation removed u* is FLAT, as d38 predicts.')
        print('  The 2.43x in the reference engine is the vote quantisation interacting with')
        print('  corners, not the corner geometry.  d38 is right about the geometry')
        print('  and the merged theory needs a quantisation-corner term instead.')
    else:
        print('  VERDICT: partial -- %.2fx clean vs %.2fx quantised.  Both effects'
              % (rc_, vq.max() / vq.min()))
        print('  are present and neither alone explains the published spread.')

    json.dump(out, open(HERE / 'd64_heading_servo_finite_alpha_2026-09-07.json', 'w'),
              indent=2)
    print('\n  total %.1f min' % ((time.time() - t0) / 60))
    print('-> d64_heading_servo_finite_alpha_2026-09-07.json')


if __name__ == '__main__':
    main()
