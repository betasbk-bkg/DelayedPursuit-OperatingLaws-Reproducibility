#!/usr/bin/env python3
"""
d66 -- the CROWD leg of u*(alpha), under d65's protocol, with a free intercept.

THEORY_GAP_CLOSURE section 3 leaves an explicit debt.  It puts two exponents side
by side --

    deterministic heading-servo   q = 0.843 +- 0.026   (d65, intercept FREE)
    the reference engine's crowd engine      q = 0.669            (d64, intercept PINNED)

-- and then says, correctly, that they are not comparable: pinning the intercept
to 0.5 biases q downward (d65 section 2.4 measured that bias at 0.05-0.08), so
part of the "quantisation weakens the angle dependence" gap could be my own fit.

There is a second, larger incomparability that section 3 does not name.  d64's
crowd numbers were taken BEFORE d65 found the truncation artefact: d64 ran the
engine's default duration, with no lap-coverage gate and a fixed settle fraction,
which is exactly the regime where the alpha = 120 deg point measured an agent that
never reached a corner.  Refitting those numbers with a free intercept would give
a better fit to contaminated data.

So this re-runs the crowd leg under d65's protocol EXACTLY -- 240 s, the >= 1.5
lap gate, RMSE scored after the first lap, segment length fixed so seg/L does not
move with alpha -- and changes one line relative to d65:

    clean (d65):   cur = ideal
    crowd (here):  cur = normalize(mean(DIRS[gen_votes(...)]))

Then both legs are fitted the same way, with A free.  Any remaining difference in
q is then quantisation, and nothing else.

Cost: the vote layer is stochastic, so each speed is averaged over MC seeds.
About 35 min for MC = 5 on the 6-polygon, 39-speed grid.

Usage:  python3 d66_crowd_alpha_law_free_intercept_2026-09-08.py [--mc 5]
"""
import argparse
import glob
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


class NGon:
    """Identical to d65's, including the side= option that holds seg/L fixed."""

    def __init__(self, n, R=10.0, side=None):
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


def simulate(traj, V, delay_f, frames, quantised, seed=181, N=150, troll=0.05):
    """d65's loop, with the vote layer as the ONLY optional difference."""
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
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
                ideal = ideal / n
            if quantised:
                ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
                votes = sm.gen_votes(ia, prev, troll, N, rng)
                blend = sm.DIRS[votes].mean(axis=0)
                g = float(np.linalg.norm(blend))
                cur = blend / g if g > 1e-10 else np.array([1.0, 0.0])
                prev = ia
            else:
                cur = ideal
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


def fit_law(alphas_deg, us):
    """A cos^q(alpha/2), both free -- and the pinned version, for the record."""
    cs = np.array([math.cos(math.radians(a) / 2) for a in alphas_deg])
    us = np.asarray(us, float)
    M = np.vstack([np.log(cs), np.ones_like(cs)]).T
    cf, *_ = np.linalg.lstsq(M, np.log(us), rcond=None)
    q, A = float(cf[0]), float(np.exp(cf[1]))
    yh = M @ cf
    r2 = float(1 - ((np.log(us) - yh) ** 2).sum()
               / ((np.log(us) - np.log(us).mean()) ** 2).sum())
    rms = float(100 * np.sqrt(np.mean((np.exp(yh) / us - 1) ** 2)))
    q_pin = float(np.sum(np.log(cs) * np.log(us / 0.5)) / np.sum(np.log(cs) ** 2))
    # a crude interval: leave one polygon out
    qs = []
    for i in range(len(us)):
        k = [j for j in range(len(us)) if j != i]
        if len(k) < 3:
            continue
        c2, *_ = np.linalg.lstsq(M[k], np.log(us[k]), rcond=None)
        qs.append(float(c2[0]))
    return {'q': q, 'A': A, 'r2': r2, 'rms_pct': rms, 'q_pinned': q_pin,
            'q_loo_sd': float(np.std(qs, ddof=1)) if len(qs) > 2 else None,
            'n': len(us)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dur', type=float, default=240.0)
    ap.add_argument('--side', type=float, default=10.0)
    ap.add_argument('--tau', type=int, default=433)
    ap.add_argument('--vmax', type=float, default=4.0)
    ap.add_argument('--mc', type=int, default=5)
    a = ap.parse_args()
    frames = int(a.dur / sm.DT)
    speeds = np.arange(0.20, a.vmax + 1e-9, 0.10)
    df = int(round(a.tau / 1000.0 / sm.DT))
    tt = a.tau / 1000.0 + TAU_EXTRA

    geoms = [NGon(n, side=a.side) for n in (3, 4, 6, 8, 12, 24)]
    print('=' * 78)
    print('d66 -- crowd leg of u*(alpha), d65 protocol, free intercept')
    print('=' * 78)
    print('  DUR %.0f s (%d frames), gate >= 1.5 laps, RMSE after the first lap'
          % (a.dur, frames))
    print('  side %.1f m fixed -> seg/L %.2f for every alpha' % (a.side, a.side / LOOK))
    print('  tau %d ms -> tau_tot %.4f s;  MC %d seeds per speed;  N %d agents'
          % (a.tau, tt, a.mc, 150))
    print()
    print('  %7s %5s %9s %9s %10s %8s' % ('alpha', 'n', 'v*', 'how', 'u*', 'valid'))

    t0 = time.time()
    rows = []
    for g in geoms:
        rr, lp = [], []
        for v in speeds:
            es, ls = [], []
            for k in range(a.mc):
                e, lo = simulate(g, v, df, frames, True, seed=k * 31 + 150)
                es.append(e)
                ls.append(lo)
            rr.append(float(np.mean(es)) if np.all(np.isfinite(es)) else float('nan'))
            lp.append(float(np.mean(ls)))
        v, how = locate(speeds, rr)
        if v is None:
            print('  %7.0f %5d   no valid points' % (g.alpha_deg, g.n))
            continue
        u = v * tt / LOOK
        rows.append({'n': g.n, 'alpha_deg': g.alpha_deg, 'seg_over_L': g.seg_over_L,
                     'perimeter_m': g.circ, 'v_star': v, 'how': how, 'u_star': u,
                     'n_valid': int(np.isfinite(rr).sum()),
                     'rmse': [None if not np.isfinite(x) else float(x) for x in rr],
                     'laps': lp})
        print('  %7.0f %5d %9.4f %9s %10.4f %8d  (%.1f min)'
              % (g.alpha_deg, g.n, v, how, u, int(np.isfinite(rr).sum()),
                 (time.time() - t0) / 60))

    ok = [r for r in rows if r['how'] == 'parabola']
    out = {'dur_s': a.dur, 'tau_ms': a.tau, 'tau_tot': tt, 'side': a.side,
           'mc': a.mc, 'speeds': speeds.tolist(), 'rows': rows}
    if len(ok) >= 3:
        crowd = fit_law([r['alpha_deg'] for r in ok], [r['u_star'] for r in ok])
        out['fit_crowd'] = crowd
        print()
        print('  CROWD   A = %.4f  q = %.3f  r2 = %.4f  rms %.2f%%  (LOO sd %s, n %d)'
              % (crowd['A'], crowd['q'], crowd['r2'], crowd['rms_pct'],
                 ('%.3f' % crowd['q_loo_sd']) if crowd['q_loo_sd'] else 'n/a',
                 crowd['n']))
        print('          intercept pinned to 0.5 would give q = %.3f'
              % crowd['q_pinned'])

        # the paired clean leg, refitted from d65's saved output with the same code
        cand = sorted(glob.glob(str(HERE / ('d65_heading_servo_alpha_law_side%.0f_tau%d_*.json'
                                            % (a.side, a.tau)))))
        if cand:
            d65 = json.load(open(cand[-1]))
            ok5 = [r for r in d65['rows'] if r['how'] == 'parabola']
            clean = fit_law([r['alpha_deg'] for r in ok5], [r['u_star'] for r in ok5])
            out['fit_clean_d65'] = clean
            out['clean_source'] = pathlib.Path(cand[-1]).name
            print('  CLEAN   A = %.4f  q = %.3f  r2 = %.4f  rms %.2f%%  (from %s)'
                  % (clean['A'], clean['q'], clean['r2'], clean['rms_pct'],
                     pathlib.Path(cand[-1]).name))
            print()
            print('  SAME PROTOCOL, SAME FIT, one line different:')
            print('    q  clean %.3f  ->  crowd %.3f   (%+.1f%%)'
                  % (clean['q'], crowd['q'], 100 * (crowd['q'] / clean['q'] - 1)))
            print('    A  clean %.4f ->  crowd %.4f  (%+.1f%%)'
                  % (clean['A'], crowd['A'], 100 * (crowd['A'] / clean['A'] - 1)))
            sd = crowd['q_loo_sd']
            if sd:
                print('    separation in q: %.2f LOO sd'
                      % (abs(crowd['q'] - clean['q']) / sd))
        else:
            print('  (no d65 side%.0f tau%d file found; clean leg not refitted)'
                  % (a.side, a.tau))
    else:
        print('  too few parabola rows to fit')

    json.dump(out, open(HERE / ('d66_crowd_alpha_law_side%.0f_tau%d_mc%d_2026-09-08.json'
                                % (a.side, a.tau, a.mc)), 'w'), indent=2)
    print('\n-> d66_crowd_alpha_law_side%.0f_tau%d_mc%d_2026-09-08.json'
          % (a.side, a.tau, a.mc))


if __name__ == '__main__':
    main()
