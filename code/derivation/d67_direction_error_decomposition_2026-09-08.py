#!/usr/bin/env python3
"""
d67 -- what the crowd's direction error actually is, and why the reference engine's
       quantisation model cannot be the explanation of it.

the reference engine, Results, Direction Error (ii):

    "The theoretical direction error from 8-direction quantization, modeled as a
     uniform distribution over [-22.5, +22.5] deg, has RMS value
     22.5/sqrt(6) = 9.19 deg"

and that 9.19 is then said to agree with the measured 8.60 deg "within 6%".

Two things are wrong, and the second is the one that matters.

  (1) ARITHMETIC.  A uniform distribution on [-a, a] has RMS a/sqrt(3), so the
      number is 22.5/sqrt(3) = 12.99 deg.  a/sqrt(6) is the RMS of a TRIANGULAR
      distribution on [-a, a].  The text names one distribution and evaluates the
      other.  With the correct value the claimed agreement is 51%, not 6%.

  (2) MODEL.  Even corrected, the single-vote quantisation error is the wrong
      quantity.  The plant does not follow one agent's quantised vote; it follows
      the NORMALISED MEAN of N = 150 votes.  Averaging shrinks the error, and the
      composition (70% accurate with +-3 deg noise, 20% one step behind, the rest
      +-30 deg, plus 5% trolls voting at random) means the votes are not even
      centred on the same direction.  There is no reason for the single-vote
      figure to match the blend figure, and this script shows it does not.

So the correction is not "9.19 -> 12.99".  It is that the sentence should report
the error of the quantity the dynamics actually use.  This script measures that,
so the corrected manuscript has a number to state rather than a gap.

Usage:  python3 d67_direction_error_decomposition_2026-09-08.py [--n 150]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2] / "engine" / "crowd_control_engine"   # the engine as it ships in this package
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402

PAPER_MEASURED = 8.60      # deg, as reported
PAPER_CLAIMED = 9.19       # deg, 22.5/sqrt(6)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def on_trajectory(side, V, tau_ms, dur, troll, n_agents, seed):
    """The same statistic, measured where the manuscript measures it: along a run.

    The synthetic draws below have to invent a previous heading for the "slow"
    agents, and any choice there is an assumption.  On an actual run there is no
    choice -- prev is last tick's ideal angle, which equals this tick's on a
    straight and differs only through a corner.  So this is the faithful number,
    and the synthetic one is an upper bound.
    """
    n_side = 4
    R = side / (2.0 * math.sin(math.pi / n_side))
    th = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
    c = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
    c = np.vstack([c, c[:1]])
    segs = [(c[i], c[i + 1]) for i in range(n_side)]
    lens = [float(np.linalg.norm(b - a)) for a, b in segs]
    circ = float(sum(lens))
    cum = np.array([0.0] + list(np.cumsum(lens)))

    def closest(p):
        bd, ba = 1e18, 0.0
        for i, (a, b) in enumerate(segs):
            v = b - a
            t = float(np.clip((p - a) @ v / float(v @ v), 0, 1))
            d = float(np.linalg.norm(p - (a + t * v)))
            if d < bd:
                bd, ba = d, cum[i] + t * lens[i]
        return ba

    def at(arc):
        arc = arc % circ
        for i in range(n_side):
            if arc <= cum[i + 1] + 1e-9:
                t = (arc - cum[i]) / lens[i]
                a, b = segs[i]
                return a + np.clip(t, 0, 1) * (b - a)
        return c[-1]

    rng = np.random.default_rng(seed)
    frames = int(dur / sm.DT)
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    pos = c[0].copy()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    errs, corner = [], []
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            di = max(0, len(hist) - 1 - delay_f)
            dp = hist[di]
            arc = closest(dp)
            ideal = at(arc + sm.LOOK) - dp
            g = float(np.linalg.norm(ideal))
            if g > 1e-10:
                ideal = ideal / g
            ia = float(np.degrees(np.arctan2(ideal[1], ideal[0])))
            votes = sm.gen_votes(ia, prev, troll, n_agents, rng)
            blend = sm.DIRS[votes].mean(axis=0)
            gb = float(np.linalg.norm(blend))
            if gb > 1e-10:
                cur = blend / gb
                errs.append(wrap(math.degrees(math.atan2(cur[1], cur[0])) - ia))
                # is this tick inside a corner?  the lookahead point and the
                # nearest point are on different edges when it is.
                corner.append(int(np.searchsorted(cum, arc, 'right')
                                  != np.searchsorted(cum, (arc + sm.LOOK) % circ,
                                                     'right')))
            prev = ia
        vel += sm.SMOOTH * (cur * V - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
    e = np.array(errs)
    cm = np.array(corner, bool)
    return {'rms_deg': float(np.sqrt(np.mean(e ** 2))),
            'rms_straight_deg': float(np.sqrt(np.mean(e[~cm] ** 2))) if (~cm).any() else None,
            'rms_corner_deg': float(np.sqrt(np.mean(e[cm] ** 2))) if cm.any() else None,
            'n_ticks': int(e.size), 'corner_frac': float(cm.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--troll', type=float, default=0.05)
    ap.add_argument('--draws', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--side', type=float, default=10.0)
    ap.add_argument('--tau-ms', type=int, default=433, dest='tau_ms')
    ap.add_argument('--dur', type=float, default=240.0)
    ap.add_argument('--speeds', type=float, nargs='+',
                    default=[1.0, 1.5, 2.0, 2.5])
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    step = 360.0 / len(sm.DIRS)
    half = step / 2
    print('=' * 78)
    print('d67 -- direction error: single vote vs the blend the plant follows')
    print('=' * 78)
    print('  engine: %d directions (step %.1f deg), N = %d agents, troll %.0f%%'
          % (len(sm.DIRS), step, a.n, 100 * a.troll))
    print()
    print('  ANALYTIC, single quantised vote on [-%.2f, +%.2f] deg:' % (half, half))
    print('    uniform     RMS = a/sqrt(3) = %.3f deg     <- the named distribution'
          % (half / math.sqrt(3)))
    print('    triangular  RMS = a/sqrt(6) = %.3f deg     <- the evaluated formula'
          % (half / math.sqrt(6)))
    print('    the manuscript states "uniform" and prints %.2f.' % PAPER_CLAIMED)
    print()

    # ---- measure, on the engine's own vote generator -------------------------
    ideals = rng.uniform(-180, 180, a.draws)
    prevs = ideals + rng.normal(0, 10, a.draws)      # a plausible previous heading
    e_single, e_blend, e_acc = [], [], []
    for ia, pv in zip(ideals, prevs):
        votes = sm.gen_votes(float(ia), float(pv), a.troll, a.n, rng)
        d = sm.DIRS[votes]
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        e_single.append(wrap(ang - ia))
        blend = d.mean(axis=0)
        g = float(np.linalg.norm(blend))
        if g > 1e-10:
            e_blend.append(wrap(math.degrees(math.atan2(blend[1], blend[0])) - ia))
        # the accurate subgroup alone, as a reference point
        n_troll = round(a.n * a.troll)
        n_acc = round((a.n - n_troll) * 0.7368)
        e_acc.append(wrap(ang[:n_acc] - ia))

    single = np.concatenate([np.atleast_1d(x) for x in e_single])
    acc = np.concatenate([np.atleast_1d(x) for x in e_acc])
    blend = np.array(e_blend)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))   # noqa: E731

    print('  MEASURED on the engine (%d draws):' % a.draws)
    print('    one agent, any type          RMS = %8.3f deg' % rms(single))
    print('    one agent, "accurate" only   RMS = %8.3f deg' % rms(acc))
    print('    BLEND of %d votes            RMS = %8.3f deg   <- what the plant follows'
          % (a.n, rms(blend)))
    print()
    print('  against the manuscript:')
    print('    reported measurement                 %6.2f deg' % PAPER_MEASURED)
    print('    blend measured here                  %6.2f deg   (%+.1f%%)'
          % (rms(blend), 100 * (rms(blend) / PAPER_MEASURED - 1)))
    print('    single-vote uniform RMS (corrected)  %6.2f deg   (%+.1f%% vs measured)'
          % (half / math.sqrt(3),
             100 * ((half / math.sqrt(3)) / PAPER_MEASURED - 1)))
    print('    the printed 22.5/sqrt(6)             %6.2f deg   (%+.1f%% vs measured)'
          % (PAPER_CLAIMED, 100 * (PAPER_CLAIMED / PAPER_MEASURED - 1)))
    print()
    print('  ON AN ACTUAL RUN (square, side %.0f m, tau %d ms, %.0f s):'
          % (a.side, a.tau_ms, a.dur))
    traj = {}
    for V in a.speeds:
        t = on_trajectory(a.side, V, a.tau_ms, a.dur, a.troll, a.n, a.seed)
        traj['v%.2f' % V] = t
        print('    v = %.2f m/s   blend RMS %6.3f deg   (straight %6.3f, corner '
              '%6.3f, %.0f%% of ticks in a corner)'
              % (V, t['rms_deg'], t['rms_straight_deg'] or float('nan'),
                 t['rms_corner_deg'] or float('nan'), 100 * t['corner_frac']))
    tr = [t['rms_deg'] for t in traj.values()]
    print('    -> %.3f to %.3f deg across the speed range; the manuscript reports '
          '%.2f' % (min(tr), max(tr), PAPER_MEASURED))
    print()
    print('  The agreement the manuscript reports comes from comparing a blend')
    print('  measurement against a single-vote model evaluated with the wrong')
    print('  formula.  Two errors that happen to point the same way.')

    out = {'n_agents': a.n, 'troll': a.troll, 'draws': a.draws,
           'n_dirs': len(sm.DIRS), 'half_step_deg': half,
           'uniform_rms_deg': half / math.sqrt(3),
           'triangular_rms_deg': half / math.sqrt(6),
           'rms_single_deg': rms(single), 'rms_accurate_deg': rms(acc),
           'rms_blend_deg': rms(blend),
           'paper_measured_deg': PAPER_MEASURED, 'paper_claimed_deg': PAPER_CLAIMED,
           'on_trajectory': traj, 'side': a.side, 'tau_ms': a.tau_ms,
           'dur_s': a.dur}
    json.dump(out, open(HERE / 'd67_direction_error_2026-09-08.json', 'w'), indent=2)
    print('\n-> d67_direction_error_2026-09-08.json')


if __name__ == '__main__':
    main()
