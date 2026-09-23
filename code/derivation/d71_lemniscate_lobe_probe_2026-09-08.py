#!/usr/bin/env python3
"""
d71 -- can the lemniscate single-lobe artefact be reproduced, and under what
       conditions?

d70's first pass did NOT reproduce it: at v = 2.0 m/s, tau = 433 ms the agent
covered 100% of the arclength under the shipped global projection.  the reference engine
reports the opposite ("the agent revisited one lobe rather than alternating
between the two"), so before claiming any fix works, the artefact has to be
reproduced on the conditions the paper actually ran.

Two candidate reasons d70 missed it:
  * SPEED.  The engine's own runs use MSPD = 5.0 (the console banner says
    "Control: method=fixed, speed=5.0"); d70 used 2.0.  Lobe capture is a
    look-ahead-versus-curvature effect, so it should be speed-dependent.
  * the ENGINE's loop scales the commanded speed by sqrt(gamma) at vote frames,
    which d70's transcription did not.

This script sweeps speed and latency and measures lobe occupancy directly --
the fraction of frames spent on each lobe of the figure-eight -- rather than arc
coverage, which can look complete for the wrong reason (a global projection maps
positions on one lobe onto BOTH lobes near the crossing).

Usage:  python3 d71_lemniscate_lobe_probe_2026-09-08.py
"""
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402


def run(V, tau_ms, seed=181, N=150, troll=0.05, use_gamma=True):
    """As close to the engine's simulate() as the quantity needs.

    gamma is the consensus strength |mean(votes)|; the engine multiplies the
    commanded speed by sqrt(gamma) on vote frames, so it is included here behind
    a switch, to see whether it is what makes the difference.
    """
    traj = sm.Lemniscate()
    rng = np.random.default_rng(seed)
    frames = sm.FRAMES
    delay_f = int(round(tau_ms / 1000.0 / sm.DT))
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([1.0, 0.0])
    prev = 0.0
    gam = 0.5
    err = np.empty(frames)
    xs = np.empty(frames)
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, arc = traj.closest(dp)
            ideal = traj.at(arc + sm.LOOK) - dp
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
            tgt = cur * (np.sqrt(gam) if use_gamma else 1.0) * V
        else:
            tgt = cur * V
        vel += sm.SMOOTH * (tgt - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
        cp, _ = traj.closest(pos)
        err[f] = float(np.linalg.norm(pos - cp))
        xs[f] = float(pos[0])
    right = float(np.mean(xs > 0.3))
    left = float(np.mean(xs < -0.3))
    tot = right + left
    return {'rmse': float(np.sqrt(np.mean(err ** 2))),
            'frac_right': right, 'frac_left': left,
            'lobe_balance': float(min(right, left) / tot) if tot > 0 else 0.0,
            'x_min': float(xs.min()), 'x_max': float(xs.max())}


def main():
    print('=' * 78)
    print('d71 -- lemniscate lobe occupancy on the shipped projection')
    print('=' * 78)
    print('  lobe_balance = min(left, right) / (left + right).')
    print('  0.50 means the two lobes are visited equally; 0.00 means ONE lobe.')
    print('  a = 7, arclength %.2f m, LOOK %.1f m, %d s'
          % (sm.Lemniscate().circ, sm.LOOK, sm.DUR))
    print()
    print('  %6s %6s %8s | %9s %9s %12s %9s'
          % ('v', 'tau', 'gamma', 'frac L', 'frac R', 'lobe bal.', 'RMSE'))
    out = {}
    for use_gamma in (True, False):
        for V in (1.0, 2.0, 3.0, 4.0, 5.0):
            for tms in (100, 433, 1000):
                r = run(V, tms, use_gamma=use_gamma)
                out['g%d_v%.1f_t%d' % (use_gamma, V, tms)] = r
                flag = '   <-- SINGLE LOBE' if r['lobe_balance'] < 0.05 else ''
                print('  %6.1f %6d %8s | %9.3f %9.3f %12.3f %9.4f%s'
                      % (V, tms, 'on' if use_gamma else 'off',
                         r['frac_left'], r['frac_right'], r['lobe_balance'],
                         r['rmse'], flag))
        print()
    json.dump(out, open(HERE / 'd71_lemniscate_lobe_2026-09-08.json', 'w'),
              indent=2)
    print('-> d71_lemniscate_lobe_2026-09-08.json')


if __name__ == '__main__':
    main()
