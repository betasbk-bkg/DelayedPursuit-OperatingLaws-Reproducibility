#!/usr/bin/env python3
"""
d61 -- Re-run the two the reference engine experiments whose optimum fell on the edge of the
       speed grid, with the grid extended.

Two cells in the published results report v* = 5.0 m/s, which is the LAST entry of
SPEEDS.  That is not a velocity cap (speed_override bypasses MSPD); it means the
minimum was never bracketed, so v* is a lower bound, not a measurement:

    expa_results.json   n_dirs = 4,  tau = 100 ms  ->  vstar 5.0
    exp2_fixed_results  proportional, tau = 100 ms ->  vstar 5.0  (x* 1.581)

The first feeds Supplementary Figure S1 (beta vs n_dirs); the second feeds
Supplementary Table S3 and the E2 claim about x* CV.

Both are re-run here with the engine imported unchanged and everything identical
except SPEEDS, which goes 0.25 .. 8.00 in 0.25 steps instead of the published 13
points that stop at 5.0 and have 1.0 m/s gaps above 2.5.

The comparison to beat: G1 Circle, re-run the same way in d60, moved beta from
0.5111 to 0.5143 -- i.e. the grid did NOT matter there.  The question is whether
it matters here, where the edge cell is more extreme.

Usage:  python3 d61_regrid_rerun_2026-09-07.py [--mc 15] [--only expa|exp2]
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

TAU_ALL = {100: 6, 200: 12, 300: 18, 433: 26, 600: 36, 1000: 60}
N, TROLL = 150, 0.05
N_DIRS_LIST = [4, 8, 16, 36, 360]
SPEEDS_PUB = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0]


def make_dirs(n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def angle_to_dir_n(angles_deg, n):
    step = 360.0 / n
    return (np.round(np.array(angles_deg) / step) % n).astype(int)


def patched_gen_votes_n(ideal_angle, prev_angle, troll_ratio, N_agents, rng, n_dirs=8):
    n_troll = round(N_agents * troll_ratio)
    remaining = N_agents - n_troll
    n_accurate = round(remaining * 0.7368)
    n_slow = round(remaining * 0.2105)
    n_other = remaining - n_accurate - n_slow
    angles = np.empty(n_accurate + n_slow + n_other)
    idx = 0
    angles[idx:idx + n_accurate] = ideal_angle + rng.uniform(-3, 3, n_accurate)
    idx += n_accurate
    if n_slow > 0:
        diff = (ideal_angle - prev_angle + 180) % 360 - 180
        lag = rng.uniform(0.2, 0.5, n_slow)
        angles[idx:idx + n_slow] = prev_angle + diff * (1 - lag)
        idx += n_slow
    if n_other > 0:
        angles[idx:idx + n_other] = ideal_angle + rng.uniform(-30, 30, n_other)
        idx += n_other
    nt = angle_to_dir_n(angles[:idx], n_dirs)
    tv = rng.integers(0, n_dirs, n_troll) if n_troll > 0 else np.array([], dtype=int)
    return np.concatenate([nt, tv]).astype(int)


def locate(speeds, rmses):
    r = np.asarray(rmses)
    i = int(np.argmin(r))
    if i in (0, len(r) - 1):
        return float(speeds[i]), 'EDGE'
    c = np.polyfit(speeds[i - 1:i + 2], r[i - 1:i + 2], 2)
    if c[0] > 0:
        v = -c[1] / (2 * c[0])
        if speeds[i - 1] <= v <= speeds[i + 1]:
            return float(v), 'parabola'
    return float(speeds[i]), 'grid'


def sweep_ndirs(traj, tau_f, n_dirs, speeds, mc):
    og, od, odl = sm.gen_votes, sm.DIRS, sm.DELAY_F
    sm.DIRS = make_dirs(n_dirs)
    sm.DELAY_F = tau_f
    sm.gen_votes = (lambda ia, pa, tr, na, rng:
                    patched_gen_votes_n(ia, pa, tr, na, rng, n_dirs))
    try:
        rm = [sm.run_condition(traj, N, TROLL, mc, method='fixed',
                               speed_override=v, seed_base=31,
                               seed_offset=N)['rmse_mean'] for v in speeds]
    finally:
        sm.gen_votes, sm.DIRS, sm.DELAY_F = og, od, odl
    return [float(x) for x in rm]


def sweep_voteint(traj, tau_f, vote_int, speeds, mc):
    ov, odl = sm.VOTE_INT, sm.DELAY_F
    sm.VOTE_INT, sm.DELAY_F = vote_int, tau_f
    try:
        rm = [sm.run_condition(traj, N, TROLL, mc, method='fixed',
                               speed_override=v, seed_base=31,
                               seed_offset=N)['rmse_mean'] for v in speeds]
    finally:
        sm.VOTE_INT, sm.DELAY_F = ov, odl
    return [float(x) for x in rm]


def beta_of(taus_ms, vs):
    t, v = np.log(np.array(taus_ms) / 1000.0), np.log(np.array(vs))
    A = np.vstack([t, np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    yh = A @ c
    r2 = 1 - ((v - yh) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    return float(-c[0]), float(np.exp(c[1])), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mc', type=int, default=15)
    ap.add_argument('--vmax', type=float, default=8.0)
    ap.add_argument('--only', default='both')
    a = ap.parse_args()
    speeds = np.arange(0.25, a.vmax + 1e-9, 0.25)
    traj = sm.Circle()
    out = {'speeds': speeds.tolist(), 'mc': a.mc, 'published_speeds': SPEEDS_PUB}

    print('=' * 78)
    print('d61 -- re-run of the two grid-edge experiments, grid 0.25..%.2f (%d pts)'
          % (a.vmax, len(speeds)))
    print('=' * 78)
    t0 = time.time()

    if a.only in ('both', 'expa'):
        print()
        print('  EXP-A  n_dirs sweep')
        print('  %6s %8s %10s %10s %12s' % ('n_dirs', 'tau', 'v* new', 'how', 'x* new'))
        res = {}
        for nd in N_DIRS_LIST:
            res[nd] = {}
            for tms, tf in TAU_ALL.items():
                rm = sweep_ndirs(traj, tf, nd, speeds, a.mc)
                v, how = locate(speeds, rm)
                res[nd][tms] = {'vstar': v, 'how': how,
                                'x_star': v * np.sqrt(tms / 1000.0), 'rmses': rm}
                print('  %6d %8d %10.4f %10s %12.4f'
                      % (nd, tms, v, how, v * np.sqrt(tms / 1000.0)))
            print('     -> elapsed %.1f min' % ((time.time() - t0) / 60))
        out['expa'] = res

        print()
        print('  beta per n_dirs:  published vs re-gridded')
        old = json.load(open(REPO / 'expa_results.json'))
        print('  %8s %12s %12s %10s' % ('n_dirs', 'beta (pub)', 'beta (new)', 'delta'))
        bl = {}
        for nd in N_DIRS_LIST:
            tp = sorted(TAU_ALL)
            vp = [old['results'][str(nd)][str(t)]['vstar'] for t in tp]
            vn = [res[nd][t]['vstar'] for t in tp]
            bo, _, _ = beta_of(tp, vp)
            bn, _, r2 = beta_of(tp, vn)
            bl[nd] = {'beta_pub': bo, 'beta_new': bn, 'r2_new': r2}
            print('  %8d %12.4f %12.4f %+10.4f' % (nd, bo, bn, bn - bo))
        out['expa_beta'] = bl

    if a.only in ('both', 'exp2'):
        print()
        print('  E2  VOTE_INT proportional to tau  (tau/WIN held at 1.44)')
        print('  %8s %10s %10s %10s %10s' % ('tau', 'VOTE_INT', 'v* new', 'how', 'x* new'))
        res2 = {}
        for tms, tf in TAU_ALL.items():
            win_prop = (tms / 1000.0) / (0.433 / 0.3)
            vi = max(1, int(win_prop / sm.DT))
            rm = sweep_voteint(traj, tf, vi, speeds, a.mc)
            v, how = locate(speeds, rm)
            res2[tms] = {'vote_int': vi, 'vstar': v, 'how': how,
                         'x_star': v * np.sqrt(tms / 1000.0), 'rmses': rm}
            print('  %8d %10d %10.4f %10s %10.4f'
                  % (tms, vi, v, how, v * np.sqrt(tms / 1000.0)))
        out['exp2'] = res2
        old2 = json.load(open(REPO / 'exp2_fixed_results.json'))
        xn = np.array([res2[t]['x_star'] for t in sorted(TAU_ALL)])
        xo = np.array([old2[str(t)]['x_prop'] for t in sorted(TAU_ALL)])
        xb = np.array([old2[str(t)]['x_base'] for t in sorted(TAU_ALL)])
        print()
        print('  x* CV (proportional): published %.4f  ->  re-gridded %.4f'
              % (xo.std() / xo.mean(), xn.std() / xn.mean()))
        print('  x* CV (fixed base, published): %.4f' % (xb.std() / xb.mean()))
        print('  the paper reports 0.064 (base) -> 0.203 (proportional)')
        out['exp2_cv'] = {'pub_prop': float(xo.std() / xo.mean()),
                          'new_prop': float(xn.std() / xn.mean()),
                          'pub_base': float(xb.std() / xb.mean())}

    print('\n  total %.1f min' % ((time.time() - t0) / 60))
    json.dump(out, open(HERE / 'd61_regrid_rerun_2026-09-07.json', 'w'), indent=2)
    print('-> d61_regrid_rerun_2026-09-07.json')


if __name__ == '__main__':
    main()
