#!/usr/bin/env python3
"""
d76 -- the missing denominators: MC standard errors for the audit's four
       unstated-basis checks.

AUDIT_TAXONOMY section 3.1 found four audit checks that compare a Monte-Carlo
quantity against a number printed in the reference engine, with tolerances (0.30, 0.02,
0.02, 0.15) that came from nowhere:

    line 140  open-loop swept sigma        10.9456  vs manuscript 11.06   -1.0%
    line 142  closed-loop sigma at v=2.0    8.5517  vs published   8.54   +0.1%
    line 144  attenuation at v=2.0          0.7813  vs manuscript 0.778   +0.4%
    line 186  attenuation at u=0.71         0.7899  vs manuscript 0.778   +1.5%

They pass, but passing means nothing when the tolerance is chosen rather than
measured -- the same defect the E6 criterion had before E7 replaced it with a
measured reproducibility floor.  This script supplies the denominator for the
first three, which all come out of d3's machinery, by repeating the ENTIRE
procedure with independent seed blocks and taking the scatter across repeats.

That is the right construction: the published quantity is not one draw, it is a
statistic computed from a whole seed block (d3 concatenates six seeds before
taking a standard deviation), so its uncertainty is the variability of that
statistic under re-seeding -- not the spread within one block.

Stated before running: if the differences above are inside the measured
standard error, the manuscript's numbers are reproduced and the audit's job is
to say so with a basis; if any sits outside, it is a discrepancy to correct.

Usage:  python3 d76_mc_standard_errors_2026-09-08.py [--reps 8]
"""
import argparse
import importlib.util
import json
import math
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
_s = importlib.util.spec_from_file_location(
    'd3', HERE / 'd3_loop_attenuation_2026-08-29.py')
d3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d3)          # imports the engine and defines the functions

PUBLISHED = {'sigma_open_uniform': 11.06,
             'sigma_closed_v2': 8.543,
             'ratio_v2': 0.778}
AUDIT_TOL = {'sigma_open_uniform': 0.30,
             'sigma_closed_v2': 0.20,
             'ratio_v2': 0.02}


def one_replicate(rep):
    """d3's whole procedure, on a seed block that no other replicate uses."""
    ms, b, s = d3.transfer(seed=1000 * (rep + 1) + 11)
    sig_unif = float(np.sqrt(np.mean(b ** 2 + s ** 2)))
    IA, ER = [], []
    for sd in range(6):
        ia, er = d3.closed_loop(d3.Circle(10), 2.0, seed=100000 * (rep + 1) + sd)
        IA.append(ia)
        ER.append(er)
    ER = np.concatenate(ER)
    sig_meas = float(ER.std())
    return sig_unif, sig_meas, sig_meas / sig_unif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    a = ap.parse_args()

    print('=' * 78)
    print('d76 -- MC standard errors for the audit checks that had no basis')
    print('=' * 78)
    print('  %d independent replicates of d3\'s full procedure' % a.reps)
    print('  (transfer(): 61 offsets x 1500 draws;  closed loop: 6 seeds at v=2.0)')
    print()

    t0 = time.time()
    rows = []
    for r in range(a.reps):
        su, sm_, ra = one_replicate(r)
        rows.append((su, sm_, ra))
        print('    rep %d   sigma_open %8.4f   sigma_closed %8.4f   ratio %7.4f'
              % (r, su, sm_, ra))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    arr = np.array(rows)
    names = ['sigma_open_uniform', 'sigma_closed_v2', 'ratio_v2']
    out = {'reps': a.reps, 'raw': arr.tolist(), 'results': {}}
    print()
    print('  %-22s %9s %9s %9s %10s %10s %8s'
          % ('quantity', 'mean', 'SD(rep)', 'SE(mean)', 'published', 'diff',
             'diff/SE'))
    for i, nm in enumerate(names):
        col = arr[:, i]
        mean = float(col.mean())
        # sd is the scatter of ONE replicate; se is the uncertainty of the MEAN.
        # Testing whether a published number equals the expected value divides by
        # se, not sd -- reporting only sd (as the first version of this script
        # did) understates the discrepancy by sqrt(reps).
        sd = float(col.std(ddof=1))
        se = sd / math.sqrt(len(col))
        pub = PUBLISHED[nm]
        diff = mean - pub
        z = diff / se if se > 0 else float('inf')
        out['results'][nm] = {'mean': mean, 'sd': sd, 'se': se, 'published': pub,
                              'diff': diff, 'z': z, 'z_sd': diff / sd,
                              'audit_tol': AUDIT_TOL[nm],
                              'tol_over_sd': AUDIT_TOL[nm] / sd if sd > 0 else None}
        print('  %-22s %9.4f %9.4f %9.4f %10.4f %+10.4f %8.2f'
              % (nm, mean, sd, se, pub, diff, z))

    print()
    print('  what the audit tolerance was, against what it should have been:')
    print('  %-22s %12s %12s %10s'
          % ('quantity', 'audit tol', 'SD(rep)', 'ratio'))
    for nm in names:
        r = out['results'][nm]
        print('  %-22s %12.4f %12.4f %10.1fx'
              % (nm, r['audit_tol'], r['sd'], r['tol_over_sd']))

    print()
    for nm in names:
        r = out['results'][nm]
        verdict = ('reproduced (inside 2 SE)' if abs(r['z']) <= 2
                   else 'DISCREPANT (%.1f SE)' % abs(r['z']))
        print('  %-22s %s' % (nm, verdict))

    json.dump(out, open(HERE / 'd76_mc_standard_errors_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d76_mc_standard_errors_2026-09-08.json')


if __name__ == '__main__':
    main()
