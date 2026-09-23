#!/usr/bin/env python3
"""
d77 -- the last missing denominator: the standard error of the attenuation at
       the optimal margin u = 0.71.

AUDIT_TAXONOMY section 3.1 listed four checks whose tolerance had no stated
basis.  d76 measured three of them and turned one into a manuscript correction.
Of the two left:

  * line 219 needs no new computation.  d12 contains no random number generator,
    so "open-loop sweep from closed form (engine MC 10.95)" is not a statistical
    check at all -- it is an AGREEMENT check between a closed form and an engine
    measurement, and its basis is the closed form's own accuracy, which the
    derivation report states as rms 0.051 deg.  |10.9940 - 10.95| = 0.044 sits
    inside that, so it passes once the basis is named.

  * line 186 is genuinely statistical.  d8 builds it as
        atten(u) = std(concat of 8 surrogate seeds) / 10.9456
    and the audit compares atten(0.71) against the manuscript's 0.778 with a
    tolerance of 0.02 that came from nowhere.

This measures that denominator the same way d76 did: repeat the WHOLE 8-seed
block on disjoint seeds and take the scatter of the resulting statistic.

The divisor is left at d8's hard-coded 10.9456 so the quantity is exactly the one
the audit checks.  d76 measured that same open-loop sigma as 10.9446 +- 0.0039,
i.e. the hard-coded value is right to 0.01% -- it is the manuscript's 11.06 that
was wrong, not this.

Usage:  python3 d77_atten_u071_se_2026-09-08.py [--reps 8]
"""
import argparse
import importlib.util
import json
import math
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).parent


def _load(stem):
    hits = sorted(HERE.glob(stem + '*.py'))
    assert hits, stem
    s = importlib.util.spec_from_file_location(stem, hits[0])
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


d3 = _load('d3_loop_attenuation')
d6 = _load('d6_')

SIGMA_OPEN_HARDCODED = 10.9456      # d8's divisor
PUBLISHED = 0.778
AUDIT_TOL = 0.02
U = 0.71


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=8)
    a = ap.parse_args()

    print('=' * 78)
    print('d77 -- standard error of the attenuation at u = %.2f' % U)
    print('=' * 78)
    ms, b, s = d3.transfer(seed=11)          # d8 uses the default transfer
    tt = 0.433 + d6.WIN / 2 + d6.T_S
    v = U * d6.L / tt
    print('  surrogate at v = %.4f m/s (u = %.2f), 8 seeds per replicate,'
          % (v, U))
    print('  %d disjoint replicates;  divisor = d8\'s %.4f deg'
          % (a.reps, SIGMA_OPEN_HARDCODED))
    print()

    t0 = time.time()
    vals = []
    for r in range(a.reps):
        E = np.concatenate([d6.surrogate(v, 0.433, ms, b, s,
                                         seed=900000 * (r + 1) + i)[1]
                            for i in range(8)])
        at = float(E.std()) / SIGMA_OPEN_HARDCODED
        vals.append(at)
        print('    rep %d   sigma %8.4f   attenuation %8.5f'
              % (r, E.std(), at))
    print('  -> %.1f min' % ((time.time() - t0) / 60))

    v_ = np.array(vals)
    mean = float(v_.mean())
    sd = float(v_.std(ddof=1))          # scatter of one replicate
    se = sd / math.sqrt(len(v_))        # uncertainty of the mean -- the right
    z = (mean - PUBLISHED) / se         # denominator for "is the published
                                        # number the expected value?"
    print()
    print('  attenuation(u=%.2f) = %.5f   SD(rep) %.5f   SE(mean) %.5f'
          % (U, mean, sd, se))
    print('  manuscript %.3f:  difference %+.5f = %+.2f SE'
          % (PUBLISHED, mean - PUBLISHED, z))
    print('  audit tolerance %.3f is %.1fx SD(rep)' % (AUDIT_TOL, AUDIT_TOL / sd))
    # the manuscript's 0.778 is 8.60/11.06; with the sigma_open d76 measured it
    # becomes 8.60/10.9446 = 0.78577, and THAT is what this should be tested
    # against -- the two discrepancies are one error propagating.
    corrected = 8.60 / 10.9446
    print('  corrected reference 8.60/10.9446 = %.5f:  %+.2f SE'
          % (corrected, (mean - corrected) / se))
    print()
    print('  %s' % ('reproduced (inside 2 SE)' if abs(z) <= 2
                    else 'DISCREPANT (%.1f SE) -- this one needs a correction'
                         % abs(z)))

    json.dump({'u': U, 'reps': a.reps, 'values': vals, 'mean': mean,
               'sd': sd, 'se': se, 'corrected_reference': 8.60 / 10.9446,
               'published': PUBLISHED, 'z': z, 'audit_tol': AUDIT_TOL,
               'divisor': SIGMA_OPEN_HARDCODED},
              open(HERE / 'd77_atten_u071_se_2026-09-08.json', 'w'), indent=2)
    print('\n-> d77_atten_u071_se_2026-09-08.json')


if __name__ == '__main__':
    main()
