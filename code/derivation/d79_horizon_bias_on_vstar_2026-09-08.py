#!/usr/bin/env python3
"""
d79 -- does the fixed horizon bias v*, which is what the paper reports?

d73 and d74 asked whether a fixed 65 s window biases the RMSE at low speed, and
the answer is a loud yes: -30% at 0.8 corners.  But that is not the question the
merged paper needs answered.  The paper reports v* and beta, not RMSE, and a bias
that is the SAME at every speed shifts the whole curve without moving its
minimum.  Only a speed-DEPENDENT bias moves v*.

So the decision-relevant quantity is v*(65 s) against v*(800 s) on the same
curves -- and d74 already stored both, so this costs nothing to compute.

There is also a criterion error to fix.  d74 compared |bias| (a property of the
MEAN over MC seeds) against `scatter` (the standard deviation of ONE run).  Those
are not comparable, which is why raising MC from 5 to 20 left the scatter at 20%
instead of halving it: the standard deviation of one run does not shrink with
more seeds, only the uncertainty of the mean does.  This is the same conflation
d76/d77 had to correct.  Here the comparison is made where it belongs -- between
two locations of v*, each with its own seed scatter.

If v* does not move, the coverage rule that the path-following note (not distributed) recommends is
unnecessary and the fixed horizon can stay as it is.  If it does move, the rule
is needed and the size of the move says which speeds have to go.

Usage:  python3 d79_horizon_bias_on_vstar_2026-09-08.py
"""
import glob
import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
_s = importlib.util.spec_from_file_location(
    'd78', HERE / 'd78_bowl_relocate_2026-09-08.py')
d78 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d78)


def main():
    hits = sorted(glob.glob(str(HERE / 'd74_corner_rule_*.json')))
    assert hits, 'run d74 first'
    print('=' * 78)
    print('d79 -- does the fixed horizon move v*, or only the RMSE level?')
    print('=' * 78)

    out = {}
    for path in hits:
        d = json.load(open(path))
        rows = d['rows']
        v = [r['v'] for r in rows]
        short = [r['rmse_short'] for r in rows]
        long_ = [r['rmse_long'] for r in rows]
        tag = pathlib.Path(path).name

        vs, hs, ns = d78.bowl_locate(v, short)
        vl, hl, nl = d78.bowl_locate(v, long_)
        print()
        print('  %s   (MC = %d, horizon %.0f s vs %.0f s)'
              % (tag, d['mc'], d['horizon_s'], d['reference_s']))
        print('    v* from the %3.0f s curve : %.4f  (%s, %d pts)'
              % (d['horizon_s'], vs, hs, ns))
        print('    v* from the %3.0f s curve : %.4f  (%s, %d pts)'
              % (d['reference_s'], vl, hl, nl))
        if vs and vl:
            print('    shift: %+.2f%%' % (100 * (vs / vl - 1)))
        # and what the RMSE bias looked like, for contrast
        b = np.array([r['bias'] for r in rows])
        print('    for contrast, the RMSE bias ran from %+.1f%% to %+.1f%%'
              % (100 * b.min(), 100 * b.max()))
        out[tag] = {'v_short': vs, 'how_short': hs, 'v_long': vl,
                    'how_long': hl,
                    'shift_pct': (100 * (vs / vl - 1)) if (vs and vl) else None,
                    'mc': d['mc']}

    print()
    print('  READING: a horizon effect that moves the RMSE by tens of percent but')
    print('  leaves v* where it was is a LEVEL effect, and the paper reports v*.')
    print('  A rule that deletes grid points would then be paying real data for')
    print('  no protection.  A shift in v* is the opposite: it goes straight into')
    print('  beta and x*, and the rule earns its cost.')

    json.dump(out, open(HERE / 'd79_horizon_bias_2026-09-08.json', 'w'), indent=2)
    print('\n-> d79_horizon_bias_2026-09-08.json')


if __name__ == '__main__':
    main()
