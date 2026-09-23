#!/usr/bin/env python3
"""
d103 -- refit the corner law, because three independent legs cannot share an r².

CONTEXT.  An earlier draft of the manuscript stated

    u*(alpha) = 0.511 cos^0.88(alpha/2),   r^2 = 0.9976

"measured over three legs".  The source table reports amplitudes 0.5122, 0.5128
and 0.5071 and exponents 0.921, 0.857 and 0.866 for those legs -- three visibly
different fits -- and then the SAME r^2 = 0.9976 to four decimals for all three.
That value was flagged as implausible in review, and the script behind it could not be found
that produced the table.  Three fits to different data with different parameters
agreeing to the fourth decimal of r^2 is a transcription signature, not a
coincidence.

This refits each leg from its own stored u*(alpha) points and reports what the
three r^2 actually are.

ONE THING THE DATA ALREADY SHOWS.  The third leg's alpha = 15 deg row is stored
with how = 'EDGE' -- the locator hit the end of its speed grid and did not find
an interior optimum.  Whether that row was in the fit changes the answer, so
each leg is fitted twice: with every row, and with located rows only.

Fit: log u* = log A + q log cos(alpha/2), ordinary least squares, and r^2 is
reported on the quantity fitted (log u*) and on u* itself, because the two
differ and the table does not say which it means.

Usage:  python3 d103_corner_law_refit_2026-09-13.py
"""
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

LEGS = [
    ('d65_heading_servo_alpha_law_2026-09-07.json', 'radius-fixed, tau 433'),
    ('d65_heading_servo_alpha_law_side10_2026-09-07.json', 'side-fixed, tau 300'),
    ('d65_heading_servo_alpha_law_side10_tau433_2026-09-07.json', 'side-fixed, tau 433'),
]


def fit(alphas, us):
    x = np.array([math.log(math.cos(math.radians(a) / 2.0)) for a in alphas])
    y = np.log(np.array(us))
    q, lnA = np.polyfit(x, y, 1)
    pred_log = lnA + q * x
    ss_res = float(((y - pred_log) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_log = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    u = np.array(us)
    pu = np.exp(pred_log)
    r2_lin = 1 - float(((u - pu) ** 2).sum()) / float(((u - u.mean()) ** 2).sum())
    rms = 100 * float(np.sqrt((((u - pu) / u) ** 2).mean()))
    return math.exp(lnA), q, r2_log, r2_lin, rms


def main():
    print('=' * 78)
    print('d103 -- corner law refit: what are the three r^2 really?')
    print('=' * 78)
    out = {'legs': []}
    print('\n  %-24s %5s %8s %8s %9s %9s %8s'
          % ('leg', 'rows', 'A', 'q', 'r2(log)', 'r2(u)', 'rms %'))
    for fn, label in LEGS:
        p = HERE / fn
        if not p.exists():
            print('  NOT READ: %s' % fn)
            continue
        d = json.load(open(p, encoding='utf-8'))
        rows = d['rows']
        stored = {k: d.get(k) for k in ('amplitude_A', 'exponent_q', 'r2',
                                        'rms_about_law_pct', 'scale_on_law')}
        for tag, keep in (('all rows', lambda r: True),
                          ('located only', lambda r: r.get('how') != 'EDGE')):
            sel = [r for r in rows if keep(r)]
            if len(sel) < 3:
                continue
            A, q, r2l, r2u, rms = fit([r['alpha_deg'] for r in sel],
                                      [r['u_star'] for r in sel])
            print('  %-24s %5s %8.4f %8.4f %9.5f %9.5f %7.2f%%'
                  % ('%s / %s' % (label, tag), '%d' % len(sel), A, q, r2l, r2u, rms))
            out['legs'].append({'file': fn, 'leg': label, 'subset': tag,
                                'n': len(sel), 'A': A, 'q': q,
                                'r2_log': r2l, 'r2_u': r2u, 'rms_pct': rms,
                                'stored': stored})
        if any(v is not None for v in stored.values()):
            print('  %-24s stored in the file: %s' % ('', stored))
        n_edge = sum(1 for r in rows if r.get('how') == 'EDGE')
        if n_edge:
            print('  %-24s %d row(s) stored as EDGE (locator hit the grid end)'
                  % ('', n_edge))

    # ---- where the manuscript's 0.9976 actually comes from -----------------
    p78 = HERE / 'd78_bowl_relocate_2026-09-08.json'
    if p78.exists():
        d78 = json.load(open(p78, encoding='utf-8'))
        clean = d78['legs']['clean']
        fb, f3 = clean['fit_bowl'], clean['fit_3point']
        print()
        print('  the bowl relocation (d78) -- note it covers ONE leg, not three:')
        print('    file %s' % clean['file'])
        print('    bowl    : q %.4f  A %.4f  r2 %.5f  rms %.2f%%  n %d'
              % (fb['q'], fb['A'], fb['r2'], fb['rms_pct'], fb['n']))
        print('    3-point : q %.4f  A %.4f  r2 %.5f  rms %.2f%%  n %d'
              % (f3['q'], f3['A'], f3['r2'], f3['rms_pct'], f3['n']))
        print('    -> r2 %.5f rounds to %.4f, which is the number the manuscript'
              % (fb['r2'], round(fb['r2'], 4)))
        print('       attributes to all three legs.')
        out['d78_clean_leg'] = {'bowl': fb, 'three_point': f3, 'file': clean['file']}

    loc = [x for x in out['legs'] if x['subset'] == 'located only']
    if len(loc) == 3:
        qs = np.array([x['q'] for x in loc])
        As = np.array([x['A'] for x in loc])
        r2s = [x['r2_log'] for x in loc]
        print('\n  across the three legs, located rows only:')
        print('    q = %.4f +- %.4f   (the earlier draft said 0.88 +- 0.035)'
              % (qs.mean(), qs.std(ddof=1)))
        print('    A = %.4f +- %.4f   (the manuscript says 0.511)'
              % (As.mean(), As.std(ddof=1)))
        print('    r2 = %s' % ', '.join('%.5f' % v for v in r2s))
        same = max(r2s) - min(r2s) < 5e-5
        print('    identical to four decimals: %s' % ('YES' if same else 'NO'))
        print()
        if not same:
            print('    -> the three legs do NOT share an r^2. The manuscript\'s single')
            print('       "r^2 = 0.9976 over three legs" is a transcription of one leg\'s')
            print('       value onto all three, and must be replaced by the three.')
        sd = qs.std(ddof=1)
        z = abs(qs.mean() - 1.0) / (sd / math.sqrt(3)) if sd > 0 else float('inf')
        print('    q vs Prop. 1\'s value of 1: %.2f sigma on the mean of three legs'
              % z)
        out['summary'] = {'q_mean': float(qs.mean()), 'q_sd': float(qs.std(ddof=1)),
                          'A_mean': float(As.mean()), 'A_sd': float(As.std(ddof=1)),
                          'r2': r2s, 'identical': bool(same), 'z_vs_one': float(z)}

    json.dump(out, open(HERE / 'd103_corner_law_refit_2026-09-13.json', 'w'), indent=2)
    print('\n-> d103_corner_law_refit_2026-09-13.json')


if __name__ == '__main__':
    main()
