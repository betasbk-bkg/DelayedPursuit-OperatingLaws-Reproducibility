#!/usr/bin/env python3
"""
d104 -- the freq_iso block, run twice: what reproduces, and what the finer grid changes.

The 2026-09-13 re-run used 17 speeds per control frequency; the original
2026-09-06 run used 9, and those 9 are an exact subset of the 17.  That lets two
effects be separated that a single comparison would mix:

  SESSION   same speed, same frequency, two runs a week apart -> RMSE ratio at the
            shared speeds, and u* located on the SAME 9-point grid in both runs;
  GRID      same run, 9-point subset vs all 17 points -> u* moves by the grid alone.

Each u* is located with all three estimators of d102 (imported, not copied), so
the comparison is made per estimator.

CAVEAT carried, not hidden: `track_unknown_space = False` was added to the local
costmap between the two runs, so SESSION mixes day-to-day variation with that
configuration change until a costmap probe separates them.

Usage (WSL):  python3 d104_freq_iso_rerun_compare.py ORIGINAL.json RERUN.json [OUT.json]
"""
import json
import sys

import numpy as np

import importlib.util  # noqa: E402
import os  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(HERE, 'nav2_block_freq_iso_2026-09-06.json')
RERUN = os.path.join(HERE, 'nav2_block_freq_iso_rerun17_2026-09-13.json')
OUT = os.path.join(HERE, 'd104_freq_iso_rerun_compare_2026-09-13.json')

# d102 sits beside this script (the file name carries hyphens, so it is loaded by
# path; importing it runs nothing).  An absolute path here would break the script
# on every machine but the one it was written on -- the defect the package's
# derivation README already documents for 27 older scripts.
_spec = importlib.util.spec_from_file_location(
    'd102', os.path.join(HERE, 'd102_relocate_nav2_2026-09-13.py'))
_d102 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d102)
three_point, bowl, symmetric = _d102.three_point, _d102.bowl, _d102.symmetric


def curve(row):
    pts = sorted((c['v'], c['rmse']) for c in row['curve']
                 if c.get('rmse') is not None and c['rmse'] == c['rmse'])
    return [p[0] for p in pts], [p[1] for p in pts]


def locate(vs, rs, k):
    v3, h3 = three_point(vs, rs)
    vb, hb = bowl(vs, rs)
    vsym, hs, band = symmetric(vs, rs)
    return {'3pt': v3 * k if h3 == 'parabola' else None,
            'bowl': vb * k if vb else None,
            'sym': vsym * k if vsym else None}


def subset(vs, rs, keep):
    sel = [i for i, v in enumerate(vs) if any(abs(v / w - 1) < 1e-9 for w in keep)]
    return [vs[i] for i in sel], [rs[i] for i in sel]


def main():
    # defaults are the two files shipped beside this script; arguments override
    orig = json.load(open(sys.argv[1] if len(sys.argv) > 1 else ORIGINAL))
    new = json.load(open(sys.argv[2] if len(sys.argv) > 2 else RERUN))
    out_path = sys.argv[3] if len(sys.argv) > 3 else OUT
    by_o = {round(r['freq'], 3): r for r in orig['rows']}
    out = {'rows': []}
    all_ratio = []
    print('=' * 94)
    print('d104 -- freq_iso: session reproducibility and grid refinement, per estimator')
    print('=' * 94)
    hdr = '%7s  %-5s %8s %8s %8s | %8s %8s' % ('freq', 'est', 'orig9', 'new9', 'SESSION',
                                             'new17', 'GRID')
    for r in new['rows']:
        f = round(r['freq'], 3)
        o = by_o.get(f)
        if o is None:
            continue
        k = r['u_obs'] / r['v_obs']
        ko = o['u_obs'] / o['v_obs']
        if abs(k / ko - 1) > 1e-9:
            print('  freq %.3f: tau_tot/L differs between runs (%.6f vs %.6f)' % (f, k, ko))
        vo, ro = curve(o)
        vn, rn = curve(r)
        vn9, rn9 = subset(vn, rn, vo)
        shared = len(vn9)
        ratio = [100 * (b / a - 1) for a, b in zip(ro, rn9)] if shared == len(vo) else []
        all_ratio += ratio
        lo, l9, l17 = locate(vo, ro, ko), locate(vn9, rn9, k), locate(vn, rn, k)
        print('\n  freq %.3f Hz: %d original speeds, %d re-run speeds, %d shared'
              % (f, len(vo), len(vn), shared))
        if ratio:
            print('    RMSE at shared speeds, re-run vs original: mean %+.2f%%, sd %.2f%%, '
                  'max |%.2f%%|' % (np.mean(ratio), np.std(ratio, ddof=1),
                                    max(abs(x) for x in ratio)))
        print('   ' + hdr)
        rec = {'freq': f, 'n_orig': len(vo), 'n_new': len(vn), 'shared': shared,
               'rmse_ratio_pct': ratio, 'u': {}}
        for e in ('3pt', 'sym', 'bowl'):
            a, b, c = lo[e], l9[e], l17[e]
            ses = 100 * (b / a - 1) if a and b else None
            grd = 100 * (c / b - 1) if b and c else None
            fmt = lambda x: ('%8.4f' % x) if x is not None else '    none'
            fmp = lambda x: ('%+7.1f%%' % x) if x is not None else '    none'
            print('   %7.3f  %-5s %s %s %s | %s %s' % (f, e, fmt(a), fmt(b), fmp(ses),
                                                     fmt(c), fmp(grd)))
            rec['u'][e] = {'orig9': a, 'new9': b, 'new17': c,
                           'session_pct': ses, 'grid_pct': grd}
        out['rows'].append(rec)
    if all_ratio:
        print('\n  all shared speeds: n = %d, mean %+.2f%%, sd %.2f%%'
              % (len(all_ratio), np.mean(all_ratio), np.std(all_ratio, ddof=1)))
        out['rmse_ratio_all'] = {'n': len(all_ratio), 'mean': float(np.mean(all_ratio)),
                                 'sd': float(np.std(all_ratio, ddof=1))}
    for e in ('3pt', 'sym', 'bowl'):
        for key in ('orig9', 'new9', 'new17'):
            vals = [x['u'][e][key] for x in out['rows'] if x['u'][e][key] is not None]
            if vals:
                out.setdefault('block_mean', {}).setdefault(e, {})[key] = float(np.mean(vals))
    print('\n  block mean u* over the frequencies present (per estimator):')
    for e, d in out.get('block_mean', {}).items():
        print('    %-5s ' % e + '  '.join('%s %.4f' % kv for kv in d.items()))
    print('\n  CAVEAT: track_unknown_space changed between runs; SESSION is not pure noise.')
    if out_path:
        json.dump(out, open(out_path, 'w'), indent=2)
        print('-> %s' % out_path)


if __name__ == '__main__':
    main()
