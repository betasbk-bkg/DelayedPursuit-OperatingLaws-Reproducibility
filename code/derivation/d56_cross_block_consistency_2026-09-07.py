#!/usr/bin/env python3
"""
d56 -- Do the blocks agree with each other?

Every block was designed to sweep one thing, but they overlap: several blocks
contain a row at the SAME nominal operating point (tau = 0.20 s, side 5 m,
square, lookahead 0.6 m, 20 Hz, update_duration 0.01 s), measured on different
days, in different stacks, as part of different sweeps.  Those rows were never
meant to be compared, which is exactly what makes them a good test.

Three cross-checks, none of which the blocks were designed to pass:

  1. RAW AGREEMENT at the shared operating point.  Same v, same everything --
     do the RMSE numbers match, and is the cross-block spread larger than the
     within-day reproducibility E7 measured (u* sd 1.52%, RMSE point-to-point
     0.35%)?  If cross-block spread is much larger, something drifts between
     sessions and every between-block comparison is weaker than it looks.

  2. u* AGREEMENT.  u* is supposed to be a property of the loop, so every block
     should report the same value regardless of what it was sweeping.  Pool them
     and compare against the derived 0.306 and against E7's reproducibility.

  3. COMPOSITE DELAY AGREEMENT.  Each fit reports a leftover delay c, but c means
     different things in different blocks: the tau sweep's c absorbs T_ctrl/2 and
     T_sim/2, the freq blocks' c absorbs T_sim/2 only, the tsim block's c absorbs
     T_ctrl/2 only.  Normalised to the SAME definition -- delay beyond
     tau + T_ctrl/2 + T_sim/2 -- do they agree?  This is the check that the
     bookkeeping is consistent, and it is the one I have not done before.

Usage:  python3 d56_cross_block_consistency_2026-09-07.py
"""
import glob
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

# the default configuration every block inherits unless it sweeps that knob
DEF = {'tau': 0.20, 'side': 5.0, 'n_sides': 4, 'lookahead': 0.6,
       'freq': 20.0, 'update_duration': 0.01}


# The blocks this analysis pools are PINNED by name, not globbed.
#
# They used to be globbed with `sorted(...)[-1]`, which silently means "whatever
# sorts last today".  Three of those globs have since drifted, because later
# blocks were written into the same directory with longer names:
#
#   nav2_block_tsim_*       -> ..._tsim_iso_midpoint_2026-09-08   (a DIAGNOSTIC
#                                build with a different integrator convention)
#   nav2_block_replicate_*  -> ..._replicate_iso_2026-09-07
#   nav2_block_alpha_*      -> ..._alpha_iso_2026-09-07
#
# The stored result of this script used ..._tsim_2026-09-05, ..._replicate_
# 2026-09-06 and ..._alpha_2026-09-06, so re-running it after those files
# appeared would have produced a different pooled constant from the same script
# -- and the pooled constant is the paper's headline third-party number.  Pinning
# is therefore part of the result, not tidiness.
PINNED = {
    'nav2_block_replicate_*.json': 'nav2_block_replicate_2026-09-06.json',
    'nav2_block_nulls_*.json': 'nav2_block_nulls_2026-09-06.json',
    'nav2_block_alpha_*.json': 'nav2_block_alpha_2026-09-06.json',
    'nav2_block_tsim_*.json': 'nav2_block_tsim_2026-09-05.json',
    'nav2_block_lookahead_*.json': 'nav2_block_lookahead_2026-09-05.json',
    # the c_extra table's two inputs, pinned for the same reason
    'd48_tsim_interval_*.json': 'd48_tsim_interval_2026-09-06.json',
    'd50_tau_coefficient_*.json': 'd50_tau_coefficient_2026-09-06.json',
    # These seven already resolve uniquely -- their patterns are specific
    # enough that no later file can capture them -- but they are pinned too,
    # so that the whole analysis is deterministic and the record names every
    # file it used rather than only the ones that once drifted.
    'nav2_block_freq_2026*.json': 'nav2_block_freq_2026-09-05.json',
    'nav2_block_freq_iso_2026-09-06.json': 'nav2_block_freq_iso_2026-09-06.json',
    'nav2_block_freq_iso_tau005_*.json': 'nav2_block_freq_iso_tau005_2026-09-06.json',
    'nav2_block_freq_iso_tau020_*.json': 'nav2_block_freq_iso_tau020_2026-09-06.json',
    'd48_freq_iso_interval_*.json': 'd48_freq_iso_interval_2026-09-06.json',
    'd48_freq_iso_tau005_*.json': 'd48_freq_iso_tau005_2026-09-06.json',
    'd48_freq_iso_tau020_*.json': 'd48_freq_iso_tau020_2026-09-06.json',
}


def J(pat):
    """Resolve a block by its pinned name; fall back to the glob only if the
    caller passes a pattern that is not pinned, and say so loudly."""
    name = PINNED.get(pat)
    if name is not None:
        p = HERE / name
        if not p.exists():
            raise SystemExit('pinned block missing: %s' % name)
        return json.load(open(p)), name
    f = sorted(glob.glob(str(HERE / pat)))
    if f:
        print('   !! UNPINNED glob %s -> %s' % (pat, pathlib.Path(f[-1]).name))
        return json.load(open(f[-1])), pathlib.Path(f[-1]).name
    return None, None


def shared_rows():
    """Rows that sit at the default operating point, from any block."""
    out = []
    specs = [
        ('nav2_block_replicate_*.json', lambda r: True, 'E7 replicate'),
        ('nav2_block_nulls_*.json',
         lambda r: (r.get('costmap') == 5.0 or r.get('sample_hz') == 50.0
                    or r.get('tol') == 0.1), 'E6 nulls'),
        ('nav2_block_alpha_*.json', lambda r: r.get('n_sides') == 4, 'E5 alpha'),
        ('nav2_block_tsim_*.json',
         lambda r: abs(r.get('update_duration', 0) - 0.01) < 1e-9, 'E3 tsim'),
        ('nav2_block_lookahead_*.json',
         lambda r: abs(r.get('lookahead', 0) - 0.6) < 1e-9, 'E4 lookahead'),
    ]
    for pat, keep, label in specs:
        blk, name = J(pat)
        if not blk:
            continue
        if abs(blk['meta'].get('tau', -1) - DEF['tau']) > 1e-9:
            continue
        for i, r in enumerate(blk['rows']):
            if r.get('curve') and keep(r):
                tag = next((f'{k}={r[k]}' for k in
                            ('rep', 'knob', 'n_sides', 'update_duration', 'lookahead')
                            if k in r), f'row{i}')
                out.append({'block': label, 'file': name, 'tag': tag,
                            'curve': r['curve'], 'u_obs': r.get('u_obs')})
    return out


def main():
    print('=' * 78)
    print('d56 -- cross-block consistency')
    print('=' * 78)

    # ---------------- 1. raw agreement at the shared operating point --------
    rows = shared_rows()
    print()
    print('  [1] rows sitting at the SAME operating point '
          '(tau 0.20, side 5, square, L 0.6, 20 Hz, dt 0.01)')
    for r in rows:
        print('      %-14s %-22s %d points' % (r['block'], r['tag'], len(r['curve'])))
    print('      -> %d rows from %d different blocks'
          % (len(rows), len({r['block'] for r in rows})))

    by_v = {}
    for r in rows:
        for c in r['curve']:
            by_v.setdefault(round(c['v'], 4), []).append((r['block'], c['rmse']))
    shared_v = {v: lst for v, lst in by_v.items() if len(lst) >= 3}

    print()
    print('  RMSE at each shared speed (only speeds measured by 3+ rows):')
    print('      %8s %4s %12s %12s %10s %10s' %
          ('v', 'n', 'mean RMSE', 'sd', 'spread%', 'blocks'))
    spreads, out_rows = [], []
    for v in sorted(shared_v):
        lst = shared_v[v]
        vals = np.array([x[1] for x in lst])
        nb = len({x[0] for x in lst})
        sp = 100 * (vals.max() - vals.min()) / vals.mean()
        spreads.append(sp)
        out_rows.append({'v': v, 'n': len(vals), 'mean': float(vals.mean()),
                         'sd_pct': float(100 * vals.std(ddof=1) / vals.mean()),
                         'spread_pct': sp, 'n_blocks': nb})
        print('      %8.4f %4d %12.6f %11.2f%% %9.2f%% %10d'
              % (v, len(vals), vals.mean(),
                 100 * vals.std(ddof=1) / vals.mean(), sp, nb))

    if spreads:
        print()
        print('      cross-block spread: %.2f%% to %.2f%% (median %.2f%%)'
              % (min(spreads), max(spreads), float(np.median(spreads))))
        print('      within-day reproducibility measured by E7: 0.35%% sd on one RMSE')
        print('      -> cross-block is %.0fx the within-day figure.'
              % (np.median(spreads) / 0.7))

    # ---------------- 2. u* agreement across every block --------------------
    print()
    print('  [2] u* reported by every block, whatever it was sweeping')
    print('      %-34s %8s %8s %8s' % ('block', 'n rows', 'mean u*', 'spread%'))
    allu, per_block = [], []
    for pat, label in [('nav2_block_freq_2026*.json', 'E2 freq (seg/L 8.3)'),
                       ('nav2_block_tsim_*.json', 'E3 tsim'),
                       ('nav2_block_lookahead_*.json', 'E4 lookahead'),
                       ('nav2_block_alpha_*.json', 'E5 alpha (a<=90 only)'),
                       ('nav2_block_nulls_*.json', 'E6 nulls'),
                       ('nav2_block_replicate_*.json', 'E7 replicate'),
                       ('nav2_block_freq_iso_2026-09-06.json', 'E8 freq_iso t=.10'),
                       ('nav2_block_freq_iso_tau005_*.json', 'E11a freq_iso t=.05'),
                       ('nav2_block_freq_iso_tau020_*.json', 'E11b freq_iso t=.20')]:
        blk, _ = J(pat)
        if not blk:
            continue
        us = [r['u_obs'] for r in blk['rows']
              if r.get('u_obs') and r.get('how') == 'parabola'
              and not (label.startswith('E5') and r.get('alpha_deg', 0) > 90)]
        if not us:
            continue
        us = np.array(us)
        sp = 100 * (us.max() - us.min()) / us.mean()
        per_block.append({'block': label, 'n': len(us), 'mean': float(us.mean()),
                          'spread_pct': sp})
        allu.extend(us.tolist())
        print('      %-34s %8d %8.4f %7.1f%%' % (label, len(us), us.mean(), sp))
    allu = np.array(allu)
    print()
    print('      pooled: n=%d, mean %.4f, sd %.2f%%, range %.4f-%.4f'
          % (len(allu), allu.mean(), 100 * allu.std(ddof=1) / allu.mean(),
             allu.min(), allu.max()))
    print('      derived (d38, zero fitted parameters): 0.306034  ->  mean is %+.1f%%'
          % (100 * (allu.mean() / 0.306034 - 1)))
    print('      E7 within-day reproducibility of u*: 1.52%% sd')

    # ---------------- 3. composite delay, normalised ------------------------
    print()
    print('  [3] the leftover delay c, normalised to ONE definition')
    print('      (delay beyond tau_inject + T_ctrl/2 + T_sim/2; each fit absorbs')
    print('       a different subset, so the raw numbers are not comparable)')
    print('      %-30s %10s %10s %12s' % ('fit', 'c raw', 'absorbs', 'c extra'))
    T_CTRL_HALF_DEF = 0.025      # 20 Hz
    T_SIM_HALF_DEF = 0.005       # update_duration 0.01
    entries = []
    d41 = None
    for pat, label, absorbs in [
            ('d48_freq_iso_tau005_*.json', 'E11a freq_iso t=.05', T_SIM_HALF_DEF),
            ('d48_freq_iso_interval_*.json', 'E8   freq_iso t=.10', T_SIM_HALF_DEF),
            ('d48_freq_iso_tau020_*.json', 'E11b freq_iso t=.20', T_SIM_HALF_DEF),
            ('d48_tsim_interval_*.json', 'E3   tsim', T_CTRL_HALF_DEF)]:
        d, _ = J(pat)
        if not d:
            continue
        c = d['s_free']['c_ms']
        extra = c - 1000 * absorbs
        entries.append({'fit': label, 'c_ms': c, 'absorbs_ms': 1000 * absorbs,
                        'c_extra_ms': extra})
        print('      %-30s %+9.2f %10.1f %+11.2f' % (label, c, 1000 * absorbs, extra))
    d50, _ = J('d50_tau_coefficient_*.json')
    if d50:
        c = d50['c_ms']
        absorbs = T_CTRL_HALF_DEF + T_SIM_HALF_DEF
        extra = c - 1000 * absorbs
        entries.append({'fit': 'E1 tau sweep (PRELIM)', 'c_ms': c,
                        'absorbs_ms': 1000 * absorbs, 'c_extra_ms': extra})
        print('      %-30s %+9.2f %10.1f %+11.2f'
              % ('E1   tau sweep (PRELIM)', c, 1000 * absorbs, extra))

    if entries:
        ex = np.array([e['c_extra_ms'] for e in entries])
        print()
        print('      c_extra: %s ms' % ', '.join('%+.1f' % x for x in ex))
        print('      mean %+.2f ms, sd %.2f ms, range %.2f ms'
              % (ex.mean(), ex.std(ddof=1), ex.max() - ex.min()))
        print('      the c profile half-width measured in d54 is 6-28 ms, so a')
        print('      %.1f ms range across five independent fits is %s.'
              % (ex.max() - ex.min(),
                 'inside that' if (ex.max() - ex.min()) < 28 else 'larger than that'))

    json.dump({'shared_point_rows': [{k: v for k, v in r.items() if k != 'curve'}
                                     for r in rows],
               'shared_v': out_rows, 'u_star_by_block': per_block,
               'u_star_pooled_mean': float(allu.mean()),
               'u_star_pooled_sd_pct': float(100 * allu.std(ddof=1) / allu.mean()),
               'c_extra': entries},
              open(HERE / 'd56_cross_block_consistency_2026-09-07.json', 'w'), indent=2)
    print('\n-> d56_cross_block_consistency_2026-09-07.json')


if __name__ == '__main__':
    main()
