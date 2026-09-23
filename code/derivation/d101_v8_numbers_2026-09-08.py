#!/usr/bin/env python3
"""
d101 -- pull every number v8 will state, from the file that produced it.

P34 rule 1: a sentence about what a program computed is verified by
reading that program, not by remembering it.  P39: a value whose source cannot
be read is reported NOT READ, never inferred.  This script is the gate between
the result files and the manuscript: v8 states only what this prints.

Writing the first version of this script already earned its cost.  Four numbers
carried in the working summary did not survive contact with the files:

  * the Nav2 RPP constant was carried as "0.2997-0.3134, mean +0.4%".  That is a
    subset.  The cross-block pooled value (d56, six blocks) is 0.3029 +- 5.1%,
    i.e. -1.0% against the derived 0.30603.
  * the stadium margin was carried as 0.5083 +- 0.0034 with no condition.  It
    holds for R/L >= 1; the R/L = 0.5 row sits at 0.4529 and belongs to the
    corner regime, so the filter is part of the claim.
  * two d85 runs exist.  The MC=10 narrow-grid run put the crowd circle OUTSIDE
    its speed grid (the grid was centred on the deterministic optimum); it is
    superseded by the MC=20 run over 1.0-3.5 m/s, and only the latter is read.
  * the plant-hold coefficient has two values by integrator convention
    (upstream 1.2626, midpoint 0.7191).  Neither alone is the result; that they
    bracket 1 is.

Each of those is encoded below as a source path plus a filter, so that the
selection is auditable rather than remembered.

Usage:  python3 d101_v8_numbers_2026-09-08.py
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
OUT, MISSING = {}, []


def load(name):
    p = HERE / name
    if not p.exists():
        MISSING.append(name)
        return None
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception as e:                                       # noqa: BLE001
        MISSING.append('%s (%s)' % (name, e))
        return None


def show(label, value, source, note=''):
    OUT[label] = {'value': value, 'source': source, 'note': note}
    print('  %-44s %-26s %s' % (label, value, source))
    if note:
        print('  %-44s   %s' % ('', note))


print('=' * 78)
print('d101 -- every number v8 states, read from its source')
print('=' * 78)

# ------------------------------------------------- Theorem 1: derived constants
print('\n[Theorem 1 -- derived, zero free parameters (d38)]')
d = load('d38_results_2026-08-29.json')
if d:
    hs = d['heading_servo']['u_star_derived']
    cc = d['curvature_command']['u_star_derived']
    lh = d['stability_limits']['heading_servo_u']
    lc = d['stability_limits']['curvature_command_u']
    show('u* heading-servo', '%.6f' % hs, 'd38', 'Theorem 1 constant 1/2, error %.2f%%'
         % (100 * (hs / 0.5 - 1)))
    show('u* curvature-command', '%.6f' % cc, 'd38')
    show('stability limit heading-servo', '%.6f' % lh, 'd38', 'pi/2')
    show('stability limit curvature-command', '%.6f' % lc, 'd38')
    show('ratio of the two stability limits', '%.3f' % (lh / lc), 'd38',
         'the two classes differ by this factor -- one derivation, two systems')

# ------------------------------------------ Theorem 1 on third-party code (RPP)
print('\n[Theorem 1 -- Nav2 RPP, unmodified deployed controller]')
d = load('d56_cross_block_consistency_2026-09-07.json')
if d:
    m = d['u_star_pooled_mean']
    sd = d['u_star_pooled_sd_pct']
    show('RPP pooled u* (6 blocks)', '%.4f +- %.1f%%' % (m, sd), 'd56',
         'vs derived 0.306034: %+.1f%%' % (100 * (m / 0.3060336118512221 - 1)))
    for b in d['u_star_by_block']:
        show('  block %s' % b['block'], '%.4f (n=%d, spread %.1f%%)'
             % (b['mean'], b['n'], b['spread_pct']), 'd56')

# ------------------------------------------------------ Theorem 2 coefficients
print('\n[Theorem 2 -- each term measured by varying it alone, on Nav2]')
for label, fn, key in (
        ('s_tau  (pure transport delay)', 'd62_tau_iso_coefficient_2026-09-07.json', None),
        ('s_ctrl (controller hold)', 'd48_freq_iso_interval_2026-09-06.json', 's_free'),
        ('s_act  (first-order actuator lag)', 'd48_actuator_2026-09-08.json', 's_free'),
        ('s_sim  (plant hold, upstream)', 'd48_tsim_iso_2026-09-07.json', 's_free'),
        ('s_sim  (plant hold, midpoint)', 'd48_tsim_iso_midpoint_2026-09-08.json', 's_free')):
    d = load(fn)
    if d is None:
        print('  %-44s NOT READ' % label)
        continue
    if key is None:
        s = d['best']['s_tau']
        ci = d['noise_ci']
        show(label, '%.4f [%.4f, %.4f]' % (s, ci[0], ci[1]), fn.split('_2026')[0])
    else:
        s = d[key]['s']
        iv = d['profile_s']['interval']
        one = d['profile_s']['contains_one']
        show(label, '%.4f [%.2f, %.2f]' % (s, iv[0], iv[1]), fn.split('_2026')[0],
             'interval %s 1.000' % ('contains' if one else 'EXCLUDES'))

# ----------------------------------------------------------------- bicycle
print('\n[Ackermann steering: the bound, not the kinematics, is what differs]')
d = load('nav2_bicycle_2026-09-08.json')
if d:
    for r in d.get('rows', []):
        v, tt = r.get('v_obs'), (r.get('tau_tot') or r.get('tau_tot_nominal'))
        L = r.get('L_nominal', 0.6)
        u = (v * tt / L) if (v and tt) else None
        show('max_steer %.2f rad (kappa_max %s)'
             % (r['max_steer'], ('%.3f' % r['kappa_max']) if r.get('kappa_max') else 'inf'),
             ('u* %.4f' % u) if u else str(r.get('how')), 'nav2_bicycle', r.get('how', ''))

# ---------------------------------------------------------------- graceful
print('\n[A second controller of the same class: the (A,B) family]')
d = load('d97_graceful_class_2026-09-08.json')
if d:
    show('d97 self-test: (A,B)=(2,0) vs d38', '%.2e' % d['self_test'], 'd97',
         'the family reduces to the RPP integrator exactly')
    a = d['rpp_anchor']
    show('d97 RPP anchor u* / u_c', '%.4f / %.4f' % (a['u_star'], a['u_c']), 'd97',
         'published 0.306034 / 0.520494')
    dec = d['decomposition_A6_B0']
    show('decomposition (A,B)=(6,0)', '%.4f [%s]' % (dec['u_star'], dec['how']), 'd97',
         "graceful's gain WITHOUT its tangent feedforward: the optimum survives")
d = load('d99_graceful_nonlinear_2026-09-08.json')
if d:
    show('d99 acceptance: RPP through the exact engine', '%.4f' % d['rpp_u_star'], 'd99',
         'known 0.3060; the engine is only allowed to predict after reproducing this')
    for r in d['rows']:
        show('d99 prediction k_phi %.1f' % r['k_phi'],
             ('u* %.4f' % r['u_star']) if r.get('how') == 'bowl' else 'no interior optimum',
             'd99')
d = load('d100_graceful_verdict_2026-09-08.json')
if d:
    hit = 0
    tot = 0
    for r in d['rows']:
        v, pr = r['verdict'], (r.get('pred') or {})
        if r.get('beta') != 0.0:
            continue
        got = v.get('call') == 'optimum'
        if pr:
            tot += 1
            hit += 1 if got == pr['exists'] else 0
        txt = ('u* %.4f [%.4f, %.4f]' % (v['u_star'], v['u_band'][0], v['u_band'][1])
               if got else 'no interior optimum')
        show('measured k_phi %.1f (A %.0f, B %.0f)' % (r['k_phi'], r['A'], r['B']),
             txt, 'd100', 'predicted %s' % (('%.4f' % pr['u_exact']) if pr.get('u_exact')
                                            else 'none'))
    show('pre-registered binary calls correct', '%d of %d' % (hit, tot), 'd100')

# ----------------------------------------------------------------- Stanley
print('\n[The boundary: a law with no look-ahead]')
d = load('d94_stanley_2026-09-08.json')
if d:
    rows = d['rows']
    loc = [r for r in rows if r.get('v_star') and r.get('how') == 'bowl']
    show('Stanley (k x tau grid): located optima', '%d of %d' % (len(loc), len(rows)),
         'd94', 'RMSE is monotone in v at every cell')

# ---------------------------------------------------------- plant separation
print('\n[Plant-layer separation: the derived constant and the published one]')
d = load('d85_crowd_ellipse_2026-09-08.json')
if d:
    show('d85 run used', 'MC=%d, %d speeds %.1f-%.1f' % (d['mc'], len(d['speeds']),
         d['speeds'][0], d['speeds'][-1]), 'd85',
         'the MC=10 narrow-grid run is superseded: it put the crowd circle outside its grid')
    got = {}
    for r in d['rows']:
        got[(r['geometry'], r['layer'])] = r
        show('%s / %s' % (r['geometry'], r['layer']), 'u* %.4f (%s, %d pts)'
             % (r['u_star'], r['how'], r['n_pts']), 'd85')
    for g, pub in (('circle', 0.7109), ('ellipse', 0.562)):
        det = got.get((g, 'deterministic'))
        cr = got.get((g, 'crowd'))
        if det and cr:
            show('%s: vote-layer shift' % g, '%+.1f%%'
                 % (100 * (cr['u_star'] / det['u_star'] - 1)), 'd85',
                 'crowd %.4f vs published %.4f = %+.1f%%'
                 % (cr['u_star'], pub, 100 * (cr['u_star'] / pub - 1)))
d = load('d90_vote_shift_tau_2026-09-08.json')
if d:
    for r in d['rows']:
        show('shift at tau %d ms' % r['tau_ms'], '%.1f%% (turn/vote %.3f cells)'
             % (r['shift_pct'], r['turn_per_vote_cells']), 'd90',
             'u_det %.4f -> u_crowd %.4f' % (r['u_det'], r['u_crowd']))
d = load('d95_lockin_period_2026-09-08.json')
if d:
    show('lock-in: k_peak ~ v^s', '%.4f' % d['slope'], 'd95', 'a lock-in requires -1')
    show('lock-in: k_obs / k_pred', '%.4f' % d['ratio_mean'], 'd95',
         "measured on the main paper's own recorded autocorrelation data")

# ---------------------------------------------------------------- geometry
print('\n[The smooth-geometry margin, and where it stops]')
d = load('d84_stadium_sweep_2026-09-08.json')
if d:
    rows = d['rows']
    keep = [r for r in rows if r['R_over_L'] >= 1.0]
    us = [r['u_star'] for r in keep]
    show('stadium, R/L >= 1', '%.4f +- %.4f (n=%d)'
         % (float(np.mean(us)), float(np.std(us, ddof=1)), len(us)), 'd84',
         'R/L 1-6 x S/L 2.5-20')
    low = [r for r in rows if r['R_over_L'] < 1.0]
    for r in low:
        show('stadium, R/L = %.1f' % r['R_over_L'], 'u* %.4f' % r['u_star'], 'd84',
             'below the effective boundary R/L = 1: the corner regime begins')

print('\n' + '=' * 78)
if MISSING:
    print('NOT READ -- these are findings, not omissions (P39):')
    for m in MISSING:
        print('   ', m)
else:
    print('every source file read')
print('%d quantities pulled' % len(OUT))
json.dump({'quantities': OUT, 'not_read': MISSING},
          open(HERE / 'd101_v8_numbers_2026-09-08.json', 'w'), indent=2)
print('-> d101_v8_numbers_2026-09-08.json')
