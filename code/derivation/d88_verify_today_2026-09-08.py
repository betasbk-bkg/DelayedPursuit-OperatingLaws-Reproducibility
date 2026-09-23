#!/usr/bin/env python3
"""
d88 -- pull every number claimed in today's documents back out of its JSON.

Today produced a lot of numbers and several of them were wrong the first time --
the SD/SE conflation, the length comparison, the 6x curvature error, the quasi-
static account, the crowd exponent.  Each was caught by measuring rather than
remembering.  Writing the documents from memory would reintroduce exactly that
class of error, so this reads the authoritative value for each claim from the
file that produced it, and prints them side by side with what the documents say.

Anything that disagrees is a document to fix, not a number to re-derive.

Usage:  python3 d88_verify_today_2026-09-08.py
"""
import glob
import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent


def load(pattern):
    hits = sorted(glob.glob(str(HERE / pattern)))
    return json.load(open(hits[-1])) if hits else None


def show(label, got, claimed, tol=0.005, unit=''):
    if got is None:
        print('  %-46s %12s  %12s   NO DATA' % (label, '-', claimed))
        return
    ok = claimed is None or abs(got - claimed) <= tol * max(abs(claimed), 1e-9)
    print('  %-46s %12.4f%s %12s   %s'
          % (label, got, unit, ('%.4f' % claimed) if claimed is not None else '-',
             'ok' if ok else '** DIFFERS **'))


print('=' * 92)
print('d88 -- today\'s claims against the files that produced them')
print('=' * 92)
print('  %-46s %12s %12s   %s' % ('quantity', 'from JSON', 'in docs', ''))
print()

# ---- Nav2: the actuator term ------------------------------------------------
d = load('d48_actuator_2026-09-08.json')
if d:
    print(' Theorem 2, actuator term (d48 on nav2_block_actuator)')
    show('s_act', d['s_free']['s'], 1.0614)
    lo, hi = d.get('noise_propagated', {}).get('p2_5'), \
        d.get('noise_propagated', {}).get('p97_5')
    show('  95% low', lo, 1.039, tol=0.01)
    show('  95% high', hi, 1.102, tol=0.01)
    show('  LOO sd', d.get('loo_sd'), 0.0157, tol=0.05)
    show('  c (ms), free s', d['s_free']['c_ms'], 24.3, tol=0.02, unit=' ms')
    show('  c (ms), s pinned to 1', d['s_pinned']['c_ms'], 26.3, tol=0.02, unit=' ms')

blk = load('nav2_block_actuator_2026-09-08.json')
if blk:
    us = [r['u_obs'] for r in blk['rows']]
    print('  u* across the six rows: %s' % ', '.join('%.4f' % u for u in us))
    print('  spread %.2f%% about the mean %.4f'
          % (100 * (max(us) - min(us)) / (sum(us) / len(us)), sum(us) / len(us)))

# ---- smooth-geometry invariance --------------------------------------------
print()
print(' smooth geometry (deterministic)')
d = load('d84_stadium_sweep_2026-09-08.json')
if d:
    us = [r['u_star'] for r in d['rows'] if r['u_star'] and r['R_over_L'] >= 1.0]
    import statistics as st
    show('stadium u* mean (R/L >= 1)', st.mean(us), 0.509, tol=0.02)
    show('stadium u* sd', st.pstdev(us), 0.004, tol=0.5)
    below = [r for r in d['rows'] if r['u_star'] and r['R_over_L'] < 1.0]
    for r in below:
        show('  R/L = %.2f (below the R>=L bound)' % r['R_over_L'],
             r['u_star'], None)

d = load('d85_crowd_ellipse_2026-09-08.json')
if d:
    print()
    print(' plant layer (d85, MC %d, grid %.2f..%.2f)'
          % (d['mc'], d['speeds'][0], d['speeds'][-1]))
    g = {(r['geometry'], r['layer']): r['u_star'] for r in d['rows']}
    show('circle  deterministic', g.get(('circle', 'deterministic')), 0.5164)
    show('circle  crowd', g.get(('circle', 'crowd')), 0.7206)
    show('ellipse deterministic', g.get(('ellipse', 'deterministic')), 0.5103)
    show('ellipse crowd', g.get(('ellipse', 'crowd')), 0.5252)
    cd, cc = g.get(('circle', 'deterministic')), g.get(('circle', 'crowd'))
    ed, ec = g.get(('ellipse', 'deterministic')), g.get(('ellipse', 'crowd'))
    if None not in (cd, cc, ed, ec):
        print('  vote layer shifts the circle  by %+.1f%%' % (100 * (cc / cd - 1)))
        print('  vote layer shifts the ellipse by %+.1f%%' % (100 * (ec / ed - 1)))
        print('  published circle 0.7109 -> crowd circle is %+.1f%% from it'
              % (100 * (cc / 0.7109 - 1)))
        print('  published ellipse 0.562 -> crowd ellipse is %+.1f%% from it'
              % (100 * (ec / 0.562 - 1)))

d2 = load('d85_crowd_ellipse_mc10_narrowgrid_2026-09-08.json')
if d2:
    g2 = {(r['geometry'], r['layer']): r['u_star'] for r in d2['rows']}
    a, b = g2.get(('ellipse', 'crowd')), g.get(('ellipse', 'crowd'))
    if a and b:
        print('  crowd ellipse across the two runs: %.4f and %.4f  (%.1f%% apart)'
              % (a, b, 100 * abs(a / b - 1)))

# ---- zigzag ----------------------------------------------------------------
d = load('d87_zigzag_harmonised_2026-09-08.json')
if d:
    print()
    print(' zigzag (d87)')
    for r in d['rows']:
        show('%s  seg/L %.2f' % (r['label'], r['seg_over_L']), r['u_star'], None)
    print('  alpha-law prediction for 90 deg: %.4f' % d['u_square_pred'])
    h = [r for r in d['rows'] if r['label'] == 'harmonised'][0]
    s = [r for r in d['rows'] if r['label'] == 'shipped'][0]
    print('  harmonised vs prediction %+.1f%%;  size fix moved the answer %+.1f%%'
          % (100 * (h['u_star'] / d['u_square_pred'] - 1),
             100 * (h['u_star'] / s['u_star'] - 1)))

# ---- isolation / horizon ----------------------------------------------------
d = load('d80_zigzag_isolation_2026-09-08.json')
if d:
    print()
    print(' isolation window (d80)')
    print('  shipped zigzag seg/L %.3f -> u in %.3f .. %.3f, %d grid speeds'
          % (d['seg_over_L'], d['shipped_window_u'][0], d['shipped_window_u'][1],
             d['shipped_grid_isolated']))

d = load('d74_corner_rule_2026-09-08.json')
if d:
    print()
    print(' horizon bias (d74, square)')
    for r in d['rows'][:6]:
        laps = r.get('laps', r['v'] * d['horizon_s'] / d['perimeter'])
        print('    %.2f laps -> bias %+.1f%%' % (laps, 100 * r['bias']))

d = load('d70_invariance_2026-09-08.json')
if d:
    print()
    print(' arc-monotonic invariance (d70)')
    for t in ('circle', 'square'):
        rs = [r for r in d['rows'] if r['traj'] == t]
        dd = [abs(r['v_mono'] / r['v_global'] - 1) for r in rs]
        print('    %-8s mean |dv*| %.2f%%, max %.2f%%, dbeta %+.4f'
              % (t, 100 * sum(dd) / len(dd), 100 * max(dd),
                 d['%s_beta_mono' % t] - d['%s_beta_global' % t]))

# ---- audit standard errors --------------------------------------------------
d = load('d76_mc_standard_errors_2026-09-08.json')
if d:
    print()
    print(' audit denominators (d76, SE of the mean over 8 blocks)')
    for k, v in d['results'].items():
        print('    %-22s mean %.4f  sd %.4f  se %.4f  z %+.2f'
              % (k, v['mean'], v.get('sd', float('nan')), v['se'], v['z']))
d = load('d77_atten_u071_se_2026-09-08.json')
if d:
    print('    %-22s mean %.5f  sd %.5f  se %.5f  z %+.2f'
          % ('atten_u071', d['mean'], d.get('sd', float('nan')), d['se'], d['z']))
    print('      corrected reference 8.60/10.9446 = %.5f -> %+.2f SE'
          % (d['corrected_reference'],
             (d['mean'] - d['corrected_reference']) / d['se']))

print()
print('=' * 92)
