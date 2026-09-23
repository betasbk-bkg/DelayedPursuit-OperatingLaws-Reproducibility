#!/usr/bin/env python3
"""
d42 -- Term-by-term analysis of the Nav2 experiment blocks E2-E6.

The tau sweep (d40) tests Theorem 2 only as a SUM.  These blocks vary each term
of the delay chain and each geometric knob separately, the way the reference engine's
Table I did (WIN 6-fold, T_s 5-fold).  Each block asks one question:

  E2  block_freq      tau_tot must track T_ctrl/2 = 1/(2 f)         -> slope 1.0
  E3  block_tsim      tau_tot must track T_sim/2 = update_duration/2
                       slope 0.5 (output ZOH only)   vs  1.0 (output + input
                       staleness, the audit's "second T_sim/2")      -> DISCRIMINATES
  E4  block_lookahead v* must be proportional to L at fixed tau_tot -> log slope 1.0
  E5  block_alpha     u* nearly independent of corner angle (d36)   -> flat, NOT cos(a/2)
  E6  block_nulls     u* must NOT move with costmap size, search
                       distance, sample rate, transform tolerance   -> spread ~ 0

Per block the located optimum v_obs is converted to an EFFECTIVE tau_tot by
        tau_tot_eff = U_REF * L / v_obs
using a single reference constant U_REF, and tau_tot_eff is regressed on the
swept term.  The SLOPE is the test; U_REF only shifts the intercept, and is
reported as the fitted intercept's implied composite so the two can be compared.

INPUT   nav2_block_*.json copied out of WSL (/tmp/letg2/exp/block_*.json)
OUTPUT  d42_results_2026-09-05.json
"""
import argparse
import glob
import json
import math
import os

import numpy as np

L = 0.6
U_REF = 0.306          # d38 derived constant for the curvature-command plant


def slope_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = m * x + b
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-30
    return {'slope': float(m), 'intercept': float(b), 'r2': 1 - ss_res / ss_tot,
            'n': int(len(x))}


def usable(rows, key='v_obs'):
    return [r for r in rows if r.get(key) and r.get('how') not in (None, 'EDGE', 'too-few')]


def analyse_freq(rows):
    rows = usable(rows)
    x = [r['T_ctrl_half'] for r in rows]
    y = [U_REF * L / r['v_obs'] for r in rows]                 # tau_tot_eff
    fit = slope_fit(x, y)
    tau = rows[0]['tau_inject'] if rows else None
    out = {'n': len(rows), 'tau_inject': tau, 'fit_tau_tot_vs_Tctrl_half': fit,
           'rows': [{'freq': r['freq'], 'T_ctrl_half': r['T_ctrl_half'],
                     'v_pred': r['v_pred'], 'v_obs': r['v_obs'], 'how': r['how'],
                     'u_obs': r['u_obs'], 'tau_tot_eff': U_REF * L / r['v_obs']}
                    for r in rows]}
    if fit and tau is not None:
        out['implied_other_composite_ms'] = 1e3 * (fit['intercept'] - tau)
    return out


def analyse_tsim(rows):
    rows = usable(rows)
    x = [r['T_sim_half'] for r in rows]
    y = [U_REF * L / r['v_obs'] for r in rows]
    fit = slope_fit(x, y)
    tau = rows[0]['tau_inject'] if rows else None
    out = {'n': len(rows), 'tau_inject': tau, 'fit_tau_tot_vs_Tsim_half': fit,
           'verdict': None,
           'rows': [{'update_duration': r['update_duration'], 'T_sim_half': r['T_sim_half'],
                     'v_pred': r['v_pred'], 'v_obs': r['v_obs'], 'how': r['how'],
                     'u_obs': r['u_obs'], 'tau_tot_eff': U_REF * L / r['v_obs']}
                    for r in rows]}
    if fit:
        s = fit['slope']
        # slope in tau_tot per (T_sim/2): 1.0 = one ZOH term, 2.0 = two (input + output)
        out['verdict'] = ('ONE T_sim/2 term (output ZOH only)' if abs(s - 1.0) < abs(s - 2.0)
                          else 'TWO T_sim/2 terms (output ZOH + input staleness)')
        out['slope_per_Tsim_half'] = s
    if fit and tau is not None:
        out['implied_other_composite_ms'] = 1e3 * (fit['intercept'] - tau)
    return out


def analyse_lookahead(rows):
    rows = usable(rows)
    x = [math.log(r['lookahead']) for r in rows]
    y = [math.log(r['v_obs']) for r in rows]
    fit = slope_fit(x, y)
    us = [r['u_obs'] for r in rows]
    return {'n': len(rows), 'loglog_fit_v_vs_L': fit,
            'u_star_mean': float(np.mean(us)) if us else None,
            'u_star_spread_pct': 100 * (max(us) - min(us)) / np.mean(us) if len(us) > 1 else None,
            'rows': [{'lookahead': r['lookahead'], 'seg_over_L': r['seg_over_L'],
                      'v_pred': r['v_pred'], 'v_obs': r['v_obs'], 'how': r['how'],
                      'u_obs': r['u_obs']} for r in rows]}


def analyse_alpha(rows):
    rows = usable(rows)
    us = [r['u_obs'] for r in rows]
    hc = [r['half_cos'] for r in rows]
    return {'n': len(rows),
            'u_star_spread_pct': 100 * (max(us) - min(us)) / np.mean(us) if len(us) > 1 else None,
            'half_cos_spread_pct': 100 * (max(hc) - min(hc)) / np.mean(hc) if len(hc) > 1 else None,
            'rows': [{'n_sides': r['n_sides'], 'alpha_deg': r['alpha_deg'], 'v_obs': r['v_obs'],
                      'how': r['how'], 'u_obs': r['u_obs'], 'half_cos': r['half_cos'],
                      'u_over_cos': r['u_obs'] / math.cos(math.radians(r['alpha_deg']) / 2)}
                     for r in rows]}


def analyse_nulls(rows):
    rows = usable(rows)
    by = {}
    for r in rows:
        by.setdefault(r['knob'], []).append(r)
    out = {}
    for knob, rs in by.items():
        us = [r['u_obs'] for r in rs]
        out[knob] = {'n': len(rs),
                     'u_star_spread_pct': 100 * (max(us) - min(us)) / np.mean(us) if len(us) > 1 else None,
                     'rows': [{k: v for k, v in r.items() if k != 'curve'} for r in rs]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='directory containing nav2_block_*.json')
    ap.add_argument('--out', default='d42_results_2026-09-05.json')
    a = ap.parse_args()

    handlers = {'freq': analyse_freq, 'tsim': analyse_tsim, 'lookahead': analyse_lookahead,
                'alpha': analyse_alpha, 'nulls': analyse_nulls}
    res = {'U_REF': U_REF, 'L': L, 'blocks': {}}
    for path in sorted(glob.glob(os.path.join(a.dir, 'nav2_block_*.json'))):
        name = os.path.basename(path).replace('nav2_block_', '').split('_')[0].replace('.json', '')
        if name not in handlers:
            continue
        rows = json.load(open(path)).get('rows', [])
        res['blocks'][name] = handlers[name](rows)
        r = res['blocks'][name]
        print(f'\n=== {name}  ({r.get("n", "?")} usable optima) ===')
        rows = r.get('rows')
        if isinstance(rows, list) and rows:
            keys = [k for k in rows[0] if k != 'curve']
            print('   ' + ' '.join(f'{k:>14s}' for k in keys))
            for row in rows:
                cells = []
                for k in keys:
                    v = row.get(k)
                    cells.append(f'{v:14.5g}' if isinstance(v, float) else f'{str(v):>14s}')
                print('   ' + ' '.join(cells))
        for k, v in r.items():
            if k == 'rows':
                continue
            if isinstance(v, dict) and 'rows' in v:          # nulls: nested per-knob
                print(f'   [{k}] n={v.get("n")} u*_spread={v.get("u_star_spread_pct")}')
                for row in v['rows']:
                    print('      ' + ', '.join(f'{kk}={vv:.5g}' if isinstance(vv, float)
                                               else f'{kk}={vv}' for kk, vv in row.items()))
            elif isinstance(v, dict):
                print(f'   {k}: ' + ', '.join(f'{kk}={vv:.4g}' if isinstance(vv, float)
                                              else f'{kk}={vv}' for kk, vv in v.items()))
            else:
                print(f'   {k}: {v}')
    if not res['blocks']:
        print('no nav2_block_*.json found in', a.dir)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
