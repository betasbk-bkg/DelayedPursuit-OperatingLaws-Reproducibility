#!/usr/bin/env python3
"""
d93 -- predict the coefficients the EXPERIMENT measures, not the ones the
       dominant root implies.

d51, d91 and d92 all ask the same question in the same way: solve the exact delay
chain for its dominant root, ask what single delay reproduces that root, and
compare against the measured coefficient.  On the actuator term that worked
(predicted 1.0451 against measured 1.0614).  On the controller ZOH it did not
(1.0287 against 1.0814, 35%), and on the pure transport delay it cannot even
produce the right SIGN -- a pure delay has no truncation error at all, yet the
measurement is 0.9632.

d51 flagged the reason in its own docstring and it has been sitting there since:

    "it is a linear-stability argument: it matches the dominant root, whereas the
     experiment locates the minimum of RMSE(v)"

Those are different functionals.  The experiment sweeps v, integrates the corner
transit, and takes the argmin.  So compute THAT, with the exact chain in place of
Theorem 2's single-delay surrogate, and the prediction becomes commensurate with
the measurement for the first time.

METHOD.  d38 integrates the linearised corner transit in arclength with one pure
delay.  Here the delayed signal is instead the output of the real chain

    pure delay u_tau  ->  ZOH u_ctrl  ->  ZOH u_sim  ->  first-order lag u_act

(order is irrelevant: the chain is linear).  Sweeping v scales every u_i
together, so J(v) is integrated on a speed grid and its argmin located.  Then the
coefficient the experiment would report is the s that makes

    u = v* (tau + s T_ctrl/2 + T_sim/2 + T_act) / L

constant across the block's rows -- exactly the construction d48 fits.

SELF-TEST FIRST.  With u_ctrl = u_sim = u_act = 0 this must reproduce d38's J(u)
to the integrator's accuracy.  A chain integrator that fails that produces a
plausible-looking set of coefficients from nothing, which is the same trap the
stadium geometry set (d84) and the Lambert root-finder (d80) each sprang once
today.

Usage:  python3 d93_exact_chain_argmin_2026-09-08.py
"""
import importlib.util
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
L = 0.6

_s = importlib.util.spec_from_file_location(
    'd38', HERE / 'd38_corner_energy_two_plants_2026-08-29.py')
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def g_preview(x, xc):
    return d38.g_preview(x, xc)


def transit_energy_chain(u_tau, u_ctrl, u_sim, u_act,
                         plant='curvature_command', xc=4.0, xend=40.0, h=2.0e-3):
    """J = int e^2 dx for the linearised corner transit, with the REAL chain.

    The chain acts on the signals the controller sees.  A ZOH of period p holds
    the value sampled at the last multiple of p; a first-order lag of constant
    a integrates dz/dx = (in - z)/a.  Setting all three to zero reduces this to
    d38's single pure delay, which is what the self-test checks.
    """
    n = int(xend / h)
    nd = int(round(u_tau / h))
    e = np.zeros(n + 1)
    ep = np.zeros(n + 1)
    ic = int(round(xc / h))

    # ZOH state: last sampled value and the index of the last sample
    hold_e = hold_ep = hold_g = 0.0
    last1 = last2 = -1
    lag_e = lag_ep = lag_g = 0.0
    n1 = max(int(round(u_ctrl / h)), 0)
    n2 = max(int(round(u_sim / h)), 0)

    for i in range(n):
        x = i * h
        j = i - nd
        ed = e[j] if j >= 0 else 0.0
        epd = ep[j] if j >= 0 else 0.0
        gd = g_preview(x - u_tau, xc) if x - u_tau > 0 else 0.0

        # ZOH 1 (controller period)
        if n1 > 0:
            k = i // n1
            if k != last1:
                last1, hold_e, hold_ep, hold_g = k, ed, epd, gd
            ed, epd, gd = hold_e, hold_ep, hold_g
        # ZOH 2 (plant period)
        if n2 > 0:
            k = i // n2
            if k != last2:
                last2 = k
                z_e, z_ep, z_g = ed, epd, gd
                transit_energy_chain._h2 = (z_e, z_ep, z_g)
            ed, epd, gd = getattr(transit_energy_chain, '_h2', (ed, epd, gd))
        # first-order lag
        if u_act > 0:
            lag_e += h * (ed - lag_e) / u_act
            lag_ep += h * (epd - lag_ep) / u_act
            lag_g += h * (gd - lag_g) / u_act
            ed, epd, gd = lag_e, lag_ep, lag_g

        if plant == 'heading_servo':
            step = (1.0 if (x - u_tau) > xc else 0.0) - (1.0 if x > xc else 0.0)
            e[i + 1] = e[i] + h * (-ed + gd + step)
        else:
            epp = 2.0 * (-ed - epd + gd)
            ep[i + 1] = ep[i] + h * epp
            if i == ic:
                ep[i + 1] -= 1.0
            e[i + 1] = e[i] + h * ep[i + 1]
    return float(np.sum(e * e) * h)


def self_test():
    """Zero chain must reproduce d38."""
    worst = 0.0
    for u in (0.15, 0.25, 0.306, 0.40):
        a = d38.transit_energy(u, 'curvature_command', h=1.0e-3)
        b = transit_energy_chain(u, 0.0, 0.0, 0.0, h=1.0e-3, xend=60.0)
        worst = max(worst, abs(b / a - 1))
    return worst


def locate(vs, js):
    js = np.asarray(js, float)
    i = int(np.argmin(js))
    if i in (0, len(js) - 1):
        return float(vs[i]), 'EDGE'
    c = np.polyfit(vs[i - 1:i + 2], js[i - 1:i + 2], 2)
    if c[0] <= 0:
        return float(vs[i]), 'grid'
    v = -c[1] / (2 * c[0])
    return (float(v), 'parabola') if vs[i - 1] <= v <= vs[i + 1] else (float(vs[i]), 'grid')


def predicted_vstar(tau, T_ctrl, T_sim, T_act, vlo, vhi, npt=41, h=2.0e-3):
    vs = np.linspace(vlo, vhi, npt)
    js = [transit_energy_chain(v * tau / L, v * T_ctrl / L, v * T_sim / L,
                               v * T_act / L, h=h) for v in vs]
    return locate(vs, js)


def fit_s(rows, term_key, tau, T_sim_fixed, T_act_fixed=0.0):
    """The s that makes u = v*(tau + s*term + rest)/L constant across rows."""
    def spread(s):
        us = []
        for r in rows:
            T = r[term_key]
            if term_key == 'T_ctrl_half':
                tt = tau + 2.0 * s * T / 2.0 + T_sim_fixed / 2.0 + T_act_fixed
            elif term_key == 'actuator_tau':
                tt = tau + 0.05 / 2.0 + T_sim_fixed / 2.0 + s * T
            else:
                tt = s * T + 0.05 / 2.0 + T_sim_fixed / 2.0
            us.append(r['v_pred'] * tt / L)
        return float(np.std(us, ddof=1) / np.mean(us))
    ss = np.linspace(0.6, 1.6, 501)
    vals = [spread(s) for s in ss]
    return float(ss[int(np.argmin(vals))]), float(min(vals))


def main():
    print('=' * 78)
    print('d93 -- coefficients predicted from the argmin of the EXACT chain')
    print('=' * 78)
    w = self_test()
    print('  self-test (zero chain vs d38): max relative error %.2e' % w)
    assert w < 0.02, 'chain integrator does not reproduce d38'
    print()

    out = {'self_test': w, 'blocks': {}}

    # ---- (a) T_ctrl/2 -------------------------------------------------------
    blk = json.load(open(HERE / 'nav2_block_freq_iso_2026-09-06.json'))
    rows = [r for r in blk['rows'] if r.get('v_obs')]
    tau = rows[0]['tau_inject']
    print('  (a) T_ctrl/2 block: tau %.2f s, %d rows' % (tau, len(rows)))
    print('      %7s %10s %11s %11s' % ('freq', 'v* meas', 'v* pred', 'diff'))
    pr = []
    for r in rows:
        T = 1.0 / r['freq']
        v, how = predicted_vstar(tau, T, 0.01, 0.0,
                                 0.4 * r['v_obs'], 1.8 * r['v_obs'])
        pr.append({'freq': r['freq'], 'T_ctrl_half': T / 2.0,
                   'v_obs': r['v_obs'], 'v_pred': v, 'how': how})
        print('      %7.2f %10.4f %11.4f %10.1f%%'
              % (r['freq'], r['v_obs'], v, 100 * (v / r['v_obs'] - 1)))
    s_pred, sp = fit_s(pr, 'T_ctrl_half', tau, 0.01)
    print('      -> s_ctrl predicted from the argmin: %.4f  (residual spread %.2f%%)'
          % (s_pred, 100 * sp))
    print('         measured 1.0814 [1.054, 1.124];  root-matching gave 1.0287')
    out['blocks']['T_ctrl'] = {'rows': pr, 's_pred': s_pred, 'spread': sp,
                               's_measured': 1.0814, 's_root': 1.0287}

    # ---- (b) tau ------------------------------------------------------------
    p = HERE / 'nav2_block_tau_iso_2026-09-07.json'
    if p.exists():
        blk = json.load(open(p))
        rows = [r for r in blk['rows'] if r.get('v_obs')]
        print()
        print('  (b) tau block: %d rows' % len(rows))
        print('      %7s %10s %11s %11s' % ('tau', 'v* meas', 'v* pred', 'diff'))
        pr = []
        for r in rows:
            t = r['tau_inject']
            v, how = predicted_vstar(t, 0.05, 0.01, 0.0,
                                     0.4 * r['v_obs'], 1.8 * r['v_obs'])
            pr.append({'tau_inject': t, 'v_obs': r['v_obs'], 'v_pred': v})
            print('      %7.2f %10.4f %11.4f %10.1f%%'
                  % (t, r['v_obs'], v, 100 * (v / r['v_obs'] - 1)))
        s_pred, sp = fit_s(pr, 'tau_inject', 0.0, 0.01)
        print('      -> s_tau predicted from the argmin: %.4f  (spread %.2f%%)'
              % (s_pred, 100 * sp))
        print('         measured 0.9632 [0.942, 0.990];  a pure delay has NO')
        print('         truncation error, so root-matching predicts exactly 1')
        out['blocks']['tau'] = {'rows': pr, 's_pred': s_pred, 'spread': sp,
                                's_measured': 0.9632, 's_root': 1.0}

    json.dump(out, open(HERE / 'd93_exact_chain_argmin_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d93_exact_chain_argmin_2026-09-08.json')


if __name__ == '__main__':
    main()
