#!/usr/bin/env python3
"""
d92 -- redo d51's truncation test on T_ctrl/2 with the FULL chain.

d51 asked whether Theorem 2's first-moment truncation explains the measured
T_ctrl/2 coefficient of 1.0814.  It predicted 1.028 and so accounted for 2.8 of
the 8.1 points, leaving 5.3 unexplained -- and that residual has sat in
NAV2_STATUS and PLATFORM_TERM ever since as "the controller source has no
period-proportional term, so ~5% has no name".

But d51 solved a chain of TWO elements: the injected delay and the controller
ZOH.  The real chain has three -- there is also the plant's own zero-order hold
at T_sim = 0.01 s, which d51 folded into the constant rather than into the
kernel.  d91 has just shown, on the actuator block, that solving the whole chain
matters: with all four elements in the kernel the truncation predicted 1.0451
against a measured 1.0614, i.e. it accounted for the entire excess, where the
two-element version would have predicted less.

So the same question is put again with T_sim inside the kernel:

    K(lam) = exp(-lam u_tau) * ZOH(u_ctrl) * ZOH(u_sim)

and the implied coefficient on the controller term, holding the others at
Theorem 2's 1, is

    s_ctrl_pred = 2 * (u_eff - u_tau - u_sim/2) / u_ctrl

If this lands near 1.081 the residual closes.  If it stays near 1.028 the
residual is real and the two-element treatment was not the reason.

Stated before running: d91's lesson suggests the third element helps, but the
plant ZOH is 0.01 s against the controller's 0.05, so its second moment is 25x
smaller -- the honest expectation is that it moves the prediction only slightly.

Usage:  python3 d92_tctrl_full_chain_2026-09-08.py
"""
import cmath
import glob
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
L = 0.6
T_SIM = 0.01


def K_chain(lam, u_tau, u_ctrl, u_sim, exact=True, with_sim=True):
    if not exact:
        return cmath.exp(-lam * (u_tau + 0.5 * u_ctrl
                                 + (0.5 * u_sim if with_sim else 0.0)))
    out = cmath.exp(-lam * u_tau)
    els = (u_ctrl, u_sim) if with_sim else (u_ctrl,)
    for uT in els:
        out *= (1.0 if abs(lam * uT) < 1e-12
                else (1.0 - cmath.exp(-lam * uT)) / (lam * uT))
    return out


def char(lam, u_tau, u_ctrl, u_sim, with_sim):
    return lam * lam + 2.0 * K_chain(lam, u_tau, u_ctrl, u_sim,
                                     True, with_sim) * (lam + 1.0)


def dominant_root(u_tau, u_ctrl, u_sim, with_sim):
    best = None
    for x0 in np.linspace(-3.0, 1.0, 20):
        for y0 in np.linspace(0.05, 8.0, 30):
            z = complex(x0, y0)
            good = True
            for _ in range(160):
                f = char(z, u_tau, u_ctrl, u_sim, with_sim)
                h = 1e-7
                fp = (char(z + h, u_tau, u_ctrl, u_sim, with_sim)
                      - char(z - h, u_tau, u_ctrl, u_sim, with_sim)) / (2 * h)
                if abs(fp) < 1e-16:
                    good = False
                    break
                zn = z - f / fp
                if not (math.isfinite(zn.real) and math.isfinite(zn.imag)):
                    good = False
                    break
                if abs(zn - z) < 1e-13:
                    z = zn
                    break
                z = zn
            if good and abs(char(z, u_tau, u_ctrl, u_sim, with_sim)) < 1e-8 \
                    and z.imag > 1e-6:
                if best is None or z.real > best.real:
                    best = z
    return best


def main():
    # The measured 1.0814 comes from the MAIN freq_iso block (tau_inject 0.10).
    # A glob picks tau020 last alphabetically, which would compare a prediction
    # at one latency against a measurement at another.
    path = HERE / 'nav2_block_freq_iso_2026-09-06.json'
    hits = [str(path)]
    blk = json.load(open(path))
    rows_in = [r for r in blk['rows'] if r.get('curve')]
    tau = rows_in[0].get('tau_inject', blk['meta'].get('tau'))

    print('=' * 78)
    print('d92 -- T_ctrl/2 truncation with the plant ZOH inside the kernel')
    print('=' * 78)
    print('  block %s, tau_inject %.3f s, T_sim %.3f s'
          % (pathlib.Path(hits[-1]).name, tau, T_SIM))
    print('  measured s_ctrl = 1.0814  [1.054, 1.124]')
    print('  d51 (two-element chain) predicted 1.028 -> 5.3 points unexplained')
    print()
    print('  %7s %9s %9s %11s %11s %11s'
          % ('freq', 'v used', 'u_ctrl', 's (2 elem)', 's (3 elem)', 'delta'))

    out = {'tau': tau, 'T_sim': T_SIM, 'rows': []}
    p2, p3 = [], []
    for r in rows_in:
        freq = r.get('freq')
        v = r.get('v_obs')
        if not v or not freq:
            continue
        u_tau = v * tau / L
        u_ctrl = v * (1.0 / freq) / L
        u_sim = v * T_SIM / L
        vals = {}
        for tag, with_sim in (('two', False), ('three', True)):
            z = dominant_root(u_tau, u_ctrl, u_sim, with_sim)
            if z is None:
                vals[tag] = None
                continue
            K = K_chain(z, u_tau, u_ctrl, u_sim, True, with_sim)
            u_eff = (-cmath.log(K) / z).real
            base = u_tau + (0.5 * u_sim if with_sim else 0.0)
            vals[tag] = 2.0 * (u_eff - base) / u_ctrl
        if vals['two'] is None or vals['three'] is None:
            print('  %7.2f   no dominant root' % freq)
            continue
        p2.append(vals['two'])
        p3.append(vals['three'])
        out['rows'].append({'freq': freq, 'v': v, 'u_ctrl': u_ctrl,
                            's_pred_two': vals['two'], 's_pred_three': vals['three']})
        print('  %7.2f %9.4f %9.4f %11.4f %11.4f %11.4f'
              % (freq, v, u_ctrl, vals['two'], vals['three'],
                 vals['three'] - vals['two']))

    print()
    if p3:
        m2, m3 = float(np.mean(p2)), float(np.mean(p3))
        meas = 1.0814
        print('  mean prediction, two elements  : %.4f   (d51 reported 1.028)' % m2)
        print('  mean prediction, three elements: %.4f' % m3)
        print('  measured                       : %.4f  [1.054, 1.124]' % meas)
        print()
        print('  excess explained, two elements  : %.0f%%'
              % (100 * (m2 - 1.0) / (meas - 1.0)))
        print('  excess explained, three elements: %.0f%%'
              % (100 * (m3 - 1.0) / (meas - 1.0)))
        if 1.054 <= m3 <= 1.124:
            print('  -> the three-element chain reproduces the measurement; the')
            print('     5.3-point residual was an artefact of leaving the plant')
            print('     ZOH out of the kernel.')
        else:
            print('  -> still short.  The plant ZOH is %0.0fx smaller than the'
                  % ((1.0 / rows_in[0]['freq']) / T_SIM if rows_in else 5))
            print('     controller ZOH, so its second moment cannot carry the')
            print('     residual.  The residual is real and stays open.')
        out.update({'s_pred_two_mean': m2, 's_pred_three_mean': m3,
                    's_measured': meas,
                    'three_explains': bool(1.054 <= m3 <= 1.124)})

    json.dump(out, open(HERE / 'd92_tctrl_full_chain_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d92_tctrl_full_chain_2026-09-08.json')


if __name__ == '__main__':
    main()
