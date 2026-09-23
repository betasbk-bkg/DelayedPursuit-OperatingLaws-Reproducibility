#!/usr/bin/env python3
"""
d91 -- does Theorem 2's own truncation explain s_act = 1.0614?

The actuator block measured the coefficient on a first-order lag at
1.0614 [1.039, 1.102], excluding the predicted 1.000 by about 3.7 SE.  Before
looking for a physical cause, the theory's own approximation has to be ruled in
or out -- exactly as d51 did for the ZOH terms, where the truncation predicted
1.028 against a measured 1.081 and so explained 2.8 points of the 8.1.

The lag is the term where this matters most.  Theorem 2 replaces a delay chain
by its FIRST MOMENT, and a first-order lag of time constant T has first moment T
but variance T^2 -- twelve times the ZOH's T^2/12 for the same mean delay.  If
the loop responds to more than the first moment, the lag is where it should show
most.

METHOD, identical to d51.  In arclength units the curvature-command loop has

    lam^2 + 2 K(lam) (lam + 1) = 0

with the chain kernel

    pure delay      exp(-lam u_tau)
    ZOH period T    (1 - exp(-lam u_T)) / (lam u_T)
    first-order lag 1 / (1 + lam u_act)

Solve the EXACT chain for its dominant root, then ask what single delay u_eff
reproduces that root: exp(-lam u_eff) = K(lam), so u_eff = -log K(lam) / lam.
Holding the other terms at Theorem 2's coefficient of 1, the implied coefficient
on the actuator term is

    s_act_pred = (u_eff - u_tau - u_ctrl/2 - u_sim/2) / u_act

WHAT THIS CAN AND CANNOT SETTLE, carried over from d51: it matches the dominant
root of the linearised loop, whereas the experiment locates the minimum of
RMSE(v).  Those agree to the extent the transit energy is dominated by the
slowest mode.  A large predicted deviation is evidence, a small one is evidence,
and neither settles three digits.

Usage:  python3 d91_actuator_second_moment_2026-09-08.py
"""
import cmath
import glob
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
L = 0.6
FREQ = 20.0
T_SIM = 0.01


def K_chain(lam, u_tau, u_ctrl, u_sim, u_act, exact=True):
    if not exact:
        return cmath.exp(-lam * (u_tau + 0.5 * u_ctrl + 0.5 * u_sim + u_act))
    out = cmath.exp(-lam * u_tau)
    for uT in (u_ctrl, u_sim):
        out *= (1.0 if abs(lam * uT) < 1e-12
                else (1.0 - cmath.exp(-lam * uT)) / (lam * uT))
    if u_act > 0:
        out *= 1.0 / (1.0 + lam * u_act)
    return out


def char(lam, *a):
    return lam * lam + 2.0 * K_chain(lam, *a) * (lam + 1.0)


def dominant_root(u_tau, u_ctrl, u_sim, u_act, exact=True):
    best = None
    for x0 in np.linspace(-3.0, 1.0, 20):
        for y0 in np.linspace(0.05, 8.0, 30):
            z = complex(x0, y0)
            good = True
            for _ in range(160):
                f = char(z, u_tau, u_ctrl, u_sim, u_act, exact)
                h = 1e-7
                fp = (char(z + h, u_tau, u_ctrl, u_sim, u_act, exact)
                      - char(z - h, u_tau, u_ctrl, u_sim, u_act, exact)) / (2 * h)
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
            if good and abs(char(z, u_tau, u_ctrl, u_sim, u_act, exact)) < 1e-8 \
                    and z.imag > 1e-6:
                if best is None or z.real > best.real:
                    best = z
    return best


def main():
    hits = sorted(glob.glob(str(HERE / 'nav2_block_actuator_2026-09-08.json')))
    blk = json.load(open(hits[-1]))
    tau = blk['rows'][0]['tau_inject']

    print('=' * 78)
    print('d91 -- does the first-moment truncation explain s_act = 1.0614?')
    print('=' * 78)
    print('  chain: delay %.3f s + ZOH %.3f s + ZOH %.3f s + first-order lag'
          % (tau, 1.0 / FREQ, T_SIM))
    print('  measured s_act = 1.0614  [1.039, 1.102]   (d48)')
    print('  for comparison, d51 on the ZOH: predicted 1.028, measured 1.081')
    print()
    print('  %8s %9s %9s %11s %11s %12s'
          % ('T_act', 'v used', 'u_act', 'u_eff', 'booked', 's_act pred'))

    rows, preds = [], []
    for r in blk['rows']:
        T_act = r['actuator_tau']
        if T_act <= 0:
            continue
        v = r['v_obs']
        u_tau = v * tau / L
        u_ctrl = v * (1.0 / FREQ) / L
        u_sim = v * T_SIM / L
        u_act = v * T_act / L
        z = dominant_root(u_tau, u_ctrl, u_sim, u_act, exact=True)
        if z is None:
            print('  %8.3f   no dominant root' % T_act)
            continue
        K = K_chain(z, u_tau, u_ctrl, u_sim, u_act, exact=True)
        u_eff = (-cmath.log(K) / z).real
        booked = u_tau + 0.5 * u_ctrl + 0.5 * u_sim + u_act
        s_pred = (u_eff - u_tau - 0.5 * u_ctrl - 0.5 * u_sim) / u_act
        rows.append({'T_act': T_act, 'v': v, 'u_act': u_act,
                     'u_eff': u_eff, 'booked': booked, 's_act_pred': s_pred,
                     'root_re': z.real, 'root_im': z.imag})
        preds.append(s_pred)
        print('  %8.3f %9.4f %9.4f %11.5f %11.5f %12.4f'
              % (T_act, v, u_act, u_eff, booked, s_pred))

    print()
    if preds:
        m, lo, hi = float(np.mean(preds)), min(preds), max(preds)
        print('  s_act predicted by the truncation: mean %.4f, range %.4f .. %.4f'
              % (m, lo, hi))
        print('  s_act measured:                    1.0614  [1.039, 1.102]')
        inside = 1.039 <= m <= 1.102
        print()
        if inside:
            print('  -> the truncation ALONE reproduces the measured coefficient.')
            print('     The +6.1%% is Theorem 2 approximating its own chain, not a')
            print('     property of the actuator, and the same argument that')
            print('     explained 2.8 of the ZOH\'s 8.1 points explains all of this.')
        elif m > 1.0:
            frac = 100 * (m - 1.0) / (1.0614 - 1.0)
            print('  -> the truncation explains %.0f%% of the excess (%.4f of %.4f).'
                  % (frac, m - 1.0, 1.0614 - 1.0))
            print('     The remainder is not accounted for.')
        else:
            print('  -> the truncation predicts no excess; the +6.1%% is something')
            print('     else, and this rules out the first suspect.')

        json.dump({'tau': tau, 'freq': FREQ, 'T_sim': T_SIM, 'rows': rows,
                   's_pred_mean': m, 's_pred_range': [lo, hi],
                   's_measured': 1.0614, 's_measured_ci': [1.039, 1.102],
                   'truncation_explains': bool(inside)},
                  open(HERE / 'd91_actuator_second_moment_2026-09-08.json', 'w'),
                  indent=2)
        print('\n-> d91_actuator_second_moment_2026-09-08.json')


if __name__ == '__main__':
    main()
