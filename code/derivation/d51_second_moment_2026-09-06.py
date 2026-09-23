#!/usr/bin/env python3
"""
d51 -- Does Theorem 2's own truncation explain the measured coefficient 1.081?

Theorem 2 replaces a delay CHAIN by a single delay equal to the chain's first
moment.  That is an approximation, and the natural first suspect for a coefficient
of 1.081 instead of 1.000 is the approximation itself: a zero-order hold of period
T has mean delay T/2 but also variance T^2/12, and the loop may respond to more
than the first moment.

This is decidable without any new experiment, so decide it.

In arclength units (s = distance / L, u = v * tau / L) the curvature-command
pure-pursuit loop has characteristic equation

    lam^2 + 2 K(lam) (lam + 1) = 0

where K is the delay chain's kernel:

    pure delay tau      K = exp(-lam u_tau)                u_tau = v tau / L
    ZOH of period T     K = (1 - exp(-lam u_T)) / (lam u_T)  u_T   = v T / L
    the chain           K = exp(-lam u_tau) * K_zoh(lam)

Theorem 2's surrogate is K ~ exp(-lam (u_tau + u_T/2)).

So: solve the EXACT chain for its dominant root, then ask what single delay u_eff
reproduces that root.  At the exact root the surrogate must satisfy
exp(-lam u_eff) = K(lam), giving u_eff = -log K(lam) / lam.  The implied
coefficient on the T/2 term is then

    s_predicted = (u_eff - u_tau) / (u_T / 2)

Theorem 2 asserts s = 1.  If the truncation is the explanation for the measured
1.081, this number should come out near 1.081 at the frequencies E8 used.

WHAT THIS CAN AND CANNOT SETTLE.  It is a linear-stability argument: it matches
the dominant root, whereas the experiment locates the minimum of RMSE(v).  Those
agree to the extent that the transit energy is dominated by the slowest mode.  So
treat a large predicted deviation as evidence and a small one as evidence too --
what it cannot do is quantify to three digits.

Usage:  python3 d51_second_moment_2026-09-06.py
"""
import cmath
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
L = 0.6
U_STAR = 0.306


def K_chain(lam, u_tau, u_T, exact=True):
    if exact:
        if abs(lam * u_T) < 1e-12:
            zoh = 1.0
        else:
            zoh = (1.0 - cmath.exp(-lam * u_T)) / (lam * u_T)
        return cmath.exp(-lam * u_tau) * zoh
    return cmath.exp(-lam * (u_tau + 0.5 * u_T))


def char(lam, u_tau, u_T, exact):
    return lam * lam + 2.0 * K_chain(lam, u_tau, u_T, exact) * (lam + 1.0)


def dominant_root(u_tau, u_T, exact):
    best = None
    for x0 in np.linspace(-3.0, 1.0, 20):
        for y0 in np.linspace(0.05, 8.0, 30):
            z = complex(x0, y0)
            good = True
            for _ in range(160):
                f = char(z, u_tau, u_T, exact)
                h = 1e-7
                fp = (char(z + h, u_tau, u_T, exact)
                      - char(z - h, u_tau, u_T, exact)) / (2 * h)
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
            if good and abs(char(z, u_tau, u_T, exact)) < 1e-8 and z.imag > 1e-6:
                if best is None or z.real > best.real:
                    best = z
    return best


def main():
    tau_inj = 0.10
    T_sim_half = 0.005
    freqs = [20.0, 10.0, 5.0, 3.3333, 2.5, 2.0]

    print('=' * 78)
    print("d51 -- does Theorem 2's first-moment truncation produce a coefficient?")
    print('=' * 78)
    print('  E8 conditions: tau_inject = %.2f s, L = %.2f m, u* = %.3f' % (tau_inj, L, U_STAR))
    print()
    print('  %7s %9s %9s %11s %22s %12s' %
          ('f (Hz)', 'T_ctrl/2', 'tau_tot', 'u_T', 'dominant root (exact)', 's_predicted'))

    rows = []
    for f in freqs:
        T = 1.0 / f
        tt = tau_inj + 0.5 * T + T_sim_half
        v = U_STAR * L / tt                       # the row runs near its optimum
        u_tau = v * tau_inj / L
        u_T = v * T / L
        lam = dominant_root(u_tau, u_T, exact=True)
        if lam is None:
            print('  %7.3f   no root found' % f)
            continue
        Kv = K_chain(lam, u_tau, u_T, exact=True)
        u_eff = -cmath.log(Kv) / lam
        s_pred = (u_eff.real - u_tau) / (0.5 * u_T)
        rows.append({'freq': f, 'T_ctrl_half': 0.5 * T, 'tau_tot': tt, 'v': v,
                     'u_tau': u_tau, 'u_T': u_T,
                     'root_re': lam.real, 'root_im': lam.imag,
                     'u_eff_re': u_eff.real, 'u_eff_im': u_eff.imag,
                     's_predicted': s_pred})
        print('  %7.3f %9.4f %9.4f %11.4f   %8.4f %+8.4fj %12.4f'
              % (f, 0.5 * T, tt, u_T, lam.real, lam.imag, s_pred))

    ss = np.array([r['s_predicted'] for r in rows])
    print()
    print('  s_predicted over the six rows: %.4f to %.4f  (mean %.4f)'
          % (ss.min(), ss.max(), ss.mean()))
    print()
    print('  MEASURED (E8, isolated geometry):  s = 1.0814,  95%% [1.048, 1.131]')
    inside = (ss.mean() >= 1.048) and (ss.mean() <= 1.131)
    print('  Theorem 2 as stated:               s = 1.0000')
    print()
    if inside:
        print('  VERDICT: the truncation ALONE accounts for the measured coefficient.')
        print('           Theorem 2 is not wrong -- it is stated to first order, and the')
        print('           second moment of the ZOH supplies exactly the observed excess.')
        print('           The fix is a stated second-order term, not a new experiment.')
    else:
        print('  VERDICT: the truncation does NOT account for the measured coefficient.')
        print('           Predicted %.4f vs measured 1.0814 (95%% [1.048, 1.131]).'
              % ss.mean())
        print('           So the excess is not Theorem 2 being truncated; something')
        print('           in the loop carries an extra delay proportional to T_ctrl.')
        print('           That is hypothesis H1, and E11 tests it directly.')

    # how large is the second moment relative to the first, for context
    print()
    print('  for context -- the ZOH chain moments in arclength units:')
    print('  %7s %10s %10s %12s' % ('f (Hz)', 'mean u', 'sd u', 'sd / mean'))
    for r in rows:
        mu = r['u_tau'] + 0.5 * r['u_T']
        sd = r['u_T'] / math.sqrt(12.0)
        print('  %7.3f %10.4f %10.4f %12.4f' % (r['freq'], mu, sd, sd / mu))

    # ------------------------------------------------------------------
    # The same correction, applied to the E1 tau sweep -- where it predicts the
    # OPPOSITE sign, and d50 measured exactly that.
    #
    # In E1 the ZOH terms are fixed (20 Hz, 0.01 s tick) and tau_inject is swept
    # 0.05 -> 1.2.  But v* = u* L / tau_tot falls as tau grows, so u_T = v T / L
    # falls too: the second-moment correction is LARGE at small tau and SMALL at
    # large tau.  Fitting tau_eff = s_tau * tau_inject + c against a tau_eff that
    # is inflated more at the small-tau end must give s_tau BELOW 1.
    # d50 measured s_tau = 0.9713, 95% [0.949, 0.9995].  Is that the size of it?
    # ------------------------------------------------------------------
    print()
    print('=' * 78)
    print('  the same correction on the E1 tau sweep (fixed 20 Hz), where it')
    print('  predicts a coefficient BELOW 1 -- and d50 measured 0.971')
    print('=' * 78)
    T20 = 1.0 / 20.0
    taus = [0.05, 0.2, 0.6, 1.2]
    print('  %8s %9s %9s %12s %12s' %
          ('tau_inj', 'tau_tot', 'u_T', 'tau_eff (s)', 'excess (ms)'))
    xs, ys = [], []
    for t in taus:
        tt = t + 0.5 * T20 + T_sim_half
        v = U_STAR * L / tt
        u_tau = v * t / L
        u_T = v * T20 / L
        lam = dominant_root(u_tau, u_T, exact=True)
        if lam is None:
            continue
        Kv = K_chain(lam, u_tau, u_T, exact=True)
        u_eff = (-cmath.log(Kv) / lam).real
        tau_eff = u_eff * L / v          # back to seconds
        xs.append(t); ys.append(tau_eff)
        print('  %8.2f %9.4f %9.4f %12.5f %12.2f'
              % (t, tt, u_T, tau_eff, 1000 * (tau_eff - (t + 0.5 * T20))))
    A = np.vstack([np.array(xs), np.ones(len(xs))]).T
    coef, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    s_tau_pred = float(coef[0])
    print()
    print('  slope of tau_eff vs tau_inject  ->  s_tau predicted = %.4f' % s_tau_pred)
    print('  d50 measured                        s_tau           = 0.9713')
    print('                                      95%% CI            [0.9490, 0.9995]')
    agree = 0.9490 <= s_tau_pred <= 0.9995
    print()
    if agree:
        print('  The prediction lands inside the measured interval, WITH THE SIGN THAT')
        print('  looked anomalous.  One correction -- the second moment Theorem 2 drops')
        print('  -- pushes the T_ctrl coefficient ABOVE 1 and the tau coefficient BELOW')
        print('  1, because v* moves in opposite directions in the two sweeps.  That is')
        print('  a joint explanation, not two separate excuses.')
    else:
        print('  Predicted %.4f is outside the measured [0.9490, 0.9995]: the same'
              % s_tau_pred)
        print('  correction does NOT jointly explain both coefficients.  The tau')
        print('  sweep is barely affected (the excess falls from 0.5 ms to 0.03 ms')
        print('  as tau grows, because v* falls with it), so its 0.971 stays open.')
        print('  It is also the PRELIMINARY dataset -- E1prime is the way to settle it.')

    # ------------------------------------------------------------------
    # BLAST RADIUS.  The question that decides whether this is a crack in the
    # foundation or a correction term: how much does s = 1.081 instead of 1.000
    # change what the theory PREDICTS?  v* = u* L / tau_tot, so the relative
    # error in predicted v* is 0.081 * (T_ctrl/2) / tau_tot -- the share of the
    # composite that this one term carries.
    # ------------------------------------------------------------------
    print()
    print('=' * 78)
    print('  BLAST RADIUS -- what does s = 1.081 instead of 1.000 change?')
    print('=' * 78)
    print('  %7s %10s %12s %14s' %
          ('f (Hz)', 'tau_tot', 'T/2 share', 'error in v*'))
    errs = []
    for r in rows:
        share = r['T_ctrl_half'] / r['tau_tot']
        e = 0.0814 * share
        errs.append(e)
        print('  %7.3f %10.4f %11.1f%% %13.2f%%'
              % (r['freq'], r['tau_tot'], 100 * share, 100 * e))
    # a realistic deployed robot rather than a sweep row
    for name, tau_s, f in [('typical robot, 20 Hz, 50 ms sensor delay', 0.05, 20.0),
                           ('typical robot, 10 Hz, 50 ms sensor delay', 0.05, 10.0)]:
        tt = tau_s + 0.5 / f + T_sim_half
        share = (0.5 / f) / tt
        print('  %-44s  %5.2f%%' % (name, 100 * 0.0814 * share))
    print()
    print('  So the coefficient error moves predicted v* by %.1f%%-%.1f%% across the'
          % (100 * min(errs), 100 * max(errs)))
    print('  swept range.  E7 measured the reproducibility of u* at 1.5%% sd, so this')
    print('  is bigger than the noise but it is a few percent, not a structural')
    print('  failure: Theorem 1 (an interior optimum exists, v* proportional to L)')
    print('  does not depend on the composition being exact, only on tau_tot being')
    print('  well defined.  What needs restating is Theorem 2 s exactness claim.')

    json.dump({'blast_radius_pct': [100 * min(errs), 100 * max(errs)],
               'tau_sweep_s_predicted': s_tau_pred,
               'tau_sweep_s_measured': 0.9713,
               'tau_sweep_joint_ok': bool(agree),
               'tau_inject': tau_inj, 'rows': rows,
               's_predicted_mean': float(ss.mean()),
               's_predicted_range': [float(ss.min()), float(ss.max())],
               's_measured': 1.0814, 's_measured_ci': [1.048, 1.131],
               'truncation_explains': bool(inside)},
              open(HERE / 'd51_second_moment_2026-09-06.json', 'w'), indent=2)
    print('\n-> d51_second_moment_2026-09-06.json')


if __name__ == '__main__':
    main()
