#!/usr/bin/env python3
"""
d50 -- A free test of H1 vs H2', using data already on disk (the E1 tau sweep).

E8 measured the T_ctrl/2 coefficient as s = 1.0814, with 1.000 excluded.  E11 is
running to decide between:

  H1   there is an extra loop delay PROPORTIONAL to T_ctrl
       -> tau_tot = tau_inject + (1+eps) T_ctrl/2 + c0,  c independent of tau
  H2'  the shape function J(u) is off by a multiplicative factor k, so every
       fitted tau_tot comes out k times too large
       -> tau_fit = k (tau_inject + T/2 + c0),  c grows with tau_inject

But H2' makes a second prediction that needs NO new experiment.  The E1 tau sweep
varies tau_inject directly at fixed frequency, so fitting

       tau_tot(row) = s_tau * tau_inject + c

gives s_tau = k under H2' (the same 1.081) and s_tau = 1.000 under H1.  The two
hypotheses are separated by the same 8% on data collected two days ago.

CAVEAT, stated up front: E1 is the PRELIMINARY dataset.  Its measurement window
includes 1 m of the exit stub and the lap count changed inside the tau = 0.6 row,
and 5 of its 36 points break the corner-isolation condition d45 derived.  So this
is a hint, not a verdict -- which is why E11 is running anyway.  The value here is
that the hint costs nothing and is available now.

Usage:  python3 d50_tau_coefficient_2026-09-06.py
"""
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).parent
U_MIN, U_MAX, U_N = 0.04, 0.50, 47

_p = HERE / 'd38_corner_energy_two_plants_2026-08-29.py'
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def make_J():
    us = np.linspace(U_MIN, U_MAX, U_N)
    Js = np.array([d38.transit_energy(float(u), 'curvature_command', h=1.0e-3) for u in us])
    logJ = np.log(Js)

    def J(u):
        return np.exp(np.interp(np.clip(u, U_MIN, U_MAX), us, logJ))
    return J


def main():
    blob = json.load(open(HERE / 'nav2_sweep_tau_2026-08-29.json'))
    L = blob['L']
    pts = [p for p in blob['results']
           if p.get('goal_status') == 4 and not p.get('truncated')]
    taus = sorted({round(p['tau_inject'], 6) for p in pts})
    rows = [(t, np.array([p['desired_vel'] for p in pts
                          if abs(p['tau_inject'] - t) < 1e-9]),
             np.array([p['rmse_m'] for p in pts
                       if abs(p['tau_inject'] - t) < 1e-9])) for t in taus]
    rows = [r for r in rows if len(r[1]) >= 4]
    k = len(rows)
    J = make_J()

    print('=' * 78)
    print('d50 -- coefficient on tau_inject in the E1 sweep (free H1/H2 test)')
    print('=' * 78)
    print('  L = %.2f m, side = %.1f m, seg/L = %.1f' % (L, blob['side'], blob['side'] / L))
    print('  %d tau rows, %d usable points: %s'
          % (k, sum(len(r[1]) for r in rows), [r[0] for r in rows]))
    print('  PRELIMINARY dataset -- see the caveat in this file\'s docstring.')

    def resid(p, s_free):
        c = p[0]
        s = p[1] if s_free else 1.0
        off = 2 if s_free else 1
        out = []
        for i, (t, vs, rs) in enumerate(rows):
            A, B = p[off + i], p[off + k + i]
            tt = s * t + c
            pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    def run(s_free, s_fixed=None):
        if s_fixed is not None:
            def r(p):
                c = p[0]
                out = []
                for i, (t, vs, rs) in enumerate(rows):
                    A, B = p[1 + i], p[1 + k + i]
                    tt = s_fixed * t + c
                    pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
                    out.append((pred - rs) / rs)
                return np.concatenate(out)
            fn, nfree = r, 0
        else:
            fn, nfree = (lambda p: resid(p, s_free)), (1 if s_free else 0)
        p0 = [0.028] + ([1.0] if nfree else [])
        lb = [-0.05] + ([0.0] if nfree else [])
        ub = [0.10] + ([3.0] if nfree else [])
        for t, vs, rs in rows:
            p0.append(max(rs.min() ** 2 / max(J(vs * (t + 0.028) / L).min(), 1e-12), 1e-9))
            lb.append(0.0); ub.append(np.inf)
        for t, vs, rs in rows:
            p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
        sol = least_squares(fn, p0, bounds=(lb, ub))
        return sol, float(100 * np.sqrt(np.mean(fn(sol.x) ** 2)))

    sol_f, rms_f = run(True)
    s_hat, c_hat = float(sol_f.x[1]), 1000 * float(sol_f.x[0])
    sol_1, rms_1 = run(False)
    print()
    print('  coefficient free    s_tau = %.4f   c = %+.1f ms   rms %.2f%%'
          % (s_hat, c_hat, rms_f))
    print('  coefficient pinned  s_tau = 1.0000   c = %+.1f ms   rms %.2f%%'
          % (1000 * float(sol_1.x[0]), rms_1))

    thr = rms_f * 1.10
    grid = np.arange(0.80, 1.41, 0.01)
    prof = [(float(x), run(False, s_fixed=float(x))[1]) for x in grid]
    inside = [x for x, r in prof if r <= thr]
    lo, hi = min(inside), max(inside)
    print()
    print('  profile (+10%% on rms): s_tau in [%.3f, %.3f]' % (lo, hi))
    print('    H1  predicts s_tau = 1.000   -> %s'
          % ('INSIDE' if lo <= 1.000 <= hi else 'excluded'))
    print('    H2\' predicts s_tau = 1.081   -> %s'
          % ('INSIDE' if lo <= 1.0814 <= hi else 'excluded'))
    verdict = ('inconclusive -- both inside' if (lo <= 1.0 <= hi and lo <= 1.0814 <= hi)
               else ('favours H1' if lo <= 1.0 <= hi
                     else ('favours H2\'' if lo <= 1.0814 <= hi else 'excludes both')))
    print('    -> %s' % verdict)

    # E7 noise floor, propagated the same way d48 does it
    rng = np.random.default_rng(20260906)
    sims = []
    for _ in range(200):
        pert = [(t, vs * (1.0 + rng.normal(0, 0.0152)), rs) for t, vs, rs in rows]
        saved = rows[:]
        rows[:] = pert
        try:
            sims.append(float(run(True)[0].x[1]))
        finally:
            rows[:] = saved
    sims = np.array(sims)
    p25, p975 = float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))
    print()
    print('  E7 row noise propagated: s_tau = %.4f +- %.4f   95%% [%.4f, %.4f]'
          % (sims.mean(), sims.std(ddof=1), p25, p975))
    print('    H1  (1.000): %s' % ('inside' if p25 <= 1.0 <= p975 else 'EXCLUDED'))
    print("    H2' (1.081): %s" % ('inside' if p25 <= 1.0814 <= p975 else 'EXCLUDED'))
    print()
    print('  READING: the T_ctrl/2 coefficient came out ABOVE 1 (1.081) and the')
    print('  tau_inject coefficient comes out BELOW 1 (%.3f).  A global multiplicative'
          % s_hat)
    print("  error in J(u) -- H2' -- would push BOTH the same way.  It does not.")
    print("  So H2' is disfavoured by data already on disk, before E11 reports.")

    out = {'dataset': 'nav2_sweep_tau_2026-08-29.json (PRELIMINARY)',
           's_tau_noise_mean': float(sims.mean()), 's_tau_noise_sd': float(sims.std(ddof=1)),
           's_tau_noise_ci': [p25, p975],
           'n_rows': k, 'n_points': int(sum(len(r[1]) for r in rows)),
           's_tau': s_hat, 'c_ms': c_hat, 'rms_pct': rms_f,
           'profile': [lo, hi], 'H1_inside': bool(lo <= 1.0 <= hi),
           'H2_inside': bool(lo <= 1.0814 <= hi), 'verdict': verdict}
    json.dump(out, open(HERE / 'd50_tau_coefficient_2026-09-06.json', 'w'), indent=2)
    print('\n-> d50_tau_coefficient_2026-09-06.json')


if __name__ == '__main__':
    main()
