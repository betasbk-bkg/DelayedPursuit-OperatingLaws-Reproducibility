#!/usr/bin/env python3
"""
d55 -- The discriminator E11 should have used, on the data E11 already collected.

WHAT WENT WRONG WITH E11's DESIGN.  I pre-registered c(tau_inject) as the thing
that separates

    H1   an extra loop delay proportional to T_ctrl:  tau_tot = tau + (1+e) T/2 + c0
    H2'  J(u) off by a multiplicative factor k:       tau_fit = k (tau + T/2 + c0)

and claimed 14 sigma of separation.  That number came from an error bar on c that
I took from E7's single-row scatter (2.11 ms / sqrt(6) = 0.86 ms).  d54 measured
the actual profile width of c in a block fit: 6 to 28 ms, because c and s are
strongly correlated in that fit.  With the real widths the separation is 0.8 vs
1.5 sigma -- E11 as designed has essentially no power, and the three blocks cost
about eight hours of machine time to find that out.

WHAT THE SAME DATA CAN DO.  The two hypotheses differ in ONE place: whether the
tau_inject term is scaled too.

    H1   scales only the T/2 term      ->  s_tau = 1,  s_T = 1 + e
    H2'  scales everything             ->  s_tau = s_T = k

E11 varied tau_inject across blocks (0.05, 0.10, 0.20) and T_ctrl/2 within them
(0.025 .. 0.25).  Pooling all 18 rows makes those two factors independent, so
both coefficients can be fitted at once:

    tau_tot(row) = s_tau * tau_inject + s_T * (T_ctrl/2) + c

That is a two-factor design, it is already collected, and it costs nothing.

Reported: the joint fit, a profile in each coefficient, and the H1/H2' test as
the difference s_T - s_tau, which is zero under H2' and positive under H1.

Usage:  python3 d55_two_coefficient_fit_2026-09-06.py
"""
import glob
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).parent
L = 0.6
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


def load_all():
    pats = ['nav2_block_freq_iso_tau005_*.json',
            'nav2_block_freq_iso_2026-09-06.json',
            'nav2_block_freq_iso_tau020_*.json']
    rows = []
    for pat in pats:
        f = sorted(glob.glob(str(HERE / pat)))
        if not f:
            raise SystemExit('missing block: %s' % pat)
        blk = json.load(open(f[-1]))
        tau = blk['meta']['tau']
        for r in blk['rows']:
            if not r.get('curve'):
                continue
            rows.append({'tau': tau, 'term': r['T_ctrl_half'],
                         'v': np.array([c['v'] for c in r['curve']]),
                         'rmse': np.array([c['rmse'] for c in r['curve']])})
    return rows


def fit(rows, J, fix=None):
    """fix: None, or a dict pinning 's_tau' and/or 's_T'."""
    fix = fix or {}
    k = len(rows)
    free = [n for n in ('s_tau', 's_T') if n not in fix]

    def unpack(p):
        vals, i = {}, 1
        for n in free:
            vals[n] = p[i]
            i += 1
        vals.update(fix)
        return p[0], vals, i

    def resid(p):
        c, v, off = unpack(p)
        out = []
        for i, r in enumerate(rows):
            A, B = p[off + i], p[off + k + i]
            tt = v['s_tau'] * r['tau'] + v['s_T'] * r['term'] + c
            pred = np.sqrt(np.maximum(A * J(r['v'] * tt / L), 0.0) + B * B)
            out.append((pred - r['rmse']) / r['rmse'])
        return np.concatenate(out)

    p0 = [0.005] + [1.05] * len(free)
    lb = [-0.08] + [0.2] * len(free)
    ub = [0.10] + [3.0] * len(free)
    for r in rows:
        p0.append(max(r['rmse'].min() ** 2
                      / max(J(r['v'] * (r['tau'] + 0.03) / L).min(), 1e-12), 1e-9))
        lb.append(0.0); ub.append(np.inf)
    for r in rows:
        p0.append(r['rmse'].min() * 0.5); lb.append(0.0); ub.append(np.inf)
    sol = least_squares(resid, p0, bounds=(lb, ub))
    c, v, _ = unpack(sol.x)
    return {'c_ms': 1000 * c, 's_tau': v['s_tau'], 's_T': v['s_T'],
            'rms_pct': float(100 * np.sqrt(np.mean(resid(sol.x) ** 2)))}


def main():
    J = make_J()
    rows = load_all()
    taus = sorted({r['tau'] for r in rows})
    terms = sorted({r['term'] for r in rows})
    npts = sum(len(r['v']) for r in rows)

    print('=' * 78)
    print('d55 -- two-coefficient fit over all three E11/E8 blocks')
    print('=' * 78)
    print('  %d rows, %d points' % (len(rows), npts))
    print('  tau_inject varied over %s   (between blocks)' % taus)
    print('  T_ctrl/2   varied over %s   (within blocks)'
          % [round(t, 4) for t in terms])
    print('  the two factors are crossed, so both coefficients are identifiable.')

    best = fit(rows, J)
    print()
    print('  joint fit:  s_tau = %.4f   s_T = %.4f   c = %+.2f ms   rms %.2f%%'
          % (best['s_tau'], best['s_T'], best['c_ms'], best['rms_pct']))
    print()
    print('  H1  predicts s_tau = 1.000 and s_T > 1     (only the T/2 term is scaled)')
    print("  H2' predicts s_tau = s_T                   (everything is scaled)")

    thr = best['rms_pct'] * 1.10
    out = {'n_rows': len(rows), 'n_points': npts, 'taus': taus,
           'joint': best, 'threshold_rms_pct': thr}

    print()
    print('  profiles at the +10%% residual threshold (%.3f%%):' % thr)
    for name in ('s_tau', 's_T'):
        grid = np.arange(0.70, 1.61, 0.02)
        inside = [float(x) for x in grid
                  if fit(rows, J, fix={name: float(x)})['rms_pct'] <= thr]
        lo, hi = (min(inside), max(inside)) if inside else (float('nan'),) * 2
        out[name + '_profile'] = [lo, hi]
        print('    %-6s = %.4f   profile [%.3f, %.3f]   contains 1.000: %s'
              % (name, best[name], lo, hi, 'YES' if lo <= 1.0 <= hi else 'NO'))

    # the H1 vs H2' test is the DIFFERENCE, so profile that directly:
    # pin s_tau = s_T = k (this is exactly H2') and see what it costs.
    print()
    grid = np.arange(0.90, 1.31, 0.01)
    h2 = min((fit(rows, J, fix={'s_tau': float(x), 's_T': float(x)}) for x in grid),
             key=lambda r: r['rms_pct'])
    h1 = fit(rows, J, fix={'s_tau': 1.0})
    print('  the two hypotheses as CONSTRAINED fits:')
    print('    H2\'  s_tau = s_T = k free  ->  k = %.4f, c = %+.2f ms, rms %.3f%%'
          % (h2['s_tau'], h2['c_ms'], h2['rms_pct']))
    print('    H1   s_tau = 1.000 pinned  ->  s_T = %.4f, c = %+.2f ms, rms %.3f%%'
          % (h1['s_T'], h1['c_ms'], h1['rms_pct']))
    print('    unconstrained              ->  rms %.3f%%' % best['rms_pct'])
    out['H2_constrained'] = h2
    out['H1_constrained'] = h1

    dh1 = 100 * (h1['rms_pct'] - best['rms_pct']) / best['rms_pct']
    dh2 = 100 * (h2['rms_pct'] - best['rms_pct']) / best['rms_pct']
    print()
    print('    cost of imposing H1:  +%.2f%% of residual' % dh1)
    print('    cost of imposing H2\': +%.2f%% of residual' % dh2)
    out['cost_H1_pct'] = dh1
    out['cost_H2_pct'] = dh2
    if abs(dh1 - dh2) < 1.0:
        verdict = ('INCONCLUSIVE -- the two constraints cost the same, so this '
                   'design cannot separate them either')
    elif dh1 < dh2:
        verdict = 'favours H1 (an extra delay proportional to T_ctrl)'
    else:
        verdict = "favours H2' (a multiplicative error in J(u))"
    print()
    print('  VERDICT: %s' % verdict)
    out['verdict'] = verdict

    # A proper nested-model test.  Both constrained fits have exactly one fewer
    # free parameter than the unconstrained one, so an F test applies -- with the
    # caveat that residuals within a row are correlated and not iid, which makes
    # this approximate rather than exact.
    from scipy import stats as _st
    n_par = 3 + 2 * len(rows)          # c, s_tau, s_T, and A_i, B_i per row
    dof = npts - n_par
    print()
    print('  nested F test (1 constraint, %d residual dof):' % dof)
    out['f_test'] = {'dof': dof, 'n_par': n_par}
    for tag, r in (('H1  (s_tau = 1)', h1), ("H2' (s_tau = s_T)", h2)):
        ratio = (r['rms_pct'] / best['rms_pct']) ** 2
        F = (ratio - 1.0) * dof
        pval = float(1.0 - _st.f.cdf(F, 1, dof)) if F > 0 else 1.0
        out['f_test'][tag.split()[0]] = {'F': float(F), 'p': pval}
        print('    %-20s F = %6.2f   p = %.4f   %s'
              % (tag, F, pval,
                 'REJECTED at 1%' if pval < 0.01 else
                 ('rejected at 5%' if pval < 0.05 else 'not rejected')))
    print('    (approximate: residuals within a row are correlated, not iid)')

    if 's_tau_profile' in out:
        lo, hi = out['s_tau_profile']
        print()
        print('  Note what s_tau alone says: the tau_inject coefficient is %.4f'
              % best['s_tau'])
        print('  with profile [%.3f, %.3f].  d50 got %.4f on the preliminary tau'
              % (lo, hi, 0.9713))
        print('  sweep, independently.  H2\' needs this to equal s_T = %.4f.'
              % best['s_T'])

    json.dump(out, open(HERE / 'd55_two_coefficient_fit_2026-09-06.json', 'w'), indent=2)
    print('\n-> d55_two_coefficient_fit_2026-09-06.json')


if __name__ == '__main__':
    main()
