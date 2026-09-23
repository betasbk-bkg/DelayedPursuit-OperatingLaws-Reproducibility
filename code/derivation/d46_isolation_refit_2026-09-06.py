#!/usr/bin/env python3
"""
d46 -- Refit every block with the corner-isolation violation removed.

WHAT d45 FOUND.  The design constant "seg/L >= 5" was written down as a constant,
but the length a corner transient needs is a function of u, and u is the swept
variable.  Solving the loop's own characteristic equation

    lam^2 + 2 exp(-lam u) (lam + 1) = 0            (arclength units, s = dist / L)

for the dominant root gives the arclength to decay to 1%:

    u = 0.19  ->  3.49 L        u = 0.306 ->  3.95 L        u = 0.44 -> 15.85 L

The blocks ran at seg/L = 8.3 (side 5.0 m, L = 0.6), so everything above
u_iso = 0.389 has a corner transient still alive when the next corner arrives.
34 of 225 measured points (15%) are in that state.  The minimum itself is not --
u* ~ 0.31 needs about 4 L -- but d43, the PRIMARY estimator, fits the whole curve,
tail included.  So every headline number could be carrying tail contamination.

WHAT THIS SCRIPT DOES.  Re-runs the d43 fit on each block twice, with and without
the points above that block-row's own u_iso, and prints the two answers side by
side.  If the coefficients move, the published numbers are wrong and must be
replaced by the truncated ones; if they do not, the defect is documented and
harmless, and that statement is now backed rather than assumed.

The worst case a priori is E4 (lookahead): its violation count runs 0, 1, 2, 3 as
L goes 0.3 -> 0.9, i.e. the contamination is CORRELATED WITH THE SWEPT VARIABLE,
which is exactly the situation that fakes a slope.

Usage:  python3 d46_isolation_refit_2026-09-06.py
"""
import glob
import importlib.util
import json
import math
import pathlib

import numpy as np
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).parent
L_DEFAULT = 0.6
SIDE = 5.0
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


def char(lam, u):
    return lam * lam + 2.0 * np.exp(-lam * u) * (lam + 1.0)


def dominant_root(u):
    best = None
    for x0 in np.linspace(-3.0, 1.0, 25):
        for y0 in np.linspace(0.05, 8.0, 40):
            z = complex(x0, y0)
            good = True
            for _ in range(150):
                f = char(z, u)
                h = 1e-7
                fp = (char(z + h, u) - char(z - h, u)) / (2 * h)
                if abs(fp) < 1e-16:
                    good = False
                    break
                zn = z - f / fp
                if not (np.isfinite(zn.real) and np.isfinite(zn.imag)):
                    good = False
                    break
                if abs(zn - z) < 1e-13:
                    z = zn
                    break
                z = zn
            if good and abs(char(z, u)) < 1e-8 and z.imag > 1e-6:
                if best is None or z.real > best.real:
                    best = z
    return best


def settle_L(u):
    z = dominant_root(u)
    if z is None or z.real >= 0:
        return float('inf')
    return math.log(100.0) / (-z.real)


_U_ISO_CACHE = {}


def u_isolated(seg_over_L):
    key = round(seg_over_L, 4)
    if key in _U_ISO_CACHE:
        return _U_ISO_CACHE[key]
    lo, hi = 0.05, 0.5204
    if settle_L(hi) <= seg_over_L:
        _U_ISO_CACHE[key] = hi
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if settle_L(mid) <= seg_over_L:
            lo = mid
        else:
            hi = mid
    _U_ISO_CACHE[key] = lo
    return lo


def fit_block(rows, term_key, tau, J, u_cut=None):
    """d43's shared-composite fit, optionally dropping points above u_iso."""
    data, terms, kept, dropped = [], [], 0, 0
    for r in rows:
        Lr = r.get('lookahead', L_DEFAULT)
        vs = np.array([c['v'] for c in r['curve']])
        rs = np.array([c['rmse'] for c in r['curve']])
        if u_cut is not None:
            ui = u_isolated(SIDE / Lr)
            tt = (r.get('tau_inject', tau) or tau) + r.get(term_key, 0.0) + 0.028
            keep = (vs * tt / Lr) <= ui
            dropped += int((~keep).sum())
            if keep.sum() < 4:
                keep[:] = True
            vs, rs = vs[keep], rs[keep]
        kept += len(vs)
        data.append((vs, rs, Lr))
        terms.append(r.get(term_key, 0.0))
    terms = np.array(terms)
    k = len(data)

    def resid(p, s_free):
        c = p[0]
        s = p[1] if s_free else 1.0
        off = 2 if s_free else 1
        out = []
        for i, (vs, rs, Lr) in enumerate(data):
            A, B = p[off + i], p[off + k + i]
            tt = tau + s * terms[i] + c
            pred = np.sqrt(np.maximum(A * J(vs * tt / Lr), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    def run(s_free):
        p0 = [0.028] + ([1.0] if s_free else [])
        lb = [-0.05] + ([0.0] if s_free else [])
        ub = [0.10] + ([3.0] if s_free else [])
        for vs, rs, Lr in data:
            tt = tau + 0.028
            p0.append(max(rs.min() ** 2 / max(J(vs * tt / Lr).min(), 1e-12), 1e-9))
            lb.append(0.0); ub.append(np.inf)
        for vs, rs, Lr in data:
            p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
        s = least_squares(lambda q: resid(q, s_free), p0, bounds=(lb, ub))
        return s

    s1 = run(False)
    s2 = run(True)
    return {'c_ms': 1000 * float(s1.x[0]),
            'rms_pct': float(100 * np.sqrt(np.mean(resid(s1.x, False) ** 2))),
            'coeff': float(s2.x[1]), 'c_ms_free': 1000 * float(s2.x[0]),
            'rms_pct_free': float(100 * np.sqrt(np.mean(resid(s2.x, True) ** 2))),
            'n_points': kept, 'n_dropped': dropped}


def lookahead_slope(rows, tau, J, u_cut=None):
    """E4's claim: log v* vs log L has slope 1.  Refit per row, then regress."""
    Ls, vstars = [], []
    for r in rows:
        Lr = r.get('lookahead', L_DEFAULT)
        vs = np.array([c['v'] for c in r['curve']])
        rs = np.array([c['rmse'] for c in r['curve']])
        if u_cut is not None:
            ui = u_isolated(SIDE / Lr)
            keep = (vs * (tau + 0.028) / Lr) <= ui
            if keep.sum() >= 4:
                vs, rs = vs[keep], rs[keep]

        def resid(p):
            tt, A, B = p
            pred = np.sqrt(np.maximum(A * J(vs * tt / Lr), 0.0) + B * B)
            return (pred - rs) / rs
        A0 = rs.min() ** 2 / max(J(vs * (tau + 0.028) / Lr).min(), 1e-12)
        s = least_squares(resid, [tau + 0.028, A0, rs.min() * 0.5],
                          bounds=([0.05, 0.0, 0.0], [1.0, np.inf, np.inf]))
        tt = float(s.x[0])
        u_lin = float(np.linspace(U_MIN, U_MAX, U_N)[
            int(np.argmin([d38.transit_energy(float(u), 'curvature_command', h=1e-3)
                           for u in np.linspace(U_MIN, U_MAX, U_N)]))])
        Ls.append(Lr)
        vstars.append(u_lin * Lr / tt)
    x, y = np.log(Ls), np.log(vstars)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return {'slope': float(coef[0]), 'r2': float(r2),
            'L': Ls, 'v_star': [float(v) for v in vstars]}


def main():
    J = make_J()
    print('=' * 78)
    print('d46 -- block fits with and without the points that break corner isolation')
    print('=' * 78)
    print('  isolation limit u_iso by geometry:')
    for g in (16.7, 11.1, 8.3, 5.6):
        print('    seg/L = %5.1f  ->  u_iso = %.4f' % (g, u_isolated(g)))

    res = {'u_iso': {str(g): u_isolated(g) for g in (16.7, 11.1, 8.3, 5.6)}}

    specs = [('freq', 'T_ctrl_half', 'Theorem 2 T_ctrl/2 coefficient'),
             ('tsim', 'T_sim_half', 'Theorem 2 T_sim/2 coefficient')]
    for name, term, what in specs:
        f = sorted(glob.glob(str(HERE / ('nav2_block_%s_*.json' % name))))
        if not f:
            continue
        blk = json.load(open(f[-1]))
        rows = [r for r in blk['rows'] if r.get('curve')]
        tau = rows[0].get('tau_inject', blk['meta'].get('tau'))
        full = fit_block(rows, term, tau, J, u_cut=None)
        cut = fit_block(rows, term, tau, J, u_cut=True)
        res[name] = {'full': full, 'isolated': cut, 'what': what}
        print()
        print('  [%s] %s' % (name, what))
        print('    %-12s %10s %10s %10s %8s' %
              ('', 'coeff', 'c (ms)', 'rms %', 'points'))
        print('    %-12s %10.4f %10.2f %10.3f %8d'
              % ('all points', full['coeff'], full['c_ms'], full['rms_pct'],
                 full['n_points']))
        print('    %-12s %10.4f %10.2f %10.3f %8d   (dropped %d)'
              % ('isolated', cut['coeff'], cut['c_ms'], cut['rms_pct'],
                 cut['n_points'], cut['n_dropped']))
        print('    change in coefficient: %+.4f (%.1f%%)   change in c: %+.2f ms'
              % (cut['coeff'] - full['coeff'],
                 100 * (cut['coeff'] - full['coeff']) / full['coeff'],
                 cut['c_ms'] - full['c_ms']))

    f = sorted(glob.glob(str(HERE / 'nav2_block_lookahead_*.json')))
    if f:
        blk = json.load(open(f[-1]))
        rows = [r for r in blk['rows'] if r.get('curve')]
        tau = rows[0].get('tau_inject', blk['meta'].get('tau'))
        full = lookahead_slope(rows, tau, J, None)
        cut = lookahead_slope(rows, tau, J, True)
        res['lookahead'] = {'full': full, 'isolated': cut}
        print()
        print('  [lookahead] Theorem 1: v* proportional to L, slope must be 1')
        print('    THIS IS THE DANGEROUS ONE -- violations run 0,1,2,3 with L,')
        print('    i.e. the contamination is correlated with the swept variable.')
        print('    all points   slope %.4f  r2 %.4f' % (full['slope'], full['r2']))
        print('    isolated     slope %.4f  r2 %.4f' % (cut['slope'], cut['r2']))
        print('    change: %+.4f' % (cut['slope'] - full['slope']))

    json.dump(res, open(HERE / 'd46_isolation_refit_2026-09-06.json', 'w'), indent=2)
    print('\n-> d46_isolation_refit_2026-09-06.json')


if __name__ == '__main__':
    main()
