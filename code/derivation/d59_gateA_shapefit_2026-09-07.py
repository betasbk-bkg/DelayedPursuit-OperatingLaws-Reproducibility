#!/usr/bin/env python3
"""
d59 -- Gate A, the full version: does this study's heading-servo J(u) describe the
       SHAPE of the reference engine's RMSE(v) curves, not just the location of the minimum?

d57 showed this study's lag law v* = u* L /(tau + c) beats the reference engine's power law on
the reference engine's own v*(tau) data, and recovers a composite delay close to the
a-priori WIN/2 + DT/SMOOTH = 0.2333 s.  But v* is one number per curve.  The
stronger test -- the one used on Nav2 in d43 -- fits the WHOLE curve against the
corner-transit energy J(u) computed by d38, with no shape parameter at all:

    RMSE(v)^2 = A * J(v * tau_tot / L) + B^2

Per geometry: one scale A, one floor B, one composite delay c (tau_tot = tau + c),
shared across that geometry's six latencies.  J is d38's HEADING-SERVO branch,
which is the class the engine was shown to belong to (direction command, first
order lag).  Nothing about the curve shape is free.

Two things are being tested at once:
  (a) is the SHAPE right?  -> the residual
  (b) is the composite delay right? -> c against the a-priori 0.2333 s

If the shape fits, the reference engine's engine and Nav2 are the same model with different
floor sizes B, and the only open item is the alpha dependence of the optimum.
If the shape does not fit, the heading-servo J is wrong for this engine and the
merge needs more than a re-derivation at finite alpha.

Usage:  python3 d59_gateA_shapefit_2026-09-07.py
"""
import importlib.util
import json
import pathlib
import zipfile

import numpy as np
from scipy.optimize import least_squares

ZIP = pathlib.Path(__file__).resolve().parents[2] / "engine" / "crowd_control_engine"   # the engine as it ships in this package


class _DirAsZip:                       # the engine ships unpacked here; read its files the same way
    def __init__(self, root):
        self.root = pathlib.Path(root)

    def read(self, name):
        return (self.root.parent / name).read_bytes()

HERE = pathlib.Path(__file__).parent
LOOK = 2.0
WIN, DT, SMOOTH = 0.3, 1.0 / 60.0, 0.2
TAU_EXTRA = WIN / 2 + DT / SMOOTH        # 0.23333 s
U_MIN, U_MAX, U_N = 0.04, 1.20, 120      # heading servo optimum sits near 0.5

_p = HERE / 'd38_corner_energy_two_plants_2026-08-29.py'
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def make_J(plant):
    us = np.linspace(U_MIN, U_MAX, U_N)
    Js = np.array([d38.transit_energy(float(u), plant, h=1.0e-3) for u in us])
    ok = np.isfinite(Js) & (Js > 0)
    us, Js = us[ok], Js[ok]
    logJ = np.log(Js)

    def J(u):
        return np.exp(np.interp(np.clip(u, us[0], us[-1]), us, logJ))
    return J, float(us[int(np.argmin(Js))])


def load_curves():
    z = _DirAsZip(ZIP)
    d = json.loads(z.read('crowd_control_engine/unified_vstar_results.json'))
    speeds = np.array(d['params']['speeds'], float)
    out = {}
    for geom, blob in d['results'].items():
        rows = []
        for tms, trd in sorted(blob['data'].items(), key=lambda kv: int(kv[0])):
            e = trd.get('0.05') or list(trd.values())[0]
            rows.append((int(tms) / 1000.0, speeds, np.array(e['rmses'], float)))
        out[geom] = (blob['alpha'], rows)
    return out


def shapefit(rows, J, c_free=True, vmax=None):
    """One A, one B, one composite delay c for this geometry."""
    data = []
    for tau, vs, rs in rows:
        m = np.isfinite(rs) & (rs > 0)
        if vmax is not None:
            m &= vs <= vmax
        data.append((tau, vs[m], rs[m]))
    k = len(data)

    def resid(p):
        c = p[0] if c_free else TAU_EXTRA
        off = 1 if c_free else 0
        out = []
        for i, (tau, vs, rs) in enumerate(data):
            A, B = p[off + i], p[off + k + i]
            u = vs * (tau + c) / LOOK
            pred = np.sqrt(np.maximum(A * J(u), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    p0 = ([TAU_EXTRA] if c_free else [])
    lb = ([0.0] if c_free else [])
    ub = ([2.0] if c_free else [])
    for tau, vs, rs in data:
        p0.append(rs.min() ** 2); lb.append(0.0); ub.append(np.inf)
    for tau, vs, rs in data:
        p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
    s = least_squares(resid, p0, bounds=(lb, ub))
    c = float(s.x[0]) if c_free else TAU_EXTRA
    rms = float(100 * np.sqrt(np.mean(resid(s.x) ** 2)))
    off = 1 if c_free else 0
    B = np.array(s.x[off + k: off + 2 * k], float)
    return {'c_s': c, 'rms_pct': rms, 'B_mean': float(B.mean()),
            'n_points': int(sum(len(d[1]) for d in data))}


def shapefit2(rows, J, vmax=None):
    """
    The synthesis neither paper has: this study's deterministic corner term PLUS
    the reference engine's diffusive quantisation term.

        RMSE(v)^2 = A * J(v tau_tot / L)  +  (D v sqrt(tau))^2  +  B^2
                    \_ this study lag _/         \_ the reference engine _/      floor

    D is shared across the geometry's six latencies (it is a property of the
    vote quantisation, not of tau), as is c; A and B stay per-curve.
    """
    data = []
    for tau, vs, rs in rows:
        m = np.isfinite(rs) & (rs > 0)
        if vmax is not None:
            m &= vs <= vmax
        data.append((tau, vs[m], rs[m]))
    k = len(data)

    def resid(p):
        c, D = p[0], p[1]
        out = []
        for i, (tau, vs, rs) in enumerate(data):
            A, B = p[2 + i], p[2 + k + i]
            u = vs * (tau + c) / LOOK
            pred = np.sqrt(np.maximum(A * J(u), 0.0) + (D * vs * np.sqrt(tau)) ** 2
                           + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    p0 = [TAU_EXTRA, 0.05]; lb = [0.0, 0.0]; ub = [2.0, 5.0]
    for tau, vs, rs in data:
        p0.append(rs.min() ** 2); lb.append(0.0); ub.append(np.inf)
    for tau, vs, rs in data:
        p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
    s = least_squares(resid, p0, bounds=(lb, ub))
    return {'c_s': float(s.x[0]), 'D': float(s.x[1]),
            'rms_pct': float(100 * np.sqrt(np.mean(resid(s.x) ** 2)))}


def main():
    curves = load_curves()
    Jh, u_h = make_J('heading_servo')
    Jc, u_c = make_J('curvature_command')
    print('=' * 78)
    print('d59 -- Gate A: whole-curve shape fit of the reference engine against d38 J(u)')
    print('=' * 78)
    print('  d38 argmin J: heading_servo %.4f, curvature_command %.4f' % (u_h, u_c))
    print('  engine: LOOK = %.1f m, a-priori composite c = %.4f s' % (LOOK, TAU_EXTRA))
    print('  MSPD cap = 5.0 m/s; curves are also refit with v <= 4.0 to keep the')
    print('  capped points out (several results files show vstar exactly 5.0).')
    print()
    print('  %-10s %5s %7s | %8s %8s | %8s %8s | %8s'
          % ('traj', 'alpha', 'npts', 'c (s)', 'rms%', 'c<=4', 'rms%', 'curv rms%'))

    res = {}
    for geom, (alpha, rows) in curves.items():
        fh = shapefit(rows, Jh, c_free=True)
        fh4 = shapefit(rows, Jh, c_free=True, vmax=4.0)
        fc4 = shapefit(rows, Jc, c_free=True, vmax=4.0)
        f2 = shapefit2(rows, Jh, vmax=4.0)
        res[geom] = {'alpha': alpha, 'heading': fh, 'heading_v4': fh4,
                     'curvature_v4': fc4, 'synthesis_v4': f2}
        print('  %-10s %5.0f %7d | %8.4f %7.2f%% | %8.4f %7.2f%% | %7.2f%%'
              % (geom, alpha, fh['n_points'], fh['c_s'], fh['rms_pct'],
                 fh4['c_s'], fh4['rms_pct'], fc4['rms_pct']))

    print()
    print('  READING')
    cs = np.array([r['heading_v4']['c_s'] for r in res.values()])
    print('  composite delay recovered per geometry (v<=4): %s s'
          % ', '.join('%.4f' % x for x in cs))
    print('  mean %.4f vs a-priori %.4f  ->  %+.1f%%'
          % (cs.mean(), TAU_EXTRA, 100 * (cs.mean() / TAU_EXTRA - 1)))
    rh = np.array([r['heading_v4']['rms_pct'] for r in res.values()])
    rc = np.array([r['curvature_v4']['rms_pct'] for r in res.values()])
    print('  shape residual, heading-servo J : %s' % ', '.join('%.2f%%' % x for x in rh))
    print('  shape residual, curvature-cmd J : %s' % ', '.join('%.2f%%' % x for x in rc))
    better = int((rh < rc).sum())
    print('  heading-servo J fits better on %d of %d geometries' % (better, len(rh)))
    print()
    print('  For comparison, the same estimator on Nav2 (d43/d48) gives 2.4-5.2%%')
    print('  residuals, and E7 measured a 1.5%% reproducibility floor there.')
    print()
    print('  SYNTHESIS: this study corner term + the reference engine diffusive term + floor')
    print('  %-10s %9s %9s %9s %12s' % ('traj','c (s)','D','rms%','vs lag-only'))
    for geom, r in res.items():
        f2, f1 = r['synthesis_v4'], r['heading_v4']
        print('  %-10s %9.4f %9.4f %8.2f%% %11.2f%%'
              % (geom, f2['c_s'], f2['D'], f2['rms_pct'],
                 f2['rms_pct'] - f1['rms_pct']))
    d2 = np.array([r['synthesis_v4']['rms_pct'] for r in res.values()])
    c2 = np.array([r['synthesis_v4']['c_s'] for r in res.values()])
    print('  mean residual %.2f%% (lag-only %.2f%%), composite c mean %.4f vs %.4f'
          % (d2.mean(), rh.mean(), c2.mean(), TAU_EXTRA))

    json.dump(res, open(HERE / 'd59_gateA_shapefit_2026-09-07.json', 'w'),
              indent=2, default=float)
    print('\n-> d59_gateA_shapefit_2026-09-07.json')


if __name__ == '__main__':
    main()
