#!/usr/bin/env python3
"""
d43 -- Whole-curve (shape) fit of an experiment BLOCK, with one shared composite delay.

WHY, and why d42's answer is not the last word.

d42 locates each row's RMSE(v) minimum with a three-point parabola and regresses
the implied tau_tot on the swept term.  On block_freq that gives slope 1.155
(r^2 = 0.979) -- the right qualitative answer -- but the six u* values scatter by
16% and the fitted "other composite" comes out NEGATIVE (-1.5 ms), which is
unphysical.  The cause is not the theory: the RMSE(v) curves are shallow near the
optimum and bumpy above it, so the located minimum is the noisiest thing the
experiment produces.  (Same symptom as the tau sweep: minima gave c = 44.8 ms,
the whole-curve fit gave 27.8 ms.)

So fit the SHAPE instead.  d38 derived, for the curvature-command plant,

    RMSE(v)^2 = A * J(u) + B^2,     u = v * tau_tot / L

with J the corner-transit energy of the linearised loop -- a function of u alone,
computed by d38.transit_energy, with NO free shape parameter.  Each row gets its
own scale A and floor B (they differ in corner density and numerical floor), and
ALL rows share one composite delay c:

    tau_tot(row) = tau_inject + term(row) + c

For block_freq, term = T_ctrl/2 = 1/(2f); for block_tsim, term = update_duration/2.
Every point on every curve constrains c, not just the six minima.

Also reported, because it is the actual test of Theorem 2's term:
  - SLOPE test: refit with tau_tot = tau_inject + s*term + c, s free.
    Theorem 2 says the term enters with coefficient s = 1 exactly.
  - a profile of the residual in c, so the interval is honest rather than a point.

Usage:
    python3 d43_nav2_block_shapefit_2026-09-05.py --block nav2_block_freq_2026-09-05.json
    python3 d43_nav2_block_shapefit_2026-09-05.py --block nav2_block_tsim_*.json --term T_sim_half
"""
import argparse
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

L = 0.6
U_MIN, U_MAX, U_N = 0.04, 0.50, 47

_p = pathlib.Path(__file__).with_name('d38_corner_energy_two_plants_2026-08-29.py')
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def make_J():
    us = np.linspace(U_MIN, U_MAX, U_N)
    Js = np.array([d38.transit_energy(float(u), 'curvature_command', h=1.0e-3) for u in us])
    logJ = np.log(Js)

    def J(u):
        return np.exp(np.interp(np.clip(u, U_MIN, U_MAX), us, logJ))
    return J, us, Js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', required=True)
    ap.add_argument('--term', default='T_ctrl_half',
                    help='row key holding the swept delay term (T_ctrl_half | T_sim_half)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    blk = json.load(open(a.block))
    rows = [r for r in blk['rows'] if r.get('curve')]
    if not rows:
        raise SystemExit('no rows with curves')
    tau = rows[0].get('tau_inject', blk.get('meta', {}).get('tau'))

    J, us, Js = make_J()
    u_lin = float(us[int(np.argmin(Js))])
    print(f'J(u) from d38: argmin = {u_lin:.4f}')
    print(f'block: {a.block}   term: {a.term}   tau_inject = {tau}\n')

    terms = np.array([r[a.term] for r in rows])
    data = [(np.array([c['v'] for c in r['curve']]),
             np.array([c['rmse'] for c in r['curve']])) for r in rows]
    k = len(rows)

    def resid(p, s_free):
        c = p[0]
        s = p[1] if s_free else 1.0
        off = 2 if s_free else 1
        out = []
        for i, (vs, rs) in enumerate(data):
            A, B = p[off + i], p[off + k + i]
            tt = tau + s * terms[i] + c
            pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
            out.append((pred - rs) / rs)
        return np.concatenate(out)

    def seed(s_free):
        p0 = [0.006] + ([1.0] if s_free else [])
        lb = [-0.05] + ([0.0] if s_free else [])
        ub = [0.10] + ([3.0] if s_free else [])
        for i, (vs, rs) in enumerate(data):
            tt = tau + terms[i] + 0.006
            p0.append(max(rs.min() ** 2 / max(J(vs * tt / L).min(), 1e-12), 1e-9))
            lb.append(0.0); ub.append(np.inf)
        for vs, rs in data:
            p0.append(rs.min() * 0.5); lb.append(0.0); ub.append(np.inf)
        return p0, lb, ub

    res = {'block': a.block, 'term': a.term, 'tau_inject': tau, 'argmin_J': u_lin,
           'n_rows': k, 'n_points': int(sum(len(v) for v, _ in data))}

    # --- fit 1: coefficient pinned to 1 (Theorem 2 as stated) -----------------
    p0, lb, ub = seed(False)
    s1 = least_squares(lambda q: resid(q, False), p0, bounds=(lb, ub))
    c1 = float(s1.x[0]); r1 = float(100 * np.sqrt(np.mean(resid(s1.x, False) ** 2)))
    res['fit_coeff_pinned_1'] = {'c_s': c1, 'rms_resid_pct': r1}
    print(f'coefficient pinned to 1 (Theorem 2):  c = {c1*1e3:+.1f} ms   '
          f'rms residual {r1:.2f}%')

    # --- fit 2: coefficient free (is it really 1?) ---------------------------
    p0, lb, ub = seed(True)
    s2 = least_squares(lambda q: resid(q, True), p0, bounds=(lb, ub))
    c2, sc = float(s2.x[0]), float(s2.x[1])
    r2 = float(100 * np.sqrt(np.mean(resid(s2.x, True) ** 2)))
    res['fit_coeff_free'] = {'c_s': c2, 'coefficient': sc, 'rms_resid_pct': r2}
    print(f'coefficient free:                     s = {sc:.3f}  '
          f'c = {c2*1e3:+.1f} ms   rms residual {r2:.2f}%')
    print(f'   (Theorem 2 predicts s = 1.000 exactly)')

    # --- profile in c, coefficient pinned ------------------------------------
    prof = []
    p0, lb, ub = seed(False)
    for c in np.linspace(-0.02, 0.06, 41):
        s = least_squares(lambda q: resid(np.concatenate([[c], q]), False),
                          p0[1:], bounds=(lb[1:], ub[1:]))
        prof.append((float(c), float(100 * np.sqrt(
            np.mean(resid(np.concatenate([[c], s.x]), False) ** 2)))))
    best = min(prof, key=lambda t: t[1])
    within = [c for c, r in prof if r <= best[1] * 1.10]
    res['profile_c_vs_resid_pct'] = prof
    res['c_interval_10pct'] = [min(within), max(within)]
    print(f'   profile: within 10% of the best residual for c in '
          f'[{min(within)*1e3:+.0f}, {max(within)*1e3:+.0f}] ms')

    # --- per-row comparison ---------------------------------------------------
    print(f'\n{"term":>10} {"tau_tot(fit)":>13} {"v* model":>10} {"v* parabola":>12} {"diff%":>7}')
    per = []
    for i, r in enumerate(rows):
        tt = tau + terms[i] + c1
        v_model = u_lin * L / tt
        v_par = r.get('v_obs')
        d = 100 * (v_par / v_model - 1) if v_par else None
        per.append({'term': float(terms[i]), 'tau_tot_fit': tt, 'v_star_model': v_model,
                    'v_star_parabola': v_par, 'diff_pct': d})
        print(f'{terms[i]:10.4f} {tt:13.4f} {v_model:10.4f} '
              f'{(v_par if v_par else float("nan")):12.4f} {(d if d else float("nan")):7.1f}')
    res['per_row'] = per

    out = a.out or a.block.replace('nav2_block_', 'd43_shapefit_').replace('.json', '_out.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
