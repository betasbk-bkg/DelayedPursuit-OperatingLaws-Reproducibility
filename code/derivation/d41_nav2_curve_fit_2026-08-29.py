#!/usr/bin/env python3
"""
d41 -- Whole-curve fit of the Nav2 RMSE(v) rows against the derived transit energy.

WHY (audit finding #11, NAV2_AUDIT_TRIAGE §3)
  The RMSE(v) curves on the real stack are shallow near the optimum and not
  monotone at high u, so locating v* from a three-point parabola is fragile: two
  neighbouring points can swap the minimum.  The SHAPE of the whole curve carries
  far more information than the position of its lowest sample.

MODEL
  d38 derived, for the curvature-command plant (Nav2 RPP), that the lap RMSE of a
  polygon course is
        RMSE^2 = n_c * L^3 * J(u) / P  + floor^2,      u = v * tau_tot / L
  with J(u) the corner-transit energy of the linearised loop.  J is computed by
  d38.transit_energy and depends on nothing but u.  So each tau row is fitted as
        RMSE_obs(v) = sqrt( A * J(v * tau_tot / L) + B^2 )
  Two fits per row:
    (fixed)  tau_tot pinned to the a-priori value tau_inject + 0.030 -- free (A, B).
             Tests whether the derived SHAPE matches the deployed controller at all.
    (free)   tau_tot also free -- free (A, B, tau_tot).
             Gives c_shape = tau_tot_fit - tau_inject: the composite delay estimated
             from the curve's shape, independently of where its minimum sits.
  If c_shape is consistent across rows, it is the real stack's composite delay --
  the quantity the audit-corrected bookkeeping (0.036) and the fitted-minimum
  bookkeeping (0.045) disagree about.

INPUT   nav2_sweep_tau_2026-08-29.json (or any sweep with tau_inject, desired_vel, rmse_m)
OUTPUT  d41_results_2026-08-29.json
"""
import argparse
import importlib.util
import json
import os
import pathlib

import numpy as np
from scipy.optimize import least_squares

L = 0.6
T_COMPOSITE = 0.030
U_MIN, U_MAX, U_N = 0.04, 0.50, 47       # u_c = 0.5205; stay clear of the divergence

_p = pathlib.Path(__file__).with_name('d38_corner_energy_two_plants_2026-08-29.py')
_s = importlib.util.spec_from_file_location('d38', _p)
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)


def tabulate_J():
    us = np.linspace(U_MIN, U_MAX, U_N)
    Js = np.array([d38.transit_energy(float(u), 'curvature_command', h=1.0e-3) for u in us])
    return us, Js


def make_model(us, Js):
    logJ = np.log(Js)

    def J(u):
        u = np.clip(u, U_MIN, U_MAX)
        return np.exp(np.interp(u, us, logJ))
    return J


def fit_row(vs, rs, tau, J, free_tau):
    vs, rs = np.asarray(vs), np.asarray(rs)
    tt0 = tau + T_COMPOSITE

    def resid(p):
        A, B = p[0], p[1]
        tt = p[2] if free_tau else tt0
        pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
        return pred - rs

    # initial scale from the lowest point
    Jmin = J(vs * tt0 / L).min()
    A0 = max((rs.min() ** 2) / max(Jmin, 1e-12), 1e-9)
    p0 = [A0, rs.min() * 0.5] + ([tt0] if free_tau else [])
    lb = [0.0, 0.0] + ([tau] if free_tau else [])
    ub = [np.inf, np.inf] + ([tau + 0.2] if free_tau else [])
    sol = least_squares(resid, p0, bounds=(lb, ub))
    pred = resid(sol.x) + rs
    rms_pct = 100.0 * np.sqrt(np.mean(((pred - rs) / rs) ** 2))
    out = {'A': float(sol.x[0]), 'B': float(sol.x[1]), 'rms_resid_pct': float(rms_pct),
           'pred': pred.tolist()}
    if free_tau:
        out['tau_tot_fit'] = float(sol.x[2])
        out['c_shape'] = float(sol.x[2] - tau)
    # the model's own minimum, from the fitted parameters
    ug = np.linspace(U_MIN, U_MAX, 2000)
    tt = sol.x[2] if free_tau else tt0
    vg = ug * L / tt
    m = np.sqrt(sol.x[0] * J(ug) + sol.x[1] ** 2)
    out['u_star_model'] = float(ug[int(np.argmin(m))])
    out['v_star_model'] = float(vg[int(np.argmin(m))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='nav2_sweep_tau_2026-08-29.json')
    ap.add_argument('--out', default='d41_results_2026-08-29.json')
    a = ap.parse_args()

    raw = json.load(open(a.sweep))
    rows = raw['results'] if isinstance(raw, dict) else raw
    by_tau = {}
    for r in rows:
        if r.get('rmse_m') is None or r['rmse_m'] != r['rmse_m']:
            continue
        by_tau.setdefault(round(float(r['tau_inject']), 6), []).append(
            (float(r['desired_vel']), float(r['rmse_m'])))

    print('tabulating J(u) from d38 ...')
    us, Js = tabulate_J()
    J = make_model(us, Js)
    u_lin = float(us[int(np.argmin(Js))])
    print(f'   argmin J = {u_lin:.4f} (d38: 0.306)')

    res = {'L': L, 'T_composite_apriori': T_COMPOSITE, 'u_grid': us.tolist(),
           'J_grid': Js.tolist(), 'argmin_J': u_lin, 'rows': []}

    # NOTE on what is and is not identifiable here.
    #   sqrt(A*J(u)+B^2) is monotone in J, so a per-row fit's own minimum is ALWAYS
    #   argmin J -- reporting it would be a tautology, so it is not reported.
    #   Per-row tau_tot is only identifiable where the composite is a large share of
    #   tau_tot (small tau_inject); at tau=1.2 it is 2% and slides to a bound.  So the
    #   composite delay is estimated ONCE, shared across all rows, in a joint fit:
    #   params = [c, A_1..A_k, B_1..B_k], residuals concatenated.  Rows with leverage
    #   (small tau) determine c; the others constrain the shape.
    taus = sorted(by_tau)
    data = [(np.array([p[0] for p in sorted(by_tau[t])]),
             np.array([p[1] for p in sorted(by_tau[t])])) for t in taus]

    print(f'\n{"tau":>6} {"n":>3} | {"shape fit, a-priori tau_tot: resid%":>34s}')
    for t, (vs, rs) in zip(taus, data):
        fx = fit_row(vs, rs, t, J, free_tau=False)
        fx.pop('u_star_model', None); fx.pop('v_star_model', None)
        res['rows'].append({'tau_inject': t, 'v': vs.tolist(), 'rmse': rs.tolist(),
                            'fit_fixed': fx})
        print(f'{t:6.3f} {len(vs):3d} | {fx["rms_resid_pct"]:34.2f}')

    k = len(taus)

    def joint_resid(p):
        c = p[0]
        out = []
        for i, (t, (vs, rs)) in enumerate(zip(taus, data)):
            A, B = p[1 + i], p[1 + k + i]
            pred = np.sqrt(np.maximum(A * J(vs * (t + c) / L), 0.0) + B * B)
            out.append((pred - rs) / rs)          # relative residuals: rows differ in scale
        return np.concatenate(out)

    p0 = [T_COMPOSITE]
    lb = [0.0]
    ub = [0.15]
    for t, (vs, rs) in zip(taus, data):
        Jmin = J(vs * (t + T_COMPOSITE) / L).min()
        p0 += [max((rs.min() ** 2) / max(Jmin, 1e-12), 1e-9)]
        lb += [0.0]; ub += [np.inf]
    for t, (vs, rs) in zip(taus, data):
        p0 += [rs.min() * 0.5]; lb += [0.0]; ub += [np.inf]
    sol = least_squares(joint_resid, p0, bounds=(lb, ub))
    c_joint = float(sol.x[0])
    resid_pct = 100.0 * np.sqrt(np.mean(joint_resid(sol.x) ** 2))

    # profile the joint residual in c to show how well c is pinned
    prof = []
    for c in np.linspace(0.0, 0.08, 33):
        def r_at_c(q):
            return joint_resid(np.concatenate([[c], q]))
        s = least_squares(r_at_c, sol.x[1:], bounds=(lb[1:], ub[1:]))
        prof.append((float(c), float(100.0 * np.sqrt(np.mean(r_at_c(s.x) ** 2)))))

    res['joint'] = {'c_s': c_joint, 'rms_resid_pct': float(resid_pct),
                    'A': sol.x[1:1 + k].tolist(), 'B': sol.x[1 + k:].tolist(),
                    'profile_c_vs_resid_pct': prof}
    best = min(prof, key=lambda x: x[1])
    within = [c for c, r in prof if r <= best[1] * 1.10]
    print(f'\nJOINT fit, one composite delay shared by all {k} rows:')
    print(f'   c = {c_joint*1e3:.1f} ms   (rms relative residual {resid_pct:.2f}%)')
    print(f'   profile: residual within 10% of its minimum for c in '
          f'[{min(within)*1e3:.0f}, {max(within)*1e3:.0f}] ms')
    print(f'   a-priori (T_ctrl/2 + T_sim/2)             = 30.0 ms')
    print(f'   audit-corrected (+ input T_sim/2 + timer) = 36.0 ms')
    print(f'   from minimum positions (d40)              = 44.8 ms')
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
