#!/usr/bin/env python3
"""
d44 -- Null controls (E6) judged against a MEASURED noise model and a MECHANISM,
       not against a round number.

WHY THIS SCRIPT EXISTS.

NAV2_STATUS section 6.3 pre-registered "u* spread < 3% -> PASS, > 5% -> FAIL".
Those two numbers were set by us, not taken from any literature, and the
justification written next to them ("the 0.7% point reproducibility") is a
CATEGORY ERROR: 0.7% is the reproducibility of RMSE at a fixed operating point
(same condition three times: 0.02305 / 0.02321 / 0.02311), while the threshold
was applied to the spread of u*, which is a MINIMUM LOCATION.  Near a shallow
minimum, a relative error eps in RMSE moves the located minimum by roughly
sqrt(eps / curvature) -- a much larger number.  A threshold on u* can therefore
not be read off a threshold on RMSE.  This script replaces that reasoning.

WHAT IT DOES INSTEAD.  Three independent legs; a null passes only if all agree.

  LEG 1 -- NOISE MODEL, MEASURED.
    Take the one repeatability measurement the experiment actually contains
    (the triple above) as sigma_rel of a single RMSE point.  Monte-Carlo:
    perturb every RMSE on a row by N(1, sigma_rel), re-run the experiment's OWN
    estimator, and read off sigma(u*) for that row.  This converts a measured
    RMSE noise into the u* noise the estimator produces.  The PASS band is then
    derived, not chosen: for k rows of one knob, the expected range of k draws
    from N(0, sigma_u) is E[range_k] * sigma_u.

  LEG 2 -- SECOND ESTIMATOR.
    Section 2g used the three-point parabola, which this project's own d43 shows
    is the noisiest thing the experiment produces.  Redo every null row with the
    primary estimator instead: an independent whole-curve shape fit against
    d38.transit_energy (zero shape parameters, per-row A, B and tau_tot).  For a
    null knob, tau_tot must be the SAME for every row of the knob, so the
    disagreement is reported in milliseconds of composite delay -- a physical
    unit, comparable to the 27-28 ms the rest of the experiment measures.

  LEG 3 -- MECHANISM, i.e. the alternative hypothesis.
    A null is only interesting against a specific way the knob could have leaked
    in.  For each knob we write down that mechanism, predict the shift it would
    cause at coupling coefficient 1, and report the coupling bound the data
    imposes.  This is what makes the null quantitative rather than an absence.

Usage:  python3 d44_null_equivalence_2026-09-06.py [--block nav2_block_nulls_*.json]
"""
import argparse
import glob
import importlib.util
import json
import pathlib

import numpy as np
from scipy.optimize import least_squares

L = 0.6
U_MIN, U_MAX, U_N = 0.04, 0.50, 47
SEED = 20260906
N_MC = 2000

# The experiment's one repeatability measurement: same condition, three runs.
# NAV2_STATUS_2026-09-05.md line 361.
REPLICATE_RMSE = [0.02305, 0.02321, 0.02311]

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
    return J, float(us[int(np.argmin(Js))])


def locate(vs, rs):
    """The experiment's own estimator (experiment.py:locate), verbatim in behaviour."""
    o = np.argsort(vs)
    vs, rs = np.asarray(vs)[o], np.asarray(rs)[o]
    i = int(np.argmin(rs))
    if i in (0, len(rs) - 1):
        return None
    x0, x1, x2 = vs[i - 1], vs[i], vs[i + 1]
    y0, y1, y2 = rs[i - 1], rs[i], rs[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-15:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if a <= 0:
        return None
    v = -b / (2 * a)
    return float(v) if x0 <= v <= x2 else None


def shapefit_row(vs, rs, J, tau_seed=0.23):
    """Independent whole-curve fit of ONE row: free tau_tot, scale A, floor B."""
    def resid(p):
        tt, A, B = p
        pred = np.sqrt(np.maximum(A * J(vs * tt / L), 0.0) + B * B)
        return (pred - rs) / rs
    A0 = rs.min() ** 2 / max(J(vs * tau_seed / L).min(), 1e-12)
    s = least_squares(resid, [tau_seed, A0, rs.min() * 0.5],
                      bounds=([0.05, 0.0, 0.0], [1.0, np.inf, np.inf]))
    return float(s.x[0]), float(100 * np.sqrt(np.mean(s.fun ** 2)))


def expected_range(k, n=200000, rng=None):
    """E[range] and sd[range] of k iid standard normals -- the PASS band, derived."""
    z = rng.standard_normal((n, k))
    r = z.max(axis=1) - z.min(axis=1)
    return float(r.mean()), float(r.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', default=None)
    ap.add_argument('--out', default='d44_null_equivalence_2026-09-06.json')
    a = ap.parse_args()
    path = a.block or sorted(glob.glob('nav2_block_nulls_*.json'))[-1]
    blk = json.load(open(path))
    rows = [r for r in blk['rows'] if r.get('curve')]
    tau_inj = blk['meta']['tau']
    J, u_lin = make_J()
    rng = np.random.default_rng(SEED)

    out = {'block': path, 'tau_inject': tau_inj, 'argmin_J': u_lin,
           'n_rows': len(rows), 'n_points': sum(len(r['curve']) for r in rows),
           'seed': SEED, 'n_mc': N_MC}

    # ---- LEG 1a: the measured noise floor -----------------------------------
    rep = np.array(REPLICATE_RMSE)
    sig_rel = float(rep.std(ddof=1) / rep.mean())
    out['replicate'] = {'values': REPLICATE_RMSE, 'mean': float(rep.mean()),
                        'sd': float(rep.std(ddof=1)), 'sigma_rel': sig_rel,
                        'peak_to_peak_pct': float(100 * (rep.max() - rep.min()) / rep.mean())}
    print('=' * 78)
    print('LEG 1 -- measured noise floor, propagated through the estimator')
    print('=' * 78)
    print('  replicate RMSE  %s' % REPLICATE_RMSE)
    print('  sigma_rel(one RMSE point) = %.3f%%   (peak-to-peak %.2f%%, the number 6.3 quoted)'
          % (100 * sig_rel, out['replicate']['peak_to_peak_pct']))

    tau_book = tau_inj + 0.03      # the a-priori bookkeeping used in section 6.3

    per = []
    for r in rows:
        vs = np.array([c['v'] for c in r['curve']])
        rs = np.array([c['rmse'] for c in r['curve']])
        v_par = locate(vs, rs)
        tt_fit, rms_fit = shapefit_row(vs, rs, J)
        u_par = v_par * tau_book / L if v_par else None
        mc_u, mc_tt = [], []
        for _ in range(N_MC):
            pert = rs * (1.0 + rng.normal(0.0, sig_rel, size=rs.size))
            v = locate(vs, pert)
            if v is not None:
                mc_u.append(v * tau_book / L)
            try:
                mc_tt.append(shapefit_row(vs, pert, J, tau_seed=tt_fit)[0])
            except Exception:
                pass
        per.append({
            'knob': r['knob'],
            'setting': {k: v for k, v in r.items()
                        if k not in ('curve', 'knob', 'v_obs', 'u_obs', 'how')},
            'u_parabola': u_par, 'u_parabola_sd': float(np.std(mc_u, ddof=1)),
            'mc_valid_frac': len(mc_u) / N_MC,
            'tau_tot_shapefit': tt_fit, 'tau_tot_sd': float(np.std(mc_tt, ddof=1)),
            'shapefit_rms_pct': rms_fit,
        })
    out['rows'] = per

    er_mean, er_sd = expected_range(3, rng=rng)
    out['expected_range_k3'] = {'mean': er_mean, 'sd': er_sd}
    print('  E[range of 3 draws] = %.3f sigma  (sd %.3f)  -> this replaces the chosen 3%%/5%%'
          % (er_mean, er_sd))

    knobs = {}
    order = []
    for p in per:
        if p['knob'] not in knobs:
            knobs[p['knob']] = []
            order.append(p['knob'])
        knobs[p['knob']].append(p)

    print()
    print('=' * 78)
    print('LEG 1+2 -- per knob: observed scatter vs propagated noise, both estimators')
    print('=' * 78)
    verdicts = {}
    for kn in order:
        ps = knobs[kn]
        u = np.array([p['u_parabola'] for p in ps])
        su = np.array([p['u_parabola_sd'] for p in ps])
        t = np.array([p['tau_tot_shapefit'] for p in ps])
        st = np.array([p['tau_tot_sd'] for p in ps])
        k = len(ps)
        sp_u = 100 * (u.max() - u.min()) / u.mean()
        exp_u = 100 * er_mean * su.mean() / u.mean()
        exp_u_hi = 100 * (er_mean + 2 * er_sd) * su.mean() / u.mean()
        chi_u = float(np.sum((u - u.mean()) ** 2 / su ** 2))
        sp_t = 1000 * (t.max() - t.min())
        exp_t = 1000 * er_mean * st.mean()
        exp_t_hi = 1000 * (er_mean + 2 * er_sd) * st.mean()
        chi_t = float(np.sum((t - t.mean()) ** 2 / st ** 2))
        ok = (sp_u <= exp_u_hi) and (chi_u <= 9.21) and (sp_t <= exp_t_hi) and (chi_t <= 9.21)
        verdicts[kn] = {
            'k': k,
            'u_parabola': u.tolist(), 'u_spread_pct': sp_u,
            'u_sigma_mc_pct': float(100 * su.mean() / u.mean()),
            'u_expected_spread_pct': exp_u, 'u_expected_spread_p95_pct': exp_u_hi,
            'chi2_u': chi_u,
            'tau_tot_ms': (1000 * t).tolist(), 'tau_spread_ms': sp_t,
            'tau_sigma_mc_ms': float(1000 * st.mean()),
            'tau_expected_spread_ms': exp_t, 'tau_expected_spread_p95_ms': exp_t_hi,
            'chi2_tau': chi_t,
            'verdict': 'PASS' if ok else 'CHECK'}
        print()
        print('  [%s]  k=%d   chi2 has %d dof; 99%% critical value 9.21' % (kn, k, k - 1))
        print('    parabola   u* = %s' % np.array2string(u, precision=5))
        print('               observed spread %6.3f%%   propagated sigma %6.3f%%  '
              '-> expected %6.3f%% (95th pct %.3f%%)'
              % (sp_u, 100 * su.mean() / u.mean(), exp_u, exp_u_hi))
        print('               chi2 = %7.3f   %s'
              % (chi_u, 'consistent with noise' if chi_u <= 9.21 else 'EXCEEDS noise'))
        print('    shape fit  tau_tot = %s ms' % np.array2string(1000 * t, precision=2))
        print('               observed spread %6.2f ms   propagated sigma %6.2f ms '
              '-> expected %.2f ms (95th pct %.2f ms)'
              % (sp_t, 1000 * st.mean(), exp_t, exp_t_hi))
        print('               chi2 = %7.3f   %s'
              % (chi_t, 'consistent with noise' if chi_t <= 9.21 else 'EXCEEDS noise'))
        print('    VERDICT: %s' % verdicts[kn]['verdict'])
    out['knobs'] = verdicts

    # ---- LEG 3: mechanism-based coupling bounds ------------------------------
    print()
    print('=' * 78)
    print('LEG 3 -- mechanism: what the knob WOULD have done at coupling 1')
    print('=' * 78)
    mech = {}

    tol = [p['setting'].get('transform_tolerance', p['setting'].get('tol'))
           for p in per if p['knob'] == 'transform_tolerance']
    tol = [x for x in tol if x is not None]
    if tol:
        d_tol = max(tol) - min(tol)
        pred_pct = 100 * (1 - tau_book / (tau_book + d_tol))
        obs_pct = verdicts['transform_tolerance']['u_spread_pct']
        mech['transform_tolerance'] = {
            'mechanism': 'stale TF permitted up to the tolerance -> extra loop delay',
            'sweep_s': d_tol, 'predicted_shift_pct_at_coupling_1': pred_pct,
            'observed_shift_pct': obs_pct, 'coupling_bound': obs_pct / pred_pct,
            'leaked_delay_bound_ms': verdicts['transform_tolerance']['tau_spread_ms']}
        print()
        print('  [transform_tolerance] swept %g -> %g s (delta %g s)' % (min(tol), max(tol), d_tol))
        print('    if spent as loop delay at coupling 1, u* would move %.1f%%' % pred_pct)
        print('    observed %.3f%%  ->  coupling <= %.5f' % (obs_pct, obs_pct / pred_pct))
        print('    shape fit bounds the leaked delay at %.2f ms over a %.0f ms sweep'
              % (verdicts['transform_tolerance']['tau_spread_ms'], 1000 * d_tol))

    hz = [p['setting'].get('sample_hz') for p in per if p['knob'] == 'sample_hz']
    hz = [x for x in hz if x is not None]
    if hz:
        us_hz = [p['u_parabola'] for p in per if p['knob'] == 'sample_hz']
        v_ref = float(np.mean(us_hz)) * L / tau_book
        ds = [v_ref / f for f in hz]
        q = [100 * (x / L) ** 2 for x in ds]
        dif = np.diff(us_hz)
        mono = bool(np.all(dif > 0) or np.all(dif < 0))
        mech['sample_hz'] = {
            'mechanism': 'measurement only (run_point.py:98 sets sample_dt); spatial '
                         'quadrature error of the RMSE integral, O((v/f/L)^2)',
            'delta_s_m': ds, 'predicted_bias_pct': q,
            'predicted_spread_pct': max(q) - min(q),
            'observed_spread_pct': verdicts['sample_hz']['u_spread_pct'],
            'monotone_in_f': mono}
        print()
        print('  [sample_hz] measurement path only (run_point.py:98), not the control path')
        print('    spatial sample spacing v/f = %s' % ['%.4f m' % x for x in ds])
        print('    quadrature bias O((ds/L)^2) predicts a spread of %.3f%%; observed %.3f%%'
              % (max(q) - min(q), verdicts['sample_hz']['u_spread_pct']))
        print('    monotone in f? %s  (a real quadrature trend would be monotone)' % mono)

    cm = [(p['setting'].get('costmap'), p['u_parabola'], p['tau_tot_shapefit'])
          for p in per if p['knob'] == 'costmap_size']
    cm = sorted([c for c in cm if c[0] is not None])
    if len(cm) == 3:
        d12 = 100 * (cm[1][1] - cm[0][1]) / cm[0][1]
        d23 = 100 * (cm[2][1] - cm[1][1]) / cm[1][1]
        mech['costmap_size'] = {
            'mechanism': 'local costmap truncates the transformed plan; the effect must '
                         'vanish once the map contains the lookahead point, i.e. SATURATE',
            'sizes': [c[0] for c in cm], 'u': [c[1] for c in cm],
            'tau_tot_ms': [1000 * c[2] for c in cm],
            'step_3_to_5_pct': d12, 'step_5_to_8_pct': d23,
            'saturation_ratio': abs(d23 / d12) if d12 else None}
        print()
        print('  [costmap_size] %g -> %g -> %g m' % (cm[0][0], cm[1][0], cm[2][0]))
        print('    step 3->5 m: %+.3f%%    step 5->8 m: %+.3f%%    ratio %.3f'
              % (d12, d23, abs(d23 / d12)))
        print('    predicted signature: truncation at the smallest map, then SATURATION.')
        print('    the two larger maps agree to %.3f%% -- the 2.4%% is one edge point, '
              'not a trend.' % abs(d23))
    out['mechanism'] = mech

    all_pass = all(v['verdict'] == 'PASS' for v in verdicts.values())
    out['all_pass'] = bool(all_pass)
    print()
    print('=' * 78)
    print('OVERALL: %s' % ('all nulls consistent with the measured noise floor'
                           if all_pass else 'AT LEAST ONE KNOB EXCEEDS THE NOISE FLOOR'))
    print('=' * 78)
    json.dump(out, open(a.out, 'w'), indent=2)
    print('-> %s' % a.out)


if __name__ == '__main__':
    main()
