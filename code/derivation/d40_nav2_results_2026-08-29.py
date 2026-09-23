#!/usr/bin/env python3
"""
d40 -- Analysis of the Nav2 sweep: does Theorem 2 hold on the deployed controller?

INPUT   the sweep produced by letg2_nav2_validation/sweep.py on the real ROS2 Jazzy
        stack (nav2_regulated_pure_pursuit_controller + nav2_loopback_sim), i.e.
        /tmp/letg2/sweep.json inside WSL, copied here as nav2_sweep_<date>.json
OUTPUT  d40_results_2026-08-29.json

This deliberately mirrors d37_theorem2_on_rpp_2026-08-29.py so the Nav2 numbers can be
put beside the Python pre-test numbers without re-deriving anything.  d37 asked, of a
transcribed RPP, which bookkeeping of tau_tot makes the dimensionless optimum

        u* = v* * tau_tot / L

constant as tau_inject is swept.  It found:

        (a) no additivity      tau_tot = tau_inject                 spread 35.1 %
        (b) Theorem 2, 0 free  tau_tot = tau_inject + 0.030         spread  6.5 %
        (c) one fitted delay   tau_tot = tau_inject + 0.0253        spread  1.7 %

d40 asks the same question of the ACTUAL Nav2 stack.  The claim under test is NOT the
value of the constant -- that is plant-specific (0.306 derived in d38 for a
curvature-command plant, against 1/2 for the heading-servo plant of Theorem 1).  The
claim is the INVARIANCE: that delays compose by their first moments, so that one
bookkeeping collapses the spread and the others do not.

v* is located per tau by a parabola through the three lowest RMSE points, which is the
same estimator d36/d37 used.  Points where the minimum sits on the edge of the swept
range are reported as EDGE and excluded from the invariance statistics, because a
boundary minimum is not a located optimum.
"""
import argparse
import json
import math
import os

L = 0.6
T_CTRL_HALF = 0.025
T_SIM_HALF = 0.005
T_COMPOSITE = T_CTRL_HALF + T_SIM_HALF          # 0.030 s, a priori, zero free parameters
U_PRETEST = 0.3143                              # d36, square alpha=90 deg
U_DERIVED = 0.3060                              # d38, argmin of the corner-transit energy


def parabola_min(xs, ys):
    """Vertex of the parabola through three points; None if not a genuine minimum."""
    (x0, y0), (x1, y1), (x2, y2) = zip(xs, ys)
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-15:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if a <= 0:
        return None
    return -b / (2 * a)


def locate_optimum(points):
    """points: list of (v, rmse) sorted by v.  Returns (v_star, status)."""
    pts = sorted((p for p in points if p[1] == p[1]), key=lambda p: p[0])   # drop NaN
    if len(pts) < 3:
        return None, 'too-few-points'
    vs = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    i = min(range(len(rs)), key=lambda k: rs[k])
    if i == 0 or i == len(rs) - 1:
        return vs[i], 'EDGE'
    v = parabola_min(vs[i - 1:i + 2], rs[i - 1:i + 2])
    if v is None or not (vs[i - 1] <= v <= vs[i + 1]):
        return vs[i], 'grid'
    return v, 'parabola'


def spread(us):
    return 100.0 * (max(us) - min(us)) / (sum(us) / len(us))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='nav2_sweep_2026-08-29.json')
    ap.add_argument('--out', default='d40_results_2026-08-29.json')
    a = ap.parse_args()

    if not os.path.exists(a.sweep):
        raise SystemExit(f'sweep file not found: {a.sweep}\n'
                         f'copy it out of WSL first, e.g.\n'
                         f"  wsl -e bash -lc 'cat /tmp/letg2/sweep.json' > {a.sweep}")
    with open(a.sweep) as f:
        raw = json.load(f)
    rows = raw['results'] if isinstance(raw, dict) else raw

    by_tau = {}
    for r in rows:
        if r.get('rmse_m') is None:
            continue
        by_tau.setdefault(round(float(r['tau_inject']), 6), []).append(
            (float(r['desired_vel']), float(r['rmse_m'])))

    per_tau, usable = [], []
    print(f'{"tau":>7} {"n":>3} {"v* (m/s)":>9} {"how":>9} '
          f'{"u* (Thm2)":>10} {"u* (naive)":>11}')
    for tau in sorted(by_tau):
        pts = by_tau[tau]
        v, how = locate_optimum(pts)
        if v is None:
            print(f'{tau:7.3f} {len(pts):3d}    -- {how}')
            continue
        u_thm2 = v * (tau + T_COMPOSITE) / L
        u_naive = v * tau / L
        rec = {'tau_inject': tau, 'n_points': len(pts), 'v_star': v, 'how': how,
               'u_theorem2': u_thm2, 'u_no_additivity': u_naive,
               'rmse_min': min(p[1] for p in pts),
               'curve': sorted(pts)}
        per_tau.append(rec)
        if how != 'EDGE':
            usable.append(rec)
        print(f'{tau:7.3f} {len(pts):3d} {v:9.4f} {how:>9} {u_thm2:10.4f} {u_naive:11.4f}')

    out = {'source': os.path.basename(a.sweep), 'L': L,
           'T_composite_apriori': T_COMPOSITE,
           'dataset_caveat': (
               'nav2_sweep_tau_2026-08-29.json was measured BEFORE two harness fixes '
               '(NAV2_AUDIT_TRIAGE §6): (i) tail_exclude_m=2.0 < EXIT_M=3.0, so 1.0 m of '
               'the straight exit stub lay inside every measurement window -- common-mode '
               'across points (index-based window), first-order irrelevant to the located '
               'minimum, RMSE ~0.7% low; (ii) laps varied within the tau=0.6 row at its two '
               'fastest points (lap-invariance verified at 0.42%). Treat as PRELIMINARY; a '
               'clean tau re-run follows the E2-E6 blocks.'),
           'u_pretest_d36': U_PRETEST, 'u_derived_d38': U_DERIVED,
           'per_tau': per_tau, 'n_usable': len(usable)}

    if len(usable) >= 2:
        vstars = [r['v_star'] for r in usable]
        taus = [r['tau_inject'] for r in usable]

        u_naive = [v * t / L for v, t in zip(vstars, taus)]
        u_thm2 = [v * (t + T_COMPOSITE) / L for v, t in zip(vstars, taus)]

        # one fitted composite delay: choose c to flatten u*, same estimator as d37
        best_c, best_s = 0.0, float('inf')
        c = 0.0
        while c <= 0.120001:
            us = [v * (t + c) / L for v, t in zip(vstars, taus)]
            s = spread(us)
            if s < best_s:
                best_s, best_c = s, c
            c += 0.0002
        u_fit = [v * (t + best_c) / L for v, t in zip(vstars, taus)]

        # (d) audit-corrected a-priori composite.  The experiment audit (2026-08-29)
        # identified two first-moment terms the pre-registration omitted, both in
        # the real stack and absent from the Python pre-test:
        #   - the controller reads a pose that is up to one plant tick old
        #     (input-side staleness): a SECOND T_sim/2 = 0.005 s
        #   - the delay node's release timer quantises release times: measured
        #     +0.8..1.4 ms excess over the requested tau, ~ T_release/2 = 0.001 s
        # Together: 0.025 + 0.005 + 0.005 + 0.001 = 0.036 s.  Still zero FITTED
        # parameters -- every term is a documented period -- but registered
        # AFTER the tau sweep was run, so it is a post-hoc bookkeeping, and is
        # labelled as such.  The E2/E3 blocks test the second T_sim/2 directly.
        T_AUDIT = T_CTRL_HALF + T_SIM_HALF + T_SIM_HALF + 0.001
        u_audit = [v * (t + T_AUDIT) / L for v, t in zip(vstars, taus)]

        out['bookkeeping'] = {
            'a_no_additivity': {'c_s': 0.0, 'u_star': u_naive,
                                'mean': sum(u_naive) / len(u_naive),
                                'spread_pct': spread(u_naive)},
            'b_theorem2_zero_free_params': {'c_s': T_COMPOSITE, 'u_star': u_thm2,
                                            'mean': sum(u_thm2) / len(u_thm2),
                                            'spread_pct': spread(u_thm2)},
            'c_one_fitted_constant': {'c_s': best_c, 'u_star': u_fit,
                                      'mean': sum(u_fit) / len(u_fit),
                                      'spread_pct': best_s},
            'd_audit_corrected_apriori_POSTHOC': {'c_s': T_AUDIT, 'u_star': u_audit,
                                                  'mean': sum(u_audit) / len(u_audit),
                                                  'spread_pct': spread(u_audit)},
        }
        out['tau_range_factor'] = max(taus) / min(taus)
        mean_thm2 = sum(u_thm2) / len(u_thm2)
        out['vs_pretest_pct'] = 100.0 * (mean_thm2 / U_PRETEST - 1)
        out['vs_derived_pct'] = 100.0 * (mean_thm2 / U_DERIVED - 1)

        print()
        print(f'tau range {min(taus):.3f}-{max(taus):.3f} s '
              f'({out["tau_range_factor"]:.0f}x), {len(usable)} located optima')
        for k, v in out['bookkeeping'].items():
            print(f'  {k:32s} c={v["c_s"]:.4f}  mean u*={v["mean"]:.4f}  '
                  f'spread={v["spread_pct"]:.1f}%')
        print(f'  mean u* (Theorem 2) vs pre-test {U_PRETEST}: {out["vs_pretest_pct"]:+.1f}%')
        print(f'  mean u* (Theorem 2) vs derived  {U_DERIVED}: {out["vs_derived_pct"]:+.1f}%')
    else:
        print('\nnot enough located optima for the invariance statistics '
              f'({len(usable)} usable; edge minima are excluded by design)')

    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
