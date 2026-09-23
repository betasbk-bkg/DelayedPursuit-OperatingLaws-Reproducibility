#!/usr/bin/env python3
"""
d97 -- where does nav2_graceful_controller sit, and what must its optimum be?

WRITTEN BEFORE THE SWEEP.  Every number this prints is a prediction.

THE LAW, FROM THE SOURCE.  smooth_control_law.cpp:114-122 computes the commanded
curvature from egocentric polar coordinates (r, phi, delta) of the motion target:

    kappa = -(1/r) [ k_delta (delta - atan(-k_phi phi))
                     + (1 + k_phi/(1 + (k_phi phi)^2)) sin(delta) ]

with r the distance to the target, delta the robot's heading relative to the
line of sight, and phi the TARGET's heading relative to that same line
(ego_polar_coords.hpp:60-70).

LINEARISE IN d38's VARIABLES.  x = arclength / L, e = lateral error / L,
psi = e' the heading error, g(x) the preview integral, and

    P(x) = Psi_p(x+1) - Psi_p(x)        the path's turn over one look-ahead,
                                        = +alpha on (x_c-1, x_c) for a corner:
                                        the look-ahead point is already past the
                                        corner while the robot is not.

    THE SIGN OF P IS LOAD-BEARING AND WAS WRONG IN THE FIRST DRAFT OF THIS FILE.
    This paragraph said -alpha, and P_turn() returned -1.0, until d38's own
    convention (a +alpha turn drops the heading error by alpha) settled it at
    +alpha.  The term B*P(x-u) is the whole mechanism the paper attributes the
    transition to, so the reader is owed the sensitivity: flipping the sign back
    sends ALL FOUR rows to EDGE_LOW, including the two the sweep measures with an
    optimum.  The sign is therefore not a detail, and it is fixed by the same
    convention that fixes d38's corner kick -- not by which answer looks better.

The target sits one look-ahead along the path, so with r = L,

    delta = e + psi - g,        phi = delta - psi + P = e - g + P

and, writing A = k_delta + 1 + k_phi and B = k_delta k_phi,

    kappa L = -A (e + e' - g) - B (e - g + P)

so the delayed transit obeys

    e'' = -(A+B) e(x-u) - A e'(x-u) + (A+B) g(x-u) - B P(x-u)          (*)

with the usual corner kick psi -> psi - alpha at x_c (kinematics, not control).

THE POINT.  Set A = 2, B = 0 and (*) becomes

    e'' = 2[-e(x-u) - e'(x-u) + g(x-u)]

which is verbatim d38's curvature-command plant, i.e. Nav2 RPP.  So RPP and the
graceful controller are TWO MEMBERS OF ONE FAMILY, indexed by (A, B), and the
second is not a new class but a different point in the same parameter space --
with gains three times RPP's (A = 6) plus a preview term RPP does not have
(B = 6).  That makes k_phi and k_delta, both exposed as ROS parameters, into a
knob that moves u* along a curve the theory can draw in advance.  A single
matching number could be luck; a matching CURVE cannot.

    characteristic equation:   lam^2 + exp(-lam u) (A lam + (A+B)) = 0

PREDICTIONS (printed below, before any Nav2 run):
    u*(A, B) from the transit-energy argmin, and u_c(A, B) from the stability
    boundary, for the shipped defaults and for the gain grid the sweep will use.
    Higher gain acts sooner on the delayed error, so u* must FALL as A rises.

Usage:  python3 d97_graceful_class_2026-09-08.py
"""
import importlib.util
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

_s = importlib.util.spec_from_file_location(
    'd38', HERE / 'd38_corner_energy_two_plants_2026-08-29.py')
d38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d38)

_s2 = importlib.util.spec_from_file_location(
    'd84', HERE / 'd84_stadium_sweep_2026-09-08.py')
d84 = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(d84)


def AB(k_phi, k_delta):
    return k_delta + 1.0 + k_phi, k_delta * k_phi


def P_turn(x, xc):
    """Path turn over the next look-ahead, P(x) = Psi_p(x+1) - Psi_p(x).

    For x in (xc-1, xc) the look-ahead point is already PAST the corner while
    the robot is not, so P = +alpha there and 0 elsewhere.  (First written with
    the opposite sign; d38's own convention -- heading error jumps by -alpha
    when the path turns by +alpha -- fixes it.)
    """
    return 1.0 if (xc - 1.0) < x < xc else 0.0


def transit_energy_AB(u, A, B, xc=4.0, xend=60.0, h=1.0e-3):
    """J(u) = int e^2 dx for equation (*).  Method of steps, exact corner jump."""
    n = int(xend / h)
    nd = int(round(u / h))
    e = np.zeros(n + 1)
    ep = np.zeros(n + 1)
    ic = int(round(xc / h))
    for i in range(n):
        x = i * h
        j = i - nd
        ed = e[j] if j >= 0 else 0.0
        epd = ep[j] if j >= 0 else 0.0
        xd = x - u
        gd = d38.g_preview(xd, xc) if xd > 0 else 0.0
        pd = P_turn(xd, xc) if xd > 0 else 0.0
        epp = -(A + B) * ed - A * epd + (A + B) * gd - B * pd
        ep[i + 1] = ep[i] + h * epp
        if i == ic:
            ep[i + 1] -= 1.0
        e[i + 1] = e[i] + h * ep[i + 1]
    return float(np.sum(e * e) * h)


def self_test():
    """A = 2, B = 0 must reproduce d38's curvature-command plant exactly."""
    worst = 0.0
    for u in (0.15, 0.25, 0.306, 0.40):
        a = d38.transit_energy(u, 'curvature_command', h=1.0e-3)
        b = transit_energy_AB(u, 2.0, 0.0, h=1.0e-3)
        worst = max(worst, abs(b / a - 1.0))
    return worst


def u_critical(A, B, lo=0.05, hi=1.60):
    """Largest u with every root of lam^2 + e^{-lam u}(A lam + A+B) in Re < 0.

    On the boundary lam = i w:  -w^2 + (cos wu - i sin wu)(i A w + A+B) = 0.
    Real:  -w^2 + A w sin(wu) + (A+B) cos(wu) = 0
    Imag:  A w cos(wu) - (A+B) sin(wu) = 0   ->  tan(wu) = A w / (A+B)
    """
    best = None
    for w in np.linspace(1e-3, 60.0, 200000):
        t = math.atan2(A * w, (A + B))          # principal branch, wu in (0, pi)
        u = t / w
        if not (lo <= u <= hi):
            continue
        r = -w * w + A * w * math.sin(t) + (A + B) * math.cos(t)
        if abs(r) < 1e-3 * max(1.0, w * w):
            if best is None or u < best[0]:
                best = (u, w, r)
    return best


def locate(f, us, js):
    """argmin of a SMOOTH curve: decide EDGE first, then refine locally.

    d84's bowl estimator was built for measured RMSE -- noisy, shallow, sampled
    coarsely -- and it fits every point within factor x min.  On an analytic
    J(u) two things go wrong: it returns an interior value even for a MONOTONE
    curve (it did: 0.0385 for a shipped-default curve that rises from its first
    point), and on RPP the wide asymmetric window pulled the anchor to 0.2706
    against d38's 0.3060.  So: boundary test first, and where the minimum is
    genuinely interior, refine on a narrow bracket where a parabola is exact to
    the integrator.  The acceptance test is that A=2, B=0 returns d38's 0.3060.
    """
    i = int(np.argmin(js))
    if i == 0:
        return float(us[0]), 'EDGE_LOW'
    if i == len(js) - 1:
        return float(us[-1]), 'EDGE_HIGH'
    a, b = us[i - 1], us[i + 1]
    for _ in range(3):                       # three local refinements
        fine = np.linspace(a, b, 21)
        fj = [f(u) for u in fine]
        k = int(np.argmin(fj))
        k = min(max(k, 1), len(fine) - 2)
        a, b = fine[k - 1], fine[k + 1]
    c = np.polyfit(fine[k - 1:k + 2], fj[k - 1:k + 2], 2)
    v = -c[1] / (2 * c[0]) if c[0] > 0 else fine[k]
    return float(v), 'interior'


def u_star(A, B, lo=0.004, hi=None, n=121, h=1.0e-3):
    uc = u_critical(A, B)
    hi = hi if hi is not None else (0.92 * uc[0] if uc else 0.60)
    us = np.linspace(lo, hi, n)
    f = lambda u: transit_energy_AB(u, A, B, h=h)
    js = [f(u) for u in us]
    v, how = locate(f, us, js)
    return v, how, uc, us, js


def main():
    print('=' * 78)
    print('d97 -- graceful controller: the family (A, B), predicted before data')
    print('=' * 78)
    w = self_test()
    print('  self-test  A=2, B=0 vs d38 curvature-command: max rel. error %.2e' % w)
    assert w < 5e-3, 'the (A,B) integrator does not reduce to d38'
    print('  -> RPP is the (A, B) = (2, 0) member.  Same equation, not an analogy.')
    ur0, how0, uc0, _a, _b = u_star(2.0, 0.0)
    print('  self-test  locator on (2, 0): u* %.4f, u_c %.4f  [d38: 0.3060, 0.5205]'
          % (ur0, uc0[0]))
    assert abs(ur0 / 0.3060 - 1.0) < 0.02, 'locator does not reproduce d38 u*'
    assert abs(uc0[0] / 0.520494 - 1.0) < 0.01, 'u_c solver disagrees with d38'
    print()

    # A=6,B=0 is not a reachable parameter setting; it is the DECOMPOSITION
    # row -- graceful's gain with RPP's absent preview -- and separates "the
    # gain moved u*" from "the extra tangent feedforward moved u*".
    grid = [('shipped default', 3.0, 2.0)]
    for kp in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        grid.append(('k_phi %.1f' % kp, kp, 2.0))
    for kd in (0.5, 1.0, 4.0, 6.0):
        grid.append(('k_delta %.1f' % kd, 3.0, kd))

    print('  %-16s %6s %6s %6s %6s %9s %9s %9s'
          % ('case', 'k_phi', 'k_del', 'A', 'B', 'u_c', 'u*', 'how'))
    out = {'self_test': w, 'rows': []}
    seen = set()
    for name, kp, kd in grid:
        A, B = AB(kp, kd)
        if (A, B) in seen:
            continue
        seen.add((A, B))
        us_, how, uc, ugrid, jgrid = u_star(A, B)
        out['rows'].append({'case': name, 'k_phi': kp, 'k_delta': kd,
                            'A': A, 'B': B, 'u_star': us_, 'how': how,
                            'u_c': (uc[0] if uc else None),
                            'u_grid': [float(x) for x in ugrid],
                            'J': [float(x) for x in jgrid]})
        print('  %-16s %6.1f %6.1f %6.1f %6.1f %9s %9.4f %9s'
              % (name, kp, kd, A, B,
                 ('%.4f' % uc[0]) if uc else '-', us_, how))

    # decomposition: same A as shipped, no preview term
    ad, hd, ucd, _ug, _jg = u_star(6.0, 0.0)
    out['decomposition_A6_B0'] = {'u_star': ad, 'how': hd,
                                  'u_c': (ucd[0] if ucd else None)}
    print()
    print('  decomposition (A=6, B=0; gain only, no tangent feedforward):')
    print('    u_c %s  u* %.4f  [%s]'
          % (('%.4f' % ucd[0]) if ucd else '-', ad, hd))

    # the RPP anchor, for scale
    ur, howr, ucr, _ug2, _jg2 = u_star(2.0, 0.0)
    print()
    print('  RPP anchor (A=2, B=0): u_c %.4f  u* %.4f   [published 0.3060 / 0.520494]'
          % (ucr[0] if ucr else float('nan'), ur))
    out['rpp_anchor'] = {'u_star': ur, 'u_c': (ucr[0] if ucr else None)}

    d = [r for r in out['rows'] if r['case'].startswith('k_phi')]
    if len(d) >= 3:
        a = np.array([r['A'] for r in d])
        s = np.array([r['u_star'] for r in d])
        sl = float(np.polyfit(np.log(a), np.log(s), 1)[0])
        print('  along the k_phi leg: u* ~ A^%+.3f  (a gain acting sooner must give < 0)'
              % sl)
        out['kphi_slope'] = sl

    sh = [r for r in out['rows'] if r['case'] == 'shipped default'][0]
    print()
    print('  J(u) for the shipped defaults -- an EDGE verdict has to be visible:')
    ug, jg = sh['u_grid'], sh['J']
    idx = [0] + [int(round(f * (len(ug) - 1))) for f in
                 (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0)]
    print('    %s' % '  '.join('u=%.3f' % ug[i] for i in idx))
    print('    %s' % '  '.join('%7.4f' % (jg[i] / min(jg)) for i in idx))
    print('    (values are J(u)/min J; monotone rise means no interior optimum)')
    print()
    print('  PRE-REGISTERED for the shipped defaults (k_phi 3.0, k_delta 2.0):')
    print('    u* = %.4f   against RPP\'s %.4f  (%.0f%% lower)'
          % (sh['u_star'], ur, 100 * (1 - sh['u_star'] / ur)))
    print('    stability limit u_c = %.4f' % sh['u_c'])
    print('    The sweep must (i) settle whether an interior optimum exists at')
    print('    all for this member -- an EDGE above means the calculation says it')
    print('    does NOT, which is itself the prediction -- (ii) if one exists,')
    print('    place it near this value, and (iii) move it along the k_phi leg')
    print('    with the sign printed above.')

    json.dump(out, open(HERE / 'd97_graceful_class_2026-09-08.json', 'w'), indent=2)
    print('\n-> d97_graceful_class_2026-09-08.json')


if __name__ == '__main__':
    main()
